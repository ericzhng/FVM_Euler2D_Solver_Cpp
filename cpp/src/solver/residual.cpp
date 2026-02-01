#include "fvm2d/solver/residual.hpp"

#ifdef FVM2D_USE_OPENMP
#include <omp.h>
#endif

namespace fvm2d {

VectorXd compute_numerical_flux(
    const VectorXd& U_L,
    const VectorXd& U_R,
    const Vector2d& normal,
    const PhysicsModel& physics,
    FluxType flux_type
) {
    switch (flux_type) {
        case FluxType::Roe:
            return physics.roe_flux(U_L, U_R, normal);

        case FluxType::HLLC:
            return physics.hllc_flux(U_L, U_R, normal);

        case FluxType::CentralDifference: {
            VectorXd F_L = physics.compute_flux(U_L, normal);
            VectorXd F_R = physics.compute_flux(U_R, normal);
            return 0.5 * (F_L + F_R);
        }

        default:
            return physics.hllc_flux(U_L, U_R, normal);
    }
}

VectorXd reconstruct_state(
    Index i,
    int face_idx,
    const StateArray& U,
    const GradientArray& gradients,
    const LimiterArray& limiters,
    const PartitionMesh& mesh
) {
    const int num_vars = static_cast<int>(U.cols());
    const Vector2d& centroid = mesh.cell_centroids[i];
    const Vector2d face_midpoint = mesh.connectivity.midpoint(i, face_idx);

    // Vector from cell centroid to face midpoint
    Vector2d r = face_midpoint - centroid;

    VectorXd U_reconstructed(num_vars);

    for (int m = 0; m < num_vars; ++m) {
        Scalar grad_x = gradients(i, m * 2 + 0);
        Scalar grad_y = gradients(i, m * 2 + 1);
        Scalar delta_U = grad_x * r.x() + grad_y * r.y();

        U_reconstructed(m) = U(i, m) + limiters(i, m) * delta_U;
    }

    return U_reconstructed;
}

StateArray compute_residual(
    const PartitionMesh& mesh,
    const StateArray& U,
    const GradientArray& gradients,
    const LimiterArray& limiters,
    const BoundaryConditionLookup& bcs,
    const PhysicsModel& physics,
    const SolverConfig& config
) {
    const Index num_owned = mesh.num_owned_cells;
    const int num_vars = static_cast<int>(U.cols());
    const FluxType flux_type = config.spatial.flux_type;

    StateArray residual = StateArray::Zero(num_owned, num_vars);

    #pragma omp parallel for schedule(dynamic, 64)
    for (Index i = 0; i < num_owned; ++i) {
        VectorXd flux_sum = VectorXd::Zero(num_vars);
        const int num_faces = mesh.connectivity.num_faces[i];

        for (int j = 0; j < num_faces; ++j) {
            // Get face data
            const Vector2d normal = mesh.connectivity.normal(i, j);
            const Scalar face_area = mesh.connectivity.area(i, j);
            const Index neighbor_idx = mesh.connectivity.neighbor(i, j);

            // Reconstruct left state
            VectorXd U_L = reconstruct_state(i, j, U, gradients, limiters, mesh);

            // Determine right state
            VectorXd U_R;

            if (neighbor_idx >= 0) {
                // Interior face: find the reverse face index in neighbor cell
                // For simplicity, we reconstruct from neighbor cell centroid to face
                const Vector2d& centroid_j = mesh.cell_centroids[neighbor_idx];
                const Vector2d face_midpoint = mesh.connectivity.midpoint(i, j);
                Vector2d r_j = face_midpoint - centroid_j;

                U_R.resize(num_vars);
                for (int m = 0; m < num_vars; ++m) {
                    Scalar grad_x = gradients(neighbor_idx, m * 2 + 0);
                    Scalar grad_y = gradients(neighbor_idx, m * 2 + 1);
                    Scalar delta_U = grad_x * r_j.x() + grad_y * r_j.y();
                    U_R(m) = U(neighbor_idx, m) + limiters(neighbor_idx, m) * delta_U;
                }
            } else {
                // Boundary face: apply boundary condition
                int tag = mesh.connectivity.tag(i, j);
                U_R = bcs.apply(tag, U_L, normal, physics);
            }

            // Compute numerical flux
            VectorXd flux = compute_numerical_flux(U_L, U_R, normal, physics, flux_type);

            // Accumulate
            flux_sum += flux * face_area;
        }

        // Divide by cell volume
        if (mesh.cell_volumes[i] > EPSILON) {
            residual.row(i) = flux_sum.transpose() / mesh.cell_volumes[i];
        }
    }

    return residual;
}

}  // namespace fvm2d
