#include "gradient/gaussian_gradient.hpp"
#include <cmath>

#ifdef FVM2D_USE_OPENMP
#include <omp.h>
#endif

namespace fvm2d {

GradientArray compute_gradients_gaussian(
    const PartitionMesh& mesh,
    const StateArray& U,
    Scalar over_relaxation
) {
    const Index num_cells = mesh.total_cells();
    const int num_vars = static_cast<int>(U.cols());

    // Allocate gradient array: (num_cells x num_vars*2)
    // Layout: [dU0_dx, dU0_dy, dU1_dx, dU1_dy, ...]
    GradientArray gradients = GradientArray::Zero(num_cells, num_vars * 2);

    #pragma omp parallel for schedule(static)
    for (Index i = 0; i < num_cells; ++i) {
        // Temporary storage for gradient sums
        Eigen::MatrixXd grad_sum = Eigen::MatrixXd::Zero(num_vars, 2);

        const Vector2d& centroid_i = mesh.cell_centroids[i];
        const int num_faces = mesh.connectivity.num_faces[i];

        for (int j = 0; j < num_faces; ++j) {
            const Index neighbor_idx = mesh.connectivity.neighbor(i, j);
            const Vector2d face_normal = mesh.connectivity.normal(i, j);
            const Scalar face_area = mesh.connectivity.area(i, j);
            const Vector2d face_midpoint = mesh.connectivity.midpoint(i, j);

            // Compute face value
            VectorXd U_face(num_vars);

            if (neighbor_idx >= 0) {
                // Interior face: weighted interpolation
                Scalar d_i = mesh.connectivity.distance(i, j);
                Scalar d_j = (face_midpoint - mesh.cell_centroids[neighbor_idx]).norm();

                if (d_i + d_j > EPSILON) {
                    Scalar w_i = d_j / (d_i + d_j);
                    Scalar w_j = d_i / (d_i + d_j);
                    U_face = w_i * U.row(i).transpose() + w_j * U.row(neighbor_idx).transpose();
                } else {
                    U_face = 0.5 * (U.row(i).transpose() + U.row(neighbor_idx).transpose());
                }

                // Non-orthogonal correction
                Vector2d d = mesh.cell_centroids[neighbor_idx] - centroid_i;
                Scalar d_norm = d.norm();
                if (d_norm > EPSILON) {
                    Vector2d e = d / d_norm;
                    Vector2d k = face_normal.normalized();
                    Scalar dot_dk = d.dot(k);

                    if (std::abs(dot_dk) > EPSILON) {
                        Vector2d correction_vector = (e - k * e.dot(k)) / dot_dk;
                        for (int m = 0; m < num_vars; ++m) {
                            Scalar non_orth_correction =
                                (U(neighbor_idx, m) - U(i, m)) * correction_vector.dot(d);
                            U_face(m) += over_relaxation * non_orth_correction;
                        }
                    }
                }
            } else {
                // Boundary face: use cell value
                U_face = U.row(i).transpose();
            }

            // Accumulate gradient contribution
            for (int m = 0; m < num_vars; ++m) {
                grad_sum(m, 0) += U_face(m) * face_normal.x() * face_area;
                grad_sum(m, 1) += U_face(m) * face_normal.y() * face_area;
            }
        }

        // Divide by cell volume
        if (mesh.cell_volumes[i] > EPSILON) {
            for (int m = 0; m < num_vars; ++m) {
                gradients(i, m * 2 + 0) = grad_sum(m, 0) / mesh.cell_volumes[i];
                gradients(i, m * 2 + 1) = grad_sum(m, 1) / mesh.cell_volumes[i];
            }
        }
    }

    return gradients;
}

LimiterArray compute_limiters(
    const PartitionMesh& mesh,
    const StateArray& U,
    const GradientArray& gradients,
    LimiterType limiter_type
) {
    if (limiter_type == LimiterType::None) {
        return LimiterArray::Ones(mesh.total_cells(), U.cols());
    }

    const Index num_cells = mesh.total_cells();
    const int num_vars = static_cast<int>(U.cols());

    LimiterArray limiters = LimiterArray::Ones(num_cells, num_vars);

    #pragma omp parallel for schedule(static)
    for (Index i = 0; i < num_cells; ++i) {
        const VectorXd U_i = U.row(i).transpose();
        const Vector2d& centroid_i = mesh.cell_centroids[i];
        const int num_faces = mesh.connectivity.num_faces[i];

        // Find min/max of neighbors
        VectorXd U_max = U_i;
        VectorXd U_min = U_i;

        for (int j = 0; j < num_faces; ++j) {
            Index neighbor_idx = mesh.connectivity.neighbor(i, j);
            if (neighbor_idx >= 0) {
                for (int m = 0; m < num_vars; ++m) {
                    U_max(m) = std::max(U_max(m), U(neighbor_idx, m));
                    U_min(m) = std::min(U_min(m), U(neighbor_idx, m));
                }
            }
        }

        // Compute limiter for each face
        for (int j = 0; j < num_faces; ++j) {
            Vector2d face_midpoint = mesh.connectivity.midpoint(i, j);
            Vector2d r_if = face_midpoint - centroid_i;

            for (int m = 0; m < num_vars; ++m) {
                // Extrapolated value at face
                Scalar grad_x = gradients(i, m * 2 + 0);
                Scalar grad_y = gradients(i, m * 2 + 1);
                Scalar U_face_extrap = U_i(m) + grad_x * r_if.x() + grad_y * r_if.y();

                Scalar diff = U_face_extrap - U_i(m);

                if (std::abs(diff) > EPSILON) {
                    Scalar r;
                    if (diff > 0.0) {
                        r = (U_max(m) - U_i(m)) / diff;
                    } else {
                        r = (U_min(m) - U_i(m)) / diff;
                    }

                    Scalar phi = apply_limiter(limiter_type, r);
                    limiters(i, m) = std::min(limiters(i, m), phi);
                }
            }
        }
    }

    return limiters;
}

}  // namespace fvm2d
