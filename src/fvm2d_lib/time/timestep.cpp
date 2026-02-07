#include "time/timestep.hpp"
#include <cmath>
#include <limits>

#ifdef FVM2D_USE_OPENMP
#include <omp.h>
#endif

namespace fvm2d {

Scalar calculate_local_dt(
    const PartitionMesh& mesh,
    const StateArray& U,
    const PhysicsModel& physics,
    Scalar cfl
) {
    const Index num_owned = mesh.num_owned_cells;
    Scalar global_min_dt = std::numeric_limits<Scalar>::max();

#ifdef FVM2D_USE_OPENMP
    #pragma omp parallel
    {
        Scalar local_min_dt = std::numeric_limits<Scalar>::max();

        #pragma omp for nowait
        for (Index i = 0; i < num_owned; ++i) {
            // Get cell state
            VectorXd U_i = U.row(i).transpose();

            // Compute maximum eigenvalue (wave speed)
            Scalar max_lambda = physics.max_eigenvalue(U_i);

            // Characteristic length scale (sqrt of area for 2D)
            Scalar h = std::sqrt(mesh.cell_volumes[i]);

            // Local time step
            if (max_lambda > EPSILON) {
                Scalar local_dt = cfl * h / max_lambda;
                local_min_dt = std::min(local_min_dt, local_dt);
            }
        }

        #pragma omp critical
        {
            global_min_dt = std::min(global_min_dt, local_min_dt);
        }
    }
#else
    for (Index i = 0; i < num_owned; ++i) {
        // Get cell state
        VectorXd U_i = U.row(i).transpose();

        // Compute maximum eigenvalue (wave speed)
        Scalar max_lambda = physics.max_eigenvalue(U_i);

        // Characteristic length scale (sqrt of area for 2D)
        Scalar h = std::sqrt(mesh.cell_volumes[i]);

        // Local time step
        if (max_lambda > EPSILON) {
            Scalar local_dt = cfl * h / max_lambda;
            global_min_dt = std::min(global_min_dt, local_dt);
        }
    }
#endif

    return global_min_dt;
}

Scalar calculate_adaptive_dt(
    const PartitionMesh& mesh,
    const StateArray& U,
    const PhysicsModel& physics,
    Scalar cfl,
    MPI_Comm comm
) {
    // Calculate local minimum time step
    Scalar local_dt = calculate_local_dt(mesh, U, physics, cfl);

    // Global reduction to get minimum across all ranks
    Scalar global_dt;
    MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, comm);

    return global_dt;
}

}  // namespace fvm2d
