#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "mesh/partition_mesh.hpp"
#include "physics/physics_model.hpp"
#include <mpi.h>

namespace fvm2d {

/**
 * @brief Calculate adaptive time step based on CFL condition
 *
 * dt = CFL * min_all_cells(sqrt(V_i) / max_eigenvalue_i)
 *
 * @param mesh Partition mesh
 * @param U Solution state
 * @param physics Physics model
 * @param cfl CFL number
 * @param comm MPI communicator for global reduction
 * @return Stable time step
 */
FVM_API Scalar calculate_adaptive_dt(
    const PartitionMesh& mesh,
    const StateArray& U,
    const PhysicsModel& physics,
    Scalar cfl,
    MPI_Comm comm
);

/**
 * @brief Calculate local (per-rank) time step without MPI reduction
 */
FVM_API Scalar calculate_local_dt(
    const PartitionMesh& mesh,
    const StateArray& U,
    const PhysicsModel& physics,
    Scalar cfl
);

}  // namespace fvm2d
