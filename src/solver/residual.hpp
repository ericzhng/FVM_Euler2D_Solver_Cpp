#pragma once

#include "core/types.hpp"
#include "core/config.hpp"
#include "mesh/partition_mesh.hpp"
#include "physics/physics_model.hpp"
#include "boundary/boundary_condition.hpp"

namespace fvm2d {

/**
 * @brief Compute FVM residual using MUSCL-Hancock reconstruction
 *
 * The residual is: R_i = (1/V_i) * sum_faces(F_numerical * A_face)
 *
 * @param mesh Partition mesh
 * @param U Solution state array
 * @param gradients Cell gradients
 * @param limiters Cell limiters
 * @param bcs Boundary condition lookup
 * @param physics Physics model
 * @param config Solver configuration
 * @return Residual array (num_owned_cells x num_vars)
 */
StateArray compute_residual(
    const PartitionMesh& mesh,
    const StateArray& U,
    const GradientArray& gradients,
    const LimiterArray& limiters,
    const BoundaryConditionLookup& bcs,
    const PhysicsModel& physics,
    const SolverConfig& config
);

/**
 * @brief Compute numerical flux between left and right states
 *
 * @param U_L Left state
 * @param U_R Right state
 * @param normal Face normal
 * @param physics Physics model
 * @param flux_type Type of flux scheme
 * @return Numerical flux
 */
VectorXd compute_numerical_flux(
    const VectorXd& U_L,
    const VectorXd& U_R,
    const Vector2d& normal,
    const PhysicsModel& physics,
    FluxType flux_type
);

/**
 * @brief Reconstruct state at face using MUSCL scheme
 *
 * @param i Cell index
 * @param face_idx Face index
 * @param U Solution array
 * @param gradients Gradient array
 * @param limiters Limiter array
 * @param mesh Partition mesh
 * @return Reconstructed state at face
 */
VectorXd reconstruct_state(
    Index i,
    int face_idx,
    const StateArray& U,
    const GradientArray& gradients,
    const LimiterArray& limiters,
    const PartitionMesh& mesh
);

}  // namespace fvm2d
