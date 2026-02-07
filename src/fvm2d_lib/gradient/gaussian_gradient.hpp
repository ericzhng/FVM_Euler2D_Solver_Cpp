#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "mesh/partition_mesh.hpp"
#include "limiter/limiters.hpp"

namespace fvm2d {

/**
 * @brief Compute cell-centered gradients using Gaussian method
 *
 * Uses surface integral: grad(U)_i = (1/V_i) * sum_faces(U_face * n * A)
 *
 * @param mesh Partition mesh
 * @param U Solution state array (num_cells x num_vars)
 * @param over_relaxation Over-relaxation factor for non-orthogonal correction
 * @return Gradient array (num_cells x num_vars*2)
 */
FVM_API GradientArray compute_gradients_gaussian(
    const PartitionMesh& mesh,
    const StateArray& U,
    Scalar over_relaxation = 1.0
);

/**
 * @brief Compute slope limiters for each cell
 *
 * Ensures monotonicity by limiting the reconstructed values
 *
 * @param mesh Partition mesh
 * @param U Solution state array
 * @param gradients Cell gradients
 * @param limiter_type Type of limiter to use
 * @return Limiter array (num_cells x num_vars)
 */
FVM_API LimiterArray compute_limiters(
    const PartitionMesh& mesh,
    const StateArray& U,
    const GradientArray& gradients,
    LimiterType limiter_type
);

}  // namespace fvm2d
