#pragma once

#include "fvm2d/core/types.hpp"
#include "fvm2d/mesh/partition_mesh.hpp"
#include "fvm2d/limiter/limiters.hpp"

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
GradientArray compute_gradients_gaussian(
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
LimiterArray compute_limiters(
    const PartitionMesh& mesh,
    const StateArray& U,
    const GradientArray& gradients,
    LimiterType limiter_type
);

}  // namespace fvm2d
