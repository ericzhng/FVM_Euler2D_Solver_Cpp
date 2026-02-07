#pragma once

#include "core/types.hpp"
#include <cmath>
#include <algorithm>

namespace fvm2d {

// =============================================================================
// Slope Limiter Functions (Header-only for inlining)
// =============================================================================

/**
 * @brief Minmod limiter
 * Most diffusive but very stable
 */
inline Scalar minmod_limiter(Scalar r) {
    return std::max(0.0, std::min(1.0, r));
}

/**
 * @brief Barth-Jespersen limiter
 * Commonly used in unstructured mesh codes
 */
inline Scalar barth_jespersen_limiter(Scalar r) {
    return std::min(1.0, r);
}

/**
 * @brief Superbee limiter
 * Most compressive, less stable
 */
inline Scalar superbee_limiter(Scalar r) {
    return std::max(0.0, std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0)));
}

/**
 * @brief Van Leer limiter
 * Smooth limiter, good balance
 */
inline Scalar vanleer_limiter(Scalar r) {
    if (r <= 0.0) return 0.0;
    return (r + std::abs(r)) / (1.0 + std::abs(r));
}

/**
 * @brief No limiter (first order)
 */
inline Scalar no_limiter(Scalar /*r*/) {
    return 1.0;
}

/**
 * @brief Get limiter function by type
 */
inline Scalar apply_limiter(LimiterType type, Scalar r) {
    switch (type) {
        case LimiterType::Minmod:
            return minmod_limiter(r);
        case LimiterType::BarthJespersen:
            return barth_jespersen_limiter(r);
        case LimiterType::Superbee:
            return superbee_limiter(r);
        case LimiterType::None:
            return no_limiter(r);
        default:
            return minmod_limiter(r);
    }
}

}  // namespace fvm2d
