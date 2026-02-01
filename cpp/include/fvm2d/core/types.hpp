#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>
#include <array>
#include <map>
#include <string>

namespace fvm2d {

// =============================================================================
// Fundamental Types
// =============================================================================

using Scalar = double;
using Index = std::int32_t;

// =============================================================================
// Eigen Vector and Matrix Types
// =============================================================================

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// Row-major matrices for cache efficiency when iterating over cells
template<int Rows, int Cols>
using RowMatrix = Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>;

using RowMatrixXd = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// =============================================================================
// Solution State Arrays
// =============================================================================

// Conservative state array: (num_cells x num_vars), row-major for cell iteration
using StateArray = RowMatrixXd;

// Gradient array: (num_cells x num_vars*2) where columns are [dU0/dx, dU0/dy, dU1/dx, ...]
using GradientArray = RowMatrixXd;

// Limiter array: (num_cells x num_vars)
using LimiterArray = RowMatrixXd;

// =============================================================================
// Boundary Condition Types
// =============================================================================

enum class BCType : int {
    Transmissive = 0,  // Zero-gradient (outflow)
    Inlet = 1,         // Fixed inlet values
    Outlet = 2,        // Fixed outlet pressure
    Wall = 3           // Reflective wall (slip/no-slip)
};

// =============================================================================
// Mesh Constants
// =============================================================================

// Maximum number of faces per cell (supports triangles, quads, and general polygons)
constexpr int MAX_FACES_PER_CELL = 6;

// Small epsilon for numerical comparisons
constexpr Scalar EPSILON = 1e-12;

// =============================================================================
// Flux Scheme Types
// =============================================================================

enum class FluxType {
    Roe,
    HLLC,
    CentralDifference
};

// =============================================================================
// Limiter Types
// =============================================================================

enum class LimiterType {
    Minmod,
    BarthJespersen,
    Superbee,
    None
};

// =============================================================================
// Time Integration Types
// =============================================================================

enum class TimeIntegrationType {
    ExplicitEuler,
    RK2
};

// =============================================================================
// Equation Types
// =============================================================================

enum class EquationType {
    Euler,
    ShallowWater
};

// =============================================================================
// Utility Functions
// =============================================================================

inline FluxType parse_flux_type(const std::string& s) {
    if (s == "roe") return FluxType::Roe;
    if (s == "hllc") return FluxType::HLLC;
    if (s == "central_difference") return FluxType::CentralDifference;
    return FluxType::HLLC;  // Default
}

inline LimiterType parse_limiter_type(const std::string& s) {
    if (s == "minmod") return LimiterType::Minmod;
    if (s == "barth_jespersen") return LimiterType::BarthJespersen;
    if (s == "superbee") return LimiterType::Superbee;
    if (s == "none") return LimiterType::None;
    return LimiterType::Minmod;  // Default
}

inline TimeIntegrationType parse_time_integration(const std::string& s) {
    if (s == "euler") return TimeIntegrationType::ExplicitEuler;
    if (s == "rk2") return TimeIntegrationType::RK2;
    return TimeIntegrationType::RK2;  // Default
}

inline EquationType parse_equation_type(const std::string& s) {
    if (s == "euler") return EquationType::Euler;
    if (s == "shallow_water") return EquationType::ShallowWater;
    return EquationType::Euler;  // Default
}

}  // namespace fvm2d
