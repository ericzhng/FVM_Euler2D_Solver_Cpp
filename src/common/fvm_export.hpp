/**
 * @file fvm_export.hpp
 * @brief Unified export/import macros for FVM libraries
 *
 * This header provides cross-platform support for building shared libraries.
 * On Windows, it handles __declspec(dllexport/dllimport).
 * On other platforms, it uses visibility attributes for optimization.
 */

#ifndef FVM_EXPORT_HPP
#define FVM_EXPORT_HPP

// =============================================================================
// FVM Library Export Macros
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
// Windows platform
#if defined(FVM_STATIC)
// Static library - no export/import needed
#define FVM_API
#elif defined(FVM_EXPORTS)
// Building the DLL - export symbols
#define FVM_API __declspec(dllexport)
#else
// Using the DLL - import symbols
#define FVM_API __declspec(dllimport)
#endif
#else
// Non-Windows platforms (Linux, macOS, etc.)
#if defined(FVM_EXPORTS) && defined(__GNUC__) && __GNUC__ >= 4
// Use visibility attribute for GCC/Clang
#define FVM_API __attribute__((visibility("default")))
#else
#define FVM_API
#endif
#endif

// =============================================================================
// Deprecation Warning Macros
// =============================================================================

#if defined(_MSC_VER)
#define FVM_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__) || defined(__clang__)
#define FVM_DEPRECATED __attribute__((deprecated))
#else
#define FVM_DEPRECATED
#endif

// =============================================================================
// Numeric Type Configuration
// =============================================================================
//
// These macros control the precision and size of numeric types used throughout
// the FVM library. Define these before including any FVM headers to customize.
//
// FVM_USE_SINGLE_PRECISION - Use float instead of double for Real type
// FVM_USE_32BIT_INT        - Use 32-bit integers for Id/Sid (default is 64-bit)
//

#include <cstdint>
#include <cstddef>

namespace fvm
{

    // -----------------------------------------------------------------------------
    // Floating-point type configuration
    // -----------------------------------------------------------------------------

#ifdef FVM_USE_SINGLE_PRECISION
    /// Floating-point type for coordinates and field values
    using Real = float;
#define FVM_REAL_SIZE 4
#define FVM_REAL_MAX 3.402823466e+38F
#define FVM_REAL_MIN 1.175494351e-38F
#define FVM_REAL_EPS 1.192092896e-07F
#else
    /// Floating-point type for coordinates and field values
    using Real = double;
#define FVM_REAL_SIZE 8
#define FVM_REAL_MAX 1.7976931348623158e+308
#define FVM_REAL_MIN 2.2250738585072014e-308
#define FVM_REAL_EPS 2.2204460492503131e-16
#endif

    // -----------------------------------------------------------------------------
    // Integer type configuration
    // -----------------------------------------------------------------------------
    // Id  - unsigned integer for node/element numbering and connectivity
    // Sid - signed integer for algorithms needing negative values, labels, tags

#ifdef FVM_USE_32BIT_INT
    using Index = std::int32_t;
#define FVM_INT_SIZE 4
#define FVM_INDEX_MAX 0x7FFFFFFFU
#else
    using Index = std::int64_t;
#define FVM_INT_SIZE 8
#define FVM_INDEX_MAX 0x7FFFFFFFFFFFFFFFULL
#endif

    // -----------------------------------------------------------------------------
    // Convenience constants
    // -----------------------------------------------------------------------------

    constexpr Index INVALID_INDEX = static_cast<Index>(-1);

} // namespace fvm

#endif // FVM_EXPORT_HPP
