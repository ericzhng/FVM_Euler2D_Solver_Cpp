#pragma once

/**
 * @file fvm_types.hpp
 * @brief Common type definitions shared across FVM modules.
 *
 * This header contains fundamental types used by both meshgen and polymesh
 * modules, allowing them to remain independent libraries.
 *
 * Numeric precision is controlled by macros in fvm_export.hpp:
 * - FVM_USE_SINGLE_PRECISION: Use float instead of double
 * - FVM_USE_32BIT_INT: Use 32-bit integers instead of 64-bit
 */

#include "fvm_export.hpp"
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace fvm
{
    // =============================================================================
    // Basic Geometric Types (using configurable Real type)
    // =============================================================================

    /// 2D point/vector type
    using Point2D = std::array<Real, 2>;

    /// 3D point/vector type
    using Point3D = std::array<Real, 3>;

    /// Bounding box: {min_x, min_y, min_z, max_x, max_y, max_z}
    using BoundingBox = std::array<Real, 6>;

    /// Cell connectivity (variable number of nodes)
    using CellConnectivity = std::vector<Index>;

    /// A face defined by its node indices (variable size: 3 for tri, 4 for quad, etc.)
    using FaceNodes = std::vector<Index>;

    // =============================================================================
    // Mesh Interchange Data Structure
    // =============================================================================

    /// Minimal mesh data structure for format interchange
    struct MeshInfo
    {
        // === Geometry ===
        std::vector<Point3D> nodes;

        // === Topology ===
        std::vector<CellConnectivity> elements;
        std::vector<Index> elementTypes; // VTK cell types (-1 if unknown)

        // === Named Sets ===
        std::unordered_map<std::string, std::vector<Index>> nodeSets;
        std::unordered_map<std::string, std::vector<Index>> elementSets;
        std::unordered_map<std::string, std::vector<FaceNodes>> faceSets; // boundary groups go here

        // === Cell-centered scalar data ===
        // variable_name -> values[num_cells]
        std::map<std::string, std::vector<Real>> cellData;
    };

} // namespace fvm
