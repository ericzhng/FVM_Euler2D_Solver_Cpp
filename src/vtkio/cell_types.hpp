#pragma once

/**
 * @file cell_types.hpp
 * @brief Complete VTK cell type definitions and utilities.
 *
 * This header provides all VTK cell type codes and helper functions
 * for cell type inspection. This is the canonical source for VTK
 * cell type definitions in the FVM project.
 */

#include <cstddef>
#include <string>

namespace fvm {

// =============================================================================
// VTK Cell Type Codes
// =============================================================================

/// VTK cell type codes (from VTK documentation)
namespace VTKCellType {
    // Linear cells
    constexpr int VERTEX = 1;
    constexpr int POLY_VERTEX = 2;
    constexpr int LINE = 3;
    constexpr int POLY_LINE = 4;
    constexpr int TRIANGLE = 5;
    constexpr int TRIANGLE_STRIP = 6;
    constexpr int POLYGON = 7;
    constexpr int PIXEL = 8;
    constexpr int QUAD = 9;
    constexpr int TETRA = 10;
    constexpr int VOXEL = 11;
    constexpr int HEXAHEDRON = 12;
    constexpr int WEDGE = 13;
    constexpr int PYRAMID = 14;
    constexpr int PENTAGONAL_PRISM = 15;
    constexpr int HEXAGONAL_PRISM = 16;

    // Quadratic cells
    constexpr int QUADRATIC_EDGE = 21;
    constexpr int QUADRATIC_TRIANGLE = 22;
    constexpr int QUADRATIC_QUAD = 23;
    constexpr int QUADRATIC_TETRA = 24;
    constexpr int QUADRATIC_HEXAHEDRON = 25;
    constexpr int QUADRATIC_WEDGE = 26;
    constexpr int QUADRATIC_PYRAMID = 27;
}

// =============================================================================
// Cell Type Utilities
// =============================================================================

/// Map number of nodes to VTK cell type (for 2D cells)
inline int getVTKCellType(std::size_t numNodes) {
    switch (numNodes) {
        case 3: return VTKCellType::TRIANGLE;
        case 4: return VTKCellType::QUAD;
        default: return VTKCellType::POLYGON;
    }
}

/// Map number of nodes to VTK cell type (for 3D cells)
inline int getVTKCellType3D(std::size_t numNodes) {
    switch (numNodes) {
        case 4: return VTKCellType::TETRA;
        case 5: return VTKCellType::PYRAMID;
        case 6: return VTKCellType::WEDGE;
        case 8: return VTKCellType::HEXAHEDRON;
        default: return VTKCellType::POLYGON;  // Fallback
    }
}

/// Get the name of a VTK cell type
inline std::string getVTKCellTypeName(int vtkType) {
    switch (vtkType) {
        case VTKCellType::VERTEX: return "Vertex";
        case VTKCellType::POLY_VERTEX: return "PolyVertex";
        case VTKCellType::LINE: return "Line";
        case VTKCellType::POLY_LINE: return "PolyLine";
        case VTKCellType::TRIANGLE: return "Triangle";
        case VTKCellType::TRIANGLE_STRIP: return "TriangleStrip";
        case VTKCellType::POLYGON: return "Polygon";
        case VTKCellType::PIXEL: return "Pixel";
        case VTKCellType::QUAD: return "Quad";
        case VTKCellType::TETRA: return "Tetrahedron";
        case VTKCellType::VOXEL: return "Voxel";
        case VTKCellType::HEXAHEDRON: return "Hexahedron";
        case VTKCellType::WEDGE: return "Wedge";
        case VTKCellType::PYRAMID: return "Pyramid";
        case VTKCellType::PENTAGONAL_PRISM: return "PentagonalPrism";
        case VTKCellType::HEXAGONAL_PRISM: return "HexagonalPrism";
        case VTKCellType::QUADRATIC_EDGE: return "QuadraticEdge";
        case VTKCellType::QUADRATIC_TRIANGLE: return "QuadraticTriangle";
        case VTKCellType::QUADRATIC_QUAD: return "QuadraticQuad";
        case VTKCellType::QUADRATIC_TETRA: return "QuadraticTetra";
        case VTKCellType::QUADRATIC_HEXAHEDRON: return "QuadraticHexahedron";
        case VTKCellType::QUADRATIC_WEDGE: return "QuadraticWedge";
        case VTKCellType::QUADRATIC_PYRAMID: return "QuadraticPyramid";
        default: return "Unknown";
    }
}

/// Get the number of nodes for a VTK cell type (-1 for variable)
inline int getVTKCellNodeCount(int vtkType) {
    switch (vtkType) {
        case VTKCellType::VERTEX: return 1;
        case VTKCellType::LINE: return 2;
        case VTKCellType::TRIANGLE: return 3;
        case VTKCellType::PIXEL: return 4;
        case VTKCellType::QUAD: return 4;
        case VTKCellType::TETRA: return 4;
        case VTKCellType::VOXEL: return 8;
        case VTKCellType::HEXAHEDRON: return 8;
        case VTKCellType::WEDGE: return 6;
        case VTKCellType::PYRAMID: return 5;
        case VTKCellType::PENTAGONAL_PRISM: return 10;
        case VTKCellType::HEXAGONAL_PRISM: return 12;
        case VTKCellType::QUADRATIC_EDGE: return 3;
        case VTKCellType::QUADRATIC_TRIANGLE: return 6;
        case VTKCellType::QUADRATIC_QUAD: return 8;
        case VTKCellType::QUADRATIC_TETRA: return 10;
        case VTKCellType::QUADRATIC_HEXAHEDRON: return 20;
        case VTKCellType::QUADRATIC_WEDGE: return 15;
        case VTKCellType::QUADRATIC_PYRAMID: return 13;
        default: return -1;  // Variable (polygon, polyline, etc.)
    }
}

/// Check if a VTK cell type is 2D (surface element)
inline bool isVTKCellType2D(int vtkType) {
    switch (vtkType) {
        case VTKCellType::TRIANGLE:
        case VTKCellType::TRIANGLE_STRIP:
        case VTKCellType::POLYGON:
        case VTKCellType::PIXEL:
        case VTKCellType::QUAD:
        case VTKCellType::QUADRATIC_TRIANGLE:
        case VTKCellType::QUADRATIC_QUAD:
            return true;
        default:
            return false;
    }
}

/// Check if a VTK cell type is 3D (volume element)
inline bool isVTKCellType3D(int vtkType) {
    switch (vtkType) {
        case VTKCellType::TETRA:
        case VTKCellType::VOXEL:
        case VTKCellType::HEXAHEDRON:
        case VTKCellType::WEDGE:
        case VTKCellType::PYRAMID:
        case VTKCellType::PENTAGONAL_PRISM:
        case VTKCellType::HEXAGONAL_PRISM:
        case VTKCellType::QUADRATIC_TETRA:
        case VTKCellType::QUADRATIC_HEXAHEDRON:
        case VTKCellType::QUADRATIC_WEDGE:
        case VTKCellType::QUADRATIC_PYRAMID:
            return true;
        default:
            return false;
    }
}

}  // namespace fvm
