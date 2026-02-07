#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "mesh/partition_mesh.hpp"
#include <string>
#include <vector>

namespace fvm2d {

/**
 * @brief Write solution to VTK format (legacy ASCII)
 *
 * Creates a .vtk file with cell data for visualization in ParaView
 *
 * @param mesh Partition mesh
 * @param U Solution state array
 * @param var_names Variable names for the solution components
 * @param filename Output filename (should end with .vtk)
 */
FVM_API void write_vtk(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename
);

/**
 * @brief Write solution to VTK XML format (.vtu)
 *
 * Creates a .vtu file (unstructured grid) for ParaView
 *
 * @param mesh Partition mesh
 * @param U Solution state array
 * @param var_names Variable names
 * @param filename Output filename (should end with .vtu)
 */
FVM_API void write_vtu(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename
);

}  // namespace fvm2d
