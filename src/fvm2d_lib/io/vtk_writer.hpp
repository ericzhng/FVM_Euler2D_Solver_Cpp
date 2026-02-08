#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "mesh/partition_mesh.hpp"
#include <string>
#include <vector>

namespace fvm2d {

/**
 * @brief Write solution to VTU format via vtkio library
 *
 * Converts PartitionMesh + StateArray to fvm::MeshInfo and delegates
 * to fvm::VTKWriter for actual file output.
 *
 * @param mesh Partition mesh
 * @param U Solution state array (num_owned_cells x num_vars)
 * @param var_names Variable names for the solution components
 * @param filename Output filename (should end with .vtu or .vtk)
 * @param binary Whether to use binary encoding (default: true)
 */
FVM_API void write_solution(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename,
    bool binary = true
);

}  // namespace fvm2d
