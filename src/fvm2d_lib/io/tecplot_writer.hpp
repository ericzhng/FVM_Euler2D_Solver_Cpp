#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "mesh/partition_mesh.hpp"
#include <string>
#include <vector>

namespace fvm2d {

/**
 * @brief Write solution to Tecplot ASCII format
 *
 * Creates a .dat file in FEPoint format
 *
 * @param mesh Partition mesh
 * @param U Solution state array
 * @param var_names Variable names
 * @param filename Output filename (should end with .dat)
 * @param time Current simulation time
 */
FVM_API void write_tecplot(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename,
    Scalar time = 0.0
);

}  // namespace fvm2d
