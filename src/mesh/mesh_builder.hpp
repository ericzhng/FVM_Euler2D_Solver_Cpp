#pragma once

#include "core/types.hpp"
#include "mesh/partition_mesh.hpp"
#include "common/fvm_types.hpp"
#include <string>
#include <vector>
#include <map>

namespace fvm2d {

// =============================================================================
// Partition Info (parsed from JSON)
// =============================================================================

struct PartitionInfo {
    int rank = 0;
    Index num_owned_cells = 0;
    Index num_halo_cells = 0;
    Index total_cells = 0;
    Index total_nodes = 0;

    std::vector<Index> l2g_cells;  // local cell index -> global cell ID
    std::vector<Index> l2g_nodes;  // local node index -> global node ID

    // MPI communication maps: rank -> list of local cell indices
    std::map<int, std::vector<Index>> send_map;
    std::map<int, std::vector<Index>> recv_map;
};

// =============================================================================
// Boundary Face Data (parsed from boundaries.txt)
// =============================================================================

struct BoundaryEdge {
    Index node_a;  // global node index
    Index node_b;  // global node index
};

struct BoundaryFaceData {
    // boundary name -> list of edges (global node pairs)
    std::map<std::string, std::vector<BoundaryEdge>> boundaries;
};

// =============================================================================
// Loader Functions
// =============================================================================

/**
 * @brief Load partition info from JSON file
 * @param json_path Path to partition_X.json
 * @return Parsed partition information
 */
PartitionInfo load_partition_info(const std::string& json_path);

/**
 * @brief Load boundary face definitions from boundaries text file
 * @param boundaries_path Path to *_boundaries.txt
 * @return Parsed boundary face data
 */
BoundaryFaceData load_boundary_faces(const std::string& boundaries_path);

/**
 * @brief Build a PartitionMesh from VTU mesh data, partition info, and boundary faces
 *
 * This function:
 * 1. Copies node coordinates and cell connectivity from VTU data
 * 2. Builds face/edge connectivity by detecting shared edges between cells
 * 3. Identifies boundary edges (edges with only one adjacent cell)
 * 4. Matches boundary edges to named boundaries using global node indices
 * 5. Computes geometric quantities: centroids, volumes, face normals, midpoints, areas
 *
 * @param vtu_data Mesh data read from VTU file (fvm::MeshInfo)
 * @param partition Partition info from JSON file
 * @param boundaries Boundary face definitions from boundaries.txt
 * @return Fully constructed PartitionMesh
 */
PartitionMesh build_partition_mesh(
    const fvm::MeshInfo& vtu_data,
    const PartitionInfo& partition,
    const BoundaryFaceData& boundaries
);

/**
 * @brief High-level loader: read all files from a mesh directory for a given rank
 *
 * Auto-discovers files by naming convention:
 * - partition_X.vtu (or .vtk) for geometry
 * - partition_X.json for partition info
 * - *_boundaries.txt for boundary definitions
 *
 * @param mesh_dir Directory containing mesh files
 * @param rank MPI rank
 * @return Fully constructed PartitionMesh
 */
PartitionMesh load_partition_mesh_from_dir(const std::string& mesh_dir, int rank);

}  // namespace fvm2d
