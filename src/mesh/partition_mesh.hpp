#pragma once

#include "core/types.hpp"
#include <vector>
#include <array>
#include <map>
#include <string>

namespace fvm2d {

// =============================================================================
// Cell Connectivity Data (Structure of Arrays for cache efficiency)
// =============================================================================

struct CellConnectivity {
    // Number of faces per cell
    std::vector<int> num_faces;

    // Neighbor indices: [cell_idx * MAX_FACES_PER_CELL + face_idx]
    // Value is -1 for boundary faces
    std::vector<Index> neighbors;

    // Face normals (outward pointing, unit vectors)
    std::vector<Scalar> normals_x;
    std::vector<Scalar> normals_y;

    // Face midpoints
    std::vector<Scalar> midpoints_x;
    std::vector<Scalar> midpoints_y;

    // Face areas (lengths in 2D)
    std::vector<Scalar> areas;

    // Boundary tags (0 for interior, >0 for boundary patches)
    std::vector<int> tags;

    // Distance from cell centroid to face midpoint
    std::vector<Scalar> cell_to_face_dist;

    // Resize all arrays for num_cells
    void resize(Index num_cells) {
        const size_t total = static_cast<size_t>(num_cells) * MAX_FACES_PER_CELL;
        num_faces.resize(num_cells, 0);
        neighbors.resize(total, -1);
        normals_x.resize(total, 0.0);
        normals_y.resize(total, 0.0);
        midpoints_x.resize(total, 0.0);
        midpoints_y.resize(total, 0.0);
        areas.resize(total, 0.0);
        tags.resize(total, 0);
        cell_to_face_dist.resize(total, 0.0);
    }

    // Accessor helpers (inline for performance)
    inline Index neighbor(Index cell, int face) const {
        return neighbors[cell * MAX_FACES_PER_CELL + face];
    }

    inline Vector2d normal(Index cell, int face) const {
        const size_t idx = cell * MAX_FACES_PER_CELL + face;
        return Vector2d(normals_x[idx], normals_y[idx]);
    }

    inline Vector2d midpoint(Index cell, int face) const {
        const size_t idx = cell * MAX_FACES_PER_CELL + face;
        return Vector2d(midpoints_x[idx], midpoints_y[idx]);
    }

    inline Scalar area(Index cell, int face) const {
        return areas[cell * MAX_FACES_PER_CELL + face];
    }

    inline int tag(Index cell, int face) const {
        return tags[cell * MAX_FACES_PER_CELL + face];
    }

    inline Scalar distance(Index cell, int face) const {
        return cell_to_face_dist[cell * MAX_FACES_PER_CELL + face];
    }

    // Mutable setters
    inline void set_neighbor(Index cell, int face, Index value) {
        neighbors[cell * MAX_FACES_PER_CELL + face] = value;
    }

    inline void set_normal(Index cell, int face, const Vector2d& n) {
        const size_t idx = cell * MAX_FACES_PER_CELL + face;
        normals_x[idx] = n.x();
        normals_y[idx] = n.y();
    }

    inline void set_midpoint(Index cell, int face, const Vector2d& m) {
        const size_t idx = cell * MAX_FACES_PER_CELL + face;
        midpoints_x[idx] = m.x();
        midpoints_y[idx] = m.y();
    }

    inline void set_area(Index cell, int face, Scalar a) {
        areas[cell * MAX_FACES_PER_CELL + face] = a;
    }

    inline void set_tag(Index cell, int face, int t) {
        tags[cell * MAX_FACES_PER_CELL + face] = t;
    }

    inline void set_distance(Index cell, int face, Scalar d) {
        cell_to_face_dist[cell * MAX_FACES_PER_CELL + face] = d;
    }
};

// =============================================================================
// Partition Mesh Class
// =============================================================================

class PartitionMesh {
public:
    // -------------------------------------------------------------------------
    // Geometry Data
    // -------------------------------------------------------------------------

    // Node coordinates
    std::vector<Vector2d> node_coords;

    // Cell centroids
    std::vector<Vector2d> cell_centroids;

    // Cell volumes (areas in 2D)
    std::vector<Scalar> cell_volumes;

    // Cell-to-node connectivity (for output only)
    std::vector<std::vector<Index>> cell_nodes;

    // -------------------------------------------------------------------------
    // FVM Connectivity Data
    // -------------------------------------------------------------------------

    CellConnectivity connectivity;

    // -------------------------------------------------------------------------
    // Partition Information
    // -------------------------------------------------------------------------

    // Number of cells owned by this rank (excludes halo)
    Index num_owned_cells = 0;

    // Total number of cells (owned + halo)
    Index num_total_cells = 0;

    // Local-to-global cell index mapping
    std::vector<Index> l2g_cells;

    // -------------------------------------------------------------------------
    // MPI Communication Maps
    // -------------------------------------------------------------------------

    // rank -> list of local cell indices to send
    std::map<int, std::vector<Index>> send_map;

    // rank -> list of local cell indices where received data goes
    std::map<int, std::vector<Index>> recv_map;

    // -------------------------------------------------------------------------
    // Boundary Patch Information
    // -------------------------------------------------------------------------

    // patch name -> tag ID
    std::map<std::string, int> boundary_patch_map;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    PartitionMesh() = default;

    // -------------------------------------------------------------------------
    // I/O Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Load mesh from partition file
     * @param filepath Path to partition_X.mesh file
     */
    void load(const std::string& filepath);

    /**
     * @brief Save mesh to partition file (for debugging/visualization)
     * @param filepath Path to output file
     */
    void save(const std::string& filepath) const;

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    Index num_cells() const { return num_owned_cells; }
    Index total_cells() const { return num_total_cells; }
    Index num_nodes() const { return static_cast<Index>(node_coords.size()); }

    // Get centroid of cell i
    const Vector2d& centroid(Index i) const { return cell_centroids[i]; }

    // Get volume of cell i
    Scalar volume(Index i) const { return cell_volumes[i]; }

    // Get number of faces for cell i
    int faces(Index i) const { return connectivity.num_faces[i]; }
};

// =============================================================================
// Mesh Reader Functions
// =============================================================================

/**
 * @brief Load partition mesh from simple ASCII format
 * @param filepath Path to mesh file
 * @return Loaded PartitionMesh object
 */
PartitionMesh load_partition_mesh(const std::string& filepath);

/**
 * @brief Load partition mesh for a specific rank
 * @param mesh_dir Directory containing partition files
 * @param rank MPI rank
 * @return Loaded PartitionMesh object
 */
PartitionMesh load_partition_for_rank(const std::string& mesh_dir, int rank);

}  // namespace fvm2d
