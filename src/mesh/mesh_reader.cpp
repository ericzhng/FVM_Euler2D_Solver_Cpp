#include "mesh/partition_mesh.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace fvm2d {

namespace {

// Skip comment lines and empty lines
std::string read_next_line(std::ifstream& file) {
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Skip comments
        if (line[0] == '#') continue;

        return line;
    }
    return "";
}

// Parse key-value pair "key: value"
std::pair<std::string, std::string> parse_key_value(const std::string& line) {
    size_t pos = line.find(':');
    if (pos == std::string::npos) {
        return {line, ""};
    }
    std::string key = line.substr(0, pos);
    std::string value = line.substr(pos + 1);

    // Trim
    size_t start = key.find_first_not_of(" \t");
    size_t end = key.find_last_not_of(" \t");
    if (start != std::string::npos) key = key.substr(start, end - start + 1);

    start = value.find_first_not_of(" \t");
    end = value.find_last_not_of(" \t");
    if (start != std::string::npos) value = value.substr(start, end - start + 1);

    return {key, value};
}

}  // anonymous namespace

void PartitionMesh::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open mesh file: " + filepath);
    }

    std::string line;
    Index num_nodes = 0;
    Index num_halo_cells = 0;
    int num_boundary_patches = 0;

    // Parse HEADER section
    line = read_next_line(file);
    if (line != "HEADER") {
        throw std::runtime_error("Expected HEADER section in mesh file");
    }

    while ((line = read_next_line(file)) != "" && line != "BOUNDARY_PATCHES") {
        auto [key, value] = parse_key_value(line);
        if (key == "num_nodes") {
            num_nodes = std::stoi(value);
        } else if (key == "num_owned_cells") {
            num_owned_cells = std::stoi(value);
        } else if (key == "num_halo_cells") {
            num_halo_cells = std::stoi(value);
        } else if (key == "num_boundary_patches") {
            num_boundary_patches = std::stoi(value);
        }
    }

    num_total_cells = num_owned_cells + num_halo_cells;

    // Allocate arrays
    node_coords.resize(num_nodes);
    cell_centroids.resize(num_total_cells);
    cell_volumes.resize(num_total_cells);
    cell_nodes.resize(num_total_cells);
    l2g_cells.resize(num_total_cells);
    connectivity.resize(num_total_cells);

    // Parse BOUNDARY_PATCHES section
    if (line != "BOUNDARY_PATCHES") {
        line = read_next_line(file);
    }
    if (line != "BOUNDARY_PATCHES") {
        throw std::runtime_error("Expected BOUNDARY_PATCHES section");
    }

    for (int i = 0; i < num_boundary_patches; ++i) {
        line = read_next_line(file);
        std::istringstream iss(line);
        int tag;
        std::string name;
        iss >> tag >> name;
        boundary_patch_map[name] = tag;
    }

    // Parse NODES section
    line = read_next_line(file);
    if (line != "NODES") {
        throw std::runtime_error("Expected NODES section");
    }

    for (Index i = 0; i < num_nodes; ++i) {
        line = read_next_line(file);
        std::istringstream iss(line);
        Index id;
        Scalar x, y;
        iss >> id >> x >> y;
        node_coords[id] = Vector2d(x, y);
    }

    // Parse CELLS section
    line = read_next_line(file);
    if (line != "CELLS") {
        throw std::runtime_error("Expected CELLS section");
    }

    for (Index i = 0; i < num_total_cells; ++i) {
        line = read_next_line(file);
        std::istringstream iss(line);

        Index local_id, global_id;
        int n_nodes_in_cell;
        iss >> local_id >> global_id >> n_nodes_in_cell;

        l2g_cells[local_id] = global_id;

        // Read node indices
        cell_nodes[local_id].resize(n_nodes_in_cell);
        for (int j = 0; j < n_nodes_in_cell; ++j) {
            iss >> cell_nodes[local_id][j];
        }

        // Compute cell centroid from nodes
        Vector2d centroid = Vector2d::Zero();
        for (int j = 0; j < n_nodes_in_cell; ++j) {
            centroid += node_coords[cell_nodes[local_id][j]];
        }
        centroid /= n_nodes_in_cell;
        cell_centroids[local_id] = centroid;

        // Compute cell volume (area) using shoelace formula
        Scalar area = 0.0;
        for (int j = 0; j < n_nodes_in_cell; ++j) {
            int j_next = (j + 1) % n_nodes_in_cell;
            const Vector2d& p1 = node_coords[cell_nodes[local_id][j]];
            const Vector2d& p2 = node_coords[cell_nodes[local_id][j_next]];
            area += p1.x() * p2.y() - p2.x() * p1.y();
        }
        cell_volumes[local_id] = std::abs(area) * 0.5;

        // Read face data
        int n_faces;
        iss >> n_faces;
        connectivity.num_faces[local_id] = n_faces;

        for (int j = 0; j < n_faces; ++j) {
            Index neighbor;
            int tag;
            Scalar nx, ny, mx, my, face_area;
            iss >> neighbor >> tag >> nx >> ny >> mx >> my >> face_area;

            connectivity.set_neighbor(local_id, j, neighbor);
            connectivity.set_tag(local_id, j, tag);
            connectivity.set_normal(local_id, j, Vector2d(nx, ny));
            connectivity.set_midpoint(local_id, j, Vector2d(mx, my));
            connectivity.set_area(local_id, j, face_area);

            // Compute distance from cell centroid to face midpoint
            Scalar dist = (Vector2d(mx, my) - centroid).norm();
            connectivity.set_distance(local_id, j, dist);
        }
    }

    // Parse SEND_MAP section
    line = read_next_line(file);
    if (line == "SEND_MAP") {
        while ((line = read_next_line(file)) != "" && line != "RECV_MAP") {
            std::istringstream iss(line);
            int rank, count;
            iss >> rank >> count;
            send_map[rank].resize(count);
            for (int i = 0; i < count; ++i) {
                iss >> send_map[rank][i];
            }
        }
    }

    // Parse RECV_MAP section
    if (line == "RECV_MAP") {
        while ((line = read_next_line(file)) != "" && line != "END") {
            std::istringstream iss(line);
            int rank, count;
            iss >> rank >> count;
            recv_map[rank].resize(count);
            for (int i = 0; i < count; ++i) {
                iss >> recv_map[rank][i];
            }
        }
    }
}

void PartitionMesh::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create mesh file: " + filepath);
    }

    file << "# FVM2D Partition Mesh Format v1.0\n\n";

    // HEADER
    file << "HEADER\n";
    file << "num_nodes: " << node_coords.size() << "\n";
    file << "num_owned_cells: " << num_owned_cells << "\n";
    file << "num_halo_cells: " << (num_total_cells - num_owned_cells) << "\n";
    file << "num_boundary_patches: " << boundary_patch_map.size() << "\n\n";

    // BOUNDARY_PATCHES
    file << "BOUNDARY_PATCHES\n";
    for (const auto& [name, tag] : boundary_patch_map) {
        file << tag << " " << name << "\n";
    }
    file << "\n";

    // NODES
    file << "NODES\n";
    for (size_t i = 0; i < node_coords.size(); ++i) {
        file << i << " " << node_coords[i].x() << " " << node_coords[i].y() << "\n";
    }
    file << "\n";

    // CELLS
    file << "CELLS\n";
    for (Index i = 0; i < num_total_cells; ++i) {
        file << i << " " << l2g_cells[i] << " " << cell_nodes[i].size();
        for (Index n : cell_nodes[i]) {
            file << " " << n;
        }
        file << " " << connectivity.num_faces[i];
        for (int j = 0; j < connectivity.num_faces[i]; ++j) {
            Vector2d n = connectivity.normal(i, j);
            Vector2d m = connectivity.midpoint(i, j);
            file << " " << connectivity.neighbor(i, j)
                 << " " << connectivity.tag(i, j)
                 << " " << n.x() << " " << n.y()
                 << " " << m.x() << " " << m.y()
                 << " " << connectivity.area(i, j);
        }
        file << "\n";
    }
    file << "\n";

    // SEND_MAP
    file << "SEND_MAP\n";
    for (const auto& [rank, indices] : send_map) {
        file << rank << " " << indices.size();
        for (Index idx : indices) {
            file << " " << idx;
        }
        file << "\n";
    }
    file << "\n";

    // RECV_MAP
    file << "RECV_MAP\n";
    for (const auto& [rank, indices] : recv_map) {
        file << rank << " " << indices.size();
        for (Index idx : indices) {
            file << " " << idx;
        }
        file << "\n";
    }
    file << "\n";

    file << "END\n";
}

PartitionMesh load_partition_mesh(const std::string& filepath) {
    PartitionMesh mesh;
    mesh.load(filepath);
    return mesh;
}

PartitionMesh load_partition_for_rank(const std::string& mesh_dir, int rank) {
    std::string filepath = mesh_dir + "/partition_" + std::to_string(rank) + ".mesh";
    return load_partition_mesh(filepath);
}

}  // namespace fvm2d
