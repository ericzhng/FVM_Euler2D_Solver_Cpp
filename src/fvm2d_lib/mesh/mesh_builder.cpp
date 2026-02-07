#include "mesh/mesh_builder.hpp"
#include "vtkio/vtk_reader.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <filesystem>

namespace fvm2d
{
    namespace
    {
        // Hash for edge (pair of node indices), order-independent
        struct EdgeHash
        {
            std::size_t operator()(const std::pair<Index, Index> &e) const
            {
                auto h1 = std::hash<Index>{}(e.first);
                auto h2 = std::hash<Index>{}(e.second);
                return h1 ^ (h2 << 32) ^ (h2 >> 32);
            }
        };

        // Canonical edge key: always (min, max) for consistent hashing
        std::pair<Index, Index> make_edge_key(Index a, Index b)
        {
            return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        }

        // Compute outward normal for edge (a -> b) relative to cell centroid
        Vector2d compute_outward_normal(const Vector2d &a, const Vector2d &b, const Vector2d &centroid)
        {
            Vector2d edge = b - a;
            // Perpendicular (rotated 90 degrees)
            Vector2d normal(edge.y(), -edge.x());
            Scalar length = normal.norm();
            if (length < EPSILON)
                return Vector2d::Zero();
            normal /= length;

            // Ensure outward pointing: dot(normal, midpoint - centroid) > 0
            Vector2d midpoint = 0.5 * (a + b);
            if (normal.dot(midpoint - centroid) < 0.0)
            {
                normal = -normal;
            }
            return normal;
        }

        // Find boundary file in directory by pattern *_boundaries.txt
        std::string find_boundaries_file(const std::string &mesh_dir)
        {
            namespace fs = std::filesystem;
            for (const auto &entry : fs::directory_iterator(mesh_dir))
            {
                if (entry.is_regular_file())
                {
                    std::string name = entry.path().filename().string();
                    if (name.find("_boundaries.txt") != std::string::npos ||
                        name.find("_boundaries.txt") != std::string::npos)
                    {
                        return entry.path().string();
                    }
                }
            }
            // Also check for just "boundaries.txt"
            std::string fallback = mesh_dir + "/boundaries.txt";
            if (fs::exists(fallback))
                return fallback;
            return "";
        }

        // Find partition VTU/VTK file for a rank
        std::string find_partition_mesh_file(const std::string &mesh_dir, int rank)
        {
            namespace fs = std::filesystem;
            std::string base = mesh_dir + "/partition_" + std::to_string(rank);

            if (fs::exists(base + ".vtu"))
                return base + ".vtu";
            if (fs::exists(base + ".vtk"))
                return base + ".vtk";

            throw std::runtime_error("Cannot find partition mesh file for rank " +
                                     std::to_string(rank) + " in " + mesh_dir);
        }

    } // anonymous namespace

    // =============================================================================
    // Load Partition Info from JSON (parsed via yaml-cpp, which handles JSON)
    // =============================================================================

    PartitionInfo load_partition_info(const std::string &json_path)
    {
        PartitionInfo info;

        try
        {
            YAML::Node root = YAML::LoadFile(json_path);

            info.rank = root["rank"].as<int>();
            info.num_owned_cells = static_cast<Index>(root["numOwnedCells"].as<int>());
            info.num_halo_cells = static_cast<Index>(root["numHaloCells"].as<int>());
            info.total_cells = static_cast<Index>(root["totalCells"].as<int>());
            info.total_nodes = static_cast<Index>(root["totalNodes"].as<int>());

            // l2gCells
            if (root["l2gCells"])
            {
                const auto &arr = root["l2gCells"];
                info.l2g_cells.reserve(arr.size());
                for (std::size_t i = 0; i < arr.size(); ++i)
                {
                    info.l2g_cells.push_back(static_cast<Index>(arr[i].as<int>()));
                }
            }

            // l2gNodes
            if (root["l2gNodes"])
            {
                const auto &arr = root["l2gNodes"];
                info.l2g_nodes.reserve(arr.size());
                for (std::size_t i = 0; i < arr.size(); ++i)
                {
                    info.l2g_nodes.push_back(static_cast<Index>(arr[i].as<int>()));
                }
            }

            // sendMap: { "rank_id": [cell_indices...] }
            if (root["sendMap"])
            {
                for (auto it = root["sendMap"].begin(); it != root["sendMap"].end(); ++it)
                {
                    int target_rank = std::stoi(it->first.as<std::string>());
                    const auto &arr = it->second;
                    std::vector<Index> indices;
                    indices.reserve(arr.size());
                    for (std::size_t i = 0; i < arr.size(); ++i)
                    {
                        indices.push_back(static_cast<Index>(arr[i].as<int>()));
                    }
                    info.send_map[target_rank] = std::move(indices);
                }
            }

            // recvMap: { "rank_id": [cell_indices...] }
            if (root["recvMap"])
            {
                for (auto it = root["recvMap"].begin(); it != root["recvMap"].end(); ++it)
                {
                    int source_rank = std::stoi(it->first.as<std::string>());
                    const auto &arr = it->second;
                    std::vector<Index> indices;
                    indices.reserve(arr.size());
                    for (std::size_t i = 0; i < arr.size(); ++i)
                    {
                        indices.push_back(static_cast<Index>(arr[i].as<int>()));
                    }
                    info.recv_map[source_rank] = std::move(indices);
                }
            }
        }
        catch (const YAML::Exception &e)
        {
            throw std::runtime_error("Error parsing partition JSON '" + json_path + "': " + e.what());
        }

        return info;
    }

    // =============================================================================
    // Load Boundary Faces from boundaries.txt
    // =============================================================================

    BoundaryFaceData load_boundary_faces(const std::string &boundaries_path)
    {
        BoundaryFaceData data;

        std::ifstream file(boundaries_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open boundaries file: " + boundaries_path);
        }

        std::string line;
        while (std::getline(file, line))
        {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
                continue;
            line = line.substr(start);

            // Skip comments
            if (line[0] == '#')
                continue;

            // Parse boundary header: "boundary_name num_faces"
            std::istringstream header_iss(line);
            std::string boundary_name;
            int num_faces;
            header_iss >> boundary_name >> num_faces;

            if (header_iss.fail() || boundary_name.empty())
                continue;

            std::vector<BoundaryEdge> edges;
            edges.reserve(num_faces);

            for (int i = 0; i < num_faces; ++i)
            {
                if (!std::getline(file, line))
                    break;

                std::istringstream edge_iss(line);
                Index node_a, node_b;
                edge_iss >> node_a >> node_b;

                if (!edge_iss.fail())
                {
                    edges.push_back({node_a, node_b});
                }
            }

            data.boundaries[boundary_name] = std::move(edges);
        }

        return data;
    }

    // =============================================================================
    // Build PartitionMesh from VTU + JSON + Boundaries
    // =============================================================================

    PartitionMesh build_partition_mesh(
        const fvm::MeshInfo &vtu_data,
        const PartitionInfo &partition,
        const BoundaryFaceData &boundaries)
    {
        PartitionMesh mesh;

        // --- Basic partition info ---
        mesh.num_owned_cells = partition.num_owned_cells;
        mesh.num_total_cells = partition.total_cells;
        mesh.l2g_cells = partition.l2g_cells;
        mesh.l2g_nodes = partition.l2g_nodes;
        mesh.send_map = partition.send_map;
        mesh.recv_map = partition.recv_map;

        const Index num_nodes = static_cast<Index>(vtu_data.nodes.size());
        const Index num_cells = partition.total_cells;

        // --- Copy node coordinates (from VTU Point3D to Vector2d) ---
        mesh.node_coords.resize(num_nodes);
        for (Index i = 0; i < num_nodes; ++i)
        {
            mesh.node_coords[i] = Vector2d(vtu_data.nodes[i][0], vtu_data.nodes[i][1]);
        }

        // --- Copy cell-node connectivity (from VTU elements) ---
        mesh.cell_nodes.resize(num_cells);
        for (Index i = 0; i < num_cells; ++i)
        {
            const auto &elem = vtu_data.elements[i];
            mesh.cell_nodes[i].resize(elem.size());
            for (std::size_t j = 0; j < elem.size(); ++j)
            {
                mesh.cell_nodes[i][j] = static_cast<Index>(elem[j]);
            }
        }

        // --- Compute cell centroids and volumes ---
        mesh.cell_centroids.resize(num_cells);
        mesh.cell_volumes.resize(num_cells);

        for (Index i = 0; i < num_cells; ++i)
        {
            const auto &nodes = mesh.cell_nodes[i];
            int n = static_cast<int>(nodes.size());

            // Centroid: average of node positions
            Vector2d centroid = Vector2d::Zero();
            for (int j = 0; j < n; ++j)
            {
                centroid += mesh.node_coords[nodes[j]];
            }
            centroid /= n;
            mesh.cell_centroids[i] = centroid;

            // Volume (area): shoelace formula
            Scalar area = 0.0;
            for (int j = 0; j < n; ++j)
            {
                int j_next = (j + 1) % n;
                const Vector2d &p1 = mesh.node_coords[nodes[j]];
                const Vector2d &p2 = mesh.node_coords[nodes[j_next]];
                area += p1.x() * p2.y() - p2.x() * p1.y();
            }
            mesh.cell_volumes[i] = std::abs(area) * 0.5;
        }

        // --- Build global-node-pair → boundary-tag lookup ---
        // First, build boundary_patch_map (name → tag) with sequential tags starting from 1
        int tag_counter = 1;
        for (const auto &[name, edges] : boundaries.boundaries)
        {
            mesh.boundary_patch_map[name] = tag_counter++;
        }

        // Build a set for each boundary: set of (min_global_node, max_global_node)
        using EdgeKey = std::pair<Index, Index>;
        std::unordered_map<std::size_t, int> global_edge_to_tag;

        for (const auto &[name, edges] : boundaries.boundaries)
        {
            int tag = mesh.boundary_patch_map[name];
            for (const auto &edge : edges)
            {
                auto key = make_edge_key(edge.node_a, edge.node_b);
                auto hash = EdgeHash{}(key);
                global_edge_to_tag[hash] = tag;
            }
        }

        // Build local-to-global node lookup for matching boundary edges
        // l2g_nodes[local_idx] = global_node_id
        const auto &l2g_nodes = partition.l2g_nodes;

        // --- Build edge → cell adjacency map ---
        // For each cell, enumerate edges and record which cells share each edge
        struct FaceInfo
        {
            Index cell;
            int local_face; // which face of the cell
            Index node_a;   // local node index
            Index node_b;   // local node index
        };

        std::unordered_map<std::size_t, std::vector<FaceInfo>> edge_map;

        for (Index i = 0; i < num_cells; ++i)
        {
            const auto &nodes = mesh.cell_nodes[i];
            int n = static_cast<int>(nodes.size());
            for (int j = 0; j < n; ++j)
            {
                int j_next = (j + 1) % n;
                Index na = nodes[j];
                Index nb = nodes[j_next];
                auto key = make_edge_key(na, nb);
                auto hash = EdgeHash{}(key);

                FaceInfo fi;
                fi.cell = i;
                fi.local_face = j;
                fi.node_a = na;
                fi.node_b = nb;
                edge_map[hash].push_back(fi);
            }
        }

        // --- Allocate connectivity ---
        mesh.connectivity.resize(num_cells);

        // Set num_faces per cell (= number of nodes for polygon cells)
        for (Index i = 0; i < num_cells; ++i)
        {
            mesh.connectivity.num_faces[i] = static_cast<int>(mesh.cell_nodes[i].size());
        }

        // --- Fill face data ---
        for (auto &[hash, faces] : edge_map)
        {
            if (faces.size() == 2)
            {
                // Interior face: two cells share this edge
                auto &f0 = faces[0];
                auto &f1 = faces[1];

                // For f0's face: neighbor is f1's cell
                const Vector2d &a0 = mesh.node_coords[f0.node_a];
                const Vector2d &b0 = mesh.node_coords[f0.node_b];
                Vector2d mid = 0.5 * (a0 + b0);
                Scalar face_area = (b0 - a0).norm();
                Vector2d normal0 = compute_outward_normal(a0, b0, mesh.cell_centroids[f0.cell]);

                mesh.connectivity.set_neighbor(f0.cell, f0.local_face, f1.cell);
                mesh.connectivity.set_tag(f0.cell, f0.local_face, 0); // interior
                mesh.connectivity.set_normal(f0.cell, f0.local_face, normal0);
                mesh.connectivity.set_midpoint(f0.cell, f0.local_face, mid);
                mesh.connectivity.set_area(f0.cell, f0.local_face, face_area);
                Scalar dist0 = (mid - mesh.cell_centroids[f0.cell]).norm();
                mesh.connectivity.set_distance(f0.cell, f0.local_face, dist0);

                // For f1's face: neighbor is f0's cell
                Vector2d normal1 = compute_outward_normal(
                    mesh.node_coords[f1.node_a], mesh.node_coords[f1.node_b],
                    mesh.cell_centroids[f1.cell]);

                mesh.connectivity.set_neighbor(f1.cell, f1.local_face, f0.cell);
                mesh.connectivity.set_tag(f1.cell, f1.local_face, 0); // interior
                mesh.connectivity.set_normal(f1.cell, f1.local_face, normal1);
                mesh.connectivity.set_midpoint(f1.cell, f1.local_face, mid);
                mesh.connectivity.set_area(f1.cell, f1.local_face, face_area);
                Scalar dist1 = (mid - mesh.cell_centroids[f1.cell]).norm();
                mesh.connectivity.set_distance(f1.cell, f1.local_face, dist1);
            }
            else if (faces.size() == 1)
            {
                // Boundary face: only one cell adjacent
                auto &f = faces[0];

                const Vector2d &a = mesh.node_coords[f.node_a];
                const Vector2d &b = mesh.node_coords[f.node_b];
                Vector2d mid = 0.5 * (a + b);
                Scalar face_area = (b - a).norm();
                Vector2d normal = compute_outward_normal(a, b, mesh.cell_centroids[f.cell]);

                // Determine boundary tag by matching global node indices
                int tag = 0;
                if (!l2g_nodes.empty())
                {
                    Index global_a = l2g_nodes[f.node_a];
                    Index global_b = l2g_nodes[f.node_b];
                    auto gkey = make_edge_key(global_a, global_b);
                    auto ghash = EdgeHash{}(gkey);
                    auto it = global_edge_to_tag.find(ghash);
                    if (it != global_edge_to_tag.end())
                    {
                        tag = it->second;
                    }
                }

                mesh.connectivity.set_neighbor(f.cell, f.local_face, -1); // boundary
                mesh.connectivity.set_tag(f.cell, f.local_face, tag);
                mesh.connectivity.set_normal(f.cell, f.local_face, normal);
                mesh.connectivity.set_midpoint(f.cell, f.local_face, mid);
                mesh.connectivity.set_area(f.cell, f.local_face, face_area);
                Scalar dist = (mid - mesh.cell_centroids[f.cell]).norm();
                mesh.connectivity.set_distance(f.cell, f.local_face, dist);
            }
            // edges shared by >2 cells would indicate a mesh error, ignore
        }

        return mesh;
    }

    // =============================================================================
    // High-Level Loader
    // =============================================================================

    PartitionMesh load_partition_mesh_from_dir(const std::string &mesh_dir, int rank)
    {
        namespace fs = std::filesystem;

        // Find partition JSON
        std::string json_path = mesh_dir + "/partition_" + std::to_string(rank) + ".json";
        if (!fs::exists(json_path))
        {
            throw std::runtime_error("Partition JSON not found: " + json_path);
        }

        // Find partition mesh file (VTU or VTK)
        std::string mesh_path = find_partition_mesh_file(mesh_dir, rank);

        // Find boundaries file
        std::string boundaries_path = find_boundaries_file(mesh_dir);

        std::cout << "Loading partition " << rank << ":" << std::endl;
        std::cout << "  Mesh file: " << mesh_path << std::endl;
        std::cout << "  JSON file: " << json_path << std::endl;
        if (!boundaries_path.empty())
        {
            std::cout << "  Boundaries: " << boundaries_path << std::endl;
        }

        // Load partition info from JSON
        PartitionInfo partition = load_partition_info(json_path);

        // Load mesh geometry from VTU/VTK
        fvm::MeshInfo vtu_data = fvm::VTKReader::read(mesh_path);

        // Load boundary faces (optional - may not exist for all meshes)
        BoundaryFaceData boundaries;
        if (!boundaries_path.empty())
        {
            boundaries = load_boundary_faces(boundaries_path);
        }

        // Build and return the partition mesh
        return build_partition_mesh(vtu_data, partition, boundaries);
    }

} // namespace fvm2d
