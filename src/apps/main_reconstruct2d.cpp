#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <regex>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "CLI/CLI.hpp"
#include "common/fvm_types.hpp"
#include "vtkio/vtk_reader.hpp"
#include "vtkio/vtk_writer.hpp"
#include "fvm2d_lib/mesh/mesh_builder.hpp"

namespace fs = std::filesystem;

// Scan mesh_dir for partition_X.json files, return count
int count_partitions(const std::string& mesh_dir) {
    int count = 0;
    while (fs::exists(mesh_dir + "/partition_" + std::to_string(count) + ".json")) {
        ++count;
    }
    return count;
}

// Scan results_dir for result files and extract timestep numbers
// Filenames: {prefix}_{rank}_{step}.vtk or .vtu
std::set<int> find_timesteps(const std::string& results_dir,
                             const std::string& prefix,
                             int num_partitions,
                             std::string& detected_ext) {
    std::set<int> steps;
    detected_ext = "";

    // Pattern: prefix_rank_step.ext
    std::regex pattern(prefix + R"(_(\d+)_(\d+)\.(vtk|vtu))");

    for (const auto& entry : fs::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(name, match, pattern)) {
            int step = std::stoi(match[2].str());
            steps.insert(step);
            if (detected_ext.empty()) {
                detected_ext = match[3].str();
            }
        }
    }
    return steps;
}

// Auto-detect the filename prefix from result files
std::string detect_prefix(const std::string& results_dir) {
    // Look for files matching *_0_0.vtk or *_0_0.vtu (rank 0, step 0)
    std::regex pattern(R"((.+)_0_0\.(vtk|vtu))");
    for (const auto& entry : fs::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(name, match, pattern)) {
            return match[1].str();
        }
    }
    return "solution";
}

// Position key for node deduplication (quantized to avoid floating-point issues)
struct PositionKey {
    int64_t x, y, z;
    bool operator<(const PositionKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

PositionKey make_pos_key(const fvm::Point3D& p) {
    // Quantize to ~1e-6 precision (sufficient for mesh coordinates)
    return {
        static_cast<int64_t>(std::round(p[0] * 1e6)),
        static_cast<int64_t>(std::round(p[1] * 1e6)),
        static_cast<int64_t>(std::round(p[2] * 1e6))
    };
}

int main(int argc, char* argv[]) {
    // --- CLI Parsing ---
    std::string results_dir;
    std::string mesh_dir;
    std::string output_dir;
    std::string prefix;

    CLI::App app{
        "reconstruct2d - Merge partitioned FVM2D results into global VTU files"};

    app.footer(
        "\nExamples:\n"
        "  reconstruct2d results mesh\n"
        "  reconstruct2d results mesh --output merged --prefix solution\n"
        "\nCopyright (c) 2026 Eric Zhang. Distributed under MIT License.");

    app.add_option("results_dir", results_dir, "Directory containing partitioned result files")
        ->required()
        ->check(CLI::ExistingDirectory);

    app.add_option("mesh_dir", mesh_dir, "Directory containing partition_X.json files")
        ->required()
        ->check(CLI::ExistingDirectory);

    app.add_option("--output,-o", output_dir,
                   "Output directory for merged files (default: results_dir/merged)");

    app.add_option("--prefix,-p", prefix,
                   "Filename prefix (default: auto-detect from result files)");

    CLI11_PARSE(app, argc, argv);

    // --- Setup ---
    try {
        // Count partitions
        int num_partitions = count_partitions(mesh_dir);
        if (num_partitions == 0) {
            std::cerr << "Error: No partition JSON files found in " << mesh_dir << std::endl;
            return 1;
        }
        std::cout << "Found " << num_partitions << " partitions" << std::endl;

        // Load partition info (we only need l2g_cells and num_owned_cells)
        std::vector<fvm2d::PartitionInfo> partitions(num_partitions);
        for (int r = 0; r < num_partitions; ++r) {
            std::string json_path = mesh_dir + "/partition_" + std::to_string(r) + ".json";
            partitions[r] = fvm2d::load_partition_info(json_path);
        }

        // Compute global cell count from l2g_cells
        fvm::Index global_num_cells = 0;
        for (int r = 0; r < num_partitions; ++r) {
            for (auto gc : partitions[r].l2g_cells) {
                global_num_cells = std::max(global_num_cells, static_cast<fvm::Index>(gc) + 1);
            }
        }

        // Auto-detect prefix if not provided
        if (prefix.empty()) {
            prefix = detect_prefix(results_dir);
        }
        std::cout << "Filename prefix: " << prefix << std::endl;

        // Find all timesteps
        std::string ext;
        auto timesteps = find_timesteps(results_dir, prefix, num_partitions, ext);
        if (timesteps.empty()) {
            std::cerr << "Error: No result files found matching prefix '" << prefix
                      << "' in " << results_dir << std::endl;
            return 1;
        }
        std::cout << "Found " << timesteps.size() << " timestep(s): "
                  << *timesteps.begin() << " ... " << *timesteps.rbegin() << std::endl;

        // Setup output directory
        if (output_dir.empty()) {
            output_dir = results_dir + "/merged";
        }
        fs::create_directories(output_dir);

        // --- Build global mesh from partition VTU files ---
        // Use position-based node deduplication instead of l2g_nodes,
        // because l2g_nodes uses the partitioner's internal global numbering
        // which may differ from the VTU file's node ordering.
        std::map<PositionKey, fvm::Index> pos_to_global;
        std::vector<fvm::Point3D> global_nodes;
        std::vector<std::vector<fvm::Index>> local_to_global(num_partitions);

        std::vector<fvm::CellConnectivity> global_elements(global_num_cells);
        std::vector<fvm::Index> global_element_types(global_num_cells, 0);

        for (int r = 0; r < num_partitions; ++r) {
            // Find partition mesh file
            std::string mesh_file;
            if (fs::exists(mesh_dir + "/partition_" + std::to_string(r) + ".vtu"))
                mesh_file = mesh_dir + "/partition_" + std::to_string(r) + ".vtu";
            else if (fs::exists(mesh_dir + "/partition_" + std::to_string(r) + ".vtk"))
                mesh_file = mesh_dir + "/partition_" + std::to_string(r) + ".vtk";
            else {
                std::cerr << "Warning: No mesh VTU/VTK for partition " << r << std::endl;
                continue;
            }

            fvm::MeshInfo mesh_data = fvm::VTKReader::read(mesh_file);
            const auto& pinfo = partitions[r];

            // Build local-to-global node mapping by position deduplication
            local_to_global[r].resize(mesh_data.nodes.size());
            for (size_t i = 0; i < mesh_data.nodes.size(); ++i) {
                auto key = make_pos_key(mesh_data.nodes[i]);
                auto it = pos_to_global.find(key);
                if (it == pos_to_global.end()) {
                    fvm::Index gn = static_cast<fvm::Index>(global_nodes.size());
                    pos_to_global[key] = gn;
                    global_nodes.push_back(mesh_data.nodes[i]);
                    local_to_global[r][i] = gn;
                } else {
                    local_to_global[r][i] = it->second;
                }
            }

            // Map owned cells: remap local node indices to global via position map
            fvm::Index num_owned = pinfo.num_owned_cells;
            for (fvm::Index i = 0; i < num_owned
                 && i < static_cast<fvm::Index>(mesh_data.elements.size()); ++i) {
                auto gc = pinfo.l2g_cells[i];
                const auto& local_cell = mesh_data.elements[i];

                fvm::CellConnectivity global_cell(local_cell.size());
                for (size_t j = 0; j < local_cell.size(); ++j) {
                    global_cell[j] = local_to_global[r][local_cell[j]];
                }
                global_elements[gc] = std::move(global_cell);

                if (i < static_cast<fvm::Index>(mesh_data.elementTypes.size())) {
                    global_element_types[gc] = mesh_data.elementTypes[i];
                } else {
                    global_element_types[gc] = (local_cell.size() == 3) ? 5 : 9;
                }
            }
        }

        std::cout << "Global mesh: " << global_num_cells << " cells, "
                  << global_nodes.size() << " unique nodes" << std::endl;

        // --- Build local-to-global node maps for solution files ---
        // Solution VTU files may have a different (subset) of nodes than mesh VTUs.
        // We build the mapping on-the-fly per timestep using the same position approach.

        // --- Process each timestep ---
        for (int step : timesteps) {
            std::cout << "  Processing step " << step << "..." << std::flush;

            // Start with the pre-built global mesh (nodes + connectivity)
            fvm::MeshInfo global_mesh;
            global_mesh.nodes = global_nodes;
            global_mesh.elements = global_elements;
            global_mesh.elementTypes = global_element_types;

            // Track which variables we have
            std::map<std::string, std::vector<fvm::Real>> global_cell_data;

            // Read each partition's solution and merge cell data
            for (int r = 0; r < num_partitions; ++r) {
                std::string filename = results_dir + "/" + prefix + "_"
                    + std::to_string(r) + "_" + std::to_string(step) + "." + ext;

                if (!fs::exists(filename)) {
                    std::cerr << "\n  Warning: Missing file " << filename << std::endl;
                    continue;
                }

                fvm::MeshInfo part_mesh = fvm::VTKReader::read(filename);
                const auto& pinfo = partitions[r];

                // Map cell data from solution file using l2g_cells
                fvm::Index num_owned = pinfo.num_owned_cells;
                for (const auto& [name, values] : part_mesh.cellData) {
                    auto& gdata = global_cell_data[name];
                    if (gdata.empty()) {
                        gdata.resize(global_num_cells, 0.0);
                    }
                    for (fvm::Index i = 0; i < num_owned
                         && i < static_cast<fvm::Index>(values.size()); ++i) {
                        auto gc = pinfo.l2g_cells[i];
                        gdata[gc] = values[i];
                    }
                }
            }

            // Move cell data into global mesh
            global_mesh.cellData = std::move(global_cell_data);

            // Write merged VTU (binary)
            std::string out_filename = output_dir + "/" + prefix + "_" + std::to_string(step) + ".vtu";
            fvm::VTKWriter::writeVTU(global_mesh, out_filename, true);

            std::cout << " -> " << out_filename << std::endl;
        }

        std::cout << "\nReconstruction complete. " << timesteps.size()
                  << " file(s) written to " << output_dir << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
