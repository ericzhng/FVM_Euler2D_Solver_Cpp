#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <regex>
#include <filesystem>
#include <algorithm>
#include <stdexcept>

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

        // Load partition info
        std::vector<fvm2d::PartitionInfo> partitions(num_partitions);
        for (int r = 0; r < num_partitions; ++r) {
            std::string json_path = mesh_dir + "/partition_" + std::to_string(r) + ".json";
            partitions[r] = fvm2d::load_partition_info(json_path);
        }

        // Compute global sizes
        fvm::Index global_num_cells = 0;
        fvm::Index global_num_nodes = 0;
        for (int r = 0; r < num_partitions; ++r) {
            for (auto gc : partitions[r].l2g_cells) {
                global_num_cells = std::max(global_num_cells, static_cast<fvm::Index>(gc) + 1);
            }
            for (auto gn : partitions[r].l2g_nodes) {
                global_num_nodes = std::max(global_num_nodes, static_cast<fvm::Index>(gn) + 1);
            }
        }
        std::cout << "Global mesh: " << global_num_cells << " cells, "
                  << global_num_nodes << " nodes" << std::endl;

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

        // --- Process each timestep ---
        for (int step : timesteps) {
            std::cout << "  Processing step " << step << "..." << std::flush;

            // Allocate global arrays
            fvm::MeshInfo global_mesh;
            global_mesh.nodes.resize(global_num_nodes, {0.0, 0.0, 0.0});
            global_mesh.elements.resize(global_num_cells);
            global_mesh.elementTypes.resize(global_num_cells, 0);

            // Track which variables we have
            std::map<std::string, std::vector<fvm::Real>> global_cell_data;

            // Read each partition and merge
            for (int r = 0; r < num_partitions; ++r) {
                std::string filename = results_dir + "/" + prefix + "_"
                    + std::to_string(r) + "_" + std::to_string(step) + "." + ext;

                if (!fs::exists(filename)) {
                    std::cerr << "\n  Warning: Missing file " << filename << std::endl;
                    continue;
                }

                fvm::MeshInfo part_mesh = fvm::VTKReader::read(filename);
                const auto& pinfo = partitions[r];

                // Map nodes to global positions
                for (size_t i = 0; i < part_mesh.nodes.size() && i < pinfo.l2g_nodes.size(); ++i) {
                    auto gn = pinfo.l2g_nodes[i];
                    global_mesh.nodes[gn] = part_mesh.nodes[i];
                }

                // Map cells: remap local node indices to global
                fvm::Index num_owned = pinfo.num_owned_cells;
                for (fvm::Index i = 0; i < num_owned && i < static_cast<fvm::Index>(part_mesh.elements.size()); ++i) {
                    auto gc = pinfo.l2g_cells[i];
                    const auto& local_cell = part_mesh.elements[i];

                    fvm::CellConnectivity global_cell(local_cell.size());
                    for (size_t j = 0; j < local_cell.size(); ++j) {
                        global_cell[j] = pinfo.l2g_nodes[local_cell[j]];
                    }
                    global_mesh.elements[gc] = std::move(global_cell);

                    // Set element type
                    if (i < static_cast<fvm::Index>(part_mesh.elementTypes.size())) {
                        global_mesh.elementTypes[gc] = part_mesh.elementTypes[i];
                    } else {
                        global_mesh.elementTypes[gc] = (local_cell.size() == 3) ? 5 : 9;
                    }
                }

                // Map cell data
                for (const auto& [name, values] : part_mesh.cellData) {
                    auto& gdata = global_cell_data[name];
                    if (gdata.empty()) {
                        gdata.resize(global_num_cells, 0.0);
                    }
                    for (fvm::Index i = 0; i < num_owned && i < static_cast<fvm::Index>(values.size()); ++i) {
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
