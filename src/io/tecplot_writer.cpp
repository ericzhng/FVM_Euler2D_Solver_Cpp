#include "io/tecplot_writer.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fvm2d {

void write_tecplot(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename,
    Scalar time
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create Tecplot file: " + filename);
    }

    file << std::scientific << std::setprecision(8);

    const Index num_nodes = mesh.num_nodes();
    const Index num_cells = mesh.num_owned_cells;
    const int num_vars = static_cast<int>(U.cols());

    // Title
    file << "TITLE = \"FVM2D Solution, t=" << time << "\"\n";

    // Variables
    file << "VARIABLES = \"X\", \"Y\"";
    for (int v = 0; v < num_vars; ++v) {
        std::string name = (v < static_cast<int>(var_names.size()))
                          ? var_names[v] : "var" + std::to_string(v);
        file << ", \"" << name << "\"";
    }
    file << "\n";

    // Count triangles and quads
    Index num_tris = 0, num_quads = 0;
    for (Index i = 0; i < num_cells; ++i) {
        if (mesh.cell_nodes[i].size() == 3) {
            ++num_tris;
        } else if (mesh.cell_nodes[i].size() == 4) {
            ++num_quads;
        }
    }

    // Zone header (FEPoint format)
    std::string elem_type = (num_quads > num_tris) ? "QUADRILATERAL" : "TRIANGLE";
    file << "ZONE T=\"Solution\", N=" << num_nodes << ", E=" << num_cells
         << ", F=FEPOINT, ET=" << elem_type << "\n";

    // Node data (X, Y, and interpolated cell values)
    // For simplicity, we use node-averaged values from connected cells
    std::vector<std::vector<Scalar>> node_values(num_vars, std::vector<Scalar>(num_nodes, 0.0));
    std::vector<int> node_count(num_nodes, 0);

    // Accumulate cell values at nodes
    for (Index i = 0; i < num_cells; ++i) {
        for (Index node : mesh.cell_nodes[i]) {
            for (int v = 0; v < num_vars; ++v) {
                node_values[v][node] += U(i, v);
            }
            node_count[node]++;
        }
    }

    // Average and write
    for (Index i = 0; i < num_nodes; ++i) {
        file << mesh.node_coords[i].x() << " " << mesh.node_coords[i].y();
        for (int v = 0; v < num_vars; ++v) {
            Scalar val = (node_count[i] > 0) ? node_values[v][i] / node_count[i] : 0.0;
            file << " " << val;
        }
        file << "\n";
    }

    // Element connectivity (1-indexed for Tecplot)
    for (Index i = 0; i < num_cells; ++i) {
        const auto& nodes = mesh.cell_nodes[i];
        if (nodes.size() == 3) {
            // Triangle: output 3 nodes, repeat last for quad format if needed
            file << (nodes[0] + 1) << " " << (nodes[1] + 1) << " "
                 << (nodes[2] + 1);
            if (elem_type == "QUADRILATERAL") {
                file << " " << (nodes[2] + 1);  // Repeat last node
            }
            file << "\n";
        } else if (nodes.size() == 4) {
            // Quad
            file << (nodes[0] + 1) << " " << (nodes[1] + 1) << " "
                 << (nodes[2] + 1) << " " << (nodes[3] + 1) << "\n";
        }
    }
}

}  // namespace fvm2d
