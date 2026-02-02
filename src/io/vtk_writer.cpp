#include "io/vtk_writer.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fvm2d {

void write_vtk(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create VTK file: " + filename);
    }

    file << std::scientific << std::setprecision(8);

    // Header
    file << "# vtk DataFile Version 3.0\n";
    file << "FVM2D Solution\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n\n";

    // Points (nodes)
    const Index num_nodes = mesh.num_nodes();
    file << "POINTS " << num_nodes << " double\n";
    for (Index i = 0; i < num_nodes; ++i) {
        file << mesh.node_coords[i].x() << " "
             << mesh.node_coords[i].y() << " 0.0\n";
    }
    file << "\n";

    // Cells
    const Index num_cells = mesh.num_owned_cells;
    Index total_size = 0;
    for (Index i = 0; i < num_cells; ++i) {
        total_size += 1 + static_cast<Index>(mesh.cell_nodes[i].size());
    }

    file << "CELLS " << num_cells << " " << total_size << "\n";
    for (Index i = 0; i < num_cells; ++i) {
        file << mesh.cell_nodes[i].size();
        for (Index node : mesh.cell_nodes[i]) {
            file << " " << node;
        }
        file << "\n";
    }
    file << "\n";

    // Cell types (5 = triangle, 9 = quad)
    file << "CELL_TYPES " << num_cells << "\n";
    for (Index i = 0; i < num_cells; ++i) {
        int vtk_type = (mesh.cell_nodes[i].size() == 3) ? 5 : 9;
        file << vtk_type << "\n";
    }
    file << "\n";

    // Cell data
    file << "CELL_DATA " << num_cells << "\n";

    const int num_vars = static_cast<int>(U.cols());
    for (int v = 0; v < num_vars; ++v) {
        std::string name = (v < static_cast<int>(var_names.size()))
                          ? var_names[v] : "var" + std::to_string(v);
        file << "SCALARS " << name << " double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (Index i = 0; i < num_cells; ++i) {
            file << U(i, v) << "\n";
        }
        file << "\n";
    }
}

void write_vtu(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create VTU file: " + filename);
    }

    file << std::scientific << std::setprecision(8);

    const Index num_nodes = mesh.num_nodes();
    const Index num_cells = mesh.num_owned_cells;
    const int num_vars = static_cast<int>(U.cols());

    // XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << num_nodes
         << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Index i = 0; i < num_nodes; ++i) {
        file << "          " << mesh.node_coords[i].x() << " "
             << mesh.node_coords[i].y() << " 0.0\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells
    file << "      <Cells>\n";

    // Connectivity
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index i = 0; i < num_cells; ++i) {
        file << "          ";
        for (Index node : mesh.cell_nodes[i]) {
            file << node << " ";
        }
        file << "\n";
    }
    file << "        </DataArray>\n";

    // Offsets
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    file << "          ";
    Index offset = 0;
    for (Index i = 0; i < num_cells; ++i) {
        offset += static_cast<Index>(mesh.cell_nodes[i].size());
        file << offset << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";

    // Types
    file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    file << "          ";
    for (Index i = 0; i < num_cells; ++i) {
        int vtk_type = (mesh.cell_nodes[i].size() == 3) ? 5 : 9;
        file << vtk_type << " ";
    }
    file << "\n";
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    // Cell data
    file << "      <CellData>\n";
    for (int v = 0; v < num_vars; ++v) {
        std::string name = (v < static_cast<int>(var_names.size()))
                          ? var_names[v] : "var" + std::to_string(v);
        file << "        <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
        file << "          ";
        for (Index i = 0; i < num_cells; ++i) {
            file << U(i, v) << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";
    }
    file << "      </CellData>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

}  // namespace fvm2d
