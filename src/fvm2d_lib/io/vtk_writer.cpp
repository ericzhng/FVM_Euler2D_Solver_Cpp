#include "io/vtk_writer.hpp"
#include "vtkio/vtk_writer.hpp"
#include "common/fvm_types.hpp"

namespace fvm2d {

void write_solution(
    const PartitionMesh& mesh,
    const StateArray& U,
    const std::vector<std::string>& var_names,
    const std::string& filename,
    bool binary
) {
    // Build MeshInfo from PartitionMesh + StateArray
    fvm::MeshInfo info;

    // Copy node coordinates (Vector2d -> Point3D with z=0)
    const Index num_nodes = mesh.num_nodes();
    info.nodes.resize(num_nodes);
    for (Index i = 0; i < num_nodes; ++i) {
        info.nodes[i] = {mesh.node_coords[i].x(), mesh.node_coords[i].y(), 0.0};
    }

    // Copy cell connectivity (owned cells only)
    const Index num_cells = mesh.num_owned_cells;
    info.elements.resize(num_cells);
    info.elementTypes.resize(num_cells);
    for (Index i = 0; i < num_cells; ++i) {
        const auto& cn = mesh.cell_nodes[i];
        info.elements[i].assign(cn.begin(), cn.end());
        info.elementTypes[i] = (cn.size() == 3) ? 5 : 9;
    }

    // Copy solution data into cellData
    const int num_vars = static_cast<int>(U.cols());
    for (int v = 0; v < num_vars; ++v) {
        std::string name = (v < static_cast<int>(var_names.size()))
                          ? var_names[v] : "var" + std::to_string(v);
        std::vector<fvm::Real> values(num_cells);
        for (Index i = 0; i < num_cells; ++i) {
            values[i] = U(i, v);
        }
        info.cellData[name] = std::move(values);
    }

    // Delegate to vtkio writer
    fvm::VTKWriter::writeVTU(info, filename, binary);
}

}  // namespace fvm2d
