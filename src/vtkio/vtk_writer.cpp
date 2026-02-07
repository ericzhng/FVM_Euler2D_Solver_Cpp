#include "vtkio/vtk_writer.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace fvm
{

    void VTKWriter::writeVTK(const MeshInfo &mesh,
                             const std::string &filename,
                             bool binary)
    {
        std::ofstream ofs(filename);
        if (!ofs)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        ofs << std::fixed << std::setprecision(10);

        // Header
        writeVTKHeader(ofs, "FVM Mesh");

        if (binary)
        {
            ofs << "BINARY\n";
        }
        else
        {
            ofs << "ASCII\n";
        }

        ofs << "DATASET UNSTRUCTURED_GRID\n";

        // Points
        writeVTKPoints(ofs, mesh);

        // Cells
        writeVTKCells(ofs, mesh);

        // Cell types
        writeVTKCellTypes(ofs, mesh);

        // Cell data (boundary info, etc.)
        writeVTKCellData(ofs, mesh);

        ofs.close();
        std::cout << "VTK file written: " << filename << std::endl;
    }

    void VTKWriter::writeVTU(const MeshInfo &mesh,
                             const std::string &filename,
                             bool binary)
    {
        std::ofstream ofs(filename);
        if (!ofs)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        ofs << std::fixed << std::setprecision(10);

        // XML header
        ofs << "<?xml version=\"1.0\"?>\n";
        ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        ofs << "  <UnstructuredGrid>\n";
        ofs << "    <Piece NumberOfPoints=\"" << mesh.nodes.size()
            << "\" NumberOfCells=\"" << mesh.elements.size() << "\">\n";

        // Points
        ofs << "      <Points>\n";
        ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (const auto &node : mesh.nodes)
        {
            ofs << "          " << node[0] << " " << node[1] << " " << node[2] << "\n";
        }
        ofs << "        </DataArray>\n";
        ofs << "      </Points>\n";

        // Cells
        ofs << "      <Cells>\n";

        // Connectivity
        ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
        for (const auto &cell : mesh.elements)
        {
            ofs << "          ";
            for (auto nodeIdx : cell)
            {
                ofs << nodeIdx << " ";
            }
            ofs << "\n";
        }
        ofs << "        </DataArray>\n";

        // Offsets
        ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
        Index offset = 0;
        ofs << "          ";
        for (const auto &cell : mesh.elements)
        {
            offset += cell.size();
            ofs << offset << " ";
        }
        ofs << "\n";
        ofs << "        </DataArray>\n";

        // Types
        ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        ofs << "          ";
        for (auto cellType : mesh.elementTypes)
        {
            ofs << cellType << " ";
        }
        ofs << "\n";
        ofs << "        </DataArray>\n";

        ofs << "      </Cells>\n";

        // Cell Data
        ofs << "      <CellData>\n";

        // Cell IDs
        ofs << "        <DataArray type=\"Int32\" Name=\"CellID\" format=\"ascii\">\n";
        ofs << "          ";
        for (auto i = 0; i < mesh.elements.size(); ++i)
        {
            ofs << i << " ";
        }
        ofs << "\n";
        ofs << "        </DataArray>\n";

        ofs << "      </CellData>\n";

        // Point Data
        ofs << "      <PointData>\n";

        // Point IDs
        ofs << "        <DataArray type=\"Int32\" Name=\"PointID\" format=\"ascii\">\n";
        ofs << "          ";
        for (auto i = 0; i < mesh.nodes.size(); ++i)
        {
            ofs << i << " ";
        }
        ofs << "\n";
        ofs << "        </DataArray>\n";

        ofs << "      </PointData>\n";

        ofs << "    </Piece>\n";
        ofs << "  </UnstructuredGrid>\n";
        ofs << "</VTKFile>\n";

        ofs.close();
        // std::cout << "VTU file written: " << filename << std::endl;
    }

    void VTKWriter::writeOpenFOAM(const MeshInfo &mesh,
                                  const std::string &outputDir)
    {
        std::string polyMeshDir = outputDir + "/constant/polyMesh";
        std::filesystem::create_directories(polyMeshDir);

        auto writeFoamHeader = [](std::ofstream &ofs, const std::string &objectClass,
                                  const std::string &objectName)
        {
            ofs << "FoamFile\n";
            ofs << "{\n";
            ofs << "    version     2.0;\n";
            ofs << "    format      ascii;\n";
            ofs << "    class       " << objectClass << ";\n";
            ofs << "    object      " << objectName << ";\n";
            ofs << "}\n\n";
        };

        // Write points
        {
            std::ofstream ofs(polyMeshDir + "/points");
            writeFoamHeader(ofs, "vectorField", "points");

            ofs << std::fixed << std::setprecision(10);
            ofs << mesh.nodes.size() << "\n(\n";
            for (const auto &node : mesh.nodes)
            {
                ofs << "(" << node[0] << " " << node[1] << " " << node[2] << ")\n";
            }
            ofs << ")\n";
        }

        // For 2D meshes in OpenFOAM, we need to extrude to 3D
        // This is a simplified 2D export (single layer)
        // Note: A proper 2D OpenFOAM mesh would need extrusion

        // Write faces (for 2D, each cell edge becomes a face)
        std::vector<std::vector<Index>> allFaces;
        std::vector<Index> owner;
        std::vector<Index> neighbour;

        // Build face data from cells
        std::map<std::pair<Index, Index>, Index> edgeToFaceIdx;

        for (auto cellIdx = 0; cellIdx < mesh.elements.size(); ++cellIdx)
        {
            const auto &cell = mesh.elements[cellIdx];
            Index n = cell.size();

            for (auto i = 0; i < n; ++i)
            {
                Index n1 = cell[i];
                Index n2 = cell[(i + 1) % n];
                auto edge = std::minmax(n1, n2);

                auto it = edgeToFaceIdx.find(edge);
                if (it == edgeToFaceIdx.end())
                {
                    // New face
                    auto faceIdx = allFaces.size();
                    allFaces.push_back({n1, n2});
                    owner.push_back(static_cast<Index>(cellIdx));
                    edgeToFaceIdx[edge] = faceIdx;
                }
                else
                {
                    // Existing face - this cell is the neighbour
                    neighbour.resize(allFaces.size(), -1);
                    neighbour[it->second] = static_cast<Index>(cellIdx);
                }
            }
        }

        // Separate internal and boundary faces
        std::vector<Index> internalFaceIndices;
        std::vector<Index> boundaryFaceIndices;

        neighbour.resize(allFaces.size(), -1);
        for (auto i = 0; i < allFaces.size(); ++i)
        {
            if (neighbour[i] >= 0)
            {
                internalFaceIndices.push_back(i);
            }
            else
            {
                boundaryFaceIndices.push_back(i);
            }
        }

        // Write faces file
        {
            std::ofstream ofs(polyMeshDir + "/faces");
            writeFoamHeader(ofs, "faceList", "faces");

            Index totalFaces = internalFaceIndices.size() + boundaryFaceIndices.size();
            ofs << totalFaces << "\n(\n";

            // Internal faces first
            for (auto idx : internalFaceIndices)
            {
                const auto &face = allFaces[idx];
                ofs << face.size() << "(";
                for (auto j = 0; j < face.size(); ++j)
                {
                    if (j > 0)
                        ofs << " ";
                    ofs << face[j];
                }
                ofs << ")\n";
            }

            // Boundary faces
            for (auto idx : boundaryFaceIndices)
            {
                const auto &face = allFaces[idx];
                ofs << face.size() << "(";
                for (auto j = 0; j < face.size(); ++j)
                {
                    if (j > 0)
                        ofs << " ";
                    ofs << face[j];
                }
                ofs << ")\n";
            }

            ofs << ")\n";
        }

        // Write owner
        {
            std::ofstream ofs(polyMeshDir + "/owner");
            writeFoamHeader(ofs, "labelList", "owner");

            Index totalFaces = internalFaceIndices.size() + boundaryFaceIndices.size();
            ofs << totalFaces << "\n(\n";

            for (auto idx : internalFaceIndices)
            {
                ofs << owner[idx] << "\n";
            }
            for (auto idx : boundaryFaceIndices)
            {
                ofs << owner[idx] << "\n";
            }

            ofs << ")\n";
        }

        // Write neighbour (only for internal faces)
        {
            std::ofstream ofs(polyMeshDir + "/neighbour");
            writeFoamHeader(ofs, "labelList", "neighbour");

            ofs << internalFaceIndices.size() << "\n(\n";
            for (auto idx : internalFaceIndices)
            {
                ofs << neighbour[idx] << "\n";
            }
            ofs << ")\n";
        }

        // Write boundary
        {
            std::ofstream ofs(polyMeshDir + "/boundary");
            writeFoamHeader(ofs, "polyBoundaryMesh", "boundary");

            // Use faceSets for boundary patches
            // For now, treat all boundary faces as one patch if no faceSets defined
            std::map<std::string, std::vector<Index>> boundaryPatches;
            if (mesh.faceSets.empty())
            {
                for (auto i = 0; i < boundaryFaceIndices.size(); ++i)
                {
                    boundaryPatches["defaultPatch"].push_back(i);
                }
            }
            else
            {
                // Use faceSet names as patch names
                for (const auto &[name, faces] : mesh.faceSets)
                {
                    for (auto i = 0; i < faces.size(); ++i)
                    {
                        boundaryPatches[name].push_back(i);
                    }
                }
            }

            ofs << boundaryPatches.size() << "\n(\n";

            Index startFace = internalFaceIndices.size();
            for (const auto &[name, faces] : boundaryPatches)
            {
                ofs << "    " << name << "\n";
                ofs << "    {\n";
                ofs << "        type            patch;\n";
                ofs << "        nFaces          " << faces.size() << ";\n";
                ofs << "        startFace       " << startFace << ";\n";
                ofs << "    }\n";
                startFace += faces.size();
            }

            ofs << ")\n";
        }

        // std::cout << "OpenFOAM polyMesh written to: " << polyMeshDir << std::endl;
    }

    void VTKWriter::writeBoundaryInfo(const MeshInfo &mesh,
                                      const std::string &filename)
    {
        std::ofstream ofs(filename);
        if (!ofs)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        ofs << "# Boundary Information\n";
        ofs << "# Format: boundary_name num_faces\n";
        ofs << "#   node_indices...\n\n";

        for (const auto &[name, faces] : mesh.faceSets)
        {
            ofs << name << " " << faces.size() << "\n";
            for (const auto &face : faces)
            {
                for (auto nodeIdx : face)
                {
                    ofs << " " << nodeIdx;
                }
                ofs << "\n";
            }
        }

        ofs.close();
        // std::cout << "Boundary info written to: " << filename << std::endl;
    }

    void VTKWriter::writeVTKHeader(std::ostream &os, const std::string &title)
    {
        os << "# vtk DataFile Version 3.0\n";
        os << title << "\n";
    }

    void VTKWriter::writeVTKPoints(std::ostream &os, const MeshInfo &mesh)
    {
        os << "POINTS " << mesh.nodes.size() << " double\n";
        for (const auto &node : mesh.nodes)
        {
            os << node[0] << " " << node[1] << " " << node[2] << "\n";
        }
    }

    void VTKWriter::writeVTKCells(std::ostream &os, const MeshInfo &mesh)
    {
        // Calculate total size (num_nodes_per_cell + node_indices for each cell)
        Index totalSize = 0;
        for (const auto &cell : mesh.elements)
        {
            totalSize += 1 + cell.size(); // 1 for count, rest for indices
        }

        os << "CELLS " << mesh.elements.size() << " " << totalSize << "\n";
        for (const auto &cell : mesh.elements)
        {
            os << cell.size();
            for (auto nodeIdx : cell)
            {
                os << " " << nodeIdx;
            }
            os << "\n";
        }
    }

    void VTKWriter::writeVTKCellTypes(std::ostream &os, const MeshInfo &mesh)
    {
        os << "CELL_TYPES " << mesh.elements.size() << "\n";
        for (auto cellType : mesh.elementTypes)
        {
            os << cellType << "\n";
        }
    }

    void VTKWriter::writeVTKCellData(std::ostream &os, const MeshInfo &mesh)
    {
        os << "CELL_DATA " << mesh.elements.size() << "\n";

        // Cell IDs
        os << "SCALARS CellID int 1\n";
        os << "LOOKUP_TABLE default\n";
        for (auto i = 0; i < mesh.elements.size(); ++i)
        {
            os << i << "\n";
        }
    }

} // namespace fvm
