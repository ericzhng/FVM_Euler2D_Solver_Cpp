#include "vtkio/vtk_writer.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace fvm
{

    // =========================================================================
    // Base64 encoding for binary VTU
    // =========================================================================
    namespace
    {
        static const char base64_chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string base64_encode(const unsigned char *data, size_t len)
        {
            std::string result;
            result.reserve(((len + 2) / 3) * 4);

            for (size_t i = 0; i < len; i += 3)
            {
                unsigned int b = (data[i] << 16);
                if (i + 1 < len) b |= (data[i + 1] << 8);
                if (i + 2 < len) b |= data[i + 2];

                result.push_back(base64_chars[(b >> 18) & 0x3F]);
                result.push_back(base64_chars[(b >> 12) & 0x3F]);
                result.push_back((i + 1 < len) ? base64_chars[(b >> 6) & 0x3F] : '=');
                result.push_back((i + 2 < len) ? base64_chars[b & 0x3F] : '=');
            }
            return result;
        }

        // Encode a typed data block: [uint32 header (byte count)] + [raw data] → base64
        template <typename T>
        std::string encode_data_block(const T *data, size_t count)
        {
            uint32_t nbytes = static_cast<uint32_t>(count * sizeof(T));
            std::vector<unsigned char> buf(sizeof(uint32_t) + nbytes);
            std::memcpy(buf.data(), &nbytes, sizeof(uint32_t));
            std::memcpy(buf.data() + sizeof(uint32_t), data, nbytes);
            return base64_encode(buf.data(), buf.size());
        }

    } // namespace

    // =========================================================================
    // Legacy VTK writer (unchanged)
    // =========================================================================

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

        // Cell data (boundary info + solution data)
        writeVTKCellData(ofs, mesh);

        ofs.close();
        std::cout << "VTK file written: " << filename << std::endl;
    }

    // =========================================================================
    // VTU writer (ASCII or binary)
    // =========================================================================

    void VTKWriter::writeVTU(const MeshInfo &mesh,
                             const std::string &filename,
                             bool binary)
    {
        std::ofstream ofs(filename);
        if (!ofs)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        const auto numPoints = mesh.nodes.size();
        const auto numCells = mesh.elements.size();

        if (binary)
        {
            // ----- Binary VTU with base64-encoded appended data -----
            ofs << "<?xml version=\"1.0\"?>\n";
            ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\">\n";
            ofs << "  <UnstructuredGrid>\n";
            ofs << "    <Piece NumberOfPoints=\"" << numPoints
                << "\" NumberOfCells=\"" << numCells << "\">\n";

            // Track offsets for appended data blocks
            std::vector<std::string> encoded_blocks;

            // --- Points ---
            {
                std::vector<double> coords(numPoints * 3);
                for (size_t i = 0; i < numPoints; ++i)
                {
                    coords[i * 3 + 0] = mesh.nodes[i][0];
                    coords[i * 3 + 1] = mesh.nodes[i][1];
                    coords[i * 3 + 2] = mesh.nodes[i][2];
                }
                encoded_blocks.push_back(encode_data_block(coords.data(), coords.size()));
            }
            size_t points_block = 0;

            ofs << "      <Points>\n";
            ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                   "format=\"appended\" offset=\"0\"/>\n";
            ofs << "      </Points>\n";

            // --- Cells ---
            // Connectivity
            std::vector<int32_t> connectivity;
            for (const auto &cell : mesh.elements)
            {
                for (auto idx : cell)
                {
                    connectivity.push_back(static_cast<int32_t>(idx));
                }
            }
            encoded_blocks.push_back(encode_data_block(connectivity.data(), connectivity.size()));
            size_t conn_block = 1;

            // Offsets
            std::vector<int32_t> offsets(numCells);
            int32_t off = 0;
            for (size_t i = 0; i < numCells; ++i)
            {
                off += static_cast<int32_t>(mesh.elements[i].size());
                offsets[i] = off;
            }
            encoded_blocks.push_back(encode_data_block(offsets.data(), offsets.size()));
            size_t off_block = 2;

            // Types
            std::vector<uint8_t> types(numCells);
            for (size_t i = 0; i < numCells; ++i)
            {
                types[i] = (i < mesh.elementTypes.size())
                               ? static_cast<uint8_t>(mesh.elementTypes[i])
                               : static_cast<uint8_t>((mesh.elements[i].size() == 3) ? 5 : 9);
            }
            encoded_blocks.push_back(encode_data_block(types.data(), types.size()));
            size_t type_block = 3;

            // Compute offsets for appended data section
            // Each block contributes its base64 string length
            // But VTK appended with encoding="base64" uses inline offsets that
            // are byte offsets into the decoded stream. For simplicity, we use
            // inline base64 format instead of appended.

            // Actually, let's use format="binary" (inline base64) which is simpler
            // and well-supported by ParaView.
            // Rewrite: use inline binary (base64) encoding per DataArray

            // Clear and rewrite
            ofs.seekp(0);
            ofs << "<?xml version=\"1.0\"?>\n";
            ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\">\n";
            ofs << "  <UnstructuredGrid>\n";
            ofs << "    <Piece NumberOfPoints=\"" << numPoints
                << "\" NumberOfCells=\"" << numCells << "\">\n";

            // Points (inline base64)
            ofs << "      <Points>\n";
            ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                   "format=\"binary\">\n";
            ofs << "          " << encoded_blocks[points_block] << "\n";
            ofs << "        </DataArray>\n";
            ofs << "      </Points>\n";

            // Cells
            ofs << "      <Cells>\n";
            ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
                   "format=\"binary\">\n";
            ofs << "          " << encoded_blocks[conn_block] << "\n";
            ofs << "        </DataArray>\n";
            ofs << "        <DataArray type=\"Int32\" Name=\"offsets\" "
                   "format=\"binary\">\n";
            ofs << "          " << encoded_blocks[off_block] << "\n";
            ofs << "        </DataArray>\n";
            ofs << "        <DataArray type=\"UInt8\" Name=\"types\" "
                   "format=\"binary\">\n";
            ofs << "          " << encoded_blocks[type_block] << "\n";
            ofs << "        </DataArray>\n";
            ofs << "      </Cells>\n";

            // Cell Data
            if (!mesh.cellData.empty())
            {
                ofs << "      <CellData>\n";
                for (const auto &[name, values] : mesh.cellData)
                {
                    std::string encoded = encode_data_block(values.data(), values.size());
                    ofs << "        <DataArray type=\"Float64\" Name=\"" << name
                        << "\" format=\"binary\">\n";
                    ofs << "          " << encoded << "\n";
                    ofs << "        </DataArray>\n";
                }
                ofs << "      </CellData>\n";
            }

            ofs << "    </Piece>\n";
            ofs << "  </UnstructuredGrid>\n";
            ofs << "</VTKFile>\n";
        }
        else
        {
            // ----- ASCII VTU -----
            ofs << std::fixed << std::setprecision(10);

            ofs << "<?xml version=\"1.0\"?>\n";
            ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
                   "byte_order=\"LittleEndian\">\n";
            ofs << "  <UnstructuredGrid>\n";
            ofs << "    <Piece NumberOfPoints=\"" << numPoints
                << "\" NumberOfCells=\"" << numCells << "\">\n";

            // Points
            ofs << "      <Points>\n";
            ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                   "format=\"ascii\">\n";
            for (const auto &node : mesh.nodes)
            {
                ofs << "          " << node[0] << " " << node[1] << " " << node[2] << "\n";
            }
            ofs << "        </DataArray>\n";
            ofs << "      </Points>\n";

            // Cells
            ofs << "      <Cells>\n";

            // Connectivity
            ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
                   "format=\"ascii\">\n";
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
            ofs << "        <DataArray type=\"Int32\" Name=\"offsets\" "
                   "format=\"ascii\">\n";
            ofs << "          ";
            Index offset = 0;
            for (const auto &cell : mesh.elements)
            {
                offset += cell.size();
                ofs << offset << " ";
            }
            ofs << "\n";
            ofs << "        </DataArray>\n";

            // Types
            ofs << "        <DataArray type=\"UInt8\" Name=\"types\" "
                   "format=\"ascii\">\n";
            ofs << "          ";
            for (size_t i = 0; i < numCells; ++i)
            {
                if (i < mesh.elementTypes.size())
                    ofs << mesh.elementTypes[i] << " ";
                else
                    ofs << ((mesh.elements[i].size() == 3) ? 5 : 9) << " ";
            }
            ofs << "\n";
            ofs << "        </DataArray>\n";

            ofs << "      </Cells>\n";

            // Cell Data
            if (!mesh.cellData.empty())
            {
                ofs << "      <CellData>\n";
                for (const auto &[name, values] : mesh.cellData)
                {
                    ofs << "        <DataArray type=\"Float64\" Name=\"" << name
                        << "\" format=\"ascii\">\n";
                    ofs << "          ";
                    for (const auto &v : values)
                    {
                        ofs << v << " ";
                    }
                    ofs << "\n";
                    ofs << "        </DataArray>\n";
                }
                ofs << "      </CellData>\n";
            }

            ofs << "    </Piece>\n";
            ofs << "  </UnstructuredGrid>\n";
            ofs << "</VTKFile>\n";
        }

        ofs.close();
    }

    // =========================================================================
    // OpenFOAM writer (unchanged)
    // =========================================================================

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

        // Write faces (for 2D, each cell edge becomes a face)
        std::vector<std::vector<Index>> allFaces;
        std::vector<Index> owner;
        std::vector<Index> neighbour;

        // Build face data from cells
        std::map<std::pair<Index, Index>, Index> edgeToFaceIdx;

        for (size_t cellIdx = 0; cellIdx < mesh.elements.size(); ++cellIdx)
        {
            const auto &cell = mesh.elements[cellIdx];
            Index n = cell.size();

            for (Index i = 0; i < n; ++i)
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
        for (size_t i = 0; i < allFaces.size(); ++i)
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
                for (size_t j = 0; j < face.size(); ++j)
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
                for (size_t j = 0; j < face.size(); ++j)
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

            std::map<std::string, std::vector<Index>> boundaryPatches;
            if (mesh.faceSets.empty())
            {
                for (size_t i = 0; i < boundaryFaceIndices.size(); ++i)
                {
                    boundaryPatches["defaultPatch"].push_back(i);
                }
            }
            else
            {
                for (const auto &[name, faces] : mesh.faceSets)
                {
                    for (size_t i = 0; i < faces.size(); ++i)
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
        Index totalSize = 0;
        for (const auto &cell : mesh.elements)
        {
            totalSize += 1 + cell.size();
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
        // Only write CELL_DATA section if there's data to write
        bool hasCellData = !mesh.cellData.empty();
        if (!hasCellData)
            return;

        os << "CELL_DATA " << mesh.elements.size() << "\n";

        for (const auto &[name, values] : mesh.cellData)
        {
            os << "SCALARS " << name << " double 1\n";
            os << "LOOKUP_TABLE default\n";
            for (const auto &v : values)
            {
                os << v << "\n";
            }
        }
    }

} // namespace fvm
