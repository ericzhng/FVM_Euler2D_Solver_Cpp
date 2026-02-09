#include "vtkio/vtk_reader.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fvm
{

    namespace
    {

        // Trim whitespace from string
        std::string trim(const std::string &str)
        {
            auto start = str.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
                return "";
            auto end = str.find_last_not_of(" \t\r\n");
            return str.substr(start, end - start + 1);
        }

        // Convert string to lowercase
        std::string toLower(const std::string &str)
        {
            std::string result = str;
            std::transform(result.begin(), result.end(), result.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return result;
        }

        // Skip whitespace and comments in VTK file
        void skipWhitespace(std::istream &is)
        {
            while (is.good())
            {
                int c = is.peek();
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
                {
                    is.get();
                }
                else
                {
                    break;
                }
            }
        }

        // Read a line, skipping empty lines
        std::string readNonEmptyLine(std::istream &is)
        {
            std::string line;
            while (std::getline(is, line))
            {
                line = trim(line);
                if (!line.empty())
                {
                    return line;
                }
            }
            return "";
        }

    } // namespace

    std::string VTKReader::getExtension(const std::string &filename)
    {
        auto pos = filename.rfind('.');
        if (pos == std::string::npos)
            return "";
        return toLower(filename.substr(pos));
    }

    MeshInfo VTKReader::read(const std::string &filename)
    {
        std::string ext = getExtension(filename);
        if (ext == ".vtk")
        {
            return readVTK(filename);
        }
        else if (ext == ".vtu")
        {
            return readVTU(filename);
        }
        else
        {
            throw std::runtime_error("Unsupported file format: " + ext +
                                     " (expected .vtk or .vtu)");
        }
    }

    MeshInfo VTKReader::readVTK(const std::string &filename)
    {
        std::ifstream ifs(filename);
        if (!ifs)
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        MeshInfo mesh;
        std::string title;
        bool binary = false;

        // Parse header
        parseVTKHeader(ifs, title, binary);

        if (binary)
        {
            throw std::runtime_error("Binary VTK format not yet supported");
        }

        // Read dataset type
        std::string line = readNonEmptyLine(ifs);
        if (line.find("DATASET") == std::string::npos)
        {
            throw std::runtime_error("Expected DATASET keyword");
        }
        if (line.find("UNSTRUCTURED_GRID") == std::string::npos)
        {
            throw std::runtime_error("Only UNSTRUCTURED_GRID dataset type is supported");
        }

        // Parse sections
        Index numPoints = 0;
        Index numCells = 0;

        while (ifs.good())
        {
            line = readNonEmptyLine(ifs);
            if (line.empty())
                break;

            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;
            keyword = toLower(keyword);

            if (keyword == "points")
            {
                Index count;
                std::string dataType;
                iss >> count >> dataType;
                numPoints = count;

                mesh.nodes.reserve(count);
                for (auto i = 0; i < count; ++i)
                {
                    Point3D pt;
                    ifs >> pt[0] >> pt[1] >> pt[2];
                    mesh.nodes.push_back(pt);
                }
            }
            else if (keyword == "cells")
            {
                Index count, totalSize;
                iss >> count >> totalSize;
                numCells = count;

                mesh.elements.reserve(count);
                for (auto i = 0; i < count; ++i)
                {
                    Index numNodes;
                    ifs >> numNodes;
                    CellConnectivity cell(numNodes);
                    for (auto j = 0; j < numNodes; ++j)
                    {
                        ifs >> cell[j];
                    }
                    mesh.elements.push_back(std::move(cell));
                }
            }
            else if (keyword == "cell_types")
            {
                Index count;
                iss >> count;

                mesh.elementTypes.reserve(count);
                for (auto i = 0; i < count; ++i)
                {
                    int cellType;
                    ifs >> cellType;
                    mesh.elementTypes.push_back(cellType);
                }
            }
            else if (keyword == "cell_data")
            {
                Index count;
                iss >> count;
                parseVTKCellData(ifs, mesh, count);
            }
            else if (keyword == "point_data")
            {
                Index count;
                iss >> count;
                parseVTKPointData(ifs, mesh, count);
            }
        }

        ifs.close();
        std::cout << "VTK file read: " << filename << std::endl;
        std::cout << "  Nodes: " << mesh.nodes.size() << std::endl;
        std::cout << "  Elements: " << mesh.elements.size() << std::endl;

        return mesh;
    }

    void VTKReader::parseVTKHeader(std::istream &is, std::string &title, bool &binary)
    {
        // Line 1: version
        std::string line;
        std::getline(is, line);
        if (line.find("vtk DataFile") == std::string::npos)
        {
            throw std::runtime_error("Invalid VTK file: missing version header");
        }

        // Line 2: title
        std::getline(is, title);
        title = trim(title);

        // Line 3: ASCII or BINARY
        std::getline(is, line);
        line = trim(line);
        std::string format = toLower(line);
        if (format == "ascii")
        {
            binary = false;
        }
        else if (format == "binary")
        {
            binary = true;
        }
        else
        {
            throw std::runtime_error("Invalid VTK format: expected ASCII or BINARY");
        }
    }

    void VTKReader::parseVTKCellData(std::istream &is, MeshInfo &mesh, Index numCells)
    {
        std::string line;
        while (is.good())
        {
            std::streampos pos = is.tellg();
            line = readNonEmptyLine(is);
            if (line.empty())
                break;

            std::string keyword = toLower(line.substr(0, line.find(' ')));

            // Check if we've hit a new section
            if (keyword == "point_data" || keyword == "points" ||
                keyword == "cells" || keyword == "cell_types")
            {
                is.seekg(pos); // Rewind to let main loop handle it
                break;
            }

            if (keyword == "scalars")
            {
                // Parse variable name from "SCALARS name type ncomp"
                std::istringstream iss(line);
                std::string kw, name, dataType;
                iss >> kw >> name >> dataType;

                // Read LOOKUP_TABLE line
                readNonEmptyLine(is);

                // Read values and store in mesh.cellData
                std::vector<Real> values(numCells);
                for (Index i = 0; i < numCells; ++i)
                {
                    is >> values[i];
                }
                mesh.cellData[name] = std::move(values);
            }
            else if (keyword == "vectors")
            {
                // Skip vector values for now
                for (Index i = 0; i < numCells; ++i)
                {
                    Real x, y, z;
                    is >> x >> y >> z;
                }
            }
        }
    }

    void VTKReader::parseVTKPointData(std::istream &is, MeshInfo &mesh, Index numPoints)
    {
        // Similar to parseVTKCellData - skip for now
        std::string line;
        while (is.good())
        {
            std::streampos pos = is.tellg();
            line = readNonEmptyLine(is);
            if (line.empty())
                break;

            std::string keyword = toLower(line.substr(0, line.find(' ')));

            if (keyword == "cell_data" || keyword == "points" ||
                keyword == "cells" || keyword == "cell_types")
            {
                is.seekg(pos);
                break;
            }

            if (keyword == "scalars")
            {
                readNonEmptyLine(is); // LOOKUP_TABLE
                for (auto i = 0; i < numPoints; ++i)
                {
                    Real val;
                    is >> val;
                }
            }
            else if (keyword == "vectors")
            {
                for (auto i = 0; i < numPoints; ++i)
                {
                    Real x, y, z;
                    is >> x >> y >> z;
                }
            }
        }
    }

    // Simple XML parsing helpers for VTU
    namespace
    {

        std::string getXMLAttribute(const std::string &line, const std::string &attr)
        {
            std::string search = attr + "=\"";
            auto pos = line.find(search);
            if (pos == std::string::npos)
                return "";
            pos += search.length();
            auto endPos = line.find('"', pos);
            if (endPos == std::string::npos)
                return "";
            return line.substr(pos, endPos - pos);
        }

        bool startsWith(const std::string &str, const std::string &prefix)
        {
            return str.length() >= prefix.length() &&
                   str.compare(0, prefix.length(), prefix) == 0;
        }

        // Base64 decoding table
        static const unsigned char b64_decode_table[256] = {
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
             52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,  0,255,255,
            255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
            255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        };

        std::vector<unsigned char> base64_decode(const std::string &input)
        {
            // Strip whitespace
            std::string clean;
            clean.reserve(input.size());
            for (char c : input)
            {
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
                    clean.push_back(c);
            }

            std::vector<unsigned char> result;
            if (clean.size() < 4)
                return result;

            result.reserve(clean.size() * 3 / 4);

            for (size_t i = 0; i + 3 < clean.size(); i += 4)
            {
                unsigned char a = b64_decode_table[static_cast<unsigned char>(clean[i])];
                unsigned char b = b64_decode_table[static_cast<unsigned char>(clean[i + 1])];
                unsigned char c = b64_decode_table[static_cast<unsigned char>(clean[i + 2])];
                unsigned char d = b64_decode_table[static_cast<unsigned char>(clean[i + 3])];

                result.push_back((a << 2) | (b >> 4));
                if (clean[i + 2] != '=')
                    result.push_back((b << 4) | (c >> 2));
                if (clean[i + 3] != '=')
                    result.push_back((c << 6) | d);
            }
            return result;
        }

        // Decode a VTK binary data block: [uint32 header (byte count)] + [raw data]
        // Returns raw data bytes (after the header).
        std::vector<unsigned char> decode_vtk_binary_block(const std::string &b64)
        {
            auto bytes = base64_decode(b64);
            if (bytes.size() < sizeof(uint32_t))
                return {};

            uint32_t nbytes;
            std::memcpy(&nbytes, bytes.data(), sizeof(uint32_t));

            size_t data_start = sizeof(uint32_t);
            size_t data_end = data_start + nbytes;
            if (data_end > bytes.size())
                data_end = bytes.size();

            return std::vector<unsigned char>(bytes.begin() + data_start,
                                              bytes.begin() + data_end);
        }

        // Extract typed array from raw bytes
        template <typename T>
        std::vector<T> bytes_to_vector(const std::vector<unsigned char> &raw)
        {
            size_t count = raw.size() / sizeof(T);
            std::vector<T> result(count);
            if (count > 0)
                std::memcpy(result.data(), raw.data(), count * sizeof(T));
            return result;
        }

    } // namespace

    MeshInfo VTKReader::readVTU(const std::string &filename)
    {
        std::ifstream ifs(filename);
        if (!ifs)
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        MeshInfo mesh;
        std::string line;

        // Find Piece element to get counts
        Index numPoints = 0;
        Index numCells = 0;

        while (std::getline(ifs, line))
        {
            line = trim(line);
            if (line.find("<Piece") != std::string::npos)
            {
                std::string np = getXMLAttribute(line, "NumberOfPoints");
                std::string nc = getXMLAttribute(line, "NumberOfCells");
                if (!np.empty())
                    numPoints = std::stoull(np);
                if (!nc.empty())
                    numCells = std::stoull(nc);
                break;
            }
        }

        if (numPoints == 0 || numCells == 0)
        {
            throw std::runtime_error("Invalid VTU file: missing Piece element or counts");
        }

        mesh.nodes.reserve(numPoints);
        mesh.elements.reserve(numCells);
        mesh.elementTypes.reserve(numCells);

        // Parse data arrays
        enum class Section
        {
            None,
            Points,
            Cells,
            PointData,
            CellData
        };
        Section currentSection = Section::None;
        std::string currentArrayName;

        std::vector<Index> connectivity;
        std::vector<Index> offsets;

        while (std::getline(ifs, line))
        {
            line = trim(line);

            if (line.find("<Points>") != std::string::npos)
            {
                currentSection = Section::Points;
            }
            else if (line.find("</Points>") != std::string::npos)
            {
                currentSection = Section::None;
            }
            else if (line.find("<Cells>") != std::string::npos)
            {
                currentSection = Section::Cells;
            }
            else if (line.find("</Cells>") != std::string::npos)
            {
                currentSection = Section::None;
            }
            else if (line.find("<PointData") != std::string::npos)
            {
                currentSection = Section::PointData;
            }
            else if (line.find("</PointData>") != std::string::npos)
            {
                currentSection = Section::None;
            }
            else if (line.find("<CellData") != std::string::npos)
            {
                currentSection = Section::CellData;
            }
            else if (line.find("</CellData>") != std::string::npos)
            {
                currentSection = Section::None;
            }
            else if (line.find("<DataArray") != std::string::npos)
            {
                currentArrayName = getXMLAttribute(line, "Name");
                std::string format = getXMLAttribute(line, "format");
                std::string dataType = getXMLAttribute(line, "type");
                bool isBinary = (format == "binary");

                // Read data content (between > and </DataArray>)
                auto closeTag = line.find("</DataArray>");
                auto endBracket = line.find('>');

                std::string dataContent;
                if (closeTag != std::string::npos && endBracket != std::string::npos)
                {
                    dataContent = line.substr(endBracket + 1, closeTag - endBracket - 1);
                }
                else
                {
                    std::ostringstream oss;
                    while (std::getline(ifs, line))
                    {
                        if (line.find("</DataArray>") != std::string::npos)
                            break;
                        oss << line << " ";
                    }
                    dataContent = oss.str();
                }

                if (currentSection == Section::Points)
                {
                    if (isBinary)
                    {
                        auto raw = decode_vtk_binary_block(dataContent);
                        auto coords = bytes_to_vector<double>(raw);
                        for (size_t i = 0; i + 2 < coords.size(); i += 3)
                        {
                            mesh.nodes.push_back({coords[i], coords[i + 1], coords[i + 2]});
                        }
                    }
                    else
                    {
                        std::istringstream ds(dataContent);
                        for (Index i = 0; i < numPoints; ++i)
                        {
                            Point3D pt{};
                            ds >> pt[0] >> pt[1] >> pt[2];
                            mesh.nodes.push_back(pt);
                        }
                    }
                }
                else if (currentSection == Section::Cells)
                {
                    if (currentArrayName == "connectivity")
                    {
                        if (isBinary)
                        {
                            auto raw = decode_vtk_binary_block(dataContent);
                            auto vals = bytes_to_vector<int32_t>(raw);
                            connectivity.reserve(vals.size());
                            for (auto v : vals)
                                connectivity.push_back(static_cast<Index>(v));
                        }
                        else
                        {
                            std::istringstream ds(dataContent);
                            Index idx;
                            while (ds >> idx)
                                connectivity.push_back(idx);
                        }
                    }
                    else if (currentArrayName == "offsets")
                    {
                        if (isBinary)
                        {
                            auto raw = decode_vtk_binary_block(dataContent);
                            auto vals = bytes_to_vector<int32_t>(raw);
                            offsets.reserve(vals.size());
                            for (auto v : vals)
                                offsets.push_back(static_cast<Index>(v));
                        }
                        else
                        {
                            std::istringstream ds(dataContent);
                            Index off;
                            while (ds >> off)
                                offsets.push_back(off);
                        }
                    }
                    else if (currentArrayName == "types")
                    {
                        if (isBinary)
                        {
                            auto raw = decode_vtk_binary_block(dataContent);
                            auto vals = bytes_to_vector<uint8_t>(raw);
                            for (auto v : vals)
                                mesh.elementTypes.push_back(static_cast<int>(v));
                        }
                        else
                        {
                            std::istringstream ds(dataContent);
                            int cellType;
                            while (ds >> cellType)
                                mesh.elementTypes.push_back(cellType);
                        }
                    }
                }
                else if (currentSection == Section::CellData)
                {
                    std::vector<Real> values;
                    if (isBinary)
                    {
                        auto raw = decode_vtk_binary_block(dataContent);
                        values = bytes_to_vector<double>(raw);
                    }
                    else
                    {
                        std::istringstream ds(dataContent);
                        Real val;
                        while (ds >> val)
                            values.push_back(val);
                    }
                    if (!currentArrayName.empty() && !values.empty())
                    {
                        mesh.cellData[currentArrayName] = std::move(values);
                    }
                }
                else if (currentSection == Section::PointData)
                {
                    std::vector<Real> values;
                    if (isBinary)
                    {
                        auto raw = decode_vtk_binary_block(dataContent);
                        values = bytes_to_vector<double>(raw);
                    }
                    else
                    {
                        std::istringstream ds(dataContent);
                        Real val;
                        while (ds >> val)
                            values.push_back(val);
                    }
                    if (!currentArrayName.empty() && !values.empty())
                    {
                        mesh.pointData[currentArrayName] = std::move(values);
                    }
                }
            }

            if (line.find("</Piece>") != std::string::npos)
            {
                break;
            }
        }

        // Build cells from connectivity and offsets
        if (!connectivity.empty() && !offsets.empty())
        {
            Index prevOffset = 0;
            for (size_t i = 0; i < offsets.size(); ++i)
            {
                Index currentOffset = offsets[i];
                CellConnectivity cell;
                for (auto j = prevOffset; j < currentOffset; ++j)
                {
                    cell.push_back(connectivity[j]);
                }
                mesh.elements.push_back(std::move(cell));
                prevOffset = currentOffset;
            }
        }

        ifs.close();
        std::cout << "VTU file read: " << filename << std::endl;
        std::cout << "  Nodes: " << mesh.nodes.size() << std::endl;
        std::cout << "  Elements: " << mesh.elements.size() << std::endl;

        return mesh;
    }

} // namespace fvm
