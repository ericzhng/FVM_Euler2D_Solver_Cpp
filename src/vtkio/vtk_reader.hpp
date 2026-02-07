#pragma once

#include "common/fvm_export.hpp"
#include "common/fvm_types.hpp"
#include <string>

namespace fvm
{

    /**
     * @brief Utility class for reading mesh data from VTK formats.
     *
     * Supports legacy VTK (.vtk) and XML VTU (.vtu) formats.
     * Reads unstructured grid data into MeshInfo structure.
     */
    class FVM_API VTKReader
    {
    public:
        /**
         * @brief Read mesh data from legacy VTK format (.vtk).
         * @param filename Input filename
         * @return MeshInfo structure with loaded mesh
         * @throws std::runtime_error if file cannot be read or format is invalid
         */
        static MeshInfo readVTK(const std::string &filename);

        /**
         * @brief Read mesh data from XML VTU format (.vtu).
         * @param filename Input filename
         * @return MeshInfo structure with loaded mesh
         * @throws std::runtime_error if file cannot be read or format is invalid
         */
        static MeshInfo readVTU(const std::string &filename);

        /**
         * @brief Read mesh from file, auto-detecting format by extension.
         * @param filename Input filename (.vtk or .vtu)
         * @return MeshInfo structure with loaded mesh
         * @throws std::runtime_error if format is unsupported or read fails
         */
        static MeshInfo read(const std::string &filename);

    private:
        /// Parse VTK header and verify format
        static void parseVTKHeader(std::istream &is, std::string &title, bool &binary);

        /// Parse POINTS section
        static void parseVTKPoints(std::istream &is, MeshInfo &mesh);

        /// Parse CELLS section
        static void parseVTKCells(std::istream &is, MeshInfo &mesh);

        /// Parse CELL_TYPES section
        static void parseVTKCellTypes(std::istream &is, MeshInfo &mesh);

        /// Parse CELL_DATA section (optional)
        static void parseVTKCellData(std::istream &is, MeshInfo &mesh, Index numCells);

        /// Parse POINT_DATA section (optional)
        static void parseVTKPointData(std::istream &is, MeshInfo &mesh, Index numPoints);

        /// Helper to get file extension
        static std::string getExtension(const std::string &filename);
    };

} // namespace fvm
