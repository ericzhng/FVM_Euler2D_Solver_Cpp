# =============================================================================
# I/O Module
# =============================================================================
# Configuration parsing (YAML) and output writers (VTK, Tecplot)
# =============================================================================

set(FVM2D_IO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/config_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tecplot_writer.cpp
    PARENT_SCOPE
)
