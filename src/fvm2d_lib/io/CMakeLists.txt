# =============================================================================
# I/O Module
# =============================================================================
# Configuration parsing (YAML) and output writers (VTK, Tecplot)
# =============================================================================

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/config_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tecplot_writer.cpp
)

set(CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tecplot_writer.hpp
)

set(FVM2D_IO_SOURCES
    ${CPP_SOURCES} ${CPP_HEADERS}
    PARENT_SCOPE
)
