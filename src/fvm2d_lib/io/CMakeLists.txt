# =============================================================================
# I/O Module
# =============================================================================
# Configuration parsing (YAML) and solution output adapter
# =============================================================================

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/config_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.cpp
)

set(CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.hpp
)

set(FVM2D_IO_SOURCES
    ${CPP_SOURCES} ${CPP_HEADERS}
    PARENT_SCOPE
)
