# =============================================================================
# Tecplot I/O Module
# =============================================================================
# Tecplot format writer for FVM2D solution output
# =============================================================================

set(TECPLOT_SOURCES
    tecplot_writer.cpp
)

set(TECPLOT_HEADERS
    tecplot_writer.hpp
)

# Create library
add_library(tecplot ${FVM2D_LIBRARY_TYPE} ${TECPLOT_SOURCES} ${TECPLOT_HEADERS})

target_link_libraries(tecplot PRIVATE fvm2d::fvm2d_lib fvm2d_deps)

# Handle export definitions for shared/static builds
if(BUILD_SHARED)
    target_compile_definitions(tecplot PRIVATE FVM_EXPORTS)
else()
    target_compile_definitions(tecplot PUBLIC FVM_STATIC)
endif()

target_compile_features(tecplot PUBLIC cxx_std_17)

# Export alias
add_library(fvm2d::tecplot ALIAS tecplot)
