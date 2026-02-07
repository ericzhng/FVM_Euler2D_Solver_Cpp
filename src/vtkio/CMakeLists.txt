# =============================================================================
# VTK I/O Module
# =============================================================================
# VTK and OpenFOAM mesh I/O utilities
# =============================================================================

set(VTKIO_SOURCES
    vtk_writer.cpp
    vtk_reader.cpp
)

set(VTKIO_HEADERS
    vtk_writer.hpp
    vtk_reader.hpp
    cell_types.hpp
)

# Create library (STATIC or SHARED based on FVM_LIBRARY_TYPE)
add_library(vtkio ${FVM2D_LIBRARY_TYPE} ${VTKIO_SOURCES} ${VTKIO_HEADERS})

target_link_libraries(vtkio PRIVATE fvm2d_deps)

# Handle export definitions for shared/static builds
if(BUILD_SHARED)
    # Building shared library - define EXPORTS macro
    target_compile_definitions(vtkio PRIVATE FVM_EXPORTS)

    # Set shared library properties
    set_target_properties(vtkio PROPERTIES
        # Version information
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        # Windows: don't add "lib" prefix
        PREFIX ""
        # Enable visibility on Unix
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    # On Windows, ensure runtime library is copied to output directory
    if(WIN32)
        set_target_properties(vtkio PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
    endif()
else()
    # Building static library - define STATIC macro
    target_compile_definitions(vtkio PUBLIC FVM_STATIC)
endif()

# Set C++ standard
target_compile_features(vtkio PUBLIC cxx_std_17)

# Export alias for easier usage
add_library(fvm2d::vtkio ALIAS vtkio)
