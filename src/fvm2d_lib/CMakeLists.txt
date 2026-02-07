# =============================================================================
# FVM2D Source Directory CMakeLists.txt
# =============================================================================
# Each subdirectory defines its own OBJECT library.
# The top-level CMakeLists.txt links them into fvm2d_lib.
# =============================================================================

add_subdirectory(core)       # Header-only module
add_subdirectory(limiter)    # Header-only module
add_subdirectory(boundary)
add_subdirectory(gradient)
add_subdirectory(io)
add_subdirectory(mesh)
add_subdirectory(physics)
add_subdirectory(solver)
add_subdirectory(time)

# Collect all sources from modules
set(FVM2D_ALL_SOURCES
    ${FVM2D_BOUNDARY_SOURCES}
    ${FVM2D_GRADIENT_SOURCES}
    ${FVM2D_IO_SOURCES}
    ${FVM2D_MESH_SOURCES}
    ${FVM2D_PHYSICS_SOURCES}
    ${FVM2D_SOLVER_SOURCES}
    ${FVM2D_TIME_SOURCES}
)

# =============================================================================
# Library Target
# =============================================================================
add_library(fvm2d_lib ${FVM_LIBRARY_TYPE} ${FVM2D_ALL_SOURCES})

# Set library properties
target_include_directories(fvm2d_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link yaml-cpp (required for YAML parsing) and common definitions
target_link_libraries(fvm2d_lib PUBLIC yaml-cpp::yaml-cpp fvm2d_deps)

# Handle export definitions for shared/static builds
if(BUILD_SHARED)
    # Building shared library - define EXPORTS macro
    target_compile_definitions(fvm2d_lib PRIVATE FVM_EXPORTS)

    # Set shared library properties
    set_target_properties(fvm2d_lib PROPERTIES
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
        set_target_properties(fvm2d_lib PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
    endif()
else()
    # Building static library - define STATIC macro
    target_compile_definitions(fvm2d_lib PUBLIC FVM_STATIC)
endif()

# Add an alias for modern CMake usage
add_library(fvm2d::fvm2d_lib ALIAS fvm2d_lib)
