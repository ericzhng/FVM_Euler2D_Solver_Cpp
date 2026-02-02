# =============================================================================
# FVM2D Source Directory CMakeLists.txt
# =============================================================================
# This file aggregates all source modules into the fvm2d_lib library.
# Each subdirectory defines its own source files.
# Headers are now alongside sources in each module folder.
# =============================================================================

# Add subdirectories - each returns its source files
add_subdirectory(boundary)
add_subdirectory(core)       # Header-only module
add_subdirectory(gradient)
add_subdirectory(io)
add_subdirectory(limiter)    # Header-only module
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
    PARENT_SCOPE
)
