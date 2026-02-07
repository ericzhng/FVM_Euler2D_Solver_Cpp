# =============================================================================
# FVM2D Source Directory CMakeLists.txt
# =============================================================================
# Each subdirectory defines its own OBJECT library.
# The top-level CMakeLists.txt links them into fvm2d_lib.
# =============================================================================

add_subdirectory(boundary)
add_subdirectory(core)       # Header-only module
add_subdirectory(gradient)
add_subdirectory(io)
add_subdirectory(limiter)    # Header-only module
add_subdirectory(mesh)
add_subdirectory(physics)
add_subdirectory(solver)
add_subdirectory(time)

# =============================================================================
# Library Target
# =============================================================================
add_library(fvm2d_lib STATIC)

# Link common settings (include dirs, compile defs, external deps)
target_link_libraries(fvm2d_lib PUBLIC fvm2d_deps)

# Link OBJECT libraries from each module
target_link_libraries(fvm2d_lib
    PRIVATE
        fvm2d_boundary
        fvm2d_gradient
        fvm2d_io
        fvm2d_mesh
        fvm2d_physics
        fvm2d_solver_lib
        fvm2d_time
        fvm2d_vtkio
)

# Add an alias for modern CMake usage
add_library(fvm2d::fvm2d ALIAS fvm2d_lib)
