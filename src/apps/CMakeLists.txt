# =============================================================================
# FVM2D Solver Application
# =============================================================================

add_executable(fvm2d_solver main_solver2d.cpp)

target_include_directories(fvm2d_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Link common definitions (numeric precision options)
target_link_libraries(fvm2d_solver PRIVATE fvm2d::fvm2d_lib fvm2d::vtkio)
