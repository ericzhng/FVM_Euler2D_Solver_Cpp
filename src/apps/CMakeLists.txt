# =============================================================================
# FVM2D Applications
# =============================================================================

# --- Main solver (MPI) ---
add_executable(fvm2d_solver main_solver2d.cpp)

target_include_directories(fvm2d_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fvm2d_solver PRIVATE
    fvm2d::fvm2d_lib fvm2d::vtkio fvm2d::tecplot
)

# --- Reconstruction tool (serial, no MPI) ---
add_executable(reconstruct2d main_reconstruct2d.cpp)

target_include_directories(reconstruct2d PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(reconstruct2d PRIVATE
    fvm2d::fvm2d_lib fvm2d::vtkio
)
