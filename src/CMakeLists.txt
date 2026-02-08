# =============================================================================
# Common compile settings for all modules
# =============================================================================
# INTERFACE library that provides include dirs, compile defs, and deps
# to all OBJECT libraries defined in subdirectories.
add_library(fvm2d_deps INTERFACE)
add_library(fvm2d::fvm2d_deps ALIAS fvm2d_deps)

target_include_directories(fvm2d_deps
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(fvm2d_deps INTERFACE FVM2D_LIBRARY_TYPE)
target_link_libraries(fvm2d_deps
    INTERFACE
        MPI::MPI_CXX
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
        CLI11::CLI11
)

if(FVM2D_USE_OPENMP AND OpenMP_FOUND)
    target_link_libraries(fvm2d_deps INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(fvm2d_deps INTERFACE FVM2D_USE_OPENMP)
endif()

# Add library subdirectories
add_subdirectory(common)
add_subdirectory(fvm2d_lib)
add_subdirectory(vtkio)
add_subdirectory(tecplot)

# Add applications subdirectory (executables)
add_subdirectory(apps)
