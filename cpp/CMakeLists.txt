cmake_minimum_required(VERSION 3.18)
project(fvm2d VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Compiler Flags
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /W4 /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# =============================================================================
# Options
# =============================================================================
option(FVM2D_USE_OPENMP "Enable OpenMP parallelization" ON)
option(FVM2D_BUILD_TESTS "Build unit tests" OFF)

# =============================================================================
# Dependencies
# =============================================================================

# MPI
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_CXX_INCLUDE_DIRS}")

# OpenMP (optional)
if(FVM2D_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP found")
    else()
        message(WARNING "OpenMP not found, building without OpenMP support")
        set(FVM2D_USE_OPENMP OFF)
    endif()
endif()

# Eigen (header-only)
find_package(Eigen3 3.4 CONFIG)
if(NOT Eigen3_FOUND)
    # Try to find Eigen manually
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
        PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
            $ENV{EIGEN3_ROOT}
            ${CMAKE_SOURCE_DIR}/extern/eigen
    )
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set EIGEN3_ROOT")
    endif()
else()
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
endif()

# yaml-cpp
find_package(yaml-cpp CONFIG)
if(NOT yaml-cpp_FOUND)
    find_library(YAML_CPP_LIB yaml-cpp)
    find_path(YAML_CPP_INCLUDE yaml-cpp/yaml.h)
    if(YAML_CPP_LIB AND YAML_CPP_INCLUDE)
        message(STATUS "yaml-cpp found: ${YAML_CPP_LIB}")
        add_library(yaml-cpp UNKNOWN IMPORTED)
        set_target_properties(yaml-cpp PROPERTIES
            IMPORTED_LOCATION "${YAML_CPP_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE}"
        )
    else()
        message(FATAL_ERROR "yaml-cpp not found. Please install yaml-cpp")
    endif()
else()
    message(STATUS "yaml-cpp found")
endif()

# =============================================================================
# Source Files
# =============================================================================
set(FVM2D_SOURCES
    src/io/config_parser.cpp
    src/mesh/mesh_reader.cpp
    src/mesh/halo_exchange.cpp
    src/physics/euler_equations.cpp
    src/physics/shallow_water.cpp
    src/gradient/gaussian_gradient.cpp
    src/boundary/boundary_condition.cpp
    src/solver/residual.cpp
    src/solver/fvm_solver.cpp
    src/time/time_integrator.cpp
    src/time/timestep.cpp
    src/io/vtk_writer.cpp
    src/io/tecplot_writer.cpp
)

# =============================================================================
# Library Target
# =============================================================================
add_library(fvm2d_lib ${FVM2D_SOURCES})

target_include_directories(fvm2d_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(fvm2d_lib
    PUBLIC
        MPI::MPI_CXX
        Eigen3::Eigen
        yaml-cpp
)

if(FVM2D_USE_OPENMP AND OpenMP_FOUND)
    target_link_libraries(fvm2d_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fvm2d_lib PUBLIC FVM2D_USE_OPENMP)
endif()

# =============================================================================
# Executable
# =============================================================================
add_executable(fvm2d_solver src/main.cpp)
target_link_libraries(fvm2d_solver PRIVATE fvm2d_lib)

# =============================================================================
# Installation
# =============================================================================
install(TARGETS fvm2d_lib fvm2d_solver
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/fvm2d DESTINATION include)

# =============================================================================
# Tests (optional)
# =============================================================================
if(FVM2D_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_fvm2d
        tests/test_euler_flux.cpp
        tests/test_gradient.cpp
        tests/test_limiters.cpp
    )

    target_link_libraries(test_fvm2d
        PRIVATE
            fvm2d_lib
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(test_fvm2d)
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "FVM2D Configuration Summary:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP: ${FVM2D_USE_OPENMP}")
message(STATUS "  Build Tests: ${FVM2D_BUILD_TESTS}")
message(STATUS "")
