cmake_minimum_required(VERSION 3.18)
project(fvm2d VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# CMake Module and Prefix Path
# =============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/extern")

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
# Output Directories
# =============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# =============================================================================
# Compiler Flags
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /W4 /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    # Enable parallel compilation on MSVC
    add_compile_options(/MP)
endif()

# =============================================================================
# Options
# =============================================================================
option(FVM2D_USE_OPENMP "Enable OpenMP parallelization" ON)
option(FVM2D_BUILD_TESTS "Build unit tests" OFF)

# =============================================================================
# Dependencies
# =============================================================================

# =============================================================================
# MPI Configuration (Cross-Platform)
# =============================================================================
# Supports:
#   - Windows: MS-MPI (Microsoft MPI)
#   - Linux:   OpenMPI, MPICH, Intel MPI
#   - macOS:   OpenMPI, MPICH
# =============================================================================
set(FVM2D_MPI_FOUND FALSE)
set(FVM2D_MPI_IMPL "Unknown")

if(WIN32)
    # -------------------------------------------------------------------------
    # Windows: Try MS-MPI first, then fall back to standard find_package
    # -------------------------------------------------------------------------
    message(STATUS "Searching for MPI on Windows...")

    # Try standard CMake MPI finder first (works if msmpicc is in PATH)
    find_package(MPI QUIET)

    if(MPI_FOUND AND TARGET MPI::MPI_CXX)
        set(FVM2D_MPI_FOUND TRUE)
        set(FVM2D_MPI_IMPL "MPI (auto-detected)")
        message(STATUS "MPI found via find_package(MPI)")
    else()
        # Try our custom MS-MPI finder
        message(STATUS "Standard MPI not found, trying MS-MPI...")
        find_package(MSMPI QUIET)

        if(MSMPI_FOUND)
            set(FVM2D_MPI_FOUND TRUE)
            set(FVM2D_MPI_IMPL "MS-MPI")

            # Create MPI::MPI_CXX target if not exists
            if(NOT TARGET MPI::MPI_CXX)
                add_library(MPI::MPI_CXX UNKNOWN IMPORTED)
                set_target_properties(MPI::MPI_CXX PROPERTIES
                    IMPORTED_LOCATION "${MSMPI_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${MSMPI_INCLUDE_DIR}"
                )
            endif()

            set(MPI_CXX_INCLUDE_DIRS "${MSMPI_INCLUDE_DIRS}")
            set(MPI_CXX_LIBRARIES "${MSMPI_LIBRARIES}")
            set(MPIEXEC_EXECUTABLE "${MSMPI_MPIEXEC}")

            message(STATUS "MS-MPI found:")
            message(STATUS "  Include: ${MSMPI_INCLUDE_DIR}")
            message(STATUS "  Library: ${MSMPI_LIBRARY}")
            if(MSMPI_VERSION)
                message(STATUS "  Version: ${MSMPI_VERSION}")
            endif()
        endif()
    endif()

    if(NOT FVM2D_MPI_FOUND)
        message(FATAL_ERROR
            "MPI not found on Windows!\n"
            "Please install Microsoft MPI:\n"
            "  1. Download from: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi\n"
            "  2. Install both 'msmpisetup.exe' (runtime) and 'msmpisdk.msi' (SDK)\n"
            "  3. Ensure environment variables MSMPI_INC and MSMPI_LIB64 are set\n"
            "     (They should be set automatically after installation)"
        )
    endif()

else()
    # -------------------------------------------------------------------------
    # Unix/Linux/macOS: Use standard CMake MPI finder
    # -------------------------------------------------------------------------
    message(STATUS "Searching for MPI on Unix/Linux/macOS...")
    find_package(MPI REQUIRED)

    if(MPI_FOUND AND TARGET MPI::MPI_CXX)
        set(FVM2D_MPI_FOUND TRUE)

        # Detect MPI implementation
        if(DEFINED MPI_CXX_COMPILER)
            execute_process(
                COMMAND ${MPI_CXX_COMPILER} --version
                OUTPUT_VARIABLE _mpi_version_output
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(_mpi_version_output MATCHES "Open MPI")
                set(FVM2D_MPI_IMPL "OpenMPI")
            elseif(_mpi_version_output MATCHES "MPICH")
                set(FVM2D_MPI_IMPL "MPICH")
            elseif(_mpi_version_output MATCHES "Intel")
                set(FVM2D_MPI_IMPL "Intel MPI")
            else()
                set(FVM2D_MPI_IMPL "MPI")
            endif()
        else()
            set(FVM2D_MPI_IMPL "MPI")
        endif()

        message(STATUS "${FVM2D_MPI_IMPL} found:")
        message(STATUS "  Include: ${MPI_CXX_INCLUDE_DIRS}")
        if(MPI_CXX_VERSION)
            message(STATUS "  Version: ${MPI_CXX_VERSION}")
        endif()
    endif()
endif()

# Verify MPI target exists
if(NOT TARGET MPI::MPI_CXX)
    message(FATAL_ERROR "MPI::MPI_CXX target not available after MPI configuration")
endif()

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

# Eigen (header-only) - Use extern version directly
message(STATUS "Using Eigen3 from 'extern/eigen-3.4.1'")
add_subdirectory(extern/eigen-3.4.1 EXCLUDE_FROM_ALL)

# Suppress warnings from Eigen headers only (target the actual 'eigen' target, not the alias)
get_target_property(EIGEN_IID eigen INTERFACE_INCLUDE_DIRECTORIES)
if(EIGEN_IID)
    set_target_properties(eigen PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${EIGEN_IID}"
    )
endif()

# yaml-cpp - Use extern version directly
message(STATUS "Using yaml-cpp from 'extern/yaml-cpp-0.8.0'")
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/yaml-cpp-0.8.0 EXCLUDE_FROM_ALL)

# =============================================================================
# Source Files (Modular Structure)
# =============================================================================
# Each subdirectory defines its sources and propagates them via PARENT_SCOPE
add_subdirectory(src)

# =============================================================================
# Library Target
# =============================================================================
add_library(fvm2d_lib STATIC ${FVM2D_ALL_SOURCES})

# Add an alias for modern CMake usage
add_library(fvm2d::fvm2d ALIAS fvm2d_lib)

# Include directories (headers are now alongside sources in src/)
target_include_directories(fvm2d_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(fvm2d_lib
    PUBLIC
        MPI::MPI_CXX
        Eigen3::Eigen
        yaml-cpp
)

# OpenMP support
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
include(GNUInstallDirs)

# Install executable only (library depends on local extern targets that can't be exported)
install(TARGETS fvm2d_solver
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers (organized by module)
install(FILES src/fvm2d.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY src/core src/mesh src/physics src/boundary src/gradient src/limiter src/solver src/time src/io
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

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
message(STATUS "============================================")
message(STATUS "FVM2D Configuration Summary")
message(STATUS "============================================")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  Dependencies:")
message(STATUS "    MPI:       ${FVM2D_MPI_IMPL}")
message(STATUS "    Eigen3:    Found")
message(STATUS "    yaml-cpp:  Found")
message(STATUS "    OpenMP:    ${FVM2D_USE_OPENMP}")
message(STATUS "")
message(STATUS "  Options:")
message(STATUS "    Build Tests: ${FVM2D_BUILD_TESTS}")
message(STATUS "")
message(STATUS "  Directories:")
message(STATUS "    Include:   ${CMAKE_CURRENT_SOURCE_DIR}/src")
message(STATUS "    Output:    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "============================================")
message(STATUS "")
