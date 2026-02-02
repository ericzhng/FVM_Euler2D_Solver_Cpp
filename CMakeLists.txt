cmake_minimum_required(VERSION 3.18)
project(fvm2d VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# CMake Module Path
# =============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
option(FVM2D_USE_FVM_MESH "Use FVM_mesh library for mesh operations" ON)

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

# Eigen (header-only)
find_package(Eigen3 3.4 CONFIG)
if(NOT Eigen3_FOUND)
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
# FVM_mesh Library (external mesh library)
# =============================================================================
if(FVM2D_USE_FVM_MESH)
    # Set the path based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(FVM_MESH_DIR "${CMAKE_SOURCE_DIR}/extern/FVM_mesh/Debug")
    else()
        set(FVM_MESH_DIR "${CMAKE_SOURCE_DIR}/extern/FVM_mesh/Release")
    endif()

    # Add the FVM_mesh cmake directory to the module path
    list(APPEND CMAKE_PREFIX_PATH "${FVM_MESH_DIR}/lib/cmake/FVM_mesh")
    list(APPEND CMAKE_MODULE_PATH "${FVM_MESH_DIR}/lib/cmake/FVM_mesh/modules")

    # Find FVM_mesh package
    find_package(FVM_mesh CONFIG)
    if(FVM_mesh_FOUND)
        message(STATUS "FVM_mesh found: ${FVM_MESH_DIR}")
    else()
        message(WARNING "FVM_mesh not found at ${FVM_MESH_DIR}")
        set(FVM2D_USE_FVM_MESH OFF)
    endif()
endif()

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

# Link FVM_mesh if enabled and found
if(FVM2D_USE_FVM_MESH AND FVM_mesh_FOUND)
    target_link_libraries(fvm2d_lib PUBLIC FVM::polymesh)
    target_include_directories(fvm2d_lib PUBLIC
        $<BUILD_INTERFACE:${FVM_MESH_DIR}/include>
    )
    target_compile_definitions(fvm2d_lib PUBLIC FVM2D_HAS_FVM_MESH)
endif()

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
# Copy DLLs for Windows Runtime
# =============================================================================
if(WIN32 AND FVM2D_USE_FVM_MESH AND FVM_mesh_FOUND)
    set(FVM_MESH_DLLS
        "${FVM_MESH_DIR}/bin/polymesh.dll"
        "${FVM_MESH_DIR}/bin/meshgen.dll"
        "${FVM_MESH_DIR}/bin/gmsh-4.15.dll"
        "${FVM_MESH_DIR}/bin/metis.dll"
    )

    foreach(DLL_FILE ${FVM_MESH_DLLS})
        if(EXISTS "${DLL_FILE}")
            add_custom_command(TARGET fvm2d_solver POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL_FILE}"
                    $<TARGET_FILE_DIR:fvm2d_solver>
                COMMENT "Copying ${DLL_FILE}"
            )
        endif()
    endforeach()
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

install(TARGETS fvm2d_lib fvm2d_solver
    EXPORT fvm2dTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers (organized by module)
install(FILES src/fvm2d.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY src/core src/mesh src/physics src/boundary src/gradient src/limiter src/solver src/time src/io
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT fvm2dTargets
    FILE fvm2dTargets.cmake
    NAMESPACE fvm2d::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fvm2d
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
message(STATUS "    FVM_mesh:  ${FVM2D_USE_FVM_MESH}")
message(STATUS "")
message(STATUS "  Options:")
message(STATUS "    Build Tests: ${FVM2D_BUILD_TESTS}")
message(STATUS "")
message(STATUS "  Directories:")
message(STATUS "    Include:   ${CMAKE_CURRENT_SOURCE_DIR}/src")
message(STATUS "    Output:    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "============================================")
message(STATUS "")
