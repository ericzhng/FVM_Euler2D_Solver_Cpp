# =============================================================================
# FindMSMPI.cmake
# =============================================================================
# Find Microsoft MPI (MS-MPI) on Windows
#
# This module defines:
#   MSMPI_FOUND        - True if MS-MPI was found
#   MSMPI_INCLUDE_DIRS - MS-MPI include directories
#   MSMPI_LIBRARIES    - MS-MPI libraries to link
#   MSMPI_VERSION      - MS-MPI version (if available)
#
# This module also creates the following imported target:
#   MSMPI::MSMPI       - Imported target for MS-MPI
#
# Environment variables used:
#   MSMPI_INC          - Include directory (set by MS-MPI SDK)
#   MSMPI_LIB64        - 64-bit library directory
#   MSMPI_LIB32        - 32-bit library directory
#   MSMPI_BIN          - Binary directory (mpiexec)
#
# =============================================================================

if(NOT WIN32)
    message(FATAL_ERROR "FindMSMPI is only for Windows platforms")
endif()

# Determine architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_MSMPI_ARCH "x64")
    set(_MSMPI_LIB_ENV "MSMPI_LIB64")
else()
    set(_MSMPI_ARCH "x86")
    set(_MSMPI_LIB_ENV "MSMPI_LIB32")
endif()

# =============================================================================
# Find Include Directory
# =============================================================================
find_path(MSMPI_INCLUDE_DIR
    NAMES mpi.h
    HINTS
        ENV MSMPI_INC
        "$ENV{MSMPI_INC}"
    PATHS
        "$ENV{ProgramFiles}/Microsoft SDKs/MPI/Include"
        "$ENV{ProgramFiles\(x86\)}/Microsoft SDKs/MPI/Include"
        "$ENV{ProgramW6432}/Microsoft SDKs/MPI/Include"
        "C:/Program Files/Microsoft SDKs/MPI/Include"
        "C:/Program Files (x86)/Microsoft SDKs/MPI/Include"
    DOC "MS-MPI include directory"
)

# =============================================================================
# Find Library
# =============================================================================
find_library(MSMPI_LIBRARY
    NAMES msmpi
    HINTS
        ENV ${_MSMPI_LIB_ENV}
        "$ENV{${_MSMPI_LIB_ENV}}"
    PATHS
        "$ENV{ProgramFiles}/Microsoft SDKs/MPI/Lib/${_MSMPI_ARCH}"
        "$ENV{ProgramFiles\(x86\)}/Microsoft SDKs/MPI/Lib/${_MSMPI_ARCH}"
        "$ENV{ProgramW6432}/Microsoft SDKs/MPI/Lib/${_MSMPI_ARCH}"
        "C:/Program Files/Microsoft SDKs/MPI/Lib/${_MSMPI_ARCH}"
        "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/${_MSMPI_ARCH}"
    DOC "MS-MPI library"
)

# =============================================================================
# Find mpiexec
# =============================================================================
find_program(MSMPI_MPIEXEC
    NAMES mpiexec mpiexec.exe
    HINTS
        ENV MSMPI_BIN
        "$ENV{MSMPI_BIN}"
    PATHS
        "$ENV{ProgramFiles}/Microsoft MPI/Bin"
        "$ENV{ProgramFiles\(x86\)}/Microsoft MPI/Bin"
        "$ENV{ProgramW6432}/Microsoft MPI/Bin"
        "C:/Program Files/Microsoft MPI/Bin"
        "C:/Program Files (x86)/Microsoft MPI/Bin"
    DOC "MS-MPI mpiexec executable"
)

# =============================================================================
# Get Version
# =============================================================================
if(MSMPI_INCLUDE_DIR AND EXISTS "${MSMPI_INCLUDE_DIR}/mpi.h")
    file(STRINGS "${MSMPI_INCLUDE_DIR}/mpi.h" _msmpi_version_line
         REGEX "^#define[ \t]+MS_MPI_VERSION[ \t]+")
    if(_msmpi_version_line)
        string(REGEX REPLACE "^#define[ \t]+MS_MPI_VERSION[ \t]+0x([0-9A-Fa-f]+).*$"
               "\\1" _msmpi_version_hex "${_msmpi_version_line}")
        # Convert hex to decimal version (e.g., 0x100 = 1.0.0)
        math(EXPR _major "0x${_msmpi_version_hex} / 256")
        math(EXPR _minor "(0x${_msmpi_version_hex} % 256) / 16")
        math(EXPR _patch "0x${_msmpi_version_hex} % 16")
        set(MSMPI_VERSION "${_major}.${_minor}.${_patch}")
    endif()
endif()

# =============================================================================
# Handle REQUIRED and QUIET arguments
# =============================================================================
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MSMPI
    REQUIRED_VARS
        MSMPI_LIBRARY
        MSMPI_INCLUDE_DIR
    VERSION_VAR
        MSMPI_VERSION
)

# =============================================================================
# Set Output Variables
# =============================================================================
if(MSMPI_FOUND)
    set(MSMPI_INCLUDE_DIRS ${MSMPI_INCLUDE_DIR})
    set(MSMPI_LIBRARIES ${MSMPI_LIBRARY})

    # Create imported target
    if(NOT TARGET MSMPI::MSMPI)
        add_library(MSMPI::MSMPI UNKNOWN IMPORTED)
        set_target_properties(MSMPI::MSMPI PROPERTIES
            IMPORTED_LOCATION "${MSMPI_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MSMPI_INCLUDE_DIR}"
        )
    endif()

    # Also create MPI::MPI_CXX alias for compatibility
    if(NOT TARGET MPI::MPI_CXX)
        add_library(MPI::MPI_CXX ALIAS MSMPI::MSMPI)
    endif()
endif()

mark_as_advanced(
    MSMPI_INCLUDE_DIR
    MSMPI_LIBRARY
    MSMPI_MPIEXEC
)
