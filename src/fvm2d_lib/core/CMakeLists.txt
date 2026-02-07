# =============================================================================
# Core Module (Header-only)
# =============================================================================
# Fundamental types, enums, and configuration structures
#
# Headers:
#   - types.hpp: Basic types (Scalar, Index, Eigen types, enums)
#   - config.hpp: Configuration structures and parser declarations
# =============================================================================

# Header-only module - no source files

set(FVM2D_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.hpp
    PARENT_SCOPE
)
