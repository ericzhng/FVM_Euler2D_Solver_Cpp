# =============================================================================
# Limiter Module (Header-only)
# =============================================================================
# Slope limiter functions for FVM reconstruction (inlined for performance)
#
# Headers:
#   - limiters.hpp: Minmod, Barth-Jespersen, Superbee, Van Leer limiters
# =============================================================================

# Header-only module - no source files

set(FVM2D_LIMITER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/limiters.hpp
    PARENT_SCOPE
)
