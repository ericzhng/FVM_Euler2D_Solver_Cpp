# =============================================================================
# Physics Module
# =============================================================================
# Physics models: Euler equations and Shallow Water equations
# Includes flux computation (Roe, HLLC) and state conversions
# =============================================================================

set(FVM2D_PHYSICS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_equations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shallow_water.cpp
    PARENT_SCOPE
)
