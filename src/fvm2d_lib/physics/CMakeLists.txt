# =============================================================================
# Physics Module
# =============================================================================
# Physics models: Euler equations and Shallow Water equations
# Includes flux computation (Roe, HLLC) and state conversions
# =============================================================================

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_equations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shallow_water.cpp
)

set(CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/euler_equations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shallow_water.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics_model.hpp
)

set(FVM2D_PHYSICS_SOURCES
    ${CPP_SOURCES} ${CPP_HEADERS}
    PARENT_SCOPE
)
