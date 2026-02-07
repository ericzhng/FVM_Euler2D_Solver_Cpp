# =============================================================================
# Boundary Conditions Module
# =============================================================================
# Handles application of boundary conditions (transmissive, inlet, outlet, wall)
# =============================================================================

set(FVM2D_BOUNDARY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary_condition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/boundary_condition.hpp
    PARENT_SCOPE
)
