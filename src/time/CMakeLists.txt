# =============================================================================
# Time Integration Module
# =============================================================================
# Time integrators (Explicit Euler, RK2) and adaptive time stepping
# =============================================================================

set(FVM2D_TIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/time_integrator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timestep.cpp
    PARENT_SCOPE
)
