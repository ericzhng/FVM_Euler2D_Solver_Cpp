# =============================================================================
# Time Integration Module
# =============================================================================
# Time integrators (Explicit Euler, RK2) and adaptive time stepping
# =============================================================================

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/time_integrator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timestep.cpp
)

set(CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/time_integrator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timestep.hpp
)

set(FVM2D_TIME_SOURCES
    ${CPP_SOURCES} ${CPP_HEADERS}
    PARENT_SCOPE
)
