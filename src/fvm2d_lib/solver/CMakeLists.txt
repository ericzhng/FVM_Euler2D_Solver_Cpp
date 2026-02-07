# =============================================================================
# Solver Module
# =============================================================================
# Main FVM solver orchestrator and residual computation
# =============================================================================

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fvm_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual.cpp
)

set(CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/fvm_solver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual.hpp
)

set(FVM2D_SOLVER_SOURCES
    ${CPP_SOURCES} ${CPP_HEADERS}
    PARENT_SCOPE
)
