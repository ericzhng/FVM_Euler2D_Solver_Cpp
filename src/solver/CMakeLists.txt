# =============================================================================
# Solver Module
# =============================================================================
# Main FVM solver orchestrator and residual computation
# =============================================================================

set(FVM2D_SOLVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fvm_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual.cpp
    PARENT_SCOPE
)
