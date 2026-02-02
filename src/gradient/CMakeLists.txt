# =============================================================================
# Gradient Module
# =============================================================================
# Gaussian (Green-Gauss) gradient computation for FVM reconstruction
# =============================================================================

set(FVM2D_GRADIENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gaussian_gradient.cpp
    PARENT_SCOPE
)
