# =============================================================================
# Mesh Module
# =============================================================================
# Partition mesh loading and MPI halo exchange communication
# =============================================================================

set(FVM2D_MESH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange.cpp
    PARENT_SCOPE
)
