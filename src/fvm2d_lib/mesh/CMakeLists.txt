# =============================================================================
# Mesh Module
# =============================================================================
# Partition mesh loading and MPI halo exchange communication
# =============================================================================

set(CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_builder.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/mesh_reader.cpp
)

set(CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_builder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/partition_mesh.hpp
)

set(FVM2D_MESH_SOURCES
    ${CPP_SOURCES} ${CPP_HEADERS}
    PARENT_SCOPE
)
