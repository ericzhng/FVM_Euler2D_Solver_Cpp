#pragma once

#include "fvm2d/core/types.hpp"
#include "fvm2d/mesh/partition_mesh.hpp"
#include <mpi.h>

namespace fvm2d {

/**
 * @brief Exchange halo cell data between MPI ranks
 *
 * Uses non-blocking MPI communication for efficiency
 *
 * @param mesh Partition mesh with send/recv maps
 * @param U Solution state array (modified in-place for halo cells)
 * @param comm MPI communicator
 */
void exchange_halo_data(
    const PartitionMesh& mesh,
    StateArray& U,
    MPI_Comm comm
);

/**
 * @brief Halo exchange manager for efficient repeated exchanges
 *
 * Pre-allocates buffers and sets up persistent communication patterns
 */
class HaloExchange {
public:
    /**
     * @brief Construct halo exchange manager
     * @param mesh Partition mesh
     * @param num_vars Number of solution variables
     * @param comm MPI communicator
     */
    HaloExchange(const PartitionMesh& mesh, int num_vars, MPI_Comm comm);

    ~HaloExchange();

    /**
     * @brief Perform halo exchange
     * @param U Solution state array (modified in-place)
     */
    void exchange(StateArray& U);

private:
    const PartitionMesh& mesh_;
    int num_vars_;
    MPI_Comm comm_;

    // Pre-allocated buffers
    std::vector<std::vector<Scalar>> send_buffers_;
    std::vector<std::vector<Scalar>> recv_buffers_;

    // Neighbor ranks
    std::vector<int> neighbor_ranks_;
};

}  // namespace fvm2d
