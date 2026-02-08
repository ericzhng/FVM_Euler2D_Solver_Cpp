#include "mesh/halo_exchange.hpp"
#include <vector>

namespace fvm2d {

void exchange_halo_data(
    const PartitionMesh& mesh,
    StateArray& U,
    MPI_Comm comm
) {
    const int num_vars = static_cast<int>(U.cols());

    std::vector<MPI_Request> requests;
    std::vector<std::vector<Scalar>> send_buffers;
    std::vector<std::vector<Scalar>> recv_buffers;

    // Post non-blocking receives
    int recv_idx = 0;
    for (const auto& [neighbor_rank, local_indices] : mesh.recv_map) {
        int count = static_cast<int>(local_indices.size()) * num_vars;
        recv_buffers.emplace_back(count);

        MPI_Request req;
        MPI_Irecv(recv_buffers.back().data(), count, MPI_DOUBLE,
                  neighbor_rank, 0, comm, &req);
        requests.push_back(req);
        ++recv_idx;
    }

    // Pack and send data
    for (const auto& [neighbor_rank, local_indices] : mesh.send_map) {
        send_buffers.emplace_back();
        auto& buf = send_buffers.back();
        buf.reserve(local_indices.size() * num_vars);

        for (Index idx : local_indices) {
            for (int v = 0; v < num_vars; ++v) {
                buf.push_back(U(idx, v));
            }
        }

        MPI_Request req;
        MPI_Isend(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE,
                  neighbor_rank, 0, comm, &req);
        requests.push_back(req);
    }

    // Wait for all communications
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack received data into halo cells
    recv_idx = 0;
    for (const auto& [neighbor_rank, local_indices] : mesh.recv_map) {
        const auto& buf = recv_buffers[recv_idx];
        int offset = 0;

        for (Index idx : local_indices) {
            for (int v = 0; v < num_vars; ++v) {
                U(idx, v) = buf[offset++];
            }
        }
        ++recv_idx;
    }
}

HaloExchange::HaloExchange(const PartitionMesh& mesh, int num_vars, MPI_Comm comm)
    : mesh_(mesh), num_vars_(num_vars), comm_(comm)
{
    // Collect neighbor ranks
    for (const auto& [rank, indices] : mesh.send_map) {
        neighbor_ranks_.push_back(rank);
    }

    // Pre-allocate send buffers
    send_buffers_.resize(mesh.send_map.size());
    int idx = 0;
    for (const auto& [rank, indices] : mesh.send_map) {
        send_buffers_[idx].resize(indices.size() * num_vars);
        ++idx;
    }

    // Pre-allocate receive buffers
    recv_buffers_.resize(mesh.recv_map.size());
    idx = 0;
    for (const auto& [rank, indices] : mesh.recv_map) {
        recv_buffers_[idx].resize(indices.size() * num_vars);
        ++idx;
    }
}

HaloExchange::~HaloExchange() = default;

void HaloExchange::exchange(StateArray& U) {
    std::vector<MPI_Request> requests;

    // Post non-blocking receives
    int idx = 0;
    for (const auto& [neighbor_rank, local_indices] : mesh_.recv_map) {
        MPI_Request req;
        MPI_Irecv(recv_buffers_[idx].data(),
                  static_cast<int>(recv_buffers_[idx].size()),
                  MPI_DOUBLE, neighbor_rank, 0, comm_, &req);
        requests.push_back(req);
        ++idx;
    }

    // Pack and send data
    idx = 0;
    for (const auto& [neighbor_rank, local_indices] : mesh_.send_map) {
        auto& buf = send_buffers_[idx];
        int offset = 0;

        for (Index cell_idx : local_indices) {
            for (int v = 0; v < num_vars_; ++v) {
                buf[offset++] = U(cell_idx, v);
            }
        }

        MPI_Request req;
        MPI_Isend(buf.data(), static_cast<int>(buf.size()),
                  MPI_DOUBLE, neighbor_rank, 0, comm_, &req);
        requests.push_back(req);
        ++idx;
    }

    // Wait for all communications
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack received data
    idx = 0;
    for (const auto& [neighbor_rank, local_indices] : mesh_.recv_map) {
        const auto& buf = recv_buffers_[idx];
        int offset = 0;

        for (Index cell_idx : local_indices) {
            for (int v = 0; v < num_vars_; ++v) {
                U(cell_idx, v) = buf[offset++];
            }
        }
        ++idx;
    }
}

}  // namespace fvm2d
