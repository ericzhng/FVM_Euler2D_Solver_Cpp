#pragma once

/**
 * @brief MPI abstraction layer for optional MPI support
 *
 * When FVM2D_USE_MPI is defined, this header includes the real <mpi.h>.
 * Otherwise, it provides lightweight stub types and no-op inline functions
 * so that all solver code compiles and runs correctly in serial mode.
 */

#ifdef FVM2D_USE_MPI

#include <mpi.h>

#else  // Serial stubs

#include <chrono>
#include <cstring>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using MPI_Comm     = int;
using MPI_Request  = int;
using MPI_Datatype = int;
using MPI_Op       = int;

struct MPI_Status {};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr MPI_Comm MPI_COMM_WORLD = 0;

constexpr MPI_Datatype MPI_DOUBLE = 0;
constexpr MPI_Datatype MPI_INT    = 1;
constexpr MPI_Datatype MPI_CHAR   = 2;

constexpr MPI_Op MPI_MIN = 0;

constexpr int MPI_MAX_PROCESSOR_NAME = 256;

#define MPI_STATUSES_IGNORE nullptr

// ---------------------------------------------------------------------------
// Environment management
// ---------------------------------------------------------------------------
inline int MPI_Init(int* /*argc*/, char*** /*argv*/) { return 0; }
inline int MPI_Finalize() { return 0; }

inline int MPI_Comm_rank(MPI_Comm /*comm*/, int* rank) {
    *rank = 0;
    return 0;
}

inline int MPI_Comm_size(MPI_Comm /*comm*/, int* size) {
    *size = 1;
    return 0;
}

inline int MPI_Barrier(MPI_Comm /*comm*/) { return 0; }

inline double MPI_Wtime() {
    using clock = std::chrono::high_resolution_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

inline int MPI_Get_processor_name(char* name, int* resultlen) {
    const char* host = "localhost";
    std::strcpy(name, host);
    *resultlen = static_cast<int>(std::strlen(host));
    return 0;
}

// ---------------------------------------------------------------------------
// Collective operations (no-ops for single rank)
// ---------------------------------------------------------------------------
inline int MPI_Bcast(void* /*buffer*/, int /*count*/,
                     MPI_Datatype /*datatype*/, int /*root*/,
                     MPI_Comm /*comm*/) {
    return 0;
}

inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op /*op*/,
                         MPI_Comm /*comm*/) {
    // For a single rank, output = input
    int elem_size = 8;  // default: double
    if (datatype == MPI_INT)  elem_size = sizeof(int);
    if (datatype == MPI_CHAR) elem_size = sizeof(char);
    std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * elem_size);
    return 0;
}

// ---------------------------------------------------------------------------
// Point-to-point (no-ops — never called in serial since maps are empty)
// ---------------------------------------------------------------------------
inline int MPI_Irecv(void* /*buf*/, int /*count*/, MPI_Datatype /*datatype*/,
                     int /*source*/, int /*tag*/, MPI_Comm /*comm*/,
                     MPI_Request* /*request*/) {
    return 0;
}

inline int MPI_Isend(const void* /*buf*/, int /*count*/, MPI_Datatype /*datatype*/,
                     int /*dest*/, int /*tag*/, MPI_Comm /*comm*/,
                     MPI_Request* /*request*/) {
    return 0;
}

inline int MPI_Waitall(int /*count*/, MPI_Request* /*requests*/,
                       MPI_Status* /*statuses*/) {
    return 0;
}

#endif  // FVM2D_USE_MPI
