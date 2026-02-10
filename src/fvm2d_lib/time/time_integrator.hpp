#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "core/config.hpp"
#include "mesh/partition_mesh.hpp"
#include "physics/physics_model.hpp"
#include "boundary/boundary_condition.hpp"
#include "core/mpi_wrapper.hpp"
#include <memory>
#include <array>

namespace fvm2d {

/**
 * @brief Accumulated wall-clock timings for time integrator sub-phases
 */
struct IntegratorTimings {
    double halo_exchange = 0.0;
    double gradient      = 0.0;
    double limiter       = 0.0;
    double residual      = 0.0;
    double update        = 0.0;
};

// Forward declaration
class HaloExchange;

/**
 * @brief Abstract base class for time integrators
 */
class FVM_API TimeIntegrator {
public:
    TimeIntegrator(const PartitionMesh& mesh,
                   const PhysicsModel& physics,
                   const BoundaryConditionLookup& bcs,
                   const SolverConfig& config,
                   MPI_Comm comm);

    virtual ~TimeIntegrator();

    /**
     * @brief Advance solution by one time step
     * @param U Solution state (modified in-place)
     * @param dt Time step size
     */
    virtual void step(StateArray& U, Scalar dt) = 0;

    /**
     * @brief Get accumulated sub-phase timings
     */
    const IntegratorTimings& timings() const { return timings_; }

protected:
    const PartitionMesh& mesh_;
    const PhysicsModel& physics_;
    const BoundaryConditionLookup& bcs_;
    const SolverConfig& config_;
    MPI_Comm comm_;

    std::unique_ptr<HaloExchange> halo_exchange_;

    // Temporary arrays
    GradientArray gradients_;
    LimiterArray limiters_;

    /**
     * @brief Compute residual with gradient/limiter computation
     */
    StateArray compute_full_residual(const StateArray& U);

    /**
     * @brief Exchange halo data
     */
    void exchange_halo(StateArray& U);

    IntegratorTimings timings_;
};

/**
 * @brief Explicit Euler time integrator
 *
 * U^{n+1} = U^n - dt * R(U^n)
 */
class FVM_API ExplicitEuler : public TimeIntegrator {
public:
    using TimeIntegrator::TimeIntegrator;

    void step(StateArray& U, Scalar dt) override;
};

/**
 * @brief 2-Stage Runge-Kutta (RK2) time integrator
 *
 * Stage 1: U* = U^n - dt * R(U^n)
 * Stage 2: U^{n+1} = 0.5 * (U^n + U* - dt * R(U*))
 */
class FVM_API RungeKutta2 : public TimeIntegrator {
public:
    RungeKutta2(const PartitionMesh& mesh,
                const PhysicsModel& physics,
                const BoundaryConditionLookup& bcs,
                const SolverConfig& config,
                MPI_Comm comm);

    void step(StateArray& U, Scalar dt) override;

private:
    StateArray U_star_;  // Intermediate state
};

/**
 * @brief Factory function to create time integrator
 */
FVM_API std::unique_ptr<TimeIntegrator> create_time_integrator(
    TimeIntegrationType type,
    const PartitionMesh& mesh,
    const PhysicsModel& physics,
    const BoundaryConditionLookup& bcs,
    const SolverConfig& config,
    MPI_Comm comm
);

}  // namespace fvm2d
