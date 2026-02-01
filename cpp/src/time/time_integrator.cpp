#include "fvm2d/time/time_integrator.hpp"
#include "fvm2d/mesh/halo_exchange.hpp"
#include "fvm2d/gradient/gaussian_gradient.hpp"
#include "fvm2d/solver/residual.hpp"

namespace fvm2d {

TimeIntegrator::TimeIntegrator(
    const PartitionMesh& mesh,
    const PhysicsModel& physics,
    const BoundaryConditionLookup& bcs,
    const SolverConfig& config,
    MPI_Comm comm
)
    : mesh_(mesh)
    , physics_(physics)
    , bcs_(bcs)
    , config_(config)
    , comm_(comm)
{
    halo_exchange_ = std::make_unique<HaloExchange>(mesh, config.num_vars(), comm);
}

TimeIntegrator::~TimeIntegrator() = default;

void TimeIntegrator::exchange_halo(StateArray& U) {
    halo_exchange_->exchange(U);
}

StateArray TimeIntegrator::compute_full_residual(const StateArray& U) {
    // Compute gradients
    gradients_ = compute_gradients_gaussian(
        mesh_, U, config_.spatial.gradient_over_relaxation);

    // Compute limiters
    limiters_ = compute_limiters(
        mesh_, U, gradients_, config_.spatial.limiter_type);

    // Compute residual
    return compute_residual(mesh_, U, gradients_, limiters_, bcs_, physics_, config_);
}

// ExplicitEuler implementation
void ExplicitEuler::step(StateArray& U, Scalar dt) {
    // Exchange halo data
    exchange_halo(U);

    // Compute residual
    StateArray residual = compute_full_residual(U);

    // Update solution: U^{n+1} = U^n - dt * R(U^n)
    for (Index i = 0; i < mesh_.num_owned_cells; ++i) {
        U.row(i) -= dt * residual.row(i);
    }
}

// RungeKutta2 implementation
RungeKutta2::RungeKutta2(
    const PartitionMesh& mesh,
    const PhysicsModel& physics,
    const BoundaryConditionLookup& bcs,
    const SolverConfig& config,
    MPI_Comm comm
)
    : TimeIntegrator(mesh, physics, bcs, config, comm)
{
    // Pre-allocate intermediate state
    U_star_.resize(mesh.total_cells(), config.num_vars());
}

void RungeKutta2::step(StateArray& U, Scalar dt) {
    // Exchange halo data for stage 1
    exchange_halo(U);

    // Stage 1: U* = U^n - dt * R(U^n)
    StateArray residual = compute_full_residual(U);

    for (Index i = 0; i < mesh_.num_owned_cells; ++i) {
        U_star_.row(i) = U.row(i) - dt * residual.row(i);
    }

    // Copy halo cells (unchanged in this stage)
    for (Index i = mesh_.num_owned_cells; i < mesh_.total_cells(); ++i) {
        U_star_.row(i) = U.row(i);
    }

    // Exchange halo data for stage 2
    exchange_halo(U_star_);

    // Stage 2: U^{n+1} = 0.5 * (U^n + U* - dt * R(U*))
    residual = compute_full_residual(U_star_);

    for (Index i = 0; i < mesh_.num_owned_cells; ++i) {
        U.row(i) = 0.5 * (U.row(i) + U_star_.row(i) - dt * residual.row(i));
    }
}

std::unique_ptr<TimeIntegrator> create_time_integrator(
    TimeIntegrationType type,
    const PartitionMesh& mesh,
    const PhysicsModel& physics,
    const BoundaryConditionLookup& bcs,
    const SolverConfig& config,
    MPI_Comm comm
) {
    switch (type) {
        case TimeIntegrationType::ExplicitEuler:
            return std::make_unique<ExplicitEuler>(mesh, physics, bcs, config, comm);

        case TimeIntegrationType::RK2:
            return std::make_unique<RungeKutta2>(mesh, physics, bcs, config, comm);

        default:
            return std::make_unique<RungeKutta2>(mesh, physics, bcs, config, comm);
    }
}

}  // namespace fvm2d
