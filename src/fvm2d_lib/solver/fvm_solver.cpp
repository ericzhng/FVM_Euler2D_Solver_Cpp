#include "solver/fvm_solver.hpp"
#include "physics/euler_equations.hpp"
#include "physics/shallow_water.hpp"
#include "time/timestep.hpp"
#include "io/vtk_writer.hpp"
#include "io/tecplot_writer.hpp"
#include <iostream>
#include <filesystem>

namespace fvm2d {

FVMSolver::FVMSolver(MPI_Comm comm, const SolverConfig& config)
    : comm_(comm), config_(config)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

FVMSolver::~FVMSolver() = default;

void FVMSolver::initialize() {
    // Load partition mesh for this rank
    mesh_ = load_partition_for_rank(config_.mesh_dir, rank_);

    if (rank_ == 0) {
        std::cout << "Loaded mesh for " << size_ << " partitions" << std::endl;
        std::cout << "  Total cells (rank 0): " << mesh_.total_cells()
                  << " (owned: " << mesh_.num_owned_cells << ")" << std::endl;
    }

    // Create physics model
    if (config_.equation == EquationType::Euler) {
        physics_ = std::make_unique<EulerEquations>(config_.physics.gamma);
    } else {
        physics_ = std::make_unique<ShallowWaterEquations>(config_.physics.g);
    }

    // Initialize solution array
    U_.resize(mesh_.total_cells(), config_.num_vars());
    U_.setZero();

    // Create output directory
    if (rank_ == 0) {
        std::filesystem::create_directories(config_.output.output_dir);
    }
    MPI_Barrier(comm_);
}

void FVMSolver::setup_case() {
    // Load boundary conditions from config file if specified
    if (!config_.boundary_config_file.empty()) {
        if (rank_ == 0) {
            std::cout << "Loading boundary config: " << config_.boundary_config_file << std::endl;
        }
        auto bc_specs = parse_boundary_config(config_.boundary_config_file);
        for (const auto& [name, spec] : bc_specs) {
            boundary_conditions_.add(name, spec);
        }
    }

    if (config_.case_name == "riemann") {
        if (config_.equation == EquationType::Euler) {
            setup_euler_riemann_case();
        } else {
            setup_shallow_water_riemann_case();
        }
    }

    // Build boundary condition lookup
    boundary_conditions_.set_patch_map(mesh_.boundary_patch_map);
    boundary_conditions_.build(config_.num_vars());

    // Create time integrator
    integrator_ = create_time_integrator(
        config_.time.method, mesh_, *physics_,
        boundary_conditions_, config_, comm_);
}

void FVMSolver::setup_euler_riemann_case() {
    // 4-quadrant Riemann problem
    Scalar x_mid = 0.5, y_mid = 0.5;

    // Define states in each quadrant [rho, u, v, p]
    VectorXd state1(4), state2(4), state3(4), state4(4);
    state1 << 1.0, -0.75, -0.5, 1.0;   // top-right
    state2 << 2.0, -0.75, 0.5, 1.0;    // top-left
    state3 << 1.0, 0.75, 0.5, 1.0;     // bottom-left
    state4 << 3.0, 0.75, -0.5, 1.0;    // bottom-right

    auto* euler = dynamic_cast<EulerEquations*>(physics_.get());

    for (Index i = 0; i < mesh_.num_owned_cells; ++i) {
        Scalar x = mesh_.cell_centroids[i].x();
        Scalar y = mesh_.cell_centroids[i].y();

        VectorXd P;
        if (x >= x_mid && y >= y_mid) {
            P = state1;
        } else if (x < x_mid && y >= y_mid) {
            P = state2;
        } else if (x < x_mid && y < y_mid) {
            P = state3;
        } else {
            P = state4;
        }

        U_.row(i) = euler->prim_to_cons(P).transpose();
    }

    // Default: all boundaries transmissive (only for patches not already configured)
    if (config_.boundary_config_file.empty()) {
        for (const auto& [name, tag] : mesh_.boundary_patch_map) {
            boundary_conditions_.add(name, BoundarySpec(BCType::Transmissive, VectorXd()));
        }
    }
}

void FVMSolver::setup_shallow_water_riemann_case() {
    // 4-quadrant Riemann problem for SWE
    Scalar x_mid = 50.0, y_mid = 50.0;

    // Define states in each quadrant [h, u, v]
    VectorXd state1(3), state2(3), state3(3), state4(3);
    state1 << 2.0, 0.0, 0.0;   // top-right
    state2 << 1.0, 0.5, 0.0;   // top-left
    state3 << 1.5, -0.5, 0.0;  // bottom-left
    state4 << 0.5, 0.0, 0.5;   // bottom-right

    auto* swe = dynamic_cast<ShallowWaterEquations*>(physics_.get());

    for (Index i = 0; i < mesh_.num_owned_cells; ++i) {
        Scalar x = mesh_.cell_centroids[i].x();
        Scalar y = mesh_.cell_centroids[i].y();

        VectorXd P;
        if (x >= x_mid && y >= y_mid) {
            P = state1;
        } else if (x < x_mid && y >= y_mid) {
            P = state2;
        } else if (x < x_mid && y < y_mid) {
            P = state3;
        } else {
            P = state4;
        }

        U_.row(i) = swe->prim_to_cons(P).transpose();
    }

    // Default: all boundaries transmissive (only for patches not already configured)
    if (config_.boundary_config_file.empty()) {
        for (const auto& [name, tag] : mesh_.boundary_patch_map) {
            boundary_conditions_.add(name, BoundarySpec(BCType::Transmissive, VectorXd()));
        }
    }
}

void FVMSolver::set_initial_condition(std::function<VectorXd(Scalar, Scalar)> init_func) {
    for (Index i = 0; i < mesh_.num_owned_cells; ++i) {
        Scalar x = mesh_.cell_centroids[i].x();
        Scalar y = mesh_.cell_centroids[i].y();
        U_.row(i) = init_func(x, y).transpose();
    }
}

void FVMSolver::add_boundary_condition(const std::string& patch_name, BCType type,
                                         const VectorXd& values) {
    boundary_conditions_.add(patch_name, BoundarySpec(type, values));
}

Scalar FVMSolver::compute_dt() const {
    if (config_.time.use_adaptive_dt) {
        return calculate_adaptive_dt(mesh_, U_, *physics_, config_.time.cfl, comm_);
    }
    return config_.time.dt_initial;
}

void FVMSolver::write_output(int step, Scalar time) const {
    std::string filename = config_.output.output_dir + "/" +
                          config_.output.filename_prefix + "_" +
                          std::to_string(rank_) + "_" +
                          std::to_string(step);

    if (config_.output.format == "vtk") {
        write_vtk(mesh_, U_, physics_->variable_names(), filename + ".vtk");
    } else {
        write_tecplot(mesh_, U_, physics_->variable_names(), filename + ".dat", time);
    }
}

void FVMSolver::run() {
    Scalar t = 0.0;
    int step = 0;
    Scalar dt = config_.time.dt_initial;

    // Write initial condition
    write_output(step, t);

    if (rank_ == 0) {
        std::cout << "Starting simulation..." << std::endl;
        std::cout << "  t_end = " << config_.t_end << std::endl;
        std::cout << "  CFL = " << config_.time.cfl << std::endl;
    }

    while (t < config_.t_end) {
        // Compute time step     
        dt = compute_dt();
        dt = std::min(dt, config_.t_end - t);

        // Advance solution
        integrator_->step(U_, dt);

        t += dt;
        ++step;

        // Output at intervals
        if (step % config_.output.interval == 0) {
            write_output(step, t);

            if (rank_ == 0) {
                std::cout << "Step " << step
                          << ", t = " << t << " / " << config_.t_end
                          << ", dt = " << dt << std::endl;
            }
        }
    }

    // Final output
    write_output(step, t);

    if (rank_ == 0) {
        std::cout << "Simulation complete. Steps: " << step << std::endl;
    }
}

}  // namespace fvm2d
