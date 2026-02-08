#include "solver/fvm_solver.hpp"
#include "physics/euler_equations.hpp"
#include "physics/shallow_water.hpp"
#include "time/timestep.hpp"
#include "io/vtk_writer.hpp"
#include "tecplot/tecplot_writer.hpp"
#include <iostream>
#include <filesystem>
#include <iomanip>

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

    if (config_.output.format == "vtu") {
        write_solution(mesh_, U_, physics_->variable_names(), filename + ".vtu", true);
    } else if (config_.output.format == "vtk") {
        write_solution(mesh_, U_, physics_->variable_names(), filename + ".vtu", false);
    } else {
        write_tecplot(mesh_, U_, physics_->variable_names(), filename + ".dat", time);
    }
}

void FVMSolver::run() {
    Scalar t = 0.0;
    int step = 0;
    Scalar dt = config_.time.dt_initial;

    // Profiling accumulators (wall-clock seconds)
    double time_compute_dt = 0.0;
    double time_step       = 0.0;
    double time_io         = 0.0;
    double t0, t1;

    // Write initial condition
    t0 = MPI_Wtime();
    write_output(step, t);
    time_io += MPI_Wtime() - t0;

    if (rank_ == 0) {
        std::cout << "Starting simulation..." << std::endl;
        std::cout << "  t_end = " << config_.t_end << std::endl;
        std::cout << "  CFL = " << config_.time.cfl << std::endl;
    }

    double wall_start = MPI_Wtime();

    while (t < config_.t_end) {
        // Compute time step
        t0 = MPI_Wtime();
        dt = compute_dt();
        dt = std::min(dt, config_.t_end - t);
        t1 = MPI_Wtime();
        time_compute_dt += t1 - t0;

        // Advance solution
        t0 = t1;
        integrator_->step(U_, dt);
        t1 = MPI_Wtime();
        time_step += t1 - t0;

        t += dt;
        ++step;

        // Output at intervals
        if (step % config_.output.interval == 0) {
            t0 = MPI_Wtime();
            write_output(step, t);
            time_io += MPI_Wtime() - t0;

            if (rank_ == 0) {
                std::cout << "Step " << step
                          << ", t = " << t << " / " << config_.t_end
                          << ", dt = " << dt << std::endl;
            }
        }
    }

    // Final output
    t0 = MPI_Wtime();
    write_output(step, t);
    time_io += MPI_Wtime() - t0;

    double wall_total = MPI_Wtime() - wall_start;

    // --- Profiling Summary (rank 0 only) ---
    if (rank_ == 0) {
        const auto& ti = integrator_->timings();
        double time_integrator_sub = ti.halo_exchange + ti.gradient
                                   + ti.limiter + ti.residual + ti.update;

        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << " Profiling Summary\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Total steps:      " << step << "\n";
        std::cout << "  Wall time:        " << wall_total << " s\n";
        std::cout << "  Avg step time:    " << (step > 0 ? wall_total / step * 1000.0 : 0.0) << " ms\n";
        std::cout << "----------------------------------------\n";
        std::cout << "  High-level breakdown:\n";
        std::cout << "    Compute dt:     " << std::setw(10) << time_compute_dt
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (wall_total > 0 ? time_compute_dt / wall_total * 100 : 0) << "%)\n";
        std::cout << std::setprecision(4);
        std::cout << "    Time stepping:  " << std::setw(10) << time_step
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (wall_total > 0 ? time_step / wall_total * 100 : 0) << "%)\n";
        std::cout << std::setprecision(4);
        std::cout << "    File I/O:       " << std::setw(10) << time_io
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (wall_total > 0 ? time_io / wall_total * 100 : 0) << "%)\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::setprecision(4);
        std::cout << "  Time stepping breakdown:\n";
        std::cout << "    Halo exchange:  " << std::setw(10) << ti.halo_exchange
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (time_step > 0 ? ti.halo_exchange / time_step * 100 : 0) << "%)\n";
        std::cout << std::setprecision(4);
        std::cout << "    Gradient:       " << std::setw(10) << ti.gradient
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (time_step > 0 ? ti.gradient / time_step * 100 : 0) << "%)\n";
        std::cout << std::setprecision(4);
        std::cout << "    Limiter:        " << std::setw(10) << ti.limiter
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (time_step > 0 ? ti.limiter / time_step * 100 : 0) << "%)\n";
        std::cout << std::setprecision(4);
        std::cout << "    Residual:       " << std::setw(10) << ti.residual
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (time_step > 0 ? ti.residual / time_step * 100 : 0) << "%)\n";
        std::cout << std::setprecision(4);
        std::cout << "    Solution update:" << std::setw(10) << ti.update
                  << " s  (" << std::setw(5) << std::setprecision(1)
                  << (time_step > 0 ? ti.update / time_step * 100 : 0) << "%)\n";
        std::cout << "========================================\n";
        std::cout << "Simulation complete. Steps: " << step << std::endl;
    }
}

}  // namespace fvm2d
