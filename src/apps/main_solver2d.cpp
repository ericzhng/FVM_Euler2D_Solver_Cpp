#include <mpi.h>
#include <iostream>
#include <string>
#include <exception>

#include "fvm2d.hpp"

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <config.yaml>\n";
    std::cout << "\n";
    std::cout << "FVM2D - 2D Parallel Finite Volume Method Solver\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  config.yaml    Path to the YAML configuration file\n";
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exit_code = 0;

    try {
        // Parse command-line arguments
        std::string config_file = "config.yaml";
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "-h" || arg == "--help") {
                if (rank == 0) {
                    print_usage(argv[0]);
                }
                MPI_Finalize();
                return 0;
            }
            config_file = arg;
        }

        // Load configuration (rank 0 loads and broadcasts)
        fvm2d::SolverConfig config;
        if (rank == 0) {
            std::cout << "========================================\n";
            std::cout << " FVM2D - 2D Parallel FVM Solver\n";
            std::cout << "========================================\n";
            std::cout << "MPI processes: " << size << "\n";
            std::cout << "Loading configuration from: " << config_file << "\n";
            config = fvm2d::parse_config(config_file);
        }

        // Broadcast configuration to all ranks
        MPI_Comm comm = MPI_COMM_WORLD;
        fvm2d::broadcast_config(config, &comm);

        if (rank == 0) {
            std::cout << "Configuration loaded successfully.\n";
            std::cout << "  Equation: " << (config.equation == fvm2d::EquationType::Euler
                                            ? "Euler" : "Shallow Water") << "\n";
            std::cout << "  t_end: " << config.t_end << "\n";
            std::cout << "  CFL: " << config.time.cfl << "\n";
            std::cout << "  Mesh dir: " << config.mesh_dir << "\n";
            std::cout << "----------------------------------------\n";
        }

        // Create and run solver
        fvm2d::FVMSolver solver(MPI_COMM_WORLD, config);

        solver.initialize();
        solver.setup_case();
        solver.run();

        if (rank == 0) {
            std::cout << "========================================\n";
            std::cout << " Simulation completed successfully!\n";
            std::cout << "========================================\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
