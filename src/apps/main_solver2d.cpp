#include <mpi.h>
#include <iostream>
#include <string>
#include <exception>
#include <thread>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include "fvm2d.hpp"
#include "CLI/CLI.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

int main(int argc, char *argv[])
{
    // --- CLI Parsing ---
    std::string config_file;
    bool wait_for_debugger = false;

    CLI::App app{
        "FVM2D - 2D Parallel Finite Volume Method Solver (v1.0)"};

    app.footer(
        "\nExamples:\n"
        "  mpiexec -n 4 ./fvm2d_solver config.yaml\n"
        "  mpiexec -n 4 ./fvm2d_solver --wait-for-debugger config.yaml\n"
        "\nCopyright (c) 2026 Eric Zhang. Distributed under MIT License.");

    app.add_option("config", config_file, "Path to the YAML configuration file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_flag("--wait-for-debugger", wait_for_debugger,
                 "Wait for a debugger to attach before running (for MPI debugging)");

    CLI11_PARSE(app, argc, argv);

    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Debugger Attach (optional) ---
    if (wait_for_debugger)
    {
        char hostname[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(hostname, &len);

#ifdef _WIN32
        DWORD pid = GetCurrentProcessId();
#else
        pid_t pid = getpid();
#endif
        std::cout << "Rank " << rank << " on " << hostname
                  << ", PID=" << pid
                  << " waiting for debugger." << std::endl;

        // Write PID file so the launcher script can find it immediately
        // (mpiexec buffers stdout, so the script can't read it in time).
        const char *pid_dir = std::getenv("FVM2D_PID_DIR");
        if (pid_dir)
        {
            std::string pid_path = std::string(pid_dir) + "/rank_" + std::to_string(rank) + ".pid";
            std::ofstream(pid_path) << pid << std::endl;
        }

#ifdef _WIN32
        while (!IsDebuggerPresent())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        __debugbreak();
#else
        // In GDB: set var i = 1 to continue
        volatile int i = 0;
        while (i == 0)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
#endif
        // Sync all ranks after attach
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- Main Solver ---
    int exit_code = 0;

    try
    {
        if (rank == 0)
        {
            std::cout << "========================================\n";
            std::cout << " FVM2D - 2D Parallel FVM Solver\n";
            std::cout << "========================================\n";
            std::cout << "MPI processes: " << size << "\n";
            std::cout << "Loading config: " << config_file << "\n";
        }

        fvm2d::SolverConfig config = fvm2d::parse_config(config_file);

        if (rank == 0)
        {
            std::cout << "  Equation: "
                      << (config.equation == fvm2d::EquationType::Euler ? "Euler" : "Shallow Water") << "\n";
            std::cout << "  t_end:    " << config.t_end << "\n";
            std::cout << "  CFL:      " << config.time.cfl << "\n";
            std::cout << "  Mesh dir: " << config.mesh_dir << "\n";
            std::cout << "----------------------------------------\n";
        }

        fvm2d::FVMSolver solver(MPI_COMM_WORLD, config);
        solver.initialize();
        solver.setup_case();
        solver.run();

        if (rank == 0)
        {
            std::cout << "========================================\n";
            std::cout << " Simulation completed successfully!\n";
            std::cout << "========================================\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
