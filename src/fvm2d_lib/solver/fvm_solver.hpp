#pragma once
#include "common/fvm_export.hpp"

#include "core/types.hpp"
#include "core/config.hpp"
#include "mesh/partition_mesh.hpp"
#include "physics/physics_model.hpp"
#include "boundary/boundary_condition.hpp"
#include "time/time_integrator.hpp"
#include <memory>
#include <mpi.h>

namespace fvm2d {

/**
 * @brief Main FVM solver class
 *
 * Orchestrates the entire simulation: mesh loading, physics setup,
 * time stepping, and output.
 */
class FVM_API FVMSolver {
public:
    /**
     * @brief Construct solver with MPI communicator and configuration
     */
    FVMSolver(MPI_Comm comm, const SolverConfig& config);

    ~FVMSolver();

    /**
     * @brief Initialize mesh and physics
     *
     * Loads partition mesh for this rank and sets up physics model
     */
    void initialize();

    /**
     * @brief Set up initial and boundary conditions based on case
     */
    void setup_case();

    /**
     * @brief Run the simulation
     */
    void run();

    /**
     * @brief Get current solution state
     */
    const StateArray& solution() const { return U_; }

    /**
     * @brief Get mesh
     */
    const PartitionMesh& mesh() const { return mesh_; }

    /**
     * @brief Get configuration
     */
    const SolverConfig& config() const { return config_; }

    /**
     * @brief Set initial condition from function
     * @param init_func Function (x, y) -> U (conservative state)
     */
    void set_initial_condition(std::function<VectorXd(Scalar, Scalar)> init_func);

    /**
     * @brief Add boundary condition for a patch
     */
    void add_boundary_condition(const std::string& patch_name, BCType type,
                                 const VectorXd& values = VectorXd());

private:
    MPI_Comm comm_;
    int rank_;
    int size_;

    SolverConfig config_;
    PartitionMesh mesh_;

    std::unique_ptr<PhysicsModel> physics_;
    std::unique_ptr<TimeIntegrator> integrator_;
    BoundaryConditionLookup boundary_conditions_;

    StateArray U_;  // Current solution state

    // Private methods
    Scalar compute_dt() const;
    void write_output(int step, Scalar time) const;
    void setup_euler_riemann_case();
    void setup_shallow_water_riemann_case();
};

}  // namespace fvm2d
