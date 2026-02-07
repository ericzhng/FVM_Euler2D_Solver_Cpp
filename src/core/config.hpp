#pragma once

#include "core/types.hpp"
#include <string>
#include <map>

namespace fvm2d {

// =============================================================================
// Time Integration Configuration
// =============================================================================

struct TimeConfig {
    TimeIntegrationType method = TimeIntegrationType::RK2;
    Scalar cfl = 0.5;
    bool use_adaptive_dt = true;
    Scalar dt_initial = 1.0e-4;
};

// =============================================================================
// Spatial Discretization Configuration
// =============================================================================

struct SpatialConfig {
    FluxType flux_type = FluxType::HLLC;
    LimiterType limiter_type = LimiterType::Minmod;
    Scalar gradient_over_relaxation = 1.0;
};

// =============================================================================
// Output Configuration
// =============================================================================

struct OutputConfig {
    std::string format = "vtk";  // "vtk" or "tecplot"
    int interval = 100;
    std::string filename_prefix = "solution";
    std::string output_dir = "results";
};

// =============================================================================
// Physics Configuration
// =============================================================================

struct PhysicsConfig {
    // Euler equations
    Scalar gamma = 1.4;

    // Shallow water equations
    Scalar g = 9.806;
};

// =============================================================================
// Main Solver Configuration
// =============================================================================

struct SolverConfig {
    // Input
    std::string mesh_dir = "mesh";  // Directory containing partition mesh files
    std::string boundary_config_file;  // Path to boundary_config.yaml (optional)

    // Simulation
    Scalar t_end = 0.25;
    EquationType equation = EquationType::Euler;
    std::string case_name = "riemann";

    // Sub-configurations
    TimeConfig time;
    SpatialConfig spatial;
    OutputConfig output;
    PhysicsConfig physics;

    // Number of variables based on equation type
    int num_vars() const {
        return (equation == EquationType::Euler) ? 4 : 3;
    }
};

// =============================================================================
// Configuration Parser
// =============================================================================

/**
 * @brief Parse solver configuration from a YAML file
 * @param filepath Path to the YAML configuration file
 * @return Parsed SolverConfig structure
 */
SolverConfig parse_config(const std::string& filepath);

/**
 * @brief Broadcast configuration from rank 0 to all other ranks
 * @param config Configuration to broadcast (modified on non-root ranks)
 * @param comm MPI communicator
 */
void broadcast_config(SolverConfig& config, void* comm);

// Forward declare BoundarySpec (defined in boundary/boundary_condition.hpp)
struct BoundarySpec;

/**
 * @brief Parse boundary condition config from YAML file
 * @param filepath Path to boundary_config.yaml
 * @return Map from boundary name to BoundarySpec
 */
std::map<std::string, BoundarySpec> parse_boundary_config(const std::string& filepath);

}  // namespace fvm2d
