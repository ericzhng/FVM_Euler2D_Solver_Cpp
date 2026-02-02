#pragma once

/**
 * @file fvm2d.hpp
 * @brief Master include header for FVM2D library
 *
 * This header includes all public interfaces for the FVM2D library.
 */

// Core
#include "core/types.hpp"
#include "core/config.hpp"

// Mesh
#include "mesh/partition_mesh.hpp"
#include "mesh/halo_exchange.hpp"

// Physics
#include "physics/physics_model.hpp"
#include "physics/euler_equations.hpp"
#include "physics/shallow_water.hpp"

// Numerics
#include "gradient/gaussian_gradient.hpp"
#include "limiter/limiters.hpp"
#include "boundary/boundary_condition.hpp"

// Solver
#include "solver/residual.hpp"
#include "solver/fvm_solver.hpp"

// Time Integration
#include "time/time_integrator.hpp"
#include "time/timestep.hpp"

// I/O
#include "io/vtk_writer.hpp"
#include "io/tecplot_writer.hpp"
