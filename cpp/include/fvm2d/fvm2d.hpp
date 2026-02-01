#pragma once

/**
 * @file fvm2d.hpp
 * @brief Master include header for FVM2D library
 *
 * This header includes all public interfaces for the FVM2D library.
 */

// Core
#include "fvm2d/core/types.hpp"
#include "fvm2d/core/config.hpp"

// Mesh
#include "fvm2d/mesh/partition_mesh.hpp"
#include "fvm2d/mesh/halo_exchange.hpp"

// Physics
#include "fvm2d/physics/physics_model.hpp"
#include "fvm2d/physics/euler_equations.hpp"
#include "fvm2d/physics/shallow_water.hpp"

// Numerics
#include "fvm2d/gradient/gaussian_gradient.hpp"
#include "fvm2d/limiter/limiters.hpp"
#include "fvm2d/boundary/boundary_condition.hpp"

// Solver
#include "fvm2d/solver/residual.hpp"
#include "fvm2d/solver/fvm_solver.hpp"

// Time Integration
#include "fvm2d/time/time_integrator.hpp"
#include "fvm2d/time/timestep.hpp"

// I/O
#include "fvm2d/io/vtk_writer.hpp"
#include "fvm2d/io/tecplot_writer.hpp"
