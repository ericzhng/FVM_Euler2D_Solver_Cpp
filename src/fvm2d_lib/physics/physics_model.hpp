#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include <vector>
#include <string>

namespace fvm2d {

/**
 * @brief Abstract base class for physics models
 *
 * This class defines the interface for physics models used in the FVM solver.
 * Derived classes implement specific equation systems (Euler, Shallow Water, etc.)
 */
class FVM_API PhysicsModel {
public:
    virtual ~PhysicsModel() = default;

    // -------------------------------------------------------------------------
    // Pure Virtual Methods (must be implemented by derived classes)
    // -------------------------------------------------------------------------

    /**
     * @brief Get the number of conservative variables
     */
    virtual int num_vars() const = 0;

    /**
     * @brief Get variable names for output
     */
    virtual std::vector<std::string> variable_names() const = 0;

    /**
     * @brief Compute physical flux F(U) in the direction of the normal
     * @param U Conservative state vector
     * @param normal Face normal vector (unit)
     * @return Flux vector in normal direction
     */
    virtual VectorXd compute_flux(const VectorXd& U, const Vector2d& normal) const = 0;

    /**
     * @brief Compute maximum eigenvalue (wave speed) for CFL condition
     * @param U Conservative state vector
     * @return Maximum wave speed |v| + a
     */
    virtual Scalar max_eigenvalue(const VectorXd& U) const = 0;

    // -------------------------------------------------------------------------
    // Riemann Solvers
    // -------------------------------------------------------------------------

    /**
     * @brief Compute numerical flux using Roe's approximate Riemann solver
     * @param U_L Left state (conservative)
     * @param U_R Right state (conservative)
     * @param normal Face normal vector
     * @return Numerical flux
     */
    virtual VectorXd roe_flux(const VectorXd& U_L, const VectorXd& U_R,
                               const Vector2d& normal) const = 0;

    /**
     * @brief Compute numerical flux using HLLC Riemann solver
     * @param U_L Left state (conservative)
     * @param U_R Right state (conservative)
     * @param normal Face normal vector
     * @return Numerical flux
     */
    virtual VectorXd hllc_flux(const VectorXd& U_L, const VectorXd& U_R,
                                const Vector2d& normal) const = 0;

    // -------------------------------------------------------------------------
    // Boundary Conditions
    // -------------------------------------------------------------------------

    /**
     * @brief Apply transmissive (zero-gradient) boundary condition
     * @param U_inside Interior cell state
     * @return Ghost cell state
     */
    virtual VectorXd apply_transmissive_bc(const VectorXd& U_inside) const = 0;

    /**
     * @brief Apply inlet boundary condition
     * @param bc_value Boundary condition values (primitive)
     * @return Ghost cell state (conservative)
     */
    virtual VectorXd apply_inlet_bc(const VectorXd& bc_value) const = 0;

    /**
     * @brief Apply outlet boundary condition
     * @param U_inside Interior cell state
     * @param bc_value Boundary condition values (e.g., pressure)
     * @return Ghost cell state
     */
    virtual VectorXd apply_outlet_bc(const VectorXd& U_inside,
                                      const VectorXd& bc_value) const = 0;

    /**
     * @brief Apply wall boundary condition (slip or no-slip)
     * @param U_inside Interior cell state
     * @param normal Wall normal vector
     * @param bc_value Boundary condition parameters (e.g., slip flag)
     * @return Ghost cell state
     */
    virtual VectorXd apply_wall_bc(const VectorXd& U_inside,
                                    const Vector2d& normal,
                                    const VectorXd& bc_value) const = 0;

    // -------------------------------------------------------------------------
    // State Conversion (protected interface)
    // -------------------------------------------------------------------------

    /**
     * @brief Convert conservative to primitive variables
     * @param U Conservative state
     * @return Primitive state
     */
    virtual VectorXd cons_to_prim(const VectorXd& U) const = 0;

    /**
     * @brief Convert primitive to conservative variables
     * @param P Primitive state
     * @return Conservative state
     */
    virtual VectorXd prim_to_cons(const VectorXd& P) const = 0;
};

}  // namespace fvm2d
