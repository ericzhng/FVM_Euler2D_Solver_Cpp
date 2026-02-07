#pragma once

#include "common/fvm_export.hpp"
#include "physics/physics_model.hpp"

namespace fvm2d {

/**
 * @brief 2D Euler equations for compressible fluid flow
 *
 * Conservative variables: U = [rho, rho*u, rho*v, E]
 * Primitive variables: P = [rho, u, v, p]
 *
 * where:
 * - rho: density
 * - u, v: velocity components
 * - E: total energy per unit volume
 * - p: pressure
 */
class FVM_API EulerEquations : public PhysicsModel {
public:
    /**
     * @brief Construct Euler equations with given gamma
     * @param gamma Ratio of specific heats (default 1.4 for air)
     */
    explicit EulerEquations(Scalar gamma = 1.4);

    // -------------------------------------------------------------------------
    // PhysicsModel Interface
    // -------------------------------------------------------------------------

    int num_vars() const override { return 4; }

    std::vector<std::string> variable_names() const override;

    VectorXd compute_flux(const VectorXd& U, const Vector2d& normal) const override;

    Scalar max_eigenvalue(const VectorXd& U) const override;

    VectorXd roe_flux(const VectorXd& U_L, const VectorXd& U_R,
                       const Vector2d& normal) const override;

    VectorXd hllc_flux(const VectorXd& U_L, const VectorXd& U_R,
                        const Vector2d& normal) const override;

    VectorXd apply_transmissive_bc(const VectorXd& U_inside) const override;

    VectorXd apply_inlet_bc(const VectorXd& bc_value) const override;

    VectorXd apply_outlet_bc(const VectorXd& U_inside,
                              const VectorXd& bc_value) const override;

    VectorXd apply_wall_bc(const VectorXd& U_inside,
                            const Vector2d& normal,
                            const VectorXd& bc_value) const override;

    VectorXd cons_to_prim(const VectorXd& U) const override;

    VectorXd prim_to_cons(const VectorXd& P) const override;

    // -------------------------------------------------------------------------
    // Euler-Specific Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Compute pressure from conservative state
     */
    Scalar pressure(const VectorXd& U) const;

    /**
     * @brief Compute speed of sound
     */
    Scalar sound_speed(Scalar rho, Scalar p) const;

    /**
     * @brief Get gamma value
     */
    Scalar gamma() const { return gamma_; }

private:
    Scalar gamma_;  // Ratio of specific heats
};

}  // namespace fvm2d
