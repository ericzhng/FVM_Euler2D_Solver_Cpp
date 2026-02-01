#pragma once

#include "fvm2d/physics/physics_model.hpp"

namespace fvm2d {

/**
 * @brief 2D Shallow Water Equations
 *
 * Conservative variables: U = [h, h*u, h*v]
 * Primitive variables: P = [h, u, v]
 *
 * where:
 * - h: water depth
 * - u, v: depth-averaged velocity components
 */
class ShallowWaterEquations : public PhysicsModel {
public:
    /**
     * @brief Construct shallow water equations with given gravity
     * @param g Acceleration due to gravity (default 9.806)
     */
    explicit ShallowWaterEquations(Scalar g = 9.806);

    // -------------------------------------------------------------------------
    // PhysicsModel Interface
    // -------------------------------------------------------------------------

    int num_vars() const override { return 3; }

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
    // Shallow Water-Specific Methods
    // -------------------------------------------------------------------------

    /**
     * @brief Compute wave celerity c = sqrt(g*h)
     */
    Scalar celerity(Scalar h) const;

    /**
     * @brief Get gravity value
     */
    Scalar gravity() const { return g_; }

private:
    Scalar g_;  // Acceleration due to gravity
};

}  // namespace fvm2d
