#include "fvm2d/physics/shallow_water.hpp"
#include <cmath>
#include <algorithm>

namespace fvm2d {

ShallowWaterEquations::ShallowWaterEquations(Scalar g) : g_(g) {}

std::vector<std::string> ShallowWaterEquations::variable_names() const {
    return {"h", "hu", "hv"};
}

VectorXd ShallowWaterEquations::cons_to_prim(const VectorXd& U) const {
    VectorXd P(3);
    Scalar h = std::max(U(0), EPSILON);
    Scalar u = U(1) / h;
    Scalar v = U(2) / h;

    P << h, u, v;
    return P;
}

VectorXd ShallowWaterEquations::prim_to_cons(const VectorXd& P) const {
    VectorXd U(3);
    Scalar h = P(0);
    Scalar u = P(1);
    Scalar v = P(2);

    U << h, h*u, h*v;
    return U;
}

Scalar ShallowWaterEquations::celerity(Scalar h) const {
    return std::sqrt(g_ * std::max(h, EPSILON));
}

VectorXd ShallowWaterEquations::compute_flux(const VectorXd& U, const Vector2d& normal) const {
    VectorXd P = cons_to_prim(U);
    Scalar h = P(0);
    Scalar u = P(1);
    Scalar v = P(2);

    // Normal velocity
    Scalar vn = u * normal.x() + v * normal.y();

    VectorXd F(3);
    F(0) = h * vn;
    F(1) = h * vn * u + 0.5 * g_ * h * h * normal.x();
    F(2) = h * vn * v + 0.5 * g_ * h * h * normal.y();

    return F;
}

Scalar ShallowWaterEquations::max_eigenvalue(const VectorXd& U) const {
    VectorXd P = cons_to_prim(U);
    Scalar h = P(0);
    Scalar u = P(1);
    Scalar v = P(2);

    Scalar c = celerity(h);
    return std::sqrt(u*u + v*v) + c;
}

VectorXd ShallowWaterEquations::apply_transmissive_bc(const VectorXd& U_inside) const {
    return U_inside;
}

VectorXd ShallowWaterEquations::apply_inlet_bc(const VectorXd& bc_value) const {
    return prim_to_cons(bc_value);
}

VectorXd ShallowWaterEquations::apply_outlet_bc(const VectorXd& U_inside,
                                                  const VectorXd& bc_value) const {
    VectorXd P_inside = cons_to_prim(U_inside);
    Scalar u_in = P_inside(1);
    Scalar v_in = P_inside(2);
    Scalar h_ghost = bc_value(0);  // Prescribed outlet depth

    VectorXd P_ghost(3);
    P_ghost << h_ghost, u_in, v_in;
    return prim_to_cons(P_ghost);
}

VectorXd ShallowWaterEquations::apply_wall_bc(const VectorXd& U_inside,
                                               const Vector2d& normal,
                                               const VectorXd& bc_value) const {
    VectorXd P_inside = cons_to_prim(U_inside);
    Scalar h_in = P_inside(0);
    Scalar u_in = P_inside(1);
    Scalar v_in = P_inside(2);

    // Decompose velocity
    Scalar nx = normal.x();
    Scalar ny = normal.y();
    Scalar tx = -ny;
    Scalar ty = nx;

    Scalar vn_in = u_in * nx + v_in * ny;
    Scalar vt_in = u_in * tx + v_in * ty;

    // Reflect normal velocity
    Scalar vn_ghost = -vn_in;

    // Slip or no-slip
    bool is_slip = (bc_value.size() > 0 && bc_value(0) == 1.0);
    Scalar vt_ghost = is_slip ? vt_in : 0.0;

    // Reconstruct velocity
    Scalar u_ghost = vn_ghost * nx + vt_ghost * tx;
    Scalar v_ghost = vn_ghost * ny + vt_ghost * ty;

    VectorXd P_ghost(3);
    P_ghost << h_in, u_ghost, v_ghost;
    return prim_to_cons(P_ghost);
}

VectorXd ShallowWaterEquations::hllc_flux(const VectorXd& U_L, const VectorXd& U_R,
                                           const Vector2d& normal) const {
    Scalar nx = normal.x();
    Scalar ny = normal.y();
    Scalar tx = -ny;
    Scalar ty = nx;

    // Left state
    VectorXd P_L = cons_to_prim(U_L);
    Scalar hL = P_L(0), uL = P_L(1), vL = P_L(2);
    Scalar vnL = uL * nx + vL * ny;
    Scalar vtL = uL * tx + vL * ty;
    Scalar cL = celerity(hL);
    VectorXd FL = compute_flux(U_L, normal);

    // Right state
    VectorXd P_R = cons_to_prim(U_R);
    Scalar hR = P_R(0), uR = P_R(1), vR = P_R(2);
    Scalar vnR = uR * nx + vR * ny;
    Scalar vtR = uR * tx + vR * ty;
    Scalar cR = celerity(hR);
    VectorXd FR = compute_flux(U_R, normal);

    // Roe averages
    Scalar sqrt_hL = std::sqrt(hL);
    Scalar sqrt_hR = std::sqrt(hR);
    Scalar denom = sqrt_hL + sqrt_hR;

    Scalar u_roe = (sqrt_hL * uL + sqrt_hR * uR) / denom;
    Scalar v_roe = (sqrt_hL * vL + sqrt_hR * vR) / denom;
    Scalar vn_roe = u_roe * nx + v_roe * ny;
    Scalar c_roe = std::sqrt(g_ * 0.5 * (hL + hR));

    // Wave speed estimates
    Scalar SL = std::min(vnL - cL, vn_roe - c_roe);
    Scalar SR = std::max(vnR + cR, vn_roe + c_roe);

    // Middle wave speed
    Scalar denom_SM = hL * (SL - vnL) - hR * (SR - vnR);
    Scalar SM = (0.5 * g_ * (hR*hR - hL*hL) + hL * vnL * (SL - vnL) - hR * vnR * (SR - vnR)) /
                (std::abs(denom_SM) > EPSILON ? denom_SM : EPSILON);

    // HLLC flux
    if (SL >= 0.0) {
        return FL;
    } else if (SL < 0.0 && SM >= 0.0) {
        Scalar factor_L = hL * (SL - vnL) / (SL - SM);
        VectorXd U_star_L(3);
        U_star_L(0) = factor_L;
        U_star_L(1) = factor_L * (SM * nx + vtL * tx);
        U_star_L(2) = factor_L * (SM * ny + vtL * ty);

        return FL + SL * (U_star_L - U_L);
    } else if (SM < 0.0 && SR > 0.0) {
        Scalar factor_R = hR * (SR - vnR) / (SR - SM);
        VectorXd U_star_R(3);
        U_star_R(0) = factor_R;
        U_star_R(1) = factor_R * (SM * nx + vtR * tx);
        U_star_R(2) = factor_R * (SM * ny + vtR * ty);

        return FR + SR * (U_star_R - U_R);
    } else {
        return FR;
    }
}

VectorXd ShallowWaterEquations::roe_flux(const VectorXd& U_L, const VectorXd& U_R,
                                          const Vector2d& normal) const {
    Scalar nx = normal.x();
    Scalar ny = normal.y();
    Scalar tx = -ny;
    Scalar ty = nx;

    // Left state
    VectorXd P_L = cons_to_prim(U_L);
    Scalar hL = P_L(0), uL = P_L(1), vL = P_L(2);
    Scalar vnL = uL * nx + vL * ny;
    Scalar vtL = uL * tx + vL * ty;
    VectorXd FL = compute_flux(U_L, normal);

    // Right state
    VectorXd P_R = cons_to_prim(U_R);
    Scalar hR = P_R(0), uR = P_R(1), vR = P_R(2);
    Scalar vnR = uR * nx + vR * ny;
    Scalar vtR = uR * tx + vR * ty;
    VectorXd FR = compute_flux(U_R, normal);

    // Roe averages
    Scalar sqrt_hL = std::sqrt(hL);
    Scalar sqrt_hR = std::sqrt(hR);
    Scalar denom = sqrt_hL + sqrt_hR;

    Scalar h = 0.5 * (hL + hR);
    Scalar u = (sqrt_hL * uL + sqrt_hR * uR) / denom;
    Scalar v = (sqrt_hL * vL + sqrt_hR * vR) / denom;
    Scalar c = std::sqrt(g_ * h);
    Scalar vn = u * nx + v * ny;

    // Wave strengths
    Scalar dh = hR - hL;
    Scalar dvn = vnR - vnL;
    Scalar dvt = vtR - vtL;

    Vector3d dV;
    dV(0) = 0.5 * (dh - h / c * dvn);
    dV(1) = h * dvt;
    dV(2) = 0.5 * (dh + h / c * dvn);

    // Wave speeds
    Vector3d ws;
    ws(0) = std::abs(vn - c);
    ws(1) = std::abs(vn);
    ws(2) = std::abs(vn + c);

    // Harten's entropy fix
    Scalar delta = 0.1 * c;
    if (ws(0) < delta) ws(0) = (ws(0) * ws(0) + delta * delta) / (2.0 * delta);
    if (ws(2) < delta) ws(2) = (ws(2) * ws(2) + delta * delta) / (2.0 * delta);

    // Right eigenvectors
    Eigen::Matrix3d Rv;
    Rv << 1.0, 0.0, 1.0,
          u - c*nx, -c*ny, u + c*nx,
          v - c*ny, c*nx, v + c*ny;

    // Roe flux
    Vector3d dissipation = Rv * (ws.cwiseProduct(dV));

    return 0.5 * (FL + FR - dissipation);
}

}  // namespace fvm2d
