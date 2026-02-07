#include "physics/euler_equations.hpp"
#include <cmath>
#include <algorithm>

namespace fvm2d {

EulerEquations::EulerEquations(Scalar gamma) : gamma_(gamma) {}

std::vector<std::string> EulerEquations::variable_names() const {
    return {"rho", "rho_u", "rho_v", "E"};
}

VectorXd EulerEquations::cons_to_prim(const VectorXd& U) const {
    VectorXd P(4);
    Scalar rho = std::max(U(0), EPSILON);
    Scalar u = U(1) / rho;
    Scalar v = U(2) / rho;
    Scalar E = U(3);
    Scalar p = (gamma_ - 1.0) * (E - 0.5 * rho * (u*u + v*v));

    P << rho, u, v, p;
    return P;
}

VectorXd EulerEquations::prim_to_cons(const VectorXd& P) const {
    VectorXd U(4);
    Scalar rho = P(0);
    Scalar u = P(1);
    Scalar v = P(2);
    Scalar p = P(3);
    Scalar E = p / (gamma_ - 1.0) + 0.5 * rho * (u*u + v*v);

    U << rho, rho*u, rho*v, E;
    return U;
}

Scalar EulerEquations::pressure(const VectorXd& U) const {
    Scalar rho = std::max(U(0), EPSILON);
    Scalar u = U(1) / rho;
    Scalar v = U(2) / rho;
    return (gamma_ - 1.0) * (U(3) - 0.5 * rho * (u*u + v*v));
}

Scalar EulerEquations::sound_speed(Scalar rho, Scalar p) const {
    return std::sqrt(gamma_ * std::max(p, EPSILON) / std::max(rho, EPSILON));
}

VectorXd EulerEquations::compute_flux(const VectorXd& U, const Vector2d& normal) const {
    VectorXd P = cons_to_prim(U);
    Scalar rho = P(0);
    Scalar u = P(1);
    Scalar v = P(2);
    Scalar p = P(3);

    // Normal velocity
    Scalar vn = u * normal.x() + v * normal.y();

    // Total enthalpy H = (E + p) / rho
    Scalar H = (U(3) + p) / rho;

    VectorXd F(4);
    F(0) = rho * vn;
    F(1) = rho * vn * u + p * normal.x();
    F(2) = rho * vn * v + p * normal.y();
    F(3) = rho * vn * H;

    return F;
}

Scalar EulerEquations::max_eigenvalue(const VectorXd& U) const {
    VectorXd P = cons_to_prim(U);
    Scalar rho = P(0);
    Scalar u = P(1);
    Scalar v = P(2);
    Scalar p = P(3);

    Scalar a = sound_speed(rho, p);
    return std::sqrt(u*u + v*v) + a;
}

VectorXd EulerEquations::apply_transmissive_bc(const VectorXd& U_inside) const {
    return U_inside;
}

VectorXd EulerEquations::apply_inlet_bc(const VectorXd& bc_value) const {
    // bc_value contains primitive variables [rho, u, v, p]
    return prim_to_cons(bc_value);
}

VectorXd EulerEquations::apply_outlet_bc(const VectorXd& U_inside,
                                          const VectorXd& bc_value) const {
    VectorXd P_inside = cons_to_prim(U_inside);
    Scalar rho_in = P_inside(0);
    Scalar u_in = P_inside(1);
    Scalar v_in = P_inside(2);
    Scalar p_ghost = bc_value(3);  // Prescribed outlet pressure

    VectorXd P_ghost(4);
    P_ghost << rho_in, u_in, v_in, p_ghost;
    return prim_to_cons(P_ghost);
}

VectorXd EulerEquations::apply_wall_bc(const VectorXd& U_inside,
                                        const Vector2d& normal,
                                        const VectorXd& bc_value) const {
    VectorXd P_inside = cons_to_prim(U_inside);
    Scalar rho_in = P_inside(0);
    Scalar u_in = P_inside(1);
    Scalar v_in = P_inside(2);
    Scalar p_in = P_inside(3);

    // Decompose velocity into normal and tangential components
    Scalar nx = normal.x();
    Scalar ny = normal.y();
    Scalar tx = -ny;
    Scalar ty = nx;

    Scalar vn_in = u_in * nx + v_in * ny;
    Scalar vt_in = u_in * tx + v_in * ty;

    // Reflect normal velocity
    Scalar vn_ghost = -vn_in;

    // Tangential velocity: slip (preserve) or no-slip (zero)
    bool is_slip = (bc_value.size() > 0 && bc_value(0) == 1.0);
    Scalar vt_ghost = is_slip ? vt_in : 0.0;

    // Reconstruct velocity in Cartesian coordinates
    Scalar u_ghost = vn_ghost * nx + vt_ghost * tx;
    Scalar v_ghost = vn_ghost * ny + vt_ghost * ty;

    VectorXd P_ghost(4);
    P_ghost << rho_in, u_ghost, v_ghost, p_in;
    return prim_to_cons(P_ghost);
}

VectorXd EulerEquations::hllc_flux(const VectorXd& U_L, const VectorXd& U_R,
                                    const Vector2d& normal) const {
    Scalar nx = normal.x();
    Scalar ny = normal.y();
    Scalar tx = -ny;
    Scalar ty = nx;

    // Left state
    VectorXd P_L = cons_to_prim(U_L);
    Scalar rL = P_L(0), uL = P_L(1), vL = P_L(2), pL = P_L(3);
    Scalar vnL = uL * nx + vL * ny;
    Scalar vtL = uL * tx + vL * ty;
    Scalar aL = sound_speed(rL, pL);
    VectorXd FL = compute_flux(U_L, normal);

    // Right state
    VectorXd P_R = cons_to_prim(U_R);
    Scalar rR = P_R(0), uR = P_R(1), vR = P_R(2), pR = P_R(3);
    Scalar vnR = uR * nx + vR * ny;
    Scalar vtR = uR * tx + vR * ty;
    Scalar aR = sound_speed(rR, pR);
    VectorXd FR = compute_flux(U_R, normal);

    // Roe averages for wave speed estimates
    Scalar sqrt_rL = std::sqrt(rL);
    Scalar sqrt_rR = std::sqrt(rR);
    Scalar denom = sqrt_rL + sqrt_rR;

    Scalar u_roe = (sqrt_rL * uL + sqrt_rR * uR) / denom;
    Scalar v_roe = (sqrt_rL * vL + sqrt_rR * vR) / denom;
    Scalar vn_roe = u_roe * nx + v_roe * ny;

    Scalar hL = (U_L(3) + pL) / rL;
    Scalar hR = (U_R(3) + pR) / rR;
    Scalar h_roe = (sqrt_rL * hL + sqrt_rR * hR) / denom;

    Scalar a_roe_sq = (gamma_ - 1.0) * (h_roe - 0.5 * (u_roe*u_roe + v_roe*v_roe));
    Scalar a_roe = std::sqrt(std::max(a_roe_sq, EPSILON));

    // Wave speed estimates (Davis-Einfeldt)
    Scalar SL = std::min(vnL - aL, vn_roe - a_roe);
    Scalar SR = std::max(vnR + aR, vn_roe + a_roe);

    // Middle wave speed
    Scalar denom_SM = rL * (SL - vnL) - rR * (SR - vnR);
    Scalar SM = (pR - pL + rL * vnL * (SL - vnL) - rR * vnR * (SR - vnR)) /
                (std::abs(denom_SM) > EPSILON ? denom_SM : EPSILON);

    // HLLC flux
    if (SL >= 0.0) {
        return FL;
    } else if (SL < 0.0 && SM >= 0.0) {
        // Left star state
        Scalar factor_L = rL * (SL - vnL) / (SL - SM);
        VectorXd U_star_L(4);
        U_star_L(0) = factor_L;
        U_star_L(1) = factor_L * (SM * nx + vtL * tx);
        U_star_L(2) = factor_L * (SM * ny + vtL * ty);
        U_star_L(3) = factor_L * (U_L(3) / rL + (SM - vnL) * (SM + pL / (rL * (SL - vnL))));

        return FL + SL * (U_star_L - U_L);
    } else if (SM < 0.0 && SR > 0.0) {
        // Right star state
        Scalar factor_R = rR * (SR - vnR) / (SR - SM);
        VectorXd U_star_R(4);
        U_star_R(0) = factor_R;
        U_star_R(1) = factor_R * (SM * nx + vtR * tx);
        U_star_R(2) = factor_R * (SM * ny + vtR * ty);
        U_star_R(3) = factor_R * (U_R(3) / rR + (SM - vnR) * (SM + pR / (rR * (SR - vnR))));

        return FR + SR * (U_star_R - U_R);
    } else {
        return FR;
    }
}

VectorXd EulerEquations::roe_flux(const VectorXd& U_L, const VectorXd& U_R,
                                   const Vector2d& normal) const {
    Scalar nx = normal.x();
    Scalar ny = normal.y();
    Scalar tx = -ny;
    Scalar ty = nx;

    // Left state
    VectorXd P_L = cons_to_prim(U_L);
    Scalar rL = P_L(0), uL = P_L(1), vL = P_L(2), pL = P_L(3);
    Scalar vnL = uL * nx + vL * ny;
    Scalar vtL = uL * tx + vL * ty;
    Scalar HL = (U_L(3) + pL) / rL;
    VectorXd FL = compute_flux(U_L, normal);

    // Right state
    VectorXd P_R = cons_to_prim(U_R);
    Scalar rR = P_R(0), uR = P_R(1), vR = P_R(2), pR = P_R(3);
    Scalar vnR = uR * nx + vR * ny;
    Scalar vtR = uR * tx + vR * ty;
    Scalar HR = (U_R(3) + pR) / rR;
    VectorXd FR = compute_flux(U_R, normal);

    // Roe averages
    Scalar sqrt_rL = std::sqrt(rL);
    Scalar sqrt_rR = std::sqrt(rR);
    Scalar denom = sqrt_rL + sqrt_rR;

    Scalar r = sqrt_rL * sqrt_rR;
    Scalar u = (sqrt_rL * uL + sqrt_rR * uR) / denom;
    Scalar v = (sqrt_rL * vL + sqrt_rR * vR) / denom;
    Scalar H = (sqrt_rL * HL + sqrt_rR * HR) / denom;

    Scalar a_sq = (gamma_ - 1.0) * (H - 0.5 * (u*u + v*v));
    Scalar a = std::sqrt(std::max(a_sq, EPSILON));
    Scalar vn = u * nx + v * ny;

    // Wave strengths
    Scalar dr = rR - rL;
    Scalar dp = pR - pL;
    Scalar dvn = vnR - vnL;
    Scalar dvt = vtR - vtL;

    Vector4d dV;
    dV(0) = (dp - r * a * dvn) / (2.0 * a * a);
    dV(1) = r * dvt;
    dV(2) = dr - dp / (a * a);
    dV(3) = (dp + r * a * dvn) / (2.0 * a * a);

    // Wave speeds (eigenvalues)
    Vector4d ws;
    ws(0) = std::abs(vn - a);
    ws(1) = std::abs(vn);
    ws(2) = std::abs(vn);
    ws(3) = std::abs(vn + a);

    // Harten's entropy fix
    Scalar delta = 0.1 * a;
    for (int i = 0; i < 4; ++i) {
        if (i == 0 || i == 3) {
            if (ws(i) < delta) {
                ws(i) = (ws(i) * ws(i) + delta * delta) / (2.0 * delta);
            }
        }
    }

    // Right eigenvectors matrix
    Eigen::Matrix4d Rv;
    Rv << 1.0, 0.0, 1.0, 1.0,
          u - a*nx, -a*ny, u, u + a*nx,
          v - a*ny, a*nx, v, v + a*ny,
          H - vn*a, -(u*ny - v*nx)*a, 0.5*(u*u + v*v), H + vn*a;

    // Roe flux: F = 0.5 * (FL + FR) - 0.5 * sum(ws * dV * R)
    Vector4d dissipation = Rv * (ws.cwiseProduct(dV));

    return 0.5 * (FL + FR - dissipation);
}

}  // namespace fvm2d
