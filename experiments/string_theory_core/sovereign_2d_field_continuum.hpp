// ============================================================================
// EXP-521: 2D Spatial Manifold Continuous Metric Field \Psi(x, y, t)
// Karyon-CoRE v28.0 Sovereign Architecture
//
// Expanding spatial dimension from 1D string to 2D continuous membrane/field!
// The field \Psi(x, y, t) is expanded over a 2D Fourier-Galerkin basis:
// \Psi(x, y, t) = \sum_{k_x=1}^{M_x} \sum_{k_y=1}^{M_y} q_{k_x, k_y}(t) \sin(k_x \pi x) \sin(k_y \pi y)
//
// Physics:
// An inverted spherical pendulum (2 DOF: \theta_x, \theta_y) coupled to the
// 2D continuous elastic metric sheet with stress-energy tensor T_{\mu\nu}.
// Torques \tau_x and \tau_y emerge as boundary traction integrals along the edges!
// ============================================================================

#ifndef SOVEREIGN_2D_FIELD_CONTINUUM_HPP
#define SOVEREIGN_2D_FIELD_CONTINUUM_HPP

#include <array>
#include <cmath>
#include <algorithm>
#include <numbers>

namespace karyon {

constexpr size_t MODES_2D_X = 8;
constexpr size_t MODES_2D_Y = 8;
constexpr size_t TOTAL_MODES_2D = MODES_2D_X * MODES_2D_Y; // 64 spatial modes

struct alignas(64) FieldState2D {
    std::array<float, TOTAL_MODES_2D> q;
    std::array<float, TOTAL_MODES_2D> p;
    std::array<float, TOTAL_MODES_2D> stiffness;
    std::array<float, TOTAL_MODES_2D> damping;
};

struct Vector2D {
    float x = 0.0f;
    float y = 0.0f;
};

class Sovereign2DContinuum {
public:
    Sovereign2DContinuum() {
        for (size_t kx = 0; kx < MODES_2D_X; ++kx) {
            for (size_t ky = 0; ky < MODES_2D_Y; ++ky) {
                size_t idx = kx * MODES_2D_Y + ky;
                state_.q[idx] = 0.0f;
                state_.p[idx] = 0.0f;
                state_.stiffness[idx] = 2.0f;
                state_.damping[idx] = 1.0f;

                float kx_f = static_cast<float>(kx + 1);
                float ky_f = static_cast<float>(ky + 1);

                kx_arr_[idx] = kx_f;
                ky_arr_[idx] = ky_f;
                k_norm_sq_arr_[idx] = kx_f * kx_f + ky_f * ky_f;
                inv_k_norm_arr_[idx] = 1.0f / std::sqrt(k_norm_sq_arr_[idx]);
            }
        }
    }

    // 2D Perception-Action step with exact analytical Lie-flow per mode
    Vector2D perceive_and_act_2d(Vector2D error, Vector2D error_rate, float dt) noexcept {
        const float fe = 0.5f * (error.x * error.x + error.y * error.y) +
                         0.02f * (error_rate.x * error_rate.x + error_rate.y * error_rate.y);

        // Continuous 2D surface plucking:
        // Modes are excited according to directional spatial projection
        const float plucking_x = (error.x + 0.15f * error_rate.x) * 10.0f;
        const float plucking_y = (error.y + 0.15f * error_rate.y) * 10.0f;

        #pragma GCC ivdep
        for (size_t i = 0; i < TOTAL_MODES_2D; ++i) {
            const float kx = kx_arr_[i];
            const float ky = ky_arr_[i];

            // Directional force coupling on (kx, ky) harmonic
            const float F_k = (plucking_x * (kx / (kx + ky)) + plucking_y * (ky / (kx + ky))) * inv_k_norm_arr_[i];
            const float omega_sq = state_.stiffness[i] * k_norm_sq_arr_[i];
            const float gamma = 0.5f * state_.damping[i];

            const float q_star = F_k / (omega_sq + 1e-7f);
            const float q_rel  = state_.q[i] - q_star;
            const float p_curr = state_.p[i];

            const float disc = omega_sq - gamma * gamma;

            float q_next = 0.0f;
            float p_next = 0.0f;

            if (disc > 1e-5f) {
                const float omega_d = std::sqrt(disc);
                const float decay   = std::exp(-gamma * dt);
                const float cos_wt  = std::cos(omega_d * dt);
                const float sin_wt  = std::sin(omega_d * dt);
                const float inv_wd  = 1.0f / omega_d;

                q_next = decay * (q_rel * cos_wt + (p_curr + gamma * q_rel) * inv_wd * sin_wt) + q_star;
                p_next = decay * (p_curr * cos_wt - (omega_sq * q_rel + gamma * p_curr) * inv_wd * sin_wt);
            } else {
                const float decay = std::exp(-gamma * dt);
                q_next = decay * (q_rel + (p_curr + gamma * q_rel) * dt) + q_star;
                p_next = decay * (p_curr - gamma * (p_curr + gamma * q_rel) * dt);
            }

            state_.q[i] = std::clamp(q_next, -10.0f, 10.0f);
            state_.p[i] = std::clamp(p_next, -30.0f, 30.0f);
        }

        // Sovereign Metric Plasticity in 2D (Principle 29)
        if (fe > 1e-12f) {
            const float err_mag = std::sqrt(error.x * error.x + error.y * error.y);
            const float err_rate_mag = std::sqrt(error_rate.x * error_rate.x + error_rate.y * error_rate.y);
            const float d_stiff = 0.08f * err_mag * dt;
            const float d_damp  = 0.08f * err_rate_mag * dt;

            #pragma GCC ivdep
            for (size_t i = 0; i < TOTAL_MODES_2D; ++i) {
                state_.stiffness[i] = std::min(state_.stiffness[i] + d_stiff * inv_k_norm_arr_[i], 50.0f);
                state_.damping[i]   = std::min(state_.damping[i] + d_damp, 20.0f);
            }
        }

        // Boundary Surface Traction Integrals: \tau_x and \tau_y
        float tau_x = 0.0f;
        float tau_y = 0.0f;
        #pragma GCC ivdep
        for (size_t i = 0; i < TOTAL_MODES_2D; ++i) {
            tau_x += kx_arr_[i] * state_.stiffness[i] * state_.q[i];
            tau_y += ky_arr_[i] * state_.stiffness[i] * state_.q[i];
        }

        latest_fe_ = fe;
        latest_torque_ = { 12.0f * tau_x, 12.0f * tau_y };
        return latest_torque_;
    }

    float get_free_energy() const noexcept { return latest_fe_; }
    Vector2D get_torque() const noexcept { return latest_torque_; }

private:
    FieldState2D state_;
    alignas(64) std::array<float, TOTAL_MODES_2D> kx_arr_;
    alignas(64) std::array<float, TOTAL_MODES_2D> ky_arr_;
    alignas(64) std::array<float, TOTAL_MODES_2D> k_norm_sq_arr_;
    alignas(64) std::array<float, TOTAL_MODES_2D> inv_k_norm_arr_;

    float latest_fe_ = 0.0f;
    Vector2D latest_torque_;
};

} // namespace karyon

#endif // SOVEREIGN_2D_FIELD_CONTINUUM_HPP
