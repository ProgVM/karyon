// ============================================================================
// Karyon-CoRE v30.0 Sovereign Substrate (Pure Universal Epistemic Continuum)
// Standard: KEP v28.0 (Principles 1, 2, 4, 5, 27, 28, 29, 30)
//
// Cybernetic Axiom:
// 1. The Core is strictly task-agnostic:
//    It knows NOTHING about text, ASCII, pendulums, robotics, or Lorenz attractors.
// 2. The Core perceives a Generalized Afferent Field (Sensory Stream) and an
//    Epistemic Error Vector e(t) supplied by the environment/trainer.
// 3. Epistemic Principle of Action:
//    Error acts as a Generalized Physical Tension Force in Hamiltonian wavefield:
//    d^2 q_k / dt^2 + Gamma_k * dq_k / dt + K_k * q_k = F_sensory_k + F_error_k
// 4. Autonomous Sovereign Plasticity (Principle 29):
//    The core continuously shapes its own internal stiffness tensor K_k,
//    damping Gamma_k, and coupling tensors to minimize Free Energy without
//    task-specific heuristics.
// ============================================================================

#ifndef KARYON_SOVEREIGN_CORE_V30_H
#define KARYON_SOVEREIGN_CORE_V30_H

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <immintrin.h>

namespace karyon {

constexpr size_t UNIVERSAL_FIELD_DIM = 128;
constexpr size_t UNIVERSAL_IO_DIM = 256;

struct alignas(64) ContinuumMetrics {
    std::array<float, UNIVERSAL_FIELD_DIM> q;            // Physical mode coordinates
    std::array<float, UNIVERSAL_FIELD_DIM> p;            // Conjugate momenta
    std::array<float, UNIVERSAL_FIELD_DIM> stiffness;    // Internal metric stiffness tensor K_k
    std::array<float, UNIVERSAL_FIELD_DIM> damping;      // Endogenous dissipation tensor Gamma_k
    std::array<float, UNIVERSAL_FIELD_DIM> causal_trace; // Epistemic credit trace (Principle 5)
};

class UniversalEpistemicContinuum {
public:
    UniversalEpistemicContinuum() noexcept {
        reset();
    }

    void reset() noexcept {
        total_steps_ = 0;
        latest_fe_ = 0.0f;
        latest_surprisal_norm_ = 0.0f;

        std::mt19937 rng(3000);
        std::normal_distribution<float> norm_dist(0.0f, 0.05f);

        for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
            metrics_.q[k] = 0.0f;
            metrics_.p[k] = 0.0f;
            metrics_.stiffness[k] = 1.0f + 0.05f * static_cast<float>(k);
            metrics_.damping[k] = 0.3f + 0.01f * static_cast<float>(k);
            metrics_.causal_trace[k] = 0.0f;
        }

        // Afferent projection: maps external sensory channel into internal wavefield
        for (size_t i = 0; i < UNIVERSAL_IO_DIM * UNIVERSAL_FIELD_DIM; ++i) {
            W_in_[i] = norm_dist(rng);
            W_out_[i] = norm_dist(rng);
        }

        for (size_t i = 0; i < UNIVERSAL_IO_DIM; ++i) {
            b_out_[i] = 0.0f;
        }
    }

    // =========================================================================
    // 1. FORWARD DYNAMICS (Perceive & Project Prediction)
    // Universal interface: accepts sensory vector x of dimension N <= UNIVERSAL_IO_DIM.
    // Advances internal continuous Hamiltonian wavefield by dt.
    // Computes and writes predicted efferent/anticipation vector into out_pred.
    // =========================================================================
    void step_dynamics(const float* sensory_in, size_t in_dim, float* out_pred, size_t out_dim, float dt) noexcept {
        total_steps_++;
        const size_t act_in = std::min(in_dim, UNIVERSAL_IO_DIM);
        const size_t act_out = std::min(out_dim, UNIVERSAL_IO_DIM);

        // --- Continuous Spatial Plucking: Sensory In -> Field Force F_k ---
        std::array<float, UNIVERSAL_FIELD_DIM> F_sensory{};
        F_sensory.fill(0.0f);

        for (size_t i = 0; i < act_in; ++i) {
            const float s_val = sensory_in[i];
            if (std::abs(s_val) < 1e-9f) continue;
            const size_t row_offset = i * UNIVERSAL_FIELD_DIM;
            #pragma GCC ivdep
            for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
                F_sensory[k] += s_val * W_in_[row_offset + k];
            }
        }

        // --- Exact Lie-Flow Hamiltonian Mode Propagation ---
        #pragma GCC ivdep
        for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
            const float omega_sq = metrics_.stiffness[k];
            const float gamma = 0.5f * metrics_.damping[k];
            const float drive = F_sensory[k];

            const float q_star = drive / (omega_sq + 1e-6f);
            const float q_rel  = metrics_.q[k] - q_star;
            const float p_curr = metrics_.p[k];
            const float disc   = omega_sq - gamma * gamma;

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
                q_next = decay * (q_rel + p_curr * dt) + q_star;
                p_next = decay * (p_curr - gamma * p_curr * dt);
            }

            metrics_.q[k] = std::clamp(q_next, -10.0f, 10.0f);
            metrics_.p[k] = std::clamp(p_next, -20.0f, 20.0f);
        }

        // --- Efferent Readout: Wavefield State -> Anticipation / Prediction ---
        for (size_t j = 0; j < act_out; ++j) {
            float sum = b_out_[j];
            const size_t col_offset = j * UNIVERSAL_FIELD_DIM;
            #pragma GCC ivdep
            for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
                sum += metrics_.q[k] * W_out_[col_offset + k];
            }
            out_pred[j] = sum;
        }
    }

    // =========================================================================
    // 2. EPISTEMIC FEEDBACK & PLASTICITY (Assimilate Generalized Error)
    // Universal interface: accepts error vector e = (prediction - target)
    // supplied by the environment / objective script.
    // 
    // The core transforms this error into:
    // a) Physical reactive tension on the modes: F_error_k = sum_j e_j * W_out_{j,k}
    // b) Causal plasticity on W_in, W_out, and intrinsic stiffness/damping tensors.
    // =========================================================================
    void assimilate_error(const float* sensory_in, size_t in_dim,
                          const float* error_vec, size_t error_dim,
                          float plasticity_rate, float dt) noexcept {
        const size_t act_in = std::min(in_dim, UNIVERSAL_IO_DIM);
        const size_t act_err = std::min(error_dim, UNIVERSAL_IO_DIM);

        // 1. Backproject error to physical mode tension vector (dF / dq_k)
        std::array<float, UNIVERSAL_FIELD_DIM> mode_tension{};
        mode_tension.fill(0.0f);

        float error_sq_sum = 0.0f;
        for (size_t j = 0; j < act_err; ++j) {
            const float e_j = error_vec[j];
            error_sq_sum += e_j * e_j;
            if (std::abs(e_j) < 1e-9f) continue;

            const size_t col_offset = j * UNIVERSAL_FIELD_DIM;
            #pragma GCC ivdep
            for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
                mode_tension[k] += e_j * W_out_[col_offset + k];
            }
        }

        latest_surprisal_norm_ = std::sqrt(error_sq_sum);
        latest_fe_ = 0.5f * error_sq_sum;

        if (plasticity_rate <= 0.0f) return;

        // 2. Sovereign Internal Plasticity (Principle 29)
        // Self-adjust stiffness K_k and damping Gamma_k along causal tension
        #pragma GCC ivdep
        for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
            const float tension_k = mode_tension[k];
            const float q_k = metrics_.q[k];

            // Causal credit alignment (Principle 5): does mode motion counteract tension?
            const float alignment = -tension_k * q_k;
            metrics_.causal_trace[k] = 0.99f * metrics_.causal_trace[k] + 0.01f * alignment;

            // Dynamically tune frequency/stiffness to match environmental rhythm
            metrics_.stiffness[k] = std::clamp(metrics_.stiffness[k] + plasticity_rate * alignment * dt, 0.1f, 50.0f);
            metrics_.damping[k]   = std::clamp(metrics_.damping[k] + plasticity_rate * (std::abs(tension_k) - 0.1f) * dt, 0.05f, 20.0f);

            // Directly inject physical error tension to correct momentum immediately (Hamiltonian force)
            metrics_.p[k] -= tension_k * dt * 2.0f;
        }

        // 3. Plasticity of Readout / Anticipation Tensor W_out and Bias
        for (size_t j = 0; j < act_err; ++j) {
            const float e_j = error_vec[j];
            b_out_[j] -= plasticity_rate * e_j;

            const size_t col_offset = j * UNIVERSAL_FIELD_DIM;
            #pragma GCC ivdep
            for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
                W_out_[col_offset + k] -= plasticity_rate * e_j * metrics_.q[k];
            }
        }

        // 4. Plasticity of Afferent Receptive Tensor W_in
        for (size_t i = 0; i < act_in; ++i) {
            const float s_i = sensory_in[i];
            if (std::abs(s_i) < 1e-9f) continue;

            const size_t row_offset = i * UNIVERSAL_FIELD_DIM;
            #pragma GCC ivdep
            for (size_t k = 0; k < UNIVERSAL_FIELD_DIM; ++k) {
                // Hebbian-Predictive assimilation: associate sensory input with mode tension
                W_in_[row_offset + k] -= plasticity_rate * mode_tension[k] * s_i * 0.1f;
            }
        }
    }

    float get_free_energy() const noexcept { return latest_fe_; }
    float get_surprisal_norm() const noexcept { return latest_surprisal_norm_; }
    const ContinuumMetrics& get_metrics() const noexcept { return metrics_; }

private:
    size_t total_steps_;
    float latest_fe_;
    float latest_surprisal_norm_;

    ContinuumMetrics metrics_;

    // Universal Afferent & Efferent Coupling Tensors: [256 x 128]
    alignas(64) std::array<float, UNIVERSAL_IO_DIM * UNIVERSAL_FIELD_DIM> W_in_;
    alignas(64) std::array<float, UNIVERSAL_IO_DIM * UNIVERSAL_FIELD_DIM> W_out_;
    alignas(64) std::array<float, UNIVERSAL_IO_DIM> b_out_;
};

} // namespace karyon

#endif // KARYON_SOVEREIGN_CORE_V30_H
