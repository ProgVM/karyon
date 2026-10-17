// ============================================================================
// Karyon-CoRE v31.0 Sovereign Substrate (Autopoietic Experiential Organism)
// Standard: KEP v28.0 (Principles 1, 2, 4, 5, 27, 28, 29, 30)
//
// Core Philosophy:
// 1. Absolute Sovereign Boundary:
//    The trainer/environment NEVER computes gradients, backpropagates, or alters
//    weights inside Karyon. The trainer only presents tasks and delivers honest
//    experiential feedback ("good" / "bad", valence + magnitude).
// 2. Continuous Internal Causal Trace:
//    Every thought, mode movement, and efferent decision leaves a physical trace:
//    tau * d(trace_k)/dt = -trace_k + q_k * action_projection
// 3. Intrinsic Autonomous Experience Assimilation:
//    When feedback arrives:
//    - POSITIVE EXPERIENCE (+): Lowers Free Energy, stabilizes resonant pathways,
//      locks beneficial modes, and strengthens the causal attractor basin.
//    - NEGATIVE EXPERIENCE (-): Increases internal tension, applies adaptive
//      damping/inhibition to guilty modes, and triggers exploratory phase shifts.
// ============================================================================

#ifndef KARYON_SOVEREIGN_EXPERIENTIAL_CORE_H
#define KARYON_SOVEREIGN_EXPERIENTIAL_CORE_H

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <immintrin.h>

namespace karyon {

constexpr size_t NUM_CONTINUUM_MODES = 128;
constexpr size_t CHANNEL_IO_DIM = 256;

struct alignas(64) ExperientialModeState {
    std::array<float, NUM_CONTINUUM_MODES> q;            // Mode coordinates (latent thoughts)
    std::array<float, NUM_CONTINUUM_MODES> p;            // Mode momenta
    std::array<float, NUM_CONTINUUM_MODES> stiffness;    // Intrinsic stiffness K_k
    std::array<float, NUM_CONTINUUM_MODES> damping;      // Dissipative damping Gamma_k
    std::array<float, NUM_CONTINUUM_MODES> causal_trace; // Short-term eligibility trace
    std::array<float, NUM_CONTINUUM_MODES> long_term_credit; // Consolidated epigenetic wisdom
};

class SovereignExperientialCore {
public:
    SovereignExperientialCore() noexcept {
        reset();
    }

    void reset() noexcept {
        total_experiences_ = 0;
        positive_accumulated_ = 0.0f;
        negative_accumulated_ = 0.0f;
        current_free_energy_ = 1.0f;

        std::mt19937 rng(3100);
        std::normal_distribution<float> norm(0.0f, 0.05f);

        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            state_.q[k] = 0.0f;
            state_.p[k] = 0.0f;
            state_.stiffness[k] = 1.0f + 0.02f * static_cast<float>(k);
            state_.damping[k] = 0.4f + 0.005f * static_cast<float>(k);
            state_.causal_trace[k] = 0.0f;
            state_.long_term_credit[k] = 0.0f;
        }

        // Intrinsic receptive and expressive coupling fields
        for (size_t i = 0; i < CHANNEL_IO_DIM * NUM_CONTINUUM_MODES; ++i) {
            W_receptive_[i] = norm(rng);
            W_expressive_[i] = norm(rng);
        }

        for (size_t i = 0; i < CHANNEL_IO_DIM; ++i) {
            b_expressive_[i] = 0.0f;
            latest_probabilities_[i] = 1.0f / static_cast<float>(CHANNEL_IO_DIM);
        }
    }

    // =========================================================================
    // 1. AUTONOMOUS PERCEPTION & EXPRESSION
    // The organism perceives external sensory stimulus, unrolls its physical
    // continuous wavefield, forms an internal causal trace, and chooses an action.
    // =========================================================================
    size_t perceive_and_decide(const float* sensory_in, size_t in_dim, float dt) noexcept {
        const size_t act_in = std::min(in_dim, CHANNEL_IO_DIM);

        // --- Continuous Receptive Field Coupling ---
        std::array<float, NUM_CONTINUUM_MODES> F_receptive{};
        F_receptive.fill(0.0f);

        for (size_t i = 0; i < act_in; ++i) {
            const float s_val = sensory_in[i];
            if (std::abs(s_val) < 1e-9f) continue;
            const size_t row = i * NUM_CONTINUUM_MODES;
            #pragma GCC ivdep
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                F_receptive[k] += s_val * W_receptive_[row + k];
            }
        }

        // --- Exact Lie-Flow Hamiltonian Mode Propagation ---
        #pragma GCC ivdep
        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            const float omega_sq = state_.stiffness[k];
            const float gamma = 0.5f * state_.damping[k];
            const float drive = F_receptive[k];

            const float q_star = drive / (omega_sq + 1e-6f);
            const float q_rel  = state_.q[k] - q_star;
            const float p_curr = state_.p[k];
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

            state_.q[k] = std::clamp(q_next, -10.0f, 10.0f);
            state_.p[k] = std::clamp(p_next, -20.0f, 20.0f);
        }

        // --- Expressive Readout to Action Space ---
        float max_l = -1e9f;
        std::array<float, CHANNEL_IO_DIM> logits{};
        for (size_t j = 0; j < CHANNEL_IO_DIM; ++j) {
            float sum = b_expressive_[j];
            const size_t col = j * NUM_CONTINUUM_MODES;
            #pragma GCC ivdep
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                sum += state_.q[k] * W_expressive_[col + k];
            }
            logits[j] = sum;
            if (sum > max_l) max_l = sum;
        }

        // Softmax probabilities
        float sum_exp = 0.0f;
        for (size_t j = 0; j < CHANNEL_IO_DIM; ++j) {
            float p = std::exp(logits[j] - max_l);
            latest_probabilities_[j] = p;
            sum_exp += p;
        }

        const float inv_sum_exp = 1.0f / (sum_exp + 1e-7f);
        size_t chosen_action = 0;
        float max_prob = -1.0f;

        for (size_t j = 0; j < CHANNEL_IO_DIM; ++j) {
            latest_probabilities_[j] *= inv_sum_exp;
            if (latest_probabilities_[j] > max_prob) {
                max_prob = latest_probabilities_[j];
                chosen_action = j;
            }
        }

        latest_chosen_action_ = chosen_action;

        // --- Update Internal Causal Trace (Principle 5) ---
        // Records which physical modes directly generated this decision
        const size_t chosen_col = chosen_action * NUM_CONTINUUM_MODES;
        #pragma GCC ivdep
        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            // Causal eligibility: product of mode activation and expressive contribution
            const float contribution = state_.q[k] * W_expressive_[chosen_col + k];
            state_.causal_trace[k] = 0.90f * state_.causal_trace[k] + 0.10f * contribution;
        }

        return chosen_action;
    }

    // =========================================================================
    // 2. INTRINSIC EXPERIENCE ASSIMILATION (The Core Understands Good & Bad)
    // Trainer/Environment passes strictly:
    // - valence: +1.0 (good/correct/rewarding), -1.0 (bad/erroneous/destructive)
    // - magnitude: [0.0, 1.0] intensity of feedback
    // - correct_target_hint: optional environmental beacon (if environment demonstrates truth)
    //
    // The core autonomously learns from its own causality!
    // =========================================================================
    void assimilate_experience(float valence, float magnitude, int target_hint = -1, float plasticity = 0.05f) noexcept {
        total_experiences_++;

        const float effective_feedback = std::clamp(valence, -1.0f, 1.0f) * std::clamp(magnitude, 0.0f, 2.0f);

        if (effective_feedback > 0.0f) {
            positive_accumulated_ += effective_feedback;
            current_free_energy_ = 0.99f * current_free_energy_; // Free Energy drops on success
        } else {
            negative_accumulated_ += std::abs(effective_feedback);
            current_free_energy_ = 0.99f * current_free_energy_ + 0.01f * std::abs(effective_feedback);
        }

        const size_t chosen = latest_chosen_action_;
        const size_t chosen_col = chosen * NUM_CONTINUUM_MODES;

        // 1. Causal Substrate Modulation:
        // Adjust mode stiffness and damping based on causal responsibility
        #pragma GCC ivdep
        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            const float trace = state_.causal_trace[k];

            if (effective_feedback > 0.0f) {
                // POSITIVE: Consolidate modes that caused this success
                state_.long_term_credit[k] += plasticity * effective_feedback * std::abs(trace);
                // Lower damping on successful resonant modes (fluidity)
                state_.damping[k] = std::clamp(state_.damping[k] * 0.999f, 0.05f, 20.0f);
            } else {
                // NEGATIVE: The modes responsible for error are penalized / quenched
                state_.long_term_credit[k] -= plasticity * std::abs(effective_feedback) * std::abs(trace);
                // Increase damping to quench erroneous oscillation
                state_.damping[k] = std::clamp(state_.damping[k] * 1.005f, 0.05f, 50.0f);
                // Push mode away from error basin (exploratory perturbation)
                state_.q[k] *= 0.95f;
            }
        }

        // 2. Expressive Plasticity (Self-Directed Routing):
        // If the environment also provides a hint (demonstration of what was right):
        if (target_hint >= 0 && target_hint < static_cast<int>(CHANNEL_IO_DIM)) {
            const size_t target = static_cast<size_t>(target_hint);
            const size_t target_col = target * NUM_CONTINUUM_MODES;

            // Reinforce target beacon
            b_expressive_[target] += plasticity * 0.1f;
            #pragma GCC ivdep
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                W_expressive_[target_col + k] += plasticity * state_.q[k] * 0.1f;
            }

            // Suppress erroneous chosen action if it was wrong
            if (target != chosen) {
                b_expressive_[chosen] -= plasticity * 0.1f;
                #pragma GCC ivdep
                for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                    W_expressive_[chosen_col + k] -= plasticity * state_.q[k] * 0.1f;
                }
            }
        } else {
            // Pure reinforcement / experiential trial-and-error (no hint provided)
            b_expressive_[chosen] += plasticity * effective_feedback * 0.1f;
            #pragma GCC ivdep
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                W_expressive_[chosen_col + k] += plasticity * effective_feedback * state_.q[k] * 0.1f;
            }
        }

        // 3. Receptive Associative Grounding (Hebb-Predictive Coupling)
        // Strengthen receptive paths when outcome is positive
        if (effective_feedback > 0.0f) {
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                W_receptive_[chosen * NUM_CONTINUUM_MODES + k] += plasticity * state_.q[k] * 0.01f;
            }
        }
    }

    // Accessors
    size_t get_latest_action() const noexcept { return latest_chosen_action_; }
    float get_free_energy() const noexcept { return current_free_energy_; }
    float get_positive_experience() const noexcept { return positive_accumulated_; }
    float get_negative_experience() const noexcept { return negative_accumulated_; }
    const std::array<float, CHANNEL_IO_DIM>& get_probabilities() const noexcept { return latest_probabilities_; }

private:
    size_t total_experiences_;
    size_t latest_chosen_action_;
    float positive_accumulated_;
    float negative_accumulated_;
    float current_free_energy_;

    ExperientialModeState state_;

    alignas(64) std::array<float, CHANNEL_IO_DIM * NUM_CONTINUUM_MODES> W_receptive_;
    alignas(64) std::array<float, CHANNEL_IO_DIM * NUM_CONTINUUM_MODES> W_expressive_;
    alignas(64) std::array<float, CHANNEL_IO_DIM> b_expressive_;
    alignas(64) std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

#endif // KARYON_SOVEREIGN_EXPERIENTIAL_CORE_H
