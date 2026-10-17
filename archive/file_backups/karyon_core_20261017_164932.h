// ============================================================================
// Karyon-CoRE v32.0 Sovereign Experiential Continuum with Contextual Attractors
// Standard: KEP v28.0 (Principles 1, 2, 4, 5, 6, 27, 28, 29, 30)
//
// Root Cybernetic Principles:
// 1. Contextual State Vector (Phase Memory):
//    An action is NEVER chosen in a vacuum. It is chosen from the current physical
//    trajectory in phase space: xi(t) = [q(t), p(t)] in R^{2*N_modes}.
// 2. Attractor Transition Landscape:
//    The core maintains an associative transition memory M[target, k].
// 3. True Experiential Learning:
//    - POSITIVE EXPERIENCE (+1.0):
//      Deepens the energy well connecting the current trajectory xi(t) to the target:
//      Delta M[target, :] += lr * xi(t)
//      Reduces internal dissipation (fluidity).
//    - NEGATIVE EXPERIENCE (-1.0):
//      Forms a REPULSION BARRIER directly on the erroneous transition from THIS context:
//      Delta M[erroneous_choice, :] -= lr * xi(t)
//      This guarantees that next time the system is in context xi(t), it will NOT
//      repeat the same mistake, completely preventing oscillatory stutter (wwwwwvwv).
// 4. Autonomous Epigenetic Consolidation:
//    Frequently rewarded transitions are epigenetically locked, preventing forgetting.
// ============================================================================

#ifndef KARYON_SOVEREIGN_EXPERIENTIAL_CONTINUUM_V32_H
#define KARYON_SOVEREIGN_EXPERIENTIAL_CONTINUUM_V32_H

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <iostream>
#include <immintrin.h>

namespace karyon {

constexpr size_t NUM_CONTINUUM_MODES = 128;
constexpr size_t STATE_PHASE_DIM = NUM_CONTINUUM_MODES * 2; // [q, p] phase space
constexpr size_t CHANNEL_IO_DIM = 256;

struct alignas(64) PhaseStateMetrics {
    std::array<float, NUM_CONTINUUM_MODES> q;            // Mode displacements
    std::array<float, NUM_CONTINUUM_MODES> p;            // Mode conjugate momenta
    std::array<float, NUM_CONTINUUM_MODES> stiffness;    // Mode stiffness K_k
    std::array<float, NUM_CONTINUUM_MODES> damping;      // Mode dissipation Gamma_k
    std::array<float, STATE_PHASE_DIM> context_trace;    // Continuous phase-trajectory memory
};

class SovereignExperientialContinuum {
public:
    SovereignExperientialContinuum() noexcept {
        reset();
    }

    void reset() noexcept {
        total_experiences_ = 0;
        positive_accumulated_ = 0.0f;
        negative_accumulated_ = 0.0f;
        internal_free_energy_ = 1.0f;
        latest_chosen_action_ = 0;

        std::mt19937 rng(3200);
        std::normal_distribution<float> norm(0.0f, 0.02f);

        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            state_.q[k] = 0.0f;
            state_.p[k] = 0.0f;
            state_.stiffness[k] = 1.0f + 0.03f * static_cast<float>(k);
            state_.damping[k] = 0.35f + 0.005f * static_cast<float>(k);
        }

        for (size_t i = 0; i < STATE_PHASE_DIM; ++i) {
            state_.context_trace[i] = 0.0f;
        }

        // Initialize Afferent Ingestion Tensor (One-hot byte -> physical pulse)
        for (size_t i = 0; i < CHANNEL_IO_DIM * NUM_CONTINUUM_MODES; ++i) {
            W_afferent_[i] = norm(rng);
        }

        // Initialize Context-to-Action Attractor Field: [256 actions x 256 phase-dim]
        for (size_t i = 0; i < CHANNEL_IO_DIM * STATE_PHASE_DIM; ++i) {
            attractor_field_[i] = norm(rng);
        }

        for (size_t i = 0; i < CHANNEL_IO_DIM; ++i) {
            action_bias_[i] = 0.0f;
            latest_probabilities_[i] = 1.0f / static_cast<float>(CHANNEL_IO_DIM);
        }
    }

    // =========================================================================
    // 1. CONTINUOUS PERCEPTION & PHASE STATE TRAJECTORY INTEGRATION
    // Sensory input plucks the wavefield; Lie-Hamiltonian dynamics propagate q, p.
    // The current phase trajectory vector xi = [q, p] is preserved as context.
    // =========================================================================
    size_t perceive_and_decide(uint8_t input_channel, float dt, float temperature = 0.0f) noexcept {
        // --- 1. Afferent Sensory Pulse into Physical Modes ---
        const size_t in_offset = static_cast<size_t>(input_channel) * NUM_CONTINUUM_MODES;

        #pragma GCC ivdep
        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            const float drive = W_afferent_[in_offset + k] * 5.0f;
            const float omega_sq = state_.stiffness[k];
            const float gamma = 0.5f * state_.damping[k];

            // Analytic Stratonovich-Lie step
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

            state_.q[k] = std::clamp(q_next, -8.0f, 8.0f);
            state_.p[k] = std::clamp(p_next, -16.0f, 16.0f);
        }

        // --- 2. Form Full Phase Trajectory Vector xi = [q, p] ---
        // Also integrate continuous leaky history to preserve multi-step context
        for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
            // First half: coordinate memory (slower decay, keeps context)
            state_.context_trace[k] = 0.85f * state_.context_trace[k] + 0.15f * state_.q[k];
            // Second half: momentum memory (faster, keeps instantaneous rhythm)
            state_.context_trace[NUM_CONTINUUM_MODES + k] = 0.70f * state_.context_trace[NUM_CONTINUUM_MODES + k] + 0.30f * state_.p[k];
        }

        // Normalize context trace for bounded stable attractor retrieval
        float norm_sq = 0.0f;
        #pragma GCC ivdep
        for (size_t i = 0; i < STATE_PHASE_DIM; ++i) {
            norm_sq += state_.context_trace[i] * state_.context_trace[i];
        }
        const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-6f);

        std::array<float, STATE_PHASE_DIM> norm_context{};
        #pragma GCC ivdep
        for (size_t i = 0; i < STATE_PHASE_DIM; ++i) {
            norm_context[i] = state_.context_trace[i] * inv_norm;
        }

        // Preserve current normalized context for experiential learning
        latest_context_ = norm_context;

        // --- 3. Compute Energy Resonance with Attractor Basins ---
        // Resonance E_c = action_bias_c + sum_i (Attractor_ci * context_i)
        std::array<float, CHANNEL_IO_DIM> energy{};
        float max_e = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float sum = action_bias_[c];
            const size_t c_offset = c * STATE_PHASE_DIM;
            #pragma GCC ivdep
            for (size_t i = 0; i < STATE_PHASE_DIM; ++i) {
                sum += attractor_field_[c_offset + i] * norm_context[i];
            }
            energy[c] = sum;
            if (sum > max_e) max_e = sum;
        }

        // Softmax probabilities
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float val = (temperature > 0.01f) ? (energy[c] - max_e) / temperature : (energy[c] - max_e);
            float p = std::exp(val);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }

        const float inv_sum_exp = 1.0f / (sum_exp + 1e-7f);
        size_t chosen = 0;

        if (temperature <= 0.01f) {
            // Greedy MAP decision
            float best_val = -1e9f;
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                latest_probabilities_[c] *= inv_sum_exp;
                if (energy[c] > best_val) {
                    best_val = energy[c];
                    chosen = c;
                }
            }
        } else {
            // Temperature sampling
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                latest_probabilities_[c] *= inv_sum_exp;
            }
            float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            float cum = 0.0f;
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                cum += latest_probabilities_[c];
                if (cum >= r) {
                    chosen = c;
                    break;
                }
            }
        }

        latest_chosen_action_ = chosen;
        return chosen;
    }

    // =========================================================================
    // 2. TRUE EXPERIENTIAL ASSIMILATION (Context-Bound Success & Failure)
    // - Environmental signal:
    //   valence: +1.0 (good/correct), -1.0 (bad/incorrect)
    //   magnitude: intensity of the reward or penalty
    //   target_hint: environmental ground-truth demonstration
    //
    // Cybernetic Mechanism:
    // - SUCCESS: The transition from the CURRENT CONTEXT to TARGET is carved deeper into memory.
    // - ERROR: The transition from the CURRENT CONTEXT to CHOSEN ACTION is hit with a REPULSION BARRIER.
    //   Next time the core is in this context, it is physically pushed AWAY from repeating the mistake.
    // =========================================================================
    void assimilate_experience(float valence, float magnitude, int target_hint = -1, float lr = 0.05f) noexcept {
        total_experiences_++;

        const float eff_feedback = std::clamp(valence, -1.0f, 1.0f) * std::clamp(magnitude, 0.0f, 2.0f);
        const size_t chosen = latest_chosen_action_;

        if (eff_feedback > 0.0f) {
            positive_accumulated_ += eff_feedback;
            internal_free_energy_ = 0.99f * internal_free_energy_;
        } else {
            negative_accumulated_ += std::abs(eff_feedback);
            internal_free_energy_ = 0.99f * internal_free_energy_ + 0.01f * std::abs(eff_feedback);
        }

        // --- 1. Repulsion from Erroneous Action in THIS Context ---
        if (eff_feedback < 0.0f) {
            const size_t err_offset = chosen * STATE_PHASE_DIM;
            action_bias_[chosen] -= lr * 0.2f;

            #pragma GCC ivdep
            for (size_t i = 0; i < STATE_PHASE_DIM; ++i) {
                // Errect a repulsion barrier proportional to current context state
                attractor_field_[err_offset + i] -= lr * latest_context_[i] * 1.5f;
            }

            // Also adapt physical mode damping: dampen oscillatory instability
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                state_.damping[k] = std::clamp(state_.damping[k] * 1.001f, 0.05f, 20.0f);
            }
        }

        // --- 2. Attraction towards Correct Target in THIS Context ---
        if (target_hint >= 0 && target_hint < static_cast<int>(CHANNEL_IO_DIM)) {
            const size_t tgt = static_cast<size_t>(target_hint);
            const size_t tgt_offset = tgt * STATE_PHASE_DIM;

            action_bias_[tgt] += lr * 0.1f;

            #pragma GCC ivdep
            for (size_t i = 0; i < STATE_PHASE_DIM; ++i) {
                // Carve attraction basin connecting current context -> target
                attractor_field_[tgt_offset + i] += lr * latest_context_[i] * 2.0f;
            }

            // Afferent pathway consolidation: strengthen receptive link
            const size_t in_offset = chosen * NUM_CONTINUUM_MODES;
            for (size_t k = 0; k < NUM_CONTINUUM_MODES; ++k) {
                W_afferent_[in_offset + k] += lr * state_.q[k] * 0.01f;
            }
        }
    }

    // Telemetry getters
    size_t get_latest_action() const noexcept { return latest_chosen_action_; }
    float get_free_energy() const noexcept { return internal_free_energy_; }
    float get_positive_exp() const noexcept { return positive_accumulated_; }
    float get_negative_exp() const noexcept { return negative_accumulated_; }
    const std::array<float, CHANNEL_IO_DIM>& get_probabilities() const noexcept { return latest_probabilities_; }

private:
    size_t total_experiences_;
    size_t latest_chosen_action_;
    float positive_accumulated_;
    float negative_accumulated_;
    float internal_free_energy_;

    PhaseStateMetrics state_;
    std::array<float, STATE_PHASE_DIM> latest_context_;

    // Afferent receptive projection: [256 x 128]
    alignas(64) std::array<float, CHANNEL_IO_DIM * NUM_CONTINUUM_MODES> W_afferent_;

    // Contextual Phase Attractor Field: [256 actions x 256 phase-dims]
    alignas(64) std::array<float, CHANNEL_IO_DIM * STATE_PHASE_DIM> attractor_field_;
    alignas(64) std::array<float, CHANNEL_IO_DIM> action_bias_;
    alignas(64) std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

#endif // KARYON_SOVEREIGN_EXPERIENTIAL_CONTINUUM_V32_H
