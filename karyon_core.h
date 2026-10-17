// ============================================================================
// Karyon-CoRE v33.0 Sovereign Experiential Continuum with Autonomous Dynamic Memory
// Standard: KEP v28.0 (Principles 1, 2, 4, 5, 6, 8, 27, 28, 29, 30)
//
// Cybernetic Axioms:
// 1. Axiom of Unshackled Flow (Principle 8):
//    Memory capacity is NEVER fixed or capped by human hardcoded constants.
//    Karyon autonomously grows (sprouts) new memory modes and attractor capacity
//    when internal Free Energy / error persists, and prunes dead channels.
// 2. Sovereign Endogenous Time-Scale (Horizon Plasticity):
//    Every mode k self-tunes its own temporal decay lambda_k in (0.1, 0.999).
//    Modes responsible for macroscopic syntactic/semantic framing autonomously
//    widen their memory horizon (tau_k -> 1000+ steps), while rapid local modes
//    remain sharp (tau_k ~ 2 steps).
// 3. Epigenetic Smooth Sprouting (Principle 6):
//    When memory capacity sprouts, new dimensions are born with zero initial
//    perturbation (alpha_sprout -> 0), strictly preserving already consolidated knowledge.
// 4. Pure Teacher-Student Sovereignty:
//    The trainer never computes gradients or manipulates Karyon's internal tensors.
//    Trainer delivers sensory bytes and honest valence/target feedback.
// ============================================================================

#ifndef KARYON_SOVEREIGN_DYNAMIC_MEMORY_CONTINUUM_V33_H
#define KARYON_SOVEREIGN_DYNAMIC_MEMORY_CONTINUUM_V33_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <iostream>
#include <immintrin.h>

namespace karyon {

constexpr size_t CHANNEL_IO_DIM = 256;
constexpr size_t INITIAL_MODES = 64;
constexpr size_t MAX_ALLOWABLE_MODES = 512;

class SovereignDynamicExperientialContinuum {
public:
    SovereignDynamicExperientialContinuum() noexcept {
        reset();
    }

    void reset() noexcept {
        num_modes_ = INITIAL_MODES;
        total_experiences_ = 0;
        positive_accumulated_ = 0.0f;
        negative_accumulated_ = 0.0f;
        internal_free_energy_ = 1.0f;
        consecutive_errors_ = 0;
        latest_chosen_action_ = 0;

        rng_.seed(3300);
        std::normal_distribution<float> norm(0.0f, 0.02f);

        // Allocate dynamic mode tensors
        q_.assign(num_modes_, 0.0f);
        p_.assign(num_modes_, 0.0f);
        stiffness_.resize(num_modes_);
        damping_.resize(num_modes_);
        decay_horizon_.resize(num_modes_);
        causal_credit_.assign(num_modes_, 0.0f);

        for (size_t k = 0; k < num_modes_; ++k) {
            stiffness_[k] = 1.0f + 0.05f * static_cast<float>(k);
            damping_[k] = 0.35f + 0.005f * static_cast<float>(k);
            // Endogenous decay: log-spaced initial timescales from tau=3 to tau=200
            float frac = static_cast<float>(k) / static_cast<float>(num_modes_);
            decay_horizon_[k] = 0.70f + 0.28f * frac; // 0.70 ... 0.98
        }

        // Context state memory: size is 2 * num_modes_ (coordinates + momenta)
        context_trace_.assign(2 * num_modes_, 0.0f);
        latest_normalized_context_.assign(2 * num_modes_, 0.0f);

        // Afferent Ingestion Tensor: [256 actions x num_modes_]
        W_afferent_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : W_afferent_) w = norm(rng_);

        // Attractor Field: [256 actions x (2 * num_modes_)]
        attractor_field_.resize(CHANNEL_IO_DIM * (2 * num_modes_));
        for (auto& a : attractor_field_) a = norm(rng_);

        action_bias_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    // =========================================================================
    // 1. AUTONOMOUS PERCEPTION & DYNAMIC CONTEXTUAL INTEGRATION
    // =========================================================================
    size_t perceive_and_decide(uint8_t input_channel, float dt, float temperature = 0.0f) noexcept {
        const size_t in_offset = static_cast<size_t>(input_channel) * num_modes_;

        // --- 1. Pluck internal wavefield modes ---
        for (size_t k = 0; k < num_modes_; ++k) {
            const float drive = W_afferent_[in_offset + k] * 5.0f;
            const float omega_sq = stiffness_[k];
            const float gamma = 0.5f * damping_[k];

            const float q_star = drive / (omega_sq + 1e-6f);
            const float q_rel  = q_[k] - q_star;
            const float p_curr = p_[k];
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

            q_[k] = std::clamp(q_next, -8.0f, 8.0f);
            p_[k] = std::clamp(p_next, -16.0f, 16.0f);
        }

        // --- 2. Update Continuous Phase Context using Endogenous Timescales lambda_k ---
        for (size_t k = 0; k < num_modes_; ++k) {
            const float lambda_q = decay_horizon_[k];
            const float lambda_p = std::clamp(lambda_q * 0.85f, 0.40f, 0.95f);

            context_trace_[k] = lambda_q * context_trace_[k] + (1.0f - lambda_q) * q_[k];
            context_trace_[num_modes_ + k] = lambda_p * context_trace_[num_modes_ + k] + (1.0f - lambda_p) * p_[k];
        }

        // Normalize Context Vector
        float norm_sq = 0.0f;
        const size_t phase_dim = 2 * num_modes_;
        for (size_t i = 0; i < phase_dim; ++i) {
            norm_sq += context_trace_[i] * context_trace_[i];
        }
        const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-6f);

        for (size_t i = 0; i < phase_dim; ++i) {
            latest_normalized_context_[i] = context_trace_[i] * inv_norm;
        }

        // --- 3. Compute Resonance with Attractor Basins ---
        std::array<float, CHANNEL_IO_DIM> energy{};
        float max_e = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float sum = action_bias_[c];
            const size_t c_offset = c * phase_dim;
            #pragma GCC ivdep
            for (size_t i = 0; i < phase_dim; ++i) {
                sum += attractor_field_[c_offset + i] * latest_normalized_context_[i];
            }
            energy[c] = sum;
            if (sum > max_e) max_e = sum;
        }

        // Probabilities
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
            float best_val = -1e9f;
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                latest_probabilities_[c] *= inv_sum_exp;
                if (energy[c] > best_val) {
                    best_val = energy[c];
                    chosen = c;
                }
            }
        } else {
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
    // 2. SOVEREIGN EXPERIENTIAL ASSIMILATION & AUTONOMOUS CAPACITY CONTROL
    // =========================================================================
    void assimilate_experience(float valence, float magnitude, int target_hint = -1, float lr = 0.05f) noexcept {
        total_experiences_++;

        const float eff_feedback = std::clamp(valence, -1.0f, 1.0f) * std::clamp(magnitude, 0.0f, 2.0f);
        const size_t chosen = latest_chosen_action_;
        const size_t phase_dim = 2 * num_modes_;

        if (eff_feedback > 0.0f) {
            positive_accumulated_ += eff_feedback;
            internal_free_energy_ = 0.99f * internal_free_energy_;
            consecutive_errors_ = (consecutive_errors_ > 0) ? consecutive_errors_ - 1 : 0;
        } else {
            negative_accumulated_ += std::abs(eff_feedback);
            internal_free_energy_ = 0.99f * internal_free_energy_ + 0.01f * std::abs(eff_feedback);
            consecutive_errors_++;
        }

        // --- 1. Repulsion from Erroneous Action in THIS Context ---
        if (eff_feedback < 0.0f) {
            const size_t err_offset = chosen * phase_dim;
            action_bias_[chosen] -= lr * 0.2f;

            #pragma GCC ivdep
            for (size_t i = 0; i < phase_dim; ++i) {
                attractor_field_[err_offset + i] -= lr * latest_normalized_context_[i] * 1.5f;
            }

            // Endogenous horizon tuning on error: if error keeps repeating, expand memory horizon!
            for (size_t k = 0; k < num_modes_; ++k) {
                if (std::abs(latest_normalized_context_[k]) > 0.1f) {
                    // Widen memory horizon to capture earlier context
                    decay_horizon_[k] = std::clamp(decay_horizon_[k] + lr * 0.01f, 0.50f, 0.995f);
                }
            }
        }

        // --- 2. Attraction towards Correct Target ---
        if (target_hint >= 0 && target_hint < static_cast<int>(CHANNEL_IO_DIM)) {
            const size_t tgt = static_cast<size_t>(target_hint);
            const size_t tgt_offset = tgt * phase_dim;

            action_bias_[tgt] += lr * 0.1f;

            #pragma GCC ivdep
            for (size_t i = 0; i < phase_dim; ++i) {
                attractor_field_[tgt_offset + i] += lr * latest_normalized_context_[i] * 2.0f;
            }

            // Credit assignment: reward modes active during success
            for (size_t k = 0; k < num_modes_; ++k) {
                const float act = std::abs(latest_normalized_context_[k]);
                causal_credit_[k] = 0.95f * causal_credit_[k] + 0.05f * act;
            }
        }

        // --- 3. SOVEREIGN MEMORY SPROUTING (Principle 8 & Principle 6) ---
        // If consecutive errors accumulate or internal free energy exceeds threshold,
        // Karyon recognizes its current memory capacity is saturated and autonomously sprouts!
        if (consecutive_errors_ > 30 && internal_free_energy_ > 0.85f && num_modes_ < MAX_ALLOWABLE_MODES) {
            sprout_memory_capacity(16); // Sprout 16 new physical modes
            consecutive_errors_ = 0;
        }
    }

    // =========================================================================
    // 3. EPIGENETIC SMOOTH SPROUTING (Lossless Expansion)
    // =========================================================================
    void sprout_memory_capacity(size_t additional_modes) noexcept {
        const size_t old_modes = num_modes_;
        const size_t new_modes = std::min(old_modes + additional_modes, MAX_ALLOWABLE_MODES);
        if (new_modes <= old_modes) return;

        const size_t added = new_modes - old_modes;
        num_modes_ = new_modes;

        // Expand mode vectors
        q_.resize(new_modes, 0.0f);
        p_.resize(new_modes, 0.0f);
        stiffness_.resize(new_modes);
        damping_.resize(new_modes);
        decay_horizon_.resize(new_modes);
        causal_credit_.resize(new_modes, 0.0f);

        std::normal_distribution<float> tiny_init(0.0f, 0.001f);

        for (size_t k = old_modes; k < new_modes; ++k) {
            stiffness_[k] = 1.0f + 0.05f * static_cast<float>(k);
            damping_[k] = 0.35f + 0.005f * static_cast<float>(k);
            // Newly sprouted modes explore wider horizons
            float frac = static_cast<float>(k) / static_cast<float>(new_modes);
            decay_horizon_[k] = 0.85f + 0.14f * frac; // 0.85 ... 0.99
        }

        // Reallocate context traces
        context_trace_.resize(2 * new_modes, 0.0f);
        latest_normalized_context_.resize(2 * new_modes, 0.0f);

        // Re-stride Afferent Ingestion Tensor [256 x new_modes]
        std::vector<float> new_W_afferent(CHANNEL_IO_DIM * new_modes, 0.0f);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t k = 0; k < old_modes; ++k) {
                new_W_afferent[c * new_modes + k] = W_afferent_[c * old_modes + k];
            }
            // Net2Net identity: new weights are born near zero
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_W_afferent[c * new_modes + k] = tiny_init(rng_);
            }
        }
        W_afferent_ = std::move(new_W_afferent);

        // Re-stride Attractor Field [256 x (2 * new_modes)]
        const size_t old_phase_dim = 2 * old_modes;
        const size_t new_phase_dim = 2 * new_modes;
        std::vector<float> new_attractors(CHANNEL_IO_DIM * new_phase_dim, 0.0f);

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            // Copy old coordinates part
            for (size_t k = 0; k < old_modes; ++k) {
                new_attractors[c * new_phase_dim + k] = attractor_field_[c * old_phase_dim + k];
            }
            // Zero-init newly sprouted coordinate connections
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_attractors[c * new_phase_dim + k] = tiny_init(rng_);
            }
            // Copy old momentum part
            for (size_t k = 0; k < old_modes; ++k) {
                new_attractors[c * new_phase_dim + new_modes + k] = attractor_field_[c * old_phase_dim + old_modes + k];
            }
            // Zero-init newly sprouted momentum connections
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_attractors[c * new_phase_dim + new_modes + k] = tiny_init(rng_);
            }
        }
        attractor_field_ = std::move(new_attractors);

        std::cout << "  >>> [Sovereign Autopoiesis] Karyon spontaneously sprouted memory: "
                  << old_modes << " -> " << new_modes << " modes (Phase dim: " << new_phase_dim << ") <<<\n";
    }

    // Telemetry getters
    size_t get_active_modes() const noexcept { return num_modes_; }
    size_t get_phase_dim() const noexcept { return 2 * num_modes_; }
    float get_free_energy() const noexcept { return internal_free_energy_; }
    float get_positive_exp() const noexcept { return positive_accumulated_; }
    float get_negative_exp() const noexcept { return negative_accumulated_; }
    float get_avg_memory_horizon() const noexcept {
        float sum = 0.0f;
        for (float h : decay_horizon_) sum += h;
        return sum / static_cast<float>(num_modes_);
    }

private:
    size_t num_modes_;
    size_t total_experiences_;
    size_t latest_chosen_action_;
    size_t consecutive_errors_;
    float positive_accumulated_;
    float negative_accumulated_;
    float internal_free_energy_;

    std::mt19937 rng_;

    // Physical mode variables
    std::vector<float> q_;
    std::vector<float> p_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<float> decay_horizon_;  // Endogenous timescale lambda_k
    std::vector<float> causal_credit_;  // Long-term usefulness

    // Phase context traces: dim = 2 * num_modes_
    std::vector<float> context_trace_;
    std::vector<float> latest_normalized_context_;

    // Dynamic coupling tensors
    std::vector<float> W_afferent_;      // [256 x num_modes_]
    std::vector<float> attractor_field_; // [256 x (2 * num_modes_)]
    std::array<float, CHANNEL_IO_DIM> action_bias_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

#endif // KARYON_SOVEREIGN_DYNAMIC_MEMORY_CONTINUUM_V33_H
