// ============================================================================
// Karyon-CoRE v34.0 Sovereign Dynamic Memory & Adaptive Flow Refractory Continuum
// Standard: KEP v28.0 (Principles 1, 2, 4, 5, 6, 8, 27, 28, 29, 30)
//
// Cybernetic Axioms:
// 1. Axiom of Unshackled Flow & Anti-Stupor Dynamics (Principle 8, 29):
//    Every action/character channel possesses an ENDOGENOUS REFRACTORY FATIGUE
//    variable r_c in [0, 1]. When a mode/character discharges, its refractory
//    potential rises. Karyon dynamically tunes its own refractory recovery rate
//    and fatigue penalty so it never freezes into static crystal loops ('vvvvvv').
// 2. Continuous Multi-Scale Fast-Slow Split:
//    Modes are split into fast-acting rhythm/transition modes (rapid response)
//    and slow macroscopic contextual modes (framing). Fast modes guarantee
//    that every newly emitted token immediately alters the phase state, breaking
//    the cycle of inertia.
// 3. Thermodynamic Energy Scaling (1 / sqrt(D_phase)):
//    Attractor resonance is scaled by inverse square-root of active phase dimension,
//    ensuring identical kinetic temperature and plasticity regardless of whether
//    Karyon sprouts to 64, 512, or 2048 modes.
// 4. Autonomous Refractory Plasticity:
//    If Karyon detects repetitive oscillations or stagnant free energy, it
//    autonomously strengthens refractory inhibition, driving the wavefield
//    into fresh forward-traveling trajectories.
// ============================================================================

#ifndef KARYON_SOVEREIGN_DYNAMIC_MEMORY_CONTINUUM_V34_H
#define KARYON_SOVEREIGN_DYNAMIC_MEMORY_CONTINUUM_V34_H

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

        // Sovereign Refractory Parameters (Fully Self-Tuned by Karyon)
        refractory_strength_ = 1.2f;       // Sovereign fatigue penalty
        refractory_recovery_rate_ = 0.70f;  // Sovereign recovery dissipation
        repetitive_stagnation_counter_ = 0;

        action_refractory_fatigue_.fill(0.0f);
        action_bias_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        rng_.seed(3400);
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
            // Log-spaced timescales: FAST local modes (0.35-0.75) and SLOW semantic modes (0.85-0.98)
            float frac = static_cast<float>(k) / static_cast<float>(num_modes_);
            if (k < num_modes_ / 4) {
                decay_horizon_[k] = 0.35f + 0.35f * (static_cast<float>(k) / (num_modes_ / 4.0f)); // 0.35 ... 0.70 (FAST)
            } else {
                decay_horizon_[k] = 0.75f + 0.23f * frac; // 0.75 ... 0.98 (SLOW)
            }
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
    }

    // =========================================================================
    // 1. AUTONOMOUS PERCEPTION & DYNAMIC CONTEXTUAL INTEGRATION WITH REFRACTORY
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

        // --- 2. Update Continuous Phase Context using Endogenous Timescales ---
        for (size_t k = 0; k < num_modes_; ++k) {
            const float lambda_q = decay_horizon_[k];
            const float lambda_p = std::clamp(lambda_q * 0.85f, 0.20f, 0.95f);

            context_trace_[k] = lambda_q * context_trace_[k] + (1.0f - lambda_q) * q_[k];
            context_trace_[num_modes_ + k] = lambda_p * context_trace_[num_modes_ + k] + (1.0f - lambda_p) * p_[k];
        }

        // Thermodynamic Normalization over Active Phase Dimension (Ensures constant temperature)
        float norm_sq = 0.0f;
        const size_t phase_dim = 2 * num_modes_;
        for (size_t i = 0; i < phase_dim; ++i) {
            norm_sq += context_trace_[i] * context_trace_[i];
        }
        const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-6f);

        for (size_t i = 0; i < phase_dim; ++i) {
            latest_normalized_context_[i] = context_trace_[i] * inv_norm;
        }

        // --- 3. Compute Resonance with Attractor Basins & Refractory Barrier ---
        std::array<float, CHANNEL_IO_DIM> energy{};
        float max_e = -1e9f;
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(phase_dim));

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float sum = action_bias_[c];
            const size_t c_offset = c * phase_dim;
            #pragma GCC ivdep
            for (size_t i = 0; i < phase_dim; ++i) {
                sum += attractor_field_[c_offset + i] * latest_normalized_context_[i];
            }
            // Scale and apply endogenous refractory inhibition
            float eff_energy = (sum * scale_factor) - (refractory_strength_ * action_refractory_fatigue_[c]);
            energy[c] = eff_energy;
            if (eff_energy > max_e) max_e = eff_energy;
        }

        // --- 4. Probabilistic Softmax or MAP Selection ---
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

        // --- 5. Update Refractory Fatigue Field ---
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            action_refractory_fatigue_[c] *= refractory_recovery_rate_;
        }
        // Discharged channel accumulates refractory resistance
        action_refractory_fatigue_[chosen] += 1.0f;

        // Self-detect repetitive stagnation
        if (chosen == latest_chosen_action_) {
            repetitive_stagnation_counter_++;
            if (repetitive_stagnation_counter_ > 3) {
                // Autonomous self-tuning: strengthen refractory barrier to force escape!
                refractory_strength_ = std::min(refractory_strength_ + 0.15f, 4.0f);
            }
        } else {
            repetitive_stagnation_counter_ = 0;
            // Relax refractory strength toward sovereign baseline
            refractory_strength_ = 0.99f * refractory_strength_ + 0.01f * 1.2f;
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
            // Successful flow relaxes refractory recovery
            refractory_recovery_rate_ = std::clamp(0.999f * refractory_recovery_rate_ + 0.001f * 0.70f, 0.50f, 0.90f);
        } else {
            negative_accumulated_ += std::abs(eff_feedback);
            internal_free_energy_ = 0.99f * internal_free_energy_ + 0.01f * std::abs(eff_feedback);
            consecutive_errors_++;
            // Errors and friction accelerate refractory clearing
            refractory_recovery_rate_ = std::clamp(0.99f * refractory_recovery_rate_ + 0.01f * 0.60f, 0.40f, 0.85f);
        }

        // --- 1. Repulsion from Erroneous Action in THIS Context ---
        if (eff_feedback < 0.0f) {
            const size_t err_offset = chosen * phase_dim;
            action_bias_[chosen] -= lr * 0.2f;

            #pragma GCC ivdep
            for (size_t i = 0; i < phase_dim; ++i) {
                attractor_field_[err_offset + i] -= lr * latest_normalized_context_[i] * 1.5f;
            }

            // Endogenous horizon tuning: selectively widen slow modes and sharpen fast modes
            for (size_t k = 0; k < num_modes_; ++k) {
                if (k >= num_modes_ / 4) {
                    // Slow modes stretch deeper
                    decay_horizon_[k] = std::clamp(decay_horizon_[k] + lr * 0.005f, 0.60f, 0.995f);
                } else {
                    // Fast modes stay nimble for local transitions
                    decay_horizon_[k] = std::clamp(decay_horizon_[k] - lr * 0.005f, 0.25f, 0.70f);
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
        if (consecutive_errors_ > 40 && internal_free_energy_ > 0.85f && num_modes_ < MAX_ALLOWABLE_MODES) {
            sprout_memory_capacity(16);
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

        num_modes_ = new_modes;

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
            // Newly sprouted modes contribute to slow semantic grounding
            float frac = static_cast<float>(k) / static_cast<float>(new_modes);
            decay_horizon_[k] = 0.85f + 0.14f * frac;
        }

        context_trace_.resize(2 * new_modes, 0.0f);
        latest_normalized_context_.resize(2 * new_modes, 0.0f);

        // Re-stride Afferent Ingestion Tensor
        std::vector<float> new_W_afferent(CHANNEL_IO_DIM * new_modes, 0.0f);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t k = 0; k < old_modes; ++k) {
                new_W_afferent[c * new_modes + k] = W_afferent_[c * old_modes + k];
            }
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_W_afferent[c * new_modes + k] = tiny_init(rng_);
            }
        }
        W_afferent_ = std::move(new_W_afferent);

        // Re-stride Attractor Field
        const size_t old_phase_dim = 2 * old_modes;
        const size_t new_phase_dim = 2 * new_modes;
        std::vector<float> new_attractors(CHANNEL_IO_DIM * new_phase_dim, 0.0f);

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t k = 0; k < old_modes; ++k) {
                new_attractors[c * new_phase_dim + k] = attractor_field_[c * old_phase_dim + k];
            }
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_attractors[c * new_phase_dim + k] = tiny_init(rng_);
            }
            for (size_t k = 0; k < old_modes; ++k) {
                new_attractors[c * new_phase_dim + new_modes + k] = attractor_field_[c * old_phase_dim + old_modes + k];
            }
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_attractors[c * new_phase_dim + new_modes + k] = tiny_init(rng_);
            }
        }
        attractor_field_ = std::move(new_attractors);

        std::cout << "  >>> [Sovereign Autopoiesis] Karyon sprouted memory: "
                  << old_modes << " -> " << new_modes << " modes (Phase dim: " << new_phase_dim << ") <<<\n";
    }

    // Telemetry getters
    size_t get_active_modes() const noexcept { return num_modes_; }
    size_t get_phase_dim() const noexcept { return 2 * num_modes_; }
    float get_free_energy() const noexcept { return internal_free_energy_; }
    float get_positive_exp() const noexcept { return positive_accumulated_; }
    float get_negative_exp() const noexcept { return negative_accumulated_; }
    float get_refractory_strength() const noexcept { return refractory_strength_; }
    float get_refractory_recovery() const noexcept { return refractory_recovery_rate_; }
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
    size_t repetitive_stagnation_counter_;

    float positive_accumulated_;
    float negative_accumulated_;
    float internal_free_energy_;

    // Sovereign Refractory Flow Parameters
    float refractory_strength_;
    float refractory_recovery_rate_;

    std::mt19937 rng_;

    std::vector<float> q_;
    std::vector<float> p_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<float> decay_horizon_;
    std::vector<float> causal_credit_;

    std::vector<float> context_trace_;
    std::vector<float> latest_normalized_context_;

    std::vector<float> W_afferent_;
    std::vector<float> attractor_field_;
    std::array<float, CHANNEL_IO_DIM> action_refractory_fatigue_;
    std::array<float, CHANNEL_IO_DIM> action_bias_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

#endif // KARYON_SOVEREIGN_DYNAMIC_MEMORY_CONTINUUM_V34_H
