// ============================================================================
// Karyon-CoRE v38.0 Sovereign Full Plasticity & Endogenous Memory Scaling
// Standard: KEP v30.0 (Principles 1, 2, 4, 5, 6, 8, 27, 28, 29, 30)
//
// Complete Substrate Plasticity & Endogenous Control:
// 1. Plastic Afferent Projection W_afferent_: Sensitizes wavefield modes to inputs.
// 2. Plastic Resonant Coupling J_ij: Dynamic topological cross-talk chords.
// 3. Plastic Stiffness & Damping (stiffness_k, damping_k): Self-tuning natural frequencies.
// 4. Endogenous Phase Memory Scaling (hippocampal_scaling_gain_): Self-synthesized
//    precision gain governing resonance recall (KEP Principle 30).
// ============================================================================

#ifndef KARYON_SOVEREIGN_ENDOGENOUS_MEMORY_ORGANISM_V38_H
#define KARYON_SOVEREIGN_ENDOGENOUS_MEMORY_ORGANISM_V38_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <iostream>
#include <deque>

namespace karyon {

constexpr size_t CHANNEL_IO_DIM = 256;
constexpr size_t INITIAL_MODES = 64;
constexpr size_t MAX_ALLOWABLE_MODES = 512;
constexpr size_t PHASE_MEMORY_CAPACITY = 512;

struct FullPlasticityTelemetry {
    size_t active_modes;
    size_t phase_dim;
    float free_energy;
    float dt_eff;
    float refractory_strength;
    float refractory_recovery;
    float avg_stiffness;
    float avg_damping;
    float coupling_density;
    float mean_coupling_weight;
    float hippocampal_gain;
    size_t stored_phase_traces;
    float phase_retrieval_resonance;
    size_t rollout_depth;
};

struct PhaseMemoryItem {
    std::vector<float> phase_vector;
    uint8_t associated_action;
    float surprise;
};

class SovereignFullPlasticityOrganism {
public:
    SovereignFullPlasticityOrganism() noexcept {
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
        latest_retrieval_resonance_ = 0.0f;

        dt_eff_ = 0.010f;
        refractory_strength_ = 1.20f;
        refractory_recovery_rate_ = 0.70f;
        metric_curvature_ = 1.00f;
        rollout_depth_ = 0;
        coupling_density_ = 0.05f;
        hippocampal_scaling_gain_ = 4.0f; // Endogenous parameter

        action_refractory_fatigue_.fill(0.0f);
        action_bias_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        rng_.seed(5630);
        std::normal_distribution<float> norm(0.0f, 0.02f);

        q_.assign(num_modes_, 0.0f);
        p_.assign(num_modes_, 0.0f);
        stiffness_.resize(num_modes_);
        damping_.resize(num_modes_);
        decay_horizon_.resize(num_modes_);
        causal_credit_.assign(num_modes_, 0.0f);

        for (size_t k = 0; k < num_modes_; ++k) {
            stiffness_[k] = 1.0f + 0.05f * static_cast<float>(k);
            damping_[k] = 0.35f + 0.005f * static_cast<float>(k);
            float frac = static_cast<float>(k) / static_cast<float>(num_modes_);
            decay_horizon_[k] = (k < num_modes_ / 4) ? (0.35f + 0.35f * (k / (num_modes_ / 4.0f)))
                                                     : (0.75f + 0.23f * frac);
        }

        context_trace_.assign(2 * num_modes_, 0.0f);
        latest_normalized_context_.assign(2 * num_modes_, 0.0f);

        J_coupling_.assign(num_modes_ * num_modes_, 0.0f);
        for (size_t i = 0; i < num_modes_; ++i) {
            for (size_t j = 0; j < num_modes_; ++j) {
                if (i != j) J_coupling_[i * num_modes_ + j] = norm(rng_) * 0.05f;
            }
        }

        phase_memory_buffer_.clear();

        plasticity_gen_.resize((2 * num_modes_) * 8);
        for (auto& g : plasticity_gen_) g = norm(rng_);

        W_afferent_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : W_afferent_) w = norm(rng_);

        attractor_field_.resize(CHANNEL_IO_DIM * (2 * num_modes_));
        for (auto& a : attractor_field_) a = norm(rng_);
    }

    void synthesize_internal_laws() noexcept {
        const size_t phase_dim = 2 * num_modes_;

        std::array<float, 8> plast_out{};
        for (size_t col = 0; col < 8; ++col) {
            float sum = 0.0f;
            for (size_t i = 0; i < phase_dim; ++i) {
                sum += plasticity_gen_[i * 8 + col] * latest_normalized_context_[i];
            }
            plast_out[col] = sum;
        }

        float sig0 = 1.0f / (1.0f + std::exp(-plast_out[0]));
        dt_eff_ = 0.002f + 0.028f * sig0;

        float sig1 = 1.0f / (1.0f + std::exp(-plast_out[1]));
        refractory_strength_ = 0.50f + 3.00f * sig1;

        float sig2 = 1.0f / (1.0f + std::exp(-plast_out[2]));
        refractory_recovery_rate_ = 0.30f + 0.60f * sig2;

        float sig3 = 1.0f / (1.0f + std::exp(-plast_out[3]));
        metric_curvature_ = 0.20f + 2.00f * sig3;

        float sig4 = 1.0f / (1.0f + std::exp(-plast_out[4]));
        rollout_depth_ = static_cast<size_t>(std::floor(4.0f * sig4));

        float sig5 = 1.0f / (1.0f + std::exp(-plast_out[5]));
        coupling_density_ = 0.01f + 0.20f * sig5;

        // ENDOGENOUS MEMORY SCALING GAIN: Synthesized directly from internal phase state!
        float sig7 = 1.0f / (1.0f + std::exp(-plast_out[7]));
        hippocampal_scaling_gain_ = 1.0f + 15.0f * sig7; // Dynamic gain in [1.0, 16.0]

        for (size_t k = 0; k < num_modes_; ++k) {
            float act = std::abs(latest_normalized_context_[k]);
            float shift = 0.01f * std::tanh(plast_out[6] * act);
            decay_horizon_[k] = std::clamp(decay_horizon_[k] + shift, 0.20f, 0.995f);
        }
    }

    size_t perceive_and_decide(uint8_t input_channel, float external_dt, float temperature = 0.0f) noexcept {
        synthesize_internal_laws();

        const float dt = dt_eff_;
        const size_t in_offset = static_cast<size_t>(input_channel) * num_modes_;

        // --- Step 1: Wavefield Mode Update ---
        for (size_t k = 0; k < num_modes_; ++k) {
            float coupling_sum = 0.0f;
            const size_t k_row = k * num_modes_;
            for (size_t j = 0; j < num_modes_; ++j) {
                coupling_sum += J_coupling_[k_row + j] * q_[j];
            }

            const float drive = W_afferent_[in_offset + k] * 5.0f + coupling_density_ * coupling_sum;
            const float omega_sq = stiffness_[k] * metric_curvature_;
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

        // --- Step 2: Phase Context Update & Normalization ---
        for (size_t k = 0; k < num_modes_; ++k) {
            const float lambda_q = decay_horizon_[k];
            const float lambda_p = std::clamp(lambda_q * 0.85f, 0.20f, 0.95f);

            context_trace_[k] = lambda_q * context_trace_[k] + (1.0f - lambda_q) * q_[k];
            context_trace_[num_modes_ + k] = lambda_p * context_trace_[num_modes_ + k] + (1.0f - lambda_p) * p_[k];
        }

        float norm_sq = 0.0f;
        const size_t phase_dim = 2 * num_modes_;
        for (size_t i = 0; i < phase_dim; ++i) {
            norm_sq += context_trace_[i] * context_trace_[i];
        }
        const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-6f);

        for (size_t i = 0; i < phase_dim; ++i) {
            latest_normalized_context_[i] = context_trace_[i] * inv_norm;
        }

        // --- Step 3: Endogenously Governed Phase Memory Retrieval ---
        float memory_boost[CHANNEL_IO_DIM] = {0.0f};
        latest_retrieval_resonance_ = 0.0f;

        if (!phase_memory_buffer_.empty()) {
            float max_resonance = -1.0f;
            int best_action = -1;

            size_t check_count = std::min(phase_memory_buffer_.size(), static_cast<size_t>(64));
            for (size_t m = 0; m < check_count; ++m) {
                const auto& item = phase_memory_buffer_[phase_memory_buffer_.size() - 1 - m];
                if (item.phase_vector.size() == phase_dim) {
                    float dot = 0.0f;
                    for (size_t i = 0; i < phase_dim; ++i) {
                        dot += item.phase_vector[i] * latest_normalized_context_[i];
                    }
                    if (dot > max_resonance) {
                        max_resonance = dot;
                        best_action = item.associated_action;
                    }
                }
            }

            if (max_resonance > 0.60f && best_action >= 0) {
                // Hippocampal boost governed by ENDOGENOUS hippocampal_scaling_gain_!
                memory_boost[best_action] = (max_resonance - 0.60f) * hippocampal_scaling_gain_;
                latest_retrieval_resonance_ = max_resonance;
            }
        }

        // --- Step 4: Mental Sandbox Rollout ---
        std::vector<float> rollout_context = latest_normalized_context_;
        if (rollout_depth_ > 0 && temperature > 0.05f) {
            for (size_t sub = 0; sub < rollout_depth_; ++sub) {
                for (size_t k = 0; k < num_modes_; ++k) {
                    rollout_context[k] = 0.90f * rollout_context[k] + 0.10f * std::tanh(q_[k]);
                }
            }
        }

        // --- Step 5: Compute Energy ---
        std::array<float, CHANNEL_IO_DIM> energy{};
        float max_e = -1e9f;
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(phase_dim));

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float sum = action_bias_[c];
            const size_t c_offset = c * phase_dim;
            for (size_t i = 0; i < phase_dim; ++i) {
                sum += attractor_field_[c_offset + i] * rollout_context[i];
            }
            float eff_energy = (sum * scale_factor) + memory_boost[c] - (refractory_strength_ * action_refractory_fatigue_[c]);
            energy[c] = eff_energy;
            if (eff_energy > max_e) max_e = eff_energy;
        }

        // --- Step 6: Decision ---
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

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            action_refractory_fatigue_[c] *= refractory_recovery_rate_;
        }
        action_refractory_fatigue_[chosen] += 1.0f;

        latest_chosen_action_ = chosen;
        return chosen;
    }

    void assimilate_experience(uint8_t input_channel, float valence, float magnitude, int target_hint = -1, float lr = 0.05f) noexcept {
        total_experiences_++;

        const float eff_feedback = std::clamp(valence, -1.0f, 1.0f) * std::clamp(magnitude, 0.0f, 2.0f);
        const size_t chosen = latest_chosen_action_;
        const size_t phase_dim = 2 * num_modes_;
        const size_t in_offset = static_cast<size_t>(input_channel) * num_modes_;

        if (eff_feedback > 0.0f) {
            positive_accumulated_ += eff_feedback;
            internal_free_energy_ = 0.99f * internal_free_energy_;
            consecutive_errors_ = (consecutive_errors_ > 0) ? consecutive_errors_ - 1 : 0;
        } else {
            negative_accumulated_ += std::abs(eff_feedback);
            internal_free_energy_ = 0.99f * internal_free_energy_ + 0.01f * std::abs(eff_feedback);
            consecutive_errors_++;
        }

        // 1. Plastic Afferent Ingestion W_afferent_
        for (size_t k = 0; k < num_modes_; ++k) {
            float grad_w = eff_feedback * q_[k];
            W_afferent_[in_offset + k] += lr * 0.1f * grad_w;
        }

        // 2. Plastic Resonant Coupling J_ij
        for (size_t i = 0; i < num_modes_; ++i) {
            for (size_t j = 0; j < num_modes_; ++j) {
                if (i != j) {
                    float co_act = q_[i] * q_[j];
                    J_coupling_[i * num_modes_ + j] += lr * 0.01f * eff_feedback * co_act;
                    J_coupling_[i * num_modes_ + j] = std::clamp(J_coupling_[i * num_modes_ + j], -0.5f, 0.5f);
                }
            }
        }

        // 3. Plastic Stiffness & Damping
        for (size_t k = 0; k < num_modes_; ++k) {
            float energy_k = q_[k] * q_[k] + p_[k] * p_[k];
            stiffness_[k] += lr * 0.001f * eff_feedback * energy_k;
            stiffness_[k] = std::clamp(stiffness_[k], 0.2f, 5.0f);

            damping_[k] -= lr * 0.001f * eff_feedback * std::abs(p_[k]);
            damping_[k] = std::clamp(damping_[k], 0.05f, 2.0f);
        }

        // 4. Phase Memory Storage
        if (std::abs(eff_feedback) >= 0.8f && target_hint >= 0) {
            PhaseMemoryItem item;
            item.phase_vector = latest_normalized_context_;
            item.associated_action = static_cast<uint8_t>(target_hint);
            item.surprise = std::abs(eff_feedback);

            if (phase_memory_buffer_.size() >= PHASE_MEMORY_CAPACITY) {
                phase_memory_buffer_.pop_front();
            }
            phase_memory_buffer_.push_back(std::move(item));
        }

        // 5. Plasticity Generator Gradient
        for (size_t col = 0; col < 8; ++col) {
            for (size_t i = 0; i < phase_dim; ++i) {
                float grad = eff_feedback * latest_normalized_context_[i];
                plasticity_gen_[i * 8 + col] += lr * 0.01f * grad;
            }
        }

        // 6. Attractor Field Plasticity
        if (eff_feedback < 0.0f) {
            const size_t err_offset = chosen * phase_dim;
            action_bias_[chosen] -= lr * 0.2f;

            for (size_t i = 0; i < phase_dim; ++i) {
                attractor_field_[err_offset + i] -= lr * latest_normalized_context_[i] * 1.5f;
            }
        }

        if (target_hint >= 0 && target_hint < static_cast<int>(CHANNEL_IO_DIM)) {
            const size_t tgt = static_cast<size_t>(target_hint);
            const size_t tgt_offset = tgt * phase_dim;

            action_bias_[tgt] += lr * 0.1f;

            for (size_t i = 0; i < phase_dim; ++i) {
                attractor_field_[tgt_offset + i] += lr * latest_normalized_context_[i] * 2.0f;
            }
        }

        // Autopoietic Sprouting
        if (consecutive_errors_ > 35 && internal_free_energy_ > 0.80f && num_modes_ < MAX_ALLOWABLE_MODES) {
            sprout_memory_capacity(16);
            consecutive_errors_ = 0;
        }
    }

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
            float frac = static_cast<float>(k) / static_cast<float>(new_modes);
            decay_horizon_[k] = 0.85f + 0.14f * frac;
        }

        context_trace_.resize(2 * new_modes, 0.0f);
        latest_normalized_context_.resize(2 * new_modes, 0.0f);

        J_coupling_.resize(new_modes * new_modes, 0.0f);
        plasticity_gen_.resize((2 * new_modes) * 8, 0.0f);

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
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_attractors[c * new_phase_dim + new_modes + k] = attractor_field_[c * old_phase_dim + old_modes + k];
            }
            for (size_t k = old_modes; k < new_modes; ++k) {
                new_attractors[c * new_phase_dim + new_modes + k] = tiny_init(rng_);
            }
        }
        attractor_field_ = std::move(new_attractors);

        std::cout << "  >>> [Endogenous Autopoiesis] Karyon sprouted modes: "
                  << old_modes << " -> " << new_modes << " (Phase dim: " << new_phase_dim << ") <<<\n";
    }

    FullPlasticityTelemetry get_telemetry() const noexcept {
        float sum_stiff = 0.0f;
        float sum_damp = 0.0f;
        for (size_t k = 0; k < num_modes_; ++k) {
            sum_stiff += stiffness_[k];
            sum_damp += damping_[k];
        }

        float coupling_weight_sum = 0.0f;
        for (float j : J_coupling_) coupling_weight_sum += std::abs(j);

        return FullPlasticityTelemetry{
            num_modes_,
            2 * num_modes_,
            internal_free_energy_,
            dt_eff_,
            refractory_strength_,
            refractory_recovery_rate_,
            sum_stiff / static_cast<float>(num_modes_),
            sum_damp / static_cast<float>(num_modes_),
            coupling_density_,
            coupling_weight_sum / static_cast<float>(num_modes_ * num_modes_),
            hippocampal_scaling_gain_,
            phase_memory_buffer_.size(),
            latest_retrieval_resonance_,
            rollout_depth_
        };
    }

private:
    size_t num_modes_;
    size_t total_experiences_;
    size_t latest_chosen_action_;
    size_t consecutive_errors_;

    float positive_accumulated_;
    float negative_accumulated_;
    float internal_free_energy_;
    float latest_retrieval_resonance_;

    float dt_eff_;
    float refractory_strength_;
    float refractory_recovery_rate_;
    float metric_curvature_;
    size_t rollout_depth_;
    float coupling_density_;
    float hippocampal_scaling_gain_; // Endogenous memory gain

    std::mt19937 rng_;

    std::vector<float> q_;
    std::vector<float> p_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<float> decay_horizon_;
    std::vector<float> causal_credit_;

    std::vector<float> context_trace_;
    std::vector<float> latest_normalized_context_;

    std::vector<float> J_coupling_;
    std::deque<PhaseMemoryItem> phase_memory_buffer_;

    std::vector<float> plasticity_gen_;
    std::vector<float> W_afferent_;
    std::vector<float> attractor_field_;
    std::array<float, CHANNEL_IO_DIM> action_refractory_fatigue_;
    std::array<float, CHANNEL_IO_DIM> action_bias_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

#endif // KARYON_SOVEREIGN_ENDOGENOUS_MEMORY_ORGANISM_V38_H
