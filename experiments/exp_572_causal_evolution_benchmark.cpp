// ============================================================================
// EXP-572: Causal Credit Assignment Morphogenetic Evolutionary System (CCA-MES)
// KEP Principle 30 (Totipotent Self-Synthesis) & Principle 5 (Causal Credit Assignment)
// Standard: KEP v30.0
// ============================================================================

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <deque>
#include <memory>
#include <immintrin.h>

#include "karyon_core.h"

namespace karyon {

class CausalMorphogeneticOrganism {
public:
    CausalMorphogeneticOrganism() noexcept {
        reset();
    }

    void reset() noexcept {
        num_modes_ = INITIAL_MODES;
        total_steps_ = 0;
        successful_adaptations_ = 0;
        pruned_parasites_ = 0;
        causal_sprouts_ = 0;

        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        current_shannon_entropy_ = std::log(static_cast<float>(CHANNEL_IO_DIM));

        dt_eff_ = 0.010f;
        endogenous_beta_ = 4.0f;
        coupling_density_ = 0.05f;
        attractor_plasticity_rate_ = 0.05f;
        hippocampal_scaling_gain_ = 4.0f;

        rng_.seed(5720);
        std::normal_distribution<float> norm(0.0f, 0.02f);

        q_.assign(num_modes_, 0.0f);
        p_.assign(num_modes_, 0.0f);
        stiffness_.resize(num_modes_);
        damping_.resize(num_modes_);
        decay_horizon_.resize(num_modes_);

        causal_credit_.assign(num_modes_, 0.0f);
        epigenetic_lock_.assign(num_modes_, 0.0f);
        mode_age_.assign(num_modes_, 0);

        for (size_t k = 0; k < num_modes_; ++k) {
            stiffness_[k] = 1.0f + 0.05f * static_cast<float>(k);
            damping_[k] = 0.35f + 0.005f * static_cast<float>(k);
            float frac = static_cast<float>(k) / static_cast<float>(num_modes_);
            decay_horizon_[k] = 0.50f + 0.45f * frac;
        }

        context_trace_.assign(2 * num_modes_, 0.0f);
        latest_normalized_context_.assign(2 * num_modes_, 0.0f);

        J_coupling_.assign(num_modes_ * num_modes_, 0.0f);
        for (size_t i = 0; i < num_modes_; ++i) {
            for (size_t j = 0; j < num_modes_; ++j) {
                if (i != j) J_coupling_[i * num_modes_ + j] = norm(rng_) * 0.05f;
            }
        }

        plasticity_gen_.resize((2 * num_modes_) * 9);
        for (auto& g : plasticity_gen_) g = norm(rng_);

        W_afferent_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : W_afferent_) w = norm(rng_);

        const size_t phase_dim = 2 * num_modes_;
        attractor_field_.resize(CHANNEL_IO_DIM * phase_dim);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t i = 0; i < phase_dim; ++i) {
                attractor_field_[c * phase_dim + i] = norm(rng_);
            }
            normalize_attractor_basin(c);
        }

        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
        action_refractory_fatigue_.fill(0.0f);
    }

    void normalize_attractor_basin(size_t c) noexcept {
        const size_t phase_dim = 2 * num_modes_;
        float* basin = &attractor_field_[c * phase_dim];
        float sq = avx2_dot_product(basin, basin, phase_dim);
        float inv = 1.0f / (std::sqrt(sq) + SAFETY_EPSILON);
        for (size_t i = 0; i < phase_dim; ++i) {
            basin[i] *= inv;
        }
    }

    void synthesize_internal_laws() noexcept {
        const size_t phase_dim = 2 * num_modes_;

        std::array<float, 9> plast_out{};
        for (size_t col = 0; col < 9; ++col) {
            float sum = 0.0f;
            for (size_t i = 0; i < phase_dim; ++i) {
                sum += plasticity_gen_[i * 9 + col] * latest_normalized_context_[i];
            }
            plast_out[col] = sum;
        }

        dt_eff_ = std::clamp(0.010f * std::exp(0.5f * plast_out[0]), 0.0005f, 0.100f);
        coupling_density_ = std::clamp(0.05f * std::exp(0.5f * plast_out[5]), 0.001f, 0.50f);
        attractor_plasticity_rate_ = std::clamp(0.05f * std::exp(0.5f * plast_out[6]), 0.001f, 0.50f);
        hippocampal_scaling_gain_ = 4.0f * std::exp(0.5f * plast_out[7]);

        const float max_entropy = std::log(static_cast<float>(CHANNEL_IO_DIM));
        float entropy_ratio = current_shannon_entropy_ / (max_entropy + SAFETY_EPSILON);
        float criticality_drive = 1.0f + 3.0f * std::pow(entropy_ratio, 2.0f);

        endogenous_beta_ = (2.0f * criticality_drive) * std::exp(0.3f * plast_out[8]);
        endogenous_beta_ = std::clamp(endogenous_beta_, 0.5f, 32.0f);
    }

    size_t perceive_and_decide(uint8_t input_channel) noexcept {
        synthesize_internal_laws();

        const float dt = dt_eff_;
        const size_t in_offset = static_cast<size_t>(input_channel) * num_modes_;

        for (size_t k = 0; k < num_modes_; ++k) {
            mode_age_[k]++;
            const float coupling_sum = avx2_dot_product(&J_coupling_[k * num_modes_], q_.data(), num_modes_);
            const float drive = W_afferent_[in_offset + k] * 5.0f + coupling_density_ * coupling_sum;
            const float omega_sq = stiffness_[k];
            const float gamma = 0.5f * damping_[k];

            const float q_star = drive / (omega_sq + SAFETY_EPSILON);
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
                const float inv_wd  = 1.0f / (omega_d + SAFETY_EPSILON);

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

        for (size_t k = 0; k < num_modes_; ++k) {
            const float lambda_q = decay_horizon_[k];
            const float lambda_p = std::clamp(lambda_q * 0.85f, 0.20f, 0.95f);

            context_trace_[k] = lambda_q * context_trace_[k] + (1.0f - lambda_q) * q_[k];
            context_trace_[num_modes_ + k] = lambda_p * context_trace_[num_modes_ + k] + (1.0f - lambda_p) * p_[k];
        }

        const size_t phase_dim = 2 * num_modes_;
        float norm_sq = avx2_dot_product(context_trace_.data(), context_trace_.data(), phase_dim);
        const float inv_norm = 1.0f / (std::sqrt(norm_sq) + SAFETY_EPSILON);

        for (size_t i = 0; i < phase_dim; ++i) {
            latest_normalized_context_[i] = context_trace_[i] * inv_norm;
        }

        std::array<float, CHANNEL_IO_DIM> energy{};
        float max_e = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float cosine_sim = avx2_dot_product(&attractor_field_[c * phase_dim], latest_normalized_context_.data(), phase_dim);
            float eff_energy = cosine_sim - 1.20f * action_refractory_fatigue_[c];
            energy[c] = eff_energy;
            if (eff_energy > max_e) max_e = eff_energy;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float val = (energy[c] - max_e) * endogenous_beta_;
            float p = std::exp(val);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }

        const float inv_sum_exp = 1.0f / (sum_exp + SAFETY_EPSILON);
        float entropy = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            latest_probabilities_[c] *= inv_sum_exp;
            float p = latest_probabilities_[c];
            if (p > 1e-8f) {
                entropy -= p * std::log(p);
            }
        }
        current_shannon_entropy_ = entropy;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) {
                chosen = c;
                break;
            }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            action_refractory_fatigue_[c] *= 0.70f;
        }
        action_refractory_fatigue_[chosen] += 1.0f;

        return chosen;
    }

    void assimilate_causal_learning(uint8_t input_channel, uint8_t target_channel, float lr = 0.05f) noexcept {
        total_steps_++;

        float p_target = latest_probabilities_[target_channel];
        float cross_entropy = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * cross_entropy;

        float delta_FE = free_energy_ - prev_free_energy_;
        const size_t phase_dim = 2 * num_modes_;

        // Continuous Causal Credit Trace Vector
        const float lambda_credit = 0.95f;
        for (size_t k = 0; k < num_modes_; ++k) {
            float activity = std::abs(q_[k]) + std::abs(p_[k]);
            float credit_pulse = (-delta_FE) * activity * 100.0f; // Scale pulse for sensitive credit tracking
            causal_credit_[k] = lambda_credit * causal_credit_[k] + (1.0f - lambda_credit) * credit_pulse;

            epigenetic_lock_[k] = std::clamp(1.0f / (1.0f + std::exp(-5.0f * (causal_credit_[k] - 0.01f))), 0.0f, 1.0f);
        }

        const size_t in_offset = static_cast<size_t>(input_channel) * num_modes_;
        const float base_feedback = (p_target > 0.05f) ? 1.0f : -0.5f;

        for (size_t k = 0; k < num_modes_; ++k) {
            float lock_factor = 1.0f - 0.85f * epigenetic_lock_[k];
            float alpha_w = lr * 0.1f * base_feedback * lock_factor;
            W_afferent_[in_offset + k] += alpha_w * q_[k];

            stiffness_[k] += lr * 0.001f * base_feedback * lock_factor * (q_[k] * q_[k]);
            stiffness_[k] = std::clamp(stiffness_[k], 0.1f, 10.0f);
        }

        const float eff_lr = lr * attractor_plasticity_rate_ * 10.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float delta = (c == target_channel) ? 1.0f : 0.0f;
            float err = delta - latest_probabilities_[c];
            if (std::abs(err) > 0.001f) {
                float alpha_field = eff_lr * err;
                avx2_axpy(&attractor_field_[c * phase_dim], alpha_field, latest_normalized_context_.data(), phase_dim);
                normalize_attractor_basin(c);
            }
        }

        for (size_t col = 0; col < 9; ++col) {
            for (size_t k = 0; k < num_modes_; ++k) {
                float causal_grad = causal_credit_[k] * latest_normalized_context_[k];
                plasticity_gen_[k * 9 + col] += lr * 0.01f * causal_grad;
                plasticity_gen_[(num_modes_ + k) * 9 + col] += lr * 0.01f * causal_credit_[k] * latest_normalized_context_[num_modes_ + k];
            }
        }

        if (total_steps_ % 200 == 0) {
            evaluate_and_evolve_structure();
        }
    }

    void evaluate_and_evolve_structure() noexcept {
        std::vector<size_t> parasites_to_prune;
        for (size_t k = 0; k < num_modes_; ++k) {
            if (mode_age_[k] > 200 && causal_credit_[k] < -0.05f && epigenetic_lock_[k] < 0.10f) {
                parasites_to_prune.push_back(k);
            }
        }

        if (!parasites_to_prune.empty() && num_modes_ > 32) {
            size_t count = std::min(parasites_to_prune.size(), static_cast<size_t>(4));
            for (size_t idx = 0; idx < count; ++idx) {
                prune_mode(parasites_to_prune[idx] - idx);
            }
            pruned_parasites_ += count;
        }

        size_t best_parent = 0;
        float max_credit = -1e9f;
        for (size_t k = 0; k < num_modes_; ++k) {
            if (causal_credit_[k] > max_credit) {
                max_credit = causal_credit_[k];
                best_parent = k;
            }
        }

        if (max_credit > 0.01f && num_modes_ < MAX_ALLOWABLE_MODES) {
            directed_causal_sprout(best_parent);
            causal_sprouts_++;
        }
    }

    void prune_mode(size_t target_k) noexcept {
        if (num_modes_ <= 32 || target_k >= num_modes_) return;

        const size_t old_modes = num_modes_;
        const size_t new_modes = old_modes - 1;

        q_.erase(q_.begin() + target_k);
        p_.erase(p_.begin() + target_k);
        stiffness_.erase(stiffness_.begin() + target_k);
        damping_.erase(damping_.begin() + target_k);
        decay_horizon_.erase(decay_horizon_.begin() + target_k);
        causal_credit_.erase(causal_credit_.begin() + target_k);
        epigenetic_lock_.erase(epigenetic_lock_.begin() + target_k);
        mode_age_.erase(mode_age_.begin() + target_k);

        context_trace_.resize(2 * new_modes, 0.0f);
        latest_normalized_context_.resize(2 * new_modes, 0.0f);

        num_modes_ = new_modes;
    }

    void directed_causal_sprout(size_t parent_k) noexcept {
        if (num_modes_ >= MAX_ALLOWABLE_MODES) return;

        const size_t old_modes = num_modes_;
        const size_t new_modes = old_modes + 1;
        num_modes_ = new_modes;

        q_.push_back(q_[parent_k]);
        p_.push_back(p_[parent_k]);
        stiffness_.push_back(stiffness_[parent_k]);
        damping_.push_back(damping_[parent_k]);
        decay_horizon_.push_back(decay_horizon_[parent_k]);
        causal_credit_.push_back(0.0f);
        epigenetic_lock_.push_back(0.0f);
        mode_age_.push_back(0);

        context_trace_.resize(2 * new_modes, 0.0f);
        latest_normalized_context_.resize(2 * new_modes, 0.0f);

        J_coupling_.resize(new_modes * new_modes, 0.0f);
        plasticity_gen_.resize((2 * new_modes) * 9, 0.0f);

        std::vector<float> new_W_afferent(CHANNEL_IO_DIM * new_modes, 0.0f);
        std::normal_distribution<float> tiny_perturb(0.0f, 0.001f);

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t k = 0; k < old_modes; ++k) {
                new_W_afferent[c * new_modes + k] = W_afferent_[c * old_modes + k];
            }
            new_W_afferent[c * new_modes + old_modes] = W_afferent_[c * old_modes + parent_k] + tiny_perturb(rng_);
        }
        W_afferent_ = std::move(new_W_afferent);

        const size_t old_phase_dim = 2 * old_modes;
        const size_t new_phase_dim = 2 * new_modes;
        std::vector<float> new_attractors(CHANNEL_IO_DIM * new_phase_dim, 0.0f);

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t k = 0; k < old_modes; ++k) {
                new_attractors[c * new_phase_dim + k] = attractor_field_[c * old_phase_dim + k];
                new_attractors[c * new_phase_dim + new_modes + k] = attractor_field_[c * old_phase_dim + old_modes + k];
            }
            new_attractors[c * new_phase_dim + old_modes] = attractor_field_[c * old_phase_dim + parent_k] + tiny_perturb(rng_);
            new_attractors[c * new_phase_dim + new_modes + old_modes] = attractor_field_[c * old_phase_dim + old_modes + parent_k] + tiny_perturb(rng_);

            float* basin = &new_attractors[c * new_phase_dim];
            float sq = avx2_dot_product(basin, basin, new_phase_dim);
            float inv = 1.0f / (std::sqrt(sq) + SAFETY_EPSILON);
            for (size_t i = 0; i < new_phase_dim; ++i) basin[i] *= inv;
        }
        attractor_field_ = std::move(new_attractors);
    }

    size_t get_modes() const noexcept { return num_modes_; }
    size_t get_pruned_count() const noexcept { return pruned_parasites_; }
    size_t get_sprout_count() const noexcept { return causal_sprouts_; }
    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }

private:
    size_t num_modes_;
    size_t total_steps_;
    size_t successful_adaptations_;
    size_t pruned_parasites_;
    size_t causal_sprouts_;

    float free_energy_;
    float prev_free_energy_;
    float current_shannon_entropy_;

    float dt_eff_;
    float endogenous_beta_;
    float coupling_density_;
    float attractor_plasticity_rate_;
    float hippocampal_scaling_gain_;

    std::mt19937 rng_;

    std::vector<float> q_;
    std::vector<float> p_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<float> decay_horizon_;

    std::vector<float> causal_credit_;
    std::vector<float> epigenetic_lock_;
    std::vector<size_t> mode_age_;

    std::vector<float> context_trace_;
    std::vector<float> latest_normalized_context_;

    std::vector<float> J_coupling_;
    std::vector<float> plasticity_gen_;
    std::vector<float> W_afferent_;
    std::vector<float> attractor_field_;

    std::array<float, CHANNEL_IO_DIM> action_refractory_fatigue_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "============================================================\n";
    std::cout << "=== EXP-572: CAUSAL CREDIT ASSIGNMENT MORPHOGENETIC ENGINE ===\n";
    std::cout << "============================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) {
        text_path = "data/web_downloads/alpaca_data.json";
        file.open(text_path, std::ios::binary);
    }

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::cout << "Corpus: " << corpus.size() << " bytes loaded from " << text_path << "\n\n";
    } else {
        std::cout << "Fallback: generating synthetic sequence.\n";
        std::string fallback = "The capital of France is Paris. Artificial intelligence is evolving continuously.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    karyon::CausalMorphogeneticOrganism organism;

    const size_t train_steps = 200000;
    std::cout << ">>> Phase 1: Continuous Causal Morphogenetic Training for " << train_steps << " steps...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t input_byte = corpus[step % corpus.size()];
        uint8_t target_byte = corpus[(step + 1) % corpus.size()];

        size_t action = organism.perceive_and_decide(input_byte);
        organism.assimilate_causal_learning(input_byte, target_byte, 0.05f);

        if ((step + 1) % 50000 == 0) {
            std::cout << "  Step " << (step + 1) << " | Modes: " << organism.get_modes()
                      << " | Pruned Parasites: " << organism.get_pruned_count()
                      << " | Directed Sprouts: " << organism.get_sprout_count()
                      << " | Free Energy: " << organism.get_free_energy()
                      << " | Beta: " << organism.get_beta() << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    double steps_per_sec = static_cast<double>(train_steps) / duration;

    std::cout << "\n>>> Phase 2: Directed Text Generation Test...\n";
    std::string prompt = "The capital of France is ";
    std::cout << "Prompt: \"" << prompt << "\"\nOutput: " << prompt;

    uint8_t curr_byte = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 100; ++i) {
        size_t next_act = organism.perceive_and_decide(curr_byte);
        char ch = static_cast<char>(next_act);
        std::cout << ch << std::flush;
        curr_byte = static_cast<uint8_t>(next_act);
    }
    std::cout << "\n\n";

    std::cout << "============================================================\n";
    std::cout << "=== TELEMETRY SUMMARY (EXP-572) ===\n";
    std::cout << "Final Active Modes    : " << organism.get_modes() << " (" << organism.get_modes() * 2 << "D)\n";
    std::cout << "Pruned Epistemic Parasites : " << organism.get_pruned_count() << "\n";
    std::cout << "Directed Sprouts      : " << organism.get_sprout_count() << "\n";
    std::cout << "Final Free Energy     : " << organism.get_free_energy() << "\n";
    std::cout << "Endogenous Beta       : " << organism.get_beta() << "\n";
    std::cout << "Throughput            : " << steps_per_sec << " steps/sec\n";
    std::cout << "============================================================\n";

    return 0;
}
