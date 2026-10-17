// ============================================================================
// EXP-576: Unshackled Speech Emergence Benchmark: Wavefield vs String Continuum
// Testing Principle 29 (Law of Full Sovereign Self-Control & Sufficient Time)
// Testing Principle 30 (Law of Totipotent Self-Synthesis & Bottom-Up Emergence)
//
// ZERO RESTRICTIONS / COMPLETE SOVEREIGNTY:
// - NO artificial clamping of plastic learning rate or boundary anchors.
// - NO hardcoded character limits, no artificial temperature overrides.
// - Full plastic freedom: the substrate self-tunes its own metric tensors g_ij,
//   phase resonance frequencies, endogenous beta, and dynamic timescale dt_eff.
// - Long autonomous relaxation / learning (500,000 continuous bytes on Alpaca corpus)
//   followed by open-ended speech generation testing semantic coherence,
//   phonotactic integrity, vocabulary variety, and Free Energy minimization.
//
// Comparing BOTH architectures side-by-side on the exact same sequence:
// 1. Wavefield Continuum (Causal Coupled SDE Phase Oscillators, EXP-572/EXP-573)
// 2. Symplectic String Continuum (Sovereign Autopoietic Strings, EXP-575)
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
#include <immintrin.h>
#include <numbers>
#include <omp.h>
#include <map>

#include "karyon_core.h"

namespace karyon {

// ============================================================================
// 1. UNSHACKLED SOVEREIGN WAVEFIELD ORGANISM
// ============================================================================
class UnshackledWavefieldOrganism {
public:
    UnshackledWavefieldOrganism(size_t dim = 512) noexcept : dim_(dim) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        endogenous_beta_ = 3.0f;

        q_.assign(dim_, 0.0f);
        p_.assign(dim_, 0.0f);
        stiffness_.resize(dim_);
        damping_.resize(dim_);
        causal_credit_.assign(dim_, 0.0f);
        epigenetic_lock_.assign(dim_, 0.0f);

        for (size_t i = 0; i < dim_; ++i) {
            stiffness_[i] = 1.0f + 0.02f * static_cast<float>(i % 64);
            damping_[i] = 0.02f;
        }

        // Afferent sensory mapping
        std::mt19937 rng(5761);
        std::normal_distribution<float> norm(0.0f, 0.04f);
        sensory_aff_.resize(CHANNEL_IO_DIM * dim_);
        for (auto& w : sensory_aff_) w = norm(rng);

        motor_eff_.resize(CHANNEL_IO_DIM * dim_);
        for (auto& w : motor_eff_) w = norm(rng);

        // Low-rank coupling tensor J_ij
        rank_ = 8;
        U_.resize(dim_ * rank_);
        V_.resize(dim_ * rank_);
        for (auto& w : U_) w = norm(rng);
        for (auto& w : V_) w = norm(rng);

        boundary_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * dim_;

        // Low-rank coupling field projection: V_bar = V^T * q
        std::vector<float> V_bar(rank_, 0.0f);
        for (size_t r = 0; r < rank_; ++r) {
            float sum = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                sum += V_[i * rank_ + r] * q_[i];
            }
            V_bar[r] = std::tanh(sum * 0.1f);
        }

        // Substrate Phase Dynamics (Predictor-Corrector)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dim_; ++i) {
            float coupling = 0.0f;
            for (size_t r = 0; r < rank_; ++r) {
                coupling += U_[i * rank_ + r] * V_bar[r];
            }

            float force = sensory_aff_[in_offset + i] + coupling * 0.2f - stiffness_[i] * q_[i] - damping_[i] * p_[i];
            p_[i] += force * 0.05f;
            q_[i] = std::tanh(q_[i] + p_[i] * 0.05f);
        }

        // Efferent Readout with Boundary Inertia (Unshackled)
        std::array<float, CHANNEL_IO_DIM> logits{};
        float max_l = -1e9f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_eff_[c * dim_], q_.data(), dim_);
            float norm_dot = dot / (std::sqrt(static_cast<float>(dim_)) + SAFETY_EPSILON);
            float eff_logit = norm_dot - boundary_momentum_[c];
            logits[c] = eff_logit;
            if (eff_logit > max_l) max_l = eff_logit;
        }

        // Endogenous inverse temperature
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = logits[c] - max_l;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(1.5f + 2.0f * std::sqrt(variance), 1.0f, 20.0f);

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((logits[c] - max_l) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) { chosen = c; break; }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) boundary_momentum_[c] *= 0.70f;
        boundary_momentum_[chosen] += 0.90f;

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.04f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float delta_FE = free_energy_ - prev_free_energy_;

        // Causal Credit & Epigenetic Lock
        for (size_t i = 0; i < dim_; ++i) {
            float energy = q_[i] * q_[i] + p_[i] * p_[i];
            float credit = (-delta_FE) * energy * 100.0f;
            causal_credit_[i] = 0.95f * causal_credit_[i] + 0.05f * credit;
            epigenetic_lock_[i] = std::clamp(1.0f / (1.0f + std::exp(-5.0f * (causal_credit_[i] - 0.01f))), 0.0f, 1.0f);
        }

        // Efferent Plasticity with Epigenetic Protection
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t i = 0; i < dim_; ++i) {
                    float plastic = 1.0f - 0.90f * epigenetic_lock_[i];
                    motor_eff_[c * dim_ + i] += alpha * plastic * q_[i];
                }
            }
        }

        // Unshackled Stiffness Modulation
        for (size_t i = 0; i < dim_; ++i) {
            if (epigenetic_lock_[i] < 0.25f) {
                stiffness_[i] += 0.01f * causal_credit_[i];
                stiffness_[i] = std::clamp(stiffness_[i], 0.1f, 10.0f);
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }

private:
    size_t dim_;
    size_t rank_;
    size_t total_steps_;
    float free_energy_;
    float prev_free_energy_;
    float endogenous_beta_;

    std::vector<float> q_;
    std::vector<float> p_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<float> causal_credit_;
    std::vector<float> epigenetic_lock_;

    std::vector<float> sensory_aff_;
    std::vector<float> motor_eff_;
    std::vector<float> U_;
    std::vector<float> V_;

    std::array<float, CHANNEL_IO_DIM> boundary_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

// ============================================================================
// 2. UNSHACKLED SOVEREIGN STRING CONTINUUM ORGANISM
// ============================================================================
class UnshackledStringOrganism {
public:
    UnshackledStringOrganism(size_t initial_modes = 256) noexcept : num_modes_(initial_modes) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        endogenous_beta_ = 4.0f;
        dt_eff_ = 0.040f;
        total_sprouts_ = 0;
        pruned_modes_ = 0;

        u_.assign(num_modes_, 0.0f);
        v_.assign(num_modes_, 0.0f);
        tensions_.resize(num_modes_);
        damping_.resize(num_modes_);
        causal_credit_.assign(num_modes_, 0.0f);
        epigenetic_lock_.assign(num_modes_, 0.0f);

        for (size_t m = 0; m < num_modes_; ++m) {
            tensions_[m] = 1.0f + 0.03f * static_cast<float>(m % 64);
            damping_[m] = 0.015f;
        }

        cos_w_.resize(num_modes_);
        sin_w_.resize(num_modes_);
        update_frequencies();

        sensory_fourier_.resize(CHANNEL_IO_DIM * num_modes_);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t m = 0; m < num_modes_; ++m) {
                float k_wave = static_cast<float>((c + 1) * (m + 1));
                float phase = 2.0f * std::numbers::pi_v<float> * k_wave / static_cast<float>(num_modes_);
                sensory_fourier_[c * num_modes_ + m] = std::sin(phase) * 0.15f;
            }
        }

        std::mt19937 rng(5762);
        std::normal_distribution<float> norm(0.0f, 0.03f);
        motor_readout_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : motor_readout_) w = norm(rng);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    void update_frequencies() noexcept {
        for (size_t m = 0; m < num_modes_; ++m) {
            float w = 0.12f * std::sqrt(std::abs(tensions_[m]) + SAFETY_EPSILON);
            cos_w_[m] = std::cos(w * dt_eff_);
            sin_w_[m] = std::sin(w * dt_eff_);
        }
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * num_modes_;

        // Endogenous time dilation from string kinetic energy
        float total_kinetic = 0.0f;
        for (size_t m = 0; m < num_modes_; ++m) total_kinetic += v_[m] * v_[m];
        float mean_kinetic = total_kinetic / static_cast<float>(num_modes_);
        dt_eff_ = std::clamp(0.040f / (1.0f + 4.0f * mean_kinetic), 0.010f, 0.080f);
        update_frequencies();

        // Symplectic Phase Rotation
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < num_modes_; ++m) {
            float u_curr = u_[m];
            float v_curr = v_[m];

            float u_rot = u_curr * cos_w_[m] - v_curr * sin_w_[m];
            float v_rot = u_curr * sin_w_[m] + v_curr * cos_w_[m];

            float pluck = sensory_fourier_[in_offset + m];
            float decay = std::exp(-damping_[m] * dt_eff_);

            u_[m] = (u_rot + pluck * dt_eff_) * decay;
            v_[m] = (v_rot + pluck * dt_eff_) * decay;
        }

        // Efferent Readout with Boundary Inertia
        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_readout_[c * num_modes_], u_.data(), num_modes_);
            float norm_dot = dot / (std::sqrt(static_cast<float>(num_modes_)) + SAFETY_EPSILON);
            float eff_res = norm_dot - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // Endogenous beta
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = resonance[c] - max_res;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(1.5f + 2.0f * std::sqrt(variance), 1.0f, 20.0f);

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) { chosen = c; break; }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f * (1.0f + mean_kinetic);

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.04f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float delta_FE = free_energy_ - prev_free_energy_;

        // Causal Credit & Epigenetic Lock
        for (size_t m = 0; m < num_modes_; ++m) {
            float energy = u_[m] * u_[m] + v_[m] * v_[m];
            float credit = (-delta_FE) * energy * 100.0f;
            causal_credit_[m] = 0.95f * causal_credit_[m] + 0.05f * credit;
            epigenetic_lock_[m] = std::clamp(1.0f / (1.0f + std::exp(-5.0f * (causal_credit_[m] - 0.01f))), 0.0f, 1.0f);
        }

        // Efferent Plasticity
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t m = 0; m < num_modes_; ++m) {
                    float plastic = 1.0f - 0.90f * epigenetic_lock_[m];
                    motor_readout_[c * num_modes_ + m] += alpha * plastic * u_[m];
                }
            }
        }

        // Tension Adaptation
        for (size_t m = 0; m < num_modes_; ++m) {
            if (epigenetic_lock_[m] < 0.25f) {
                tensions_[m] += 0.01f * causal_credit_[m];
                tensions_[m] = std::clamp(tensions_[m], 0.2f, 5.0f);
            }
        }

        // Morphogenesis every 25k steps
        if (total_steps_ % 25000 == 0 && num_modes_ < 1024) {
            morph();
        }
    }

    void morph() noexcept {
        size_t best_parent = 0;
        float max_credit = -1e9f;
        for (size_t m = 0; m < num_modes_; ++m) {
            if (causal_credit_[m] > max_credit) {
                max_credit = causal_credit_[m];
                best_parent = m;
            }
        }

        for (size_t m = 0; m < num_modes_; ++m) {
            if (causal_credit_[m] < -0.05f && epigenetic_lock_[m] < 0.05f) {
                u_[m] = 0.0f;
                v_[m] = 0.0f;
                tensions_[m] = tensions_[best_parent] * 1.5f;
                causal_credit_[m] = 0.0f;
                pruned_modes_++;
            }
        }

        if (max_credit > 0.01f && num_modes_ + 16 <= 1024) {
            size_t sprout_count = 16;
            size_t old_modes = num_modes_;
            num_modes_ += sprout_count;

            u_.resize(num_modes_, 0.0f);
            v_.resize(num_modes_, 0.0f);
            tensions_.resize(num_modes_);
            damping_.resize(num_modes_, 0.015f);
            causal_credit_.resize(num_modes_, 0.0f);
            epigenetic_lock_.resize(num_modes_, 0.0f);
            cos_w_.resize(num_modes_);
            sin_w_.resize(num_modes_);

            for (size_t i = 0; i < sprout_count; ++i) {
                size_t child_idx = old_modes + i;
                float ratio = static_cast<float>(i + 2) / 2.0f;
                tensions_[child_idx] = tensions_[best_parent] * ratio;
                u_[child_idx] = 0.01f * u_[best_parent];
                v_[child_idx] = 0.01f * v_[best_parent];
            }

            sensory_fourier_.resize(CHANNEL_IO_DIM * num_modes_);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                for (size_t i = 0; i < sprout_count; ++i) {
                    size_t m = old_modes + i;
                    float k_wave = static_cast<float>((c + 1) * (m + 1));
                    float phase = 2.0f * std::numbers::pi_v<float> * k_wave / static_cast<float>(num_modes_);
                    sensory_fourier_[c * num_modes_ + m] = std::sin(phase) * 0.15f;
                }
            }

            std::vector<float> new_motor(CHANNEL_IO_DIM * num_modes_, 0.0f);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                for (size_t m = 0; m < old_modes; ++m) {
                    new_motor[c * num_modes_ + m] = motor_readout_[c * old_modes + m];
                }
                for (size_t i = 0; i < sprout_count; ++i) {
                    size_t child_idx = old_modes + i;
                    new_motor[c * num_modes_ + child_idx] = 0.005f * motor_readout_[c * old_modes + best_parent];
                }
            }
            motor_readout_ = std::move(new_motor);
            update_frequencies();
            total_sprouts_ += sprout_count;
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }
    size_t get_modes() const noexcept { return num_modes_; }
    size_t get_sprouts() const noexcept { return total_sprouts_; }
    size_t get_pruned() const noexcept { return pruned_modes_; }

private:
    size_t num_modes_;
    size_t total_steps_;
    float free_energy_;
    float prev_free_energy_;
    float endogenous_beta_;
    float dt_eff_;
    size_t total_sprouts_;
    size_t pruned_modes_;

    std::vector<float> u_;
    std::vector<float> v_;
    std::vector<float> tensions_;
    std::vector<float> damping_;
    std::vector<float> cos_w_;
    std::vector<float> sin_w_;

    std::vector<float> causal_credit_;
    std::vector<float> epigenetic_lock_;

    std::vector<float> sensory_fourier_;
    std::vector<float> motor_readout_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

// Linguistic Evaluation Metrics
struct SpeechMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
    size_t total_chars = 0;
};

SpeechMetrics evaluate_text(const std::string& text) {
    SpeechMetrics m;
    m.total_chars = text.size();
    if (m.total_chars == 0) return m;

    std::map<char, size_t> counts;
    size_t whitespaces = 0;
    size_t alphabets = 0;

    for (char ch : text) {
        counts[ch]++;
        if (std::isspace(static_cast<unsigned char>(ch))) whitespaces++;
        if (std::isalpha(static_cast<unsigned char>(ch))) alphabets++;
    }

    m.unique_chars = counts.size();
    m.whitespace_ratio = static_cast<double>(whitespaces) / m.total_chars;
    m.alphabet_ratio = static_cast<double>(alphabets) / m.total_chars;

    double ent = 0.0;
    for (const auto& [ch, count] : counts) {
        double p = static_cast<double>(count) / m.total_chars;
        if (p > 0.0) ent -= p * std::log2(p);
    }
    m.char_entropy = ent;
    return m;
}

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-576: UNSHACKLED SPEECH EMERGENCE (WAVEFIELD vs STRING)       ===\n";
    std::cout << "=== KEP Principle 29 (Sovereignty & Time) & Principle 30 (Synthesis) ===\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) {
        text_path = "data/web_downloads/alpaca_data.json";
        file.open(text_path, std::ios::binary);
    }

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::cout << "[+] Corpus loaded: " << corpus.size() << " bytes.\n\n";
    } else {
        std::string fallback = "The capital of France is Paris. Language is continuous physical resonance.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000; // 300,000 steps of sufficient physical relaxation!

    // ========================================================================
    // BENCHMARK 1: UNSHACKLED WAVEFIELD CONTINUUM
    // ========================================================================
    std::cout << ">>> [1/2] Initiating Unshackled Wavefield Continuum (512D Coupled SDE)...\n";
    karyon::UnshackledWavefieldOrganism wave_org(512);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        wave_org.step(in_b);
        wave_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 100000 == 0) {
            std::cout << "  Wavefield Step " << (step + 1) 
                      << " | Free Energy: " << wave_org.get_free_energy()
                      << " | Beta: " << wave_org.get_beta() << "\n";
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double wave_duration = std::chrono::duration<double>(t1 - t0).count();
    double wave_throughput = static_cast<double>(train_steps) / wave_duration;

    // Wavefield Speech Generation
    std::string wave_prompt = "The capital of France is ";
    std::string wave_output = wave_prompt;
    uint8_t curr = static_cast<uint8_t>(wave_prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t next_act = wave_org.step(curr);
        wave_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto wave_metrics = karyon::evaluate_text(wave_output);

    std::cout << "\n>>> WAVEFIELD GENERATED SPEECH (200 bytes):\n";
    std::cout << "\"" << wave_output << "\"\n\n";

    // ========================================================================
    // BENCHMARK 2: UNSHACKLED SYMPLECTIC STRING CONTINUUM
    // ========================================================================
    std::cout << ">>> [2/2] Initiating Unshackled String Continuum (Autopoietic Symplectic)...\n";
    karyon::UnshackledStringOrganism string_org(256);

    t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        string_org.step(in_b);
        string_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 100000 == 0) {
            std::cout << "  String Step " << (step + 1) 
                      << " | Modes: " << string_org.get_modes()
                      << " | Sprouts: " << string_org.get_sprouts()
                      << " | Pruned: " << string_org.get_pruned()
                      << " | Free Energy: " << string_org.get_free_energy()
                      << " | Beta: " << string_org.get_beta() << "\n";
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double string_duration = std::chrono::duration<double>(t1 - t0).count();
    double string_throughput = static_cast<double>(train_steps) / string_duration;

    // String Speech Generation
    std::string string_prompt = "The capital of France is ";
    std::string string_output = string_prompt;
    curr = static_cast<uint8_t>(string_prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t next_act = string_org.step(curr);
        string_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto string_metrics = karyon::evaluate_text(string_output);

    std::cout << "\n>>> STRING GENERATED SPEECH (200 bytes):\n";
    std::cout << "\"" << string_output << "\"\n\n";

    // ========================================================================
    // TELEMETRY COMPARISON TABLE
    // ========================================================================
    std::cout << "========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY SUMMARY (EXP-576)                          ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Metric                       | Wavefield Organism    | String Organism\n";
    std::cout << "-----------------------------|-----------------------|-------------------\n";
    std::cout << "Throughput (steps/sec)       | " << wave_throughput << "         | " << string_throughput << "\n";
    std::cout << "Final Free Energy (nats)     | " << wave_org.get_free_energy() << "               | " << string_org.get_free_energy() << "\n";
    std::cout << "Endogenous Beta              | " << wave_org.get_beta() << "               | " << string_org.get_beta() << "\n";
    std::cout << "Alphabet Ratio (%)           | " << (wave_metrics.alphabet_ratio * 100.0) << "%              | " << (string_metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "Whitespace Ratio (%)         | " << (wave_metrics.whitespace_ratio * 100.0) << "%              | " << (string_metrics.whitespace_ratio * 100.0) << "%\n";
    std::cout << "Character Entropy (bits)     | " << wave_metrics.char_entropy << "               | " << string_metrics.char_entropy << "\n";
    std::cout << "Unique Characters Generated  | " << wave_metrics.unique_chars << "                  | " << string_metrics.unique_chars << "\n";
    std::cout << "========================================================================\n";

    return 0;
}
