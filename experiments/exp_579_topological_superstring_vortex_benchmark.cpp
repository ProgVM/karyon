// ============================================================================
// EXP-579: Topological Superstring Vortex Continuum (TSFVC)
// The Quest for Universal Non-Biological Super-Intelligence Substrate
// Combining:
// 1. Symplectic String Unitary Phase Rotations (Liouville Volume Preservation)
// 2. Navier-Stokes Topological Vorticity Circulation (Solitonic Knot Memory)
// 3. Gauge-Field Knot Coupling:
//    - String vibrations induce Kelvin Circulation \Gamma_k = \oint u \cdot dl
//    - Fluid vorticity pressure modulates string tensions T_m(t)
// 4. KEP Principles 2, 3, 5, 6, 29, 30:
//    - Zero top-down human clamps, zero hardcoded rules.
//    - Autonomous non-blind causal budding of closed loop modes.
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

constexpr size_t VORTEX_GRID_SIZE = 16;
constexpr size_t VORTEX_CELLS = VORTEX_GRID_SIZE * VORTEX_GRID_SIZE; // 256 cells

struct KnotMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
};

KnotMetrics evaluate_knot_speech(const std::string& text) {
    KnotMetrics m;
    if (text.empty()) return m;
    std::map<char, size_t> counts;
    size_t whitespaces = 0;
    size_t alphabets = 0;

    for (char ch : text) {
        counts[ch]++;
        if (std::isspace(static_cast<unsigned char>(ch))) whitespaces++;
        if (std::isalpha(static_cast<unsigned char>(ch))) alphabets++;
    }

    m.unique_chars = counts.size();
    m.whitespace_ratio = static_cast<double>(whitespaces) / text.size();
    m.alphabet_ratio = static_cast<double>(alphabets) / text.size();

    double ent = 0.0;
    for (const auto& [ch, count] : counts) {
        double p = static_cast<double>(count) / text.size();
        if (p > 0.0) ent -= p * std::log2(p);
    }
    m.char_entropy = ent;
    return m;
}

class TopologicalSuperstringVortexOrganism {
public:
    TopologicalSuperstringVortexOrganism(size_t initial_strings = 256) noexcept 
        : num_strings_(initial_strings) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        endogenous_beta_ = 4.5f;
        dt_eff_ = 0.040f;
        total_sprouts_ = 0;
        pruned_strings_ = 0;

        // 1. Symplectic String Coordinates
        u_string_.assign(num_strings_, 0.0f);
        v_string_.assign(num_strings_, 0.0f);
        tensions_.resize(num_strings_);
        causal_credit_.assign(num_strings_, 0.0f);
        epigenetic_lock_.assign(num_strings_, 0.0f);

        for (size_t m = 0; m < num_strings_; ++m) {
            tensions_[m] = 1.0f + 0.03f * static_cast<float>(m % 64);
        }

        cos_w_.resize(num_strings_);
        sin_w_.resize(num_strings_);
        update_string_frequencies();

        // 2. Fluid Lattice Coordinates
        u_fluid_.assign(VORTEX_CELLS, 0.0f);
        v_fluid_.assign(VORTEX_CELLS, 0.0f);
        vorticity_.assign(VORTEX_CELLS, 0.0f);

        // 3. Bilateral Gauge Coupling (String <-> Fluid)
        // String modes project into fluid cells; fluid vorticity modulates string tension
        string_to_fluid_proj_.resize(num_strings_ * VORTEX_CELLS);
        std::mt19937 rng(5791);
        std::normal_distribution<float> norm(0.0f, 0.02f);
        for (auto& w : string_to_fluid_proj_) w = norm(rng);

        // 4. Sensory & Motor Projections
        sensory_fourier_.resize(CHANNEL_IO_DIM * num_strings_);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t m = 0; m < num_strings_; ++m) {
                float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>((c + 1) * (m + 1)) / static_cast<float>(num_strings_);
                sensory_fourier_[c * num_strings_ + m] = std::sin(phase) * 0.15f;
            }
        }

        motor_readout_.resize(CHANNEL_IO_DIM * num_strings_);
        for (auto& w : motor_readout_) w = norm(rng);

        vortex_readout_.resize(CHANNEL_IO_DIM * VORTEX_CELLS);
        for (auto& w : vortex_readout_) w = norm(rng);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    void update_string_frequencies() noexcept {
        for (size_t m = 0; m < num_strings_; ++m) {
            float w = 0.12f * std::sqrt(std::abs(tensions_[m]) + SAFETY_EPSILON);
            cos_w_[m] = std::cos(w * dt_eff_);
            sin_w_[m] = std::sin(w * dt_eff_);
        }
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * num_strings_;

        // 1. String Kinetic Energy & Endogenous Time Dilation
        float total_kinetic = 0.0f;
        for (size_t m = 0; m < num_strings_; ++m) total_kinetic += v_string_[m] * v_string_[m];
        float mean_kinetic = total_kinetic / static_cast<float>(num_strings_);
        dt_eff_ = std::clamp(0.040f / (1.0f + 4.0f * mean_kinetic), 0.010f, 0.080f);
        update_string_frequencies();

        // 2. Symplectic String Phase Rotation (Liouville Invariant)
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < num_strings_; ++m) {
            float u_rot = u_string_[m] * cos_w_[m] - v_string_[m] * sin_w_[m];
            float v_rot = u_string_[m] * sin_w_[m] + v_string_[m] * cos_w_[m];

            float pluck = sensory_fourier_[in_offset + m];
            u_string_[m] = (u_rot + pluck * dt_eff_) * 0.985f;
            v_string_[m] = (v_rot + pluck * dt_eff_) * 0.985f;
        }

        // 3. String-to-Fluid Gauge Current: J_fluid = W * u_string
        std::vector<float> fluid_injection(VORTEX_CELLS, 0.0f);
        for (size_t i = 0; i < VORTEX_CELLS; ++i) {
            float sum = 0.0f;
            const float* w_ptr = &string_to_fluid_proj_[i * num_strings_];
            for (size_t m = 0; m < num_strings_; ++m) {
                sum += w_ptr[m] * u_string_[m];
            }
            fluid_injection[i] = sum * 0.10f;
        }

        // 4. Hydrodynamic Vortex Advection & Toroidal Kelvin Circulation
        const int N = static_cast<int>(VORTEX_GRID_SIZE);
        std::vector<float> next_u(VORTEX_CELLS, 0.0f);
        std::vector<float> next_v(VORTEX_CELLS, 0.0f);

        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int idx = y * N + x;
                int right = y * N + ((x + 1) % N);
                int left  = y * N + ((x - 1 + N) % N);
                int up    = ((y - 1 + N) % N) * N + x;
                int down  = ((y + 1) % N) * N + x;

                float du_dx = (u_fluid_[right] - u_fluid_[left]) * 0.5f;
                float du_dy = (u_fluid_[down] - u_fluid_[up]) * 0.5f;
                float dv_dx = (v_fluid_[right] - v_fluid_[left]) * 0.5f;
                float dv_dy = (v_fluid_[down] - v_fluid_[up]) * 0.5f;

                float w = dv_dx - du_dy;
                vorticity_[idx] = w;

                float lap_u = u_fluid_[right] + u_fluid_[left] + u_fluid_[up] + u_fluid_[down] - 4.0f * u_fluid_[idx];
                float lap_v = v_fluid_[right] + v_fluid_[left] + v_fluid_[up] + v_fluid_[down] - 4.0f * v_fluid_[idx];

                float u_adv = -(u_fluid_[idx] * du_dx + v_fluid_[idx] * du_dy) + 0.015f * lap_u + fluid_injection[idx];
                float v_adv = -(u_fluid_[idx] * dv_dx + v_fluid_[idx] * dv_dy) + 0.015f * lap_v + fluid_injection[idx];

                next_u[idx] = std::tanh(u_fluid_[idx] + u_adv * dt_eff_);
                next_v[idx] = std::tanh(v_fluid_[idx] + v_adv * dt_eff_);
            }
        }
        u_fluid_ = std::move(next_u);
        v_fluid_ = std::move(next_v);

        // 5. Back-Reaction: Fluid Vorticity Modulates String Tension
        // Creating Topological Knots (Solitons)
        for (size_t m = 0; m < num_strings_; ++m) {
            size_t cell_idx = (m * 7) % VORTEX_CELLS;
            float local_vorticity = vorticity_[cell_idx];
            tensions_[m] = std::clamp(tensions_[m] + 0.005f * local_vorticity, 0.2f, 5.0f);
        }

        // 6. Unified Topological Readout (Strings + Vortex Knots)
        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            // String harmonic resonance
            float dot_str = avx2_dot_product(&motor_readout_[c * num_strings_], u_string_.data(), num_strings_);
            float norm_str = dot_str / (std::sqrt(static_cast<float>(num_strings_)) + SAFETY_EPSILON);

            // Vortex circulation resonance
            float dot_vor = avx2_dot_product(&vortex_readout_[c * VORTEX_CELLS], vorticity_.data(), VORTEX_CELLS);
            float norm_vor = dot_vor / (std::sqrt(static_cast<float>(VORTEX_CELLS)) + SAFETY_EPSILON);

            // Topological composite resonance
            float eff_res = (norm_str * 0.70f + norm_vor * 0.30f) - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // 7. Endogenous Inverse Temperature
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = resonance[c] - max_res;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(1.5f + 2.0f * std::sqrt(variance), 1.0f, 20.0f);

        // Softmax
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        // Volitional selection
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

        // 1. Causal Credit & Epigenetic Lock for strings
        for (size_t m = 0; m < num_strings_; ++m) {
            float energy = u_string_[m] * u_string_[m] + v_string_[m] * v_string_[m];
            float credit = (-delta_FE) * energy * 100.0f;
            causal_credit_[m] = 0.95f * causal_credit_[m] + 0.05f * credit;
            epigenetic_lock_[m] = std::clamp(1.0f / (1.0f + std::exp(-5.0f * (causal_credit_[m] - 0.01f))), 0.0f, 1.0f);
        }

        // 2. Dual Plastic Readout Updates
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t m = 0; m < num_strings_; ++m) {
                    float plastic = 1.0f - 0.90f * epigenetic_lock_[m];
                    motor_readout_[c * num_strings_ + m] += alpha * plastic * u_string_[m];
                }
                for (size_t i = 0; i < VORTEX_CELLS; ++i) {
                    vortex_readout_[c * VORTEX_CELLS + i] += alpha * 0.5f * vorticity_[i];
                }
            }
        }

        // 3. Autonomous Morphogenesis (every 25k steps)
        if (total_steps_ % 25000 == 0 && num_strings_ < 512) {
            morph_strings();
        }
    }

    void morph_strings() noexcept {
        size_t best_parent = 0;
        float max_credit = -1e9f;
        for (size_t m = 0; m < num_strings_; ++m) {
            if (causal_credit_[m] > max_credit) {
                max_credit = causal_credit_[m];
                best_parent = m;
            }
        }

        for (size_t m = 0; m < num_strings_; ++m) {
            if (causal_credit_[m] < -0.05f && epigenetic_lock_[m] < 0.05f) {
                u_string_[m] = 0.0f;
                v_string_[m] = 0.0f;
                tensions_[m] = tensions_[best_parent] * 1.5f;
                causal_credit_[m] = 0.0f;
                pruned_strings_++;
            }
        }

        if (max_credit > 0.01f && num_strings_ + 16 <= 512) {
            size_t sprout_count = 16;
            size_t old_strings = num_strings_;
            num_strings_ += sprout_count;

            u_string_.resize(num_strings_, 0.0f);
            v_string_.resize(num_strings_, 0.0f);
            tensions_.resize(num_strings_);
            causal_credit_.resize(num_strings_, 0.0f);
            epigenetic_lock_.resize(num_strings_, 0.0f);
            cos_w_.resize(num_strings_);
            sin_w_.resize(num_strings_);

            for (size_t i = 0; i < sprout_count; ++i) {
                size_t child_idx = old_strings + i;
                float ratio = static_cast<float>(i + 2) / 2.0f;
                tensions_[child_idx] = tensions_[best_parent] * ratio;
                u_string_[child_idx] = 0.01f * u_string_[best_parent];
                v_string_[child_idx] = 0.01f * v_string_[best_parent];
            }

            sensory_fourier_.resize(CHANNEL_IO_DIM * num_strings_);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                for (size_t i = 0; i < sprout_count; ++i) {
                    size_t m = old_strings + i;
                    float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>((c + 1) * (m + 1)) / static_cast<float>(num_strings_);
                    sensory_fourier_[c * num_strings_ + m] = std::sin(phase) * 0.15f;
                }
            }

            std::vector<float> new_motor(CHANNEL_IO_DIM * num_strings_, 0.0f);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                for (size_t m = 0; m < old_strings; ++m) {
                    new_motor[c * num_strings_ + m] = motor_readout_[c * old_strings + m];
                }
                for (size_t i = 0; i < sprout_count; ++i) {
                    size_t child_idx = old_strings + i;
                    new_motor[c * num_strings_ + child_idx] = 0.005f * motor_readout_[c * old_strings + best_parent];
                }
            }
            motor_readout_ = std::move(new_motor);

            // Expand string-to-fluid gauge matrix
            string_to_fluid_proj_.resize(num_strings_ * VORTEX_CELLS, 0.0f);

            update_string_frequencies();
            total_sprouts_ += sprout_count;
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }
    size_t get_strings() const noexcept { return num_strings_; }
    size_t get_sprouts() const noexcept { return total_sprouts_; }
    size_t get_pruned() const noexcept { return pruned_strings_; }

private:
    size_t num_strings_;
    size_t total_steps_;
    float free_energy_;
    float prev_free_energy_;
    float endogenous_beta_;
    float dt_eff_;
    size_t total_sprouts_;
    size_t pruned_strings_;

    std::vector<float> u_string_;
    std::vector<float> v_string_;
    std::vector<float> tensions_;
    std::vector<float> cos_w_;
    std::vector<float> sin_w_;
    std::vector<float> causal_credit_;
    std::vector<float> epigenetic_lock_;

    std::vector<float> u_fluid_;
    std::vector<float> v_fluid_;
    std::vector<float> vorticity_;
    std::vector<float> string_to_fluid_proj_;

    std::vector<float> sensory_fourier_;
    std::vector<float> motor_readout_;
    std::vector<float> vortex_readout_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-579: TOPOLOGICAL SUPERSTRING VORTEX CONTINUUM (TSFVC)        ===\n";
    std::cout << "=== The Quest for Universal Non-Biological Super-Intelligence        ===\n";
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
        std::string fallback = "The capital of France is Paris. Topological knot solitons unify mind and universe.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000;

    std::cout << ">>> Training Topological Superstring Vortex Organism (" << train_steps << " steps)...\n";
    karyon::TopologicalSuperstringVortexOrganism knot_org(256);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        knot_org.step(in_b);
        knot_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 75000 == 0) {
            std::cout << "  Step " << (step + 1) 
                      << " | Strings: " << knot_org.get_strings()
                      << " | Sprouts: " << knot_org.get_sprouts()
                      << " | Pruned: " << knot_org.get_pruned()
                      << " | Free Energy: " << knot_org.get_free_energy()
                      << " | Beta: " << knot_org.get_beta() << "\n";
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(train_steps) / duration;

    // Directed Speech Generation Test
    std::string prompt = "The capital of France is ";
    std::string speech_output = prompt;
    uint8_t curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t next_act = knot_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_knot_speech(speech_output);

    std::cout << "\n>>> TOPOLOGICAL SUPERSTRING VORTEX GENERATED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    std::cout << "========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-579 TSFVC vs EXP-578 CHAMPIONS)       ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Free Energy (nats) | Alphabet % | Whitespace % | Entropy\n";
    std::cout << "--------------------------------|--------------------|--------------------|------------|--------------|--------\n";
    std::cout << "TSFVC Knot Soliton (EXP-579)    | " << throughput << "          | " << knot_org.get_free_energy() << "             | " << (metrics.alphabet_ratio * 100.0) << "%      | " << (metrics.whitespace_ratio * 100.0) << "%        | " << metrics.char_entropy << "\n";
    std::cout << "Symplectic String (EXP-578)     | 48497.6            | 3.0875             | 74.22%     | 22.22%       | 4.13\n";
    std::cout << "Fluid Lattice (EXP-578)         | 8796.4             | 3.0950             | 76.44%     | 16.44%       | 4.55\n";
    std::cout << "Quantum Phase Space (EXP-578)   | 37884.2            | 3.4003             | 68.44%     | 22.67%       | 4.38\n";
    std::cout << "========================================================================\n";

    return 0;
}
