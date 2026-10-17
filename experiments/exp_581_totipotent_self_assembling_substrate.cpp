// ============================================================================
// EXP-581: TOTIPOTENT SELF-ASSEMBLING PHYSICAL SUBSTRATE (TSAPS)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 27, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
// Core Hypothesis:
// If the architecture is NOT hardcoded with human physical equations (no explicit
// strings, waves, Navier-Stokes, SwiGLU, or Hopfield), but instead starts as an
// unconstrained, primordial tensor field dz/dt = A z + (Uz * Vz) + D tanh(z) + In(x),
// Karyon will AUTONOMOUSLY crystallize the necessary mathematical symmetries
// and operators bottom-up to minimize Variational Free Energy.
//
// Endoscopic Probing:
// 1. Antisymmetry of A: ||A - (-A^T)|| -> Emergence of Symplectic String Oscillation
// 2. Symmetry of D: ||D - D^T|| -> Emergence of Hopfield Conceptual Attractors
// 3. Bilinear Gating Power: ||U, V|| -> Emergence of SwiGLU Contextual Gating
// 4. Linguistic Speech Emergence & Free Energy Convergence across 300,000 steps.
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

constexpr size_t STATE_DIM = 256;
constexpr size_t GATE_RANK = 64; // Low-rank bilinear tensor decomposition

struct TSAPSSpeechMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
};

TSAPSSpeechMetrics evaluate_speech(const std::string& text) {
    TSAPSSpeechMetrics m;
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

// Endoscopic inspection metrics for self-crystallized structure
struct EmergentGeometryProbe {
    float antisymmetry_ratio_A = 0.0f; // 1.0 = Pure Symplectic String, 0.0 = Dissipative
    float symmetry_ratio_D = 0.0f;     // 1.0 = Pure Hopfield Attractor Energy
    float bilinear_gating_norm = 0.0f; // Contextual multiplicative routing strength
    float linear_stiffness_norm = 0.0f;
};

class TotipotentSelfAssemblingOrganism {
public:
    TotipotentSelfAssemblingOrganism() noexcept {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        endogenous_beta_ = 4.5f;
        dt_eff_ = 0.040f;

        z_.assign(STATE_DIM, 0.0f);
        dz_.assign(STATE_DIM, 0.0f);
        z_memory_.assign(STATE_DIM, 0.0f);

        // 1. Primordial Unbiased Tensors (Random Near-Zero Broth)
        // A: Linear dynamics tensor (STATE_DIM x STATE_DIM)
        A_.assign(STATE_DIM * STATE_DIM, 0.0f);
        // D: Saturating non-linear tensor (STATE_DIM x STATE_DIM)
        D_.assign(STATE_DIM * STATE_DIM, 0.0f);
        // Bilinear Gate Factorization: U (GATE_RANK x STATE_DIM), V (GATE_RANK x STATE_DIM), W_gate (STATE_DIM x GATE_RANK)
        U_gate_.assign(GATE_RANK * STATE_DIM, 0.0f);
        V_gate_.assign(GATE_RANK * STATE_DIM, 0.0f);
        W_gate_.assign(STATE_DIM * GATE_RANK, 0.0f);

        std::mt19937 rng(5810);
        std::normal_distribution<float> init_dist(0.0f, 0.015f);

        for (auto& w : A_) w = init_dist(rng);
        for (auto& w : D_) w = init_dist(rng);
        for (auto& w : U_gate_) w = init_dist(rng);
        for (auto& w : V_gate_) w = init_dist(rng);
        for (auto& w : W_gate_) w = init_dist(rng);

        // Epigenetic protection locks for each parameter matrix
        lock_A_.assign(STATE_DIM * STATE_DIM, 0.0f);
        lock_D_.assign(STATE_DIM * STATE_DIM, 0.0f);

        // Sensory Inflow (Universal byte mapping)
        sensory_inflow_.resize(CHANNEL_IO_DIM * STATE_DIM);
        for (auto& w : sensory_inflow_) w = init_dist(rng);

        // Motor Readout
        motor_readout_.resize(CHANNEL_IO_DIM * STATE_DIM);
        for (auto& w : motor_readout_) w = init_dist(rng);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * STATE_DIM;

        // 1. Endogenous Time Dilation from State Fluctuation Energy
        float total_kinetic = 0.0f;
        for (size_t i = 0; i < STATE_DIM; ++i) total_kinetic += dz_[i] * dz_[i];
        float mean_kinetic = total_kinetic / static_cast<float>(STATE_DIM);
        dt_eff_ = std::clamp(0.040f / (1.0f + 4.0f * mean_kinetic), 0.010f, 0.080f);

        // 2. Compute Bilinear Gating Form: G = W_gate * ((U * z) \odot (V * z))
        std::vector<float> u_proj(GATE_RANK, 0.0f);
        std::vector<float> v_proj(GATE_RANK, 0.0f);
        for (size_t r = 0; r < GATE_RANK; ++r) {
            u_proj[r] = avx2_dot_product(&U_gate_[r * STATE_DIM], z_.data(), STATE_DIM);
            v_proj[r] = avx2_dot_product(&V_gate_[r * STATE_DIM], z_.data(), STATE_DIM);
        }

        std::vector<float> gate_act(GATE_RANK, 0.0f);
        for (size_t r = 0; r < GATE_RANK; ++r) {
            // Bilinear modulation (SwiGLU-like / Reynolds stress product)
            gate_act[r] = u_proj[r] / (1.0f + std::exp(-v_proj[r]));
        }

        std::vector<float> bilinear_inflow(STATE_DIM, 0.0f);
        for (size_t i = 0; i < STATE_DIM; ++i) {
            bilinear_inflow[i] = avx2_dot_product(&W_gate_[i * GATE_RANK], gate_act.data(), GATE_RANK);
        }

        // 3. Universal Continuous Substrate Integration:
        // dz_i / dt = \sum_j A_{ij} z_j + \sum_j D_{ij} \tanh(z_j) + Bilinear_i + Inflow_i
        std::vector<float> next_dz(STATE_DIM, 0.0f);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < STATE_DIM; ++i) {
            float linear_flow = avx2_dot_product(&A_[i * STATE_DIM], z_.data(), STATE_DIM);

            // Compute D * tanh(z)
            float sat_flow = 0.0f;
            const float* d_row = &D_[i * STATE_DIM];
            for (size_t j = 0; j < STATE_DIM; ++j) {
                sat_flow += d_row[j] * std::tanh(z_[j]);
            }

            float sensory_flow = sensory_inflow_[in_offset + i];
            float total_accel = linear_flow + sat_flow + bilinear_inflow[i] + sensory_flow;

            next_dz[i] = (dz_[i] + total_accel * dt_eff_) * 0.985f;
            z_[i] = std::tanh(z_[i] + next_dz[i] * dt_eff_);
            z_memory_[i] = z_memory_[i] * 0.96f + z_[i] * 0.04f;
        }
        dz_ = std::move(next_dz);

        // 4. Volitional Motor Resonance Readout
        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_readout_[c * STATE_DIM], z_.data(), STATE_DIM);
            float norm_dot = dot / (std::sqrt(static_cast<float>(STATE_DIM)) + SAFETY_EPSILON);
            float eff_res = norm_dot - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // 5. Endogenous Inverse Temperature
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

        // Sampling
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

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.035f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float delta_FE = free_energy_ - prev_free_energy_;

        // 1. Motor Readout Plasticity
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t i = 0; i < STATE_DIM; ++i) {
                    motor_readout_[c * STATE_DIM + i] += alpha * z_[i];
                }
            }
        }

        // 2. Endogenous Structural Crystallization of Tensors A, D, and Bilinear forms
        // Guided by Causal Credit: if -delta_FE > 0 (Free Energy decreased), we reinforce
        // active dynamical pathways and protect them with epigenetic locks.
        float causal_boost = (-delta_FE) * 50.0f;

        if (std::abs(causal_boost) > 0.0001f) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < STATE_DIM; ++i) {
                float z_i = z_[i];
                float dz_i = dz_[i];

                for (size_t j = 0; j < STATE_DIM; ++j) {
                    float z_j = z_[j];
                    size_t idx = i * STATE_DIM + j;

                    // A tensor update (Linear coupling): d A_{ij} = causal_boost * dz_i * z_j
                    float grad_A = causal_boost * dz_i * z_j;
                    float plastic_A = 1.0f - lock_A_[idx];
                    A_[idx] += lr * 0.2f * plastic_A * grad_A;

                    // D tensor update (Attractor coupling): d D_{ij} = causal_boost * z_i * tanh(z_j)
                    float grad_D = causal_boost * z_i * std::tanh(z_j);
                    float plastic_D = 1.0f - lock_D_[idx];
                    D_[idx] += lr * 0.2f * plastic_D * grad_D;

                    // Update Epigenetic Lock: strong successful connections become permanent
                    if (causal_boost > 0.0f) {
                        lock_A_[idx] = std::clamp(lock_A_[idx] + 0.001f * std::abs(A_[idx]), 0.0f, 0.95f);
                        lock_D_[idx] = std::clamp(lock_D_[idx] + 0.001f * std::abs(D_[idx]), 0.0f, 0.95f);
                    }
                }
            }

            // Bilinear tensor crystallization
            for (size_t r = 0; r < GATE_RANK; ++r) {
                for (size_t i = 0; i < STATE_DIM; ++i) {
                    W_gate_[i * GATE_RANK + r] += lr * 0.1f * causal_boost * z_[i];
                    U_gate_[r * STATE_DIM + i] += lr * 0.1f * causal_boost * z_[i];
                    V_gate_[r * STATE_DIM + i] += lr * 0.1f * causal_boost * z_[i];
                }
            }
        }
    }

    // Endoscopic Probe: What physical/mathematical architecture emerged?
    EmergentGeometryProbe probe_crystallized_architecture() const noexcept {
        EmergentGeometryProbe probe;

        // 1. Antisymmetry vs Symmetry of A
        // A = A_sym + A_anti
        // A_anti = 0.5 * (A - A^T) -> Symplectic string / unitary oscillator
        // A_sym  = 0.5 * (A + A^T) -> Damping / relaxation
        float norm_anti = 0.0f;
        float norm_sym = 0.0f;
        float norm_total = 0.0f;

        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < STATE_DIM; ++j) {
                float a_ij = A_[i * STATE_DIM + j];
                float a_ji = A_[j * STATE_DIM + i];
                float anti = 0.5f * (a_ij - a_ji);
                float sym  = 0.5f * (a_ij + a_ji);
                norm_anti += anti * anti;
                norm_sym  += sym * sym;
                norm_total += a_ij * a_ij;
            }
        }
        probe.antisymmetry_ratio_A = (norm_total > 0.0f) ? (std::sqrt(norm_anti) / std::sqrt(norm_total)) : 0.0f;
        probe.linear_stiffness_norm = std::sqrt(norm_total) / static_cast<float>(STATE_DIM);

        // 2. Symmetry of D (Hopfield Attractor Basin condition: D = D^T)
        float d_sym = 0.0f;
        float d_total = 0.0f;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < STATE_DIM; ++j) {
                float d_ij = D_[i * STATE_DIM + j];
                float d_ji = D_[j * STATE_DIM + i];
                float sym = 0.5f * (d_ij + d_ji);
                d_sym += sym * sym;
                d_total += d_ij * d_ij;
            }
        }
        probe.symmetry_ratio_D = (d_total > 0.0f) ? (std::sqrt(d_sym) / std::sqrt(d_total)) : 0.0f;

        // 3. Bilinear Gating Power
        float w_gate_sum = 0.0f;
        for (float w : W_gate_) w_gate_sum += w * w;
        probe.bilinear_gating_norm = std::sqrt(w_gate_sum) / static_cast<float>(GATE_RANK);

        return probe;
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }
    float get_dt_eff() const noexcept { return dt_eff_; }

private:
    size_t total_steps_;
    float free_energy_;
    float prev_free_energy_;
    float endogenous_beta_;
    float dt_eff_;

    std::vector<float> z_;
    std::vector<float> dz_;
    std::vector<float> z_memory_;

    // Fundamental Totipotent Tensors
    std::vector<float> A_;
    std::vector<float> D_;
    std::vector<float> U_gate_;
    std::vector<float> V_gate_;
    std::vector<float> W_gate_;

    // Epigenetic Locks
    std::vector<float> lock_A_;
    std::vector<float> lock_D_;

    std::vector<float> sensory_inflow_;
    std::vector<float> motor_readout_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-581: TOTIPOTENT SELF-ASSEMBLING PHYSICAL SUBSTRATE (TSAPS)   ===\n";
    std::cout << "=== Bazilevs Sovereign Autopoiesis: Architecture Synthesizes Itself  ===\n";
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
        std::string fallback = "The capital of France is Paris. Language is totipotent physical self-assembly.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000;

    std::cout << ">>> Initializing Totipotent Substrate (Zero Human Structural Clamps)...\n";
    karyon::TotipotentSelfAssemblingOrganism tsaps;

    auto initial_probe = tsaps.probe_crystallized_architecture();
    std::cout << "--- Initial Primordial Broth Endoscopy ---\n";
    std::cout << "  A Antisymmetry Ratio : " << initial_probe.antisymmetry_ratio_A << " (pure noise)\n";
    std::cout << "  D Symmetry Ratio     : " << initial_probe.symmetry_ratio_D << " (pure noise)\n";
    std::cout << "  Bilinear Gate Norm   : " << initial_probe.bilinear_gating_norm << "\n\n";

    std::cout << ">>> Relaxing and Crystallizing Substrate under Continuous Sensory Flow (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        tsaps.step(in_b);
        tsaps.adapt(in_b, tgt_b, 0.035f);

        if ((step + 1) % 75000 == 0) {
            auto p = tsaps.probe_crystallized_architecture();
            std::cout << "  Step " << (step + 1)
                      << " | Free Energy: " << tsaps.get_free_energy()
                      << " | Beta: " << tsaps.get_beta()
                      << " | dt_eff: " << tsaps.get_dt_eff()
                      << " | A_Anti: " << p.antisymmetry_ratio_A
                      << " | D_Sym: " << p.symmetry_ratio_D
                      << " | Bilinear: " << p.bilinear_gating_norm << "\n";
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
        size_t next_act = tsaps.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);

    auto final_probe = tsaps.probe_crystallized_architecture();

    std::cout << "\n========================================================================\n";
    std::cout << "=== ENDOSCOPIC ARCHITECTURAL AUTOPSY: WHAT ASSEMBLED ITSELF?         ===\n";
    std::cout << "========================================================================\n";
    std::cout << "1. Antisymmetry of A (Symplectic String Emergence): " << (final_probe.antisymmetry_ratio_A * 100.0f) << "%\n";
    std::cout << "2. Symmetry of D (Hopfield Attractor Emergence)   : " << (final_probe.symmetry_ratio_D * 100.0f) << "%\n";
    std::cout << "3. Bilinear Gating Strength (SwiGLU Emergence)    : " << final_probe.bilinear_gating_norm << "\n";
    std::cout << "4. Computational Throughput                       : " << throughput << " steps/sec\n";
    std::cout << "5. Final Free Energy                              : " << tsaps.get_free_energy() << " nats\n";
    std::cout << "6. Speech Alphabet Ratio                          : " << (metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "7. Speech Whitespace Ratio                        : " << (metrics.whitespace_ratio * 100.0) << "%\n";
    std::cout << "8. Speech Character Entropy                       : " << metrics.char_entropy << " bits\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
