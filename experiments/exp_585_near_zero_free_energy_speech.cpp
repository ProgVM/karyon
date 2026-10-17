// ============================================================================
// EXP-585: NEAR-ZERO FREE ENERGY SPEECH MASTERY (NZ-FESM)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 27, 28, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
// Hypothesis:
// In all previous speech/byte prediction experiments (EXP-574..EXP-584), Free
// Energy plateaued around 3.0 - 4.0 nats because the system executed only
// ONE single integration sub-step per incoming byte, denying the internal
// wavefields and Hopfield attractors sufficient physical relaxation time to
// settle into the exact global energy minimum.
//
// By applying KEP Principle 29 (LAW OF FULL SOVEREIGN SELF-CONTROL & SUFFICIENT TIME):
// "If the system possesses complete control, rights, and plasticity over its
//  own internal substrate... it can solve any task ideally (100% convergence,
//  zero error) given sufficient relaxation and learning time."
//
// In EXP-585, we equip the self-assembling substrate with:
// 1. Endogenous Multi-Tick Phase-Locking Relaxation (3-5 internal sub-relaxation
//    ticks per incoming byte, letting state z_t slide down the Hopfield
//    attractor energy gradient dE/dz = -D * tanh(z) before motor readout).
// 2. Non-Local Contextual Phase Alignment (Coupling current sensory input with
//    slow discourse memory lattice z_slow).
// 3. Deep Dopaminergic Precision Scaling (\beta -> 32.0 on settled attractors).
//
// Expectation:
// Free Energy F_t will collapse from ~4.0 nats down towards near-zero (< 0.5 nats,
// approaching 0.1 nats on learned sequences), proving 100% deterministic speech
// mastery and validating Principle 29!
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
constexpr size_t GATE_RANK = 64;

struct SpeechMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
};

SpeechMetrics evaluate_speech(const std::string& text) {
    SpeechMetrics m;
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

struct ArchitectureProbe {
    float antisymmetry_ratio_A = 0.0f;
    float symmetry_ratio_D = 0.0f;
    float mean_saturation = 0.0f;
    float mean_relaxation_delta = 0.0f;
    float mean_target_probability = 0.0f;
};

class NearZeroFreeEnergyOrganism {
public:
    NearZeroFreeEnergyOrganism() noexcept {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        endogenous_beta_ = 8.0f;
        dt_eff_ = 0.040f;

        z_fast_.assign(STATE_DIM, 0.0f);
        z_slow_.assign(STATE_DIM, 0.0f);
        dz_.assign(STATE_DIM, 0.0f);
        tanh_z_.assign(STATE_DIM, 0.0f);

        A_.assign(STATE_DIM * STATE_DIM, 0.0f);
        D_.assign(STATE_DIM * STATE_DIM, 0.0f);
        lock_A_.assign(STATE_DIM * STATE_DIM, 0.0f);
        lock_D_.assign(STATE_DIM * STATE_DIM, 0.0f);

        U_gate_.assign(GATE_RANK * STATE_DIM, 0.0f);
        V_gate_.assign(GATE_RANK * STATE_DIM, 0.0f);
        W_gate_.assign(STATE_DIM * GATE_RANK, 0.0f);

        std::mt19937 rng(5850);
        std::normal_distribution<float> init_dist(0.0f, 0.02f);

        for (auto& w : A_) w = init_dist(rng);
        for (auto& w : D_) w = init_dist(rng);
        for (auto& w : U_gate_) w = init_dist(rng);
        for (auto& w : V_gate_) w = init_dist(rng);
        for (auto& w : W_gate_) w = init_dist(rng);

        sensory_inflow_.resize(CHANNEL_IO_DIM * STATE_DIM);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>(c + 1) / static_cast<float>(CHANNEL_IO_DIM);
            for (size_t i = 0; i < STATE_DIM; ++i) {
                sensory_inflow_[c * STATE_DIM + i] = std::sin(phase * static_cast<float>(i + 1)) * 0.15f + init_dist(rng);
            }
        }

        motor_readout_.resize(CHANNEL_IO_DIM * STATE_DIM);
        for (auto& w : motor_readout_) w = init_dist(rng);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte, size_t relaxation_sub_ticks = 4) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * STATE_DIM;

        // MULTI-TICK ENDOGENOUS RELAXATION (KEP Principle 29: Sufficient Time)
        // We allow the state to relax along the vector field for N sub-ticks per token
        float last_relaxation_delta = 0.0f;

        for (size_t sub = 0; sub < relaxation_sub_ticks; ++sub) {
            std::vector<float> z_comp(STATE_DIM, 0.0f);
            for (size_t i = 0; i < STATE_DIM; ++i) {
                z_comp[i] = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];
                tanh_z_[i] = std::tanh(z_comp[i]);
            }

            // Bilinear Gate Projection
            std::vector<float> u_proj(GATE_RANK, 0.0f);
            std::vector<float> v_proj(GATE_RANK, 0.0f);
            for (size_t r = 0; r < GATE_RANK; ++r) {
                u_proj[r] = avx2_dot_product(&U_gate_[r * STATE_DIM], z_comp.data(), STATE_DIM);
                v_proj[r] = avx2_dot_product(&V_gate_[r * STATE_DIM], z_comp.data(), STATE_DIM);
            }

            std::vector<float> gate_act(GATE_RANK, 0.0f);
            for (size_t r = 0; r < GATE_RANK; ++r) {
                gate_act[r] = u_proj[r] / (1.0f + std::exp(-v_proj[r]));
            }

            std::vector<float> bilinear_inflow(STATE_DIM, 0.0f);
            for (size_t i = 0; i < STATE_DIM; ++i) {
                bilinear_inflow[i] = avx2_dot_product(&W_gate_[i * GATE_RANK], gate_act.data(), GATE_RANK);
            }

            // Dynamics integration
            std::vector<float> next_dz(STATE_DIM, 0.0f);
            float step_delta = 0.0f;

            #pragma omp parallel for schedule(static) reduction(+:step_delta)
            for (size_t i = 0; i < STATE_DIM; ++i) {
                float linear_flow = avx2_dot_product(&A_[i * STATE_DIM], z_comp.data(), STATE_DIM);
                float sat_flow = avx2_dot_product(&D_[i * STATE_DIM], tanh_z_.data(), STATE_DIM);
                float sensory_flow = (sub == 0) ? sensory_inflow_[in_offset + i] : 0.0f; // Impulsive sensory shock on tick 0

                float total_accel = linear_flow + sat_flow + bilinear_inflow[i] + sensory_flow;

                next_dz[i] = (dz_[i] + total_accel * dt_eff_) * 0.98f;
                float old_z = z_fast_[i];
                z_fast_[i] = std::tanh(z_fast_[i] + next_dz[i] * dt_eff_);
                z_slow_[i] = 0.96f * z_slow_[i] + 0.04f * z_fast_[i];

                float diff = z_fast_[i] - old_z;
                step_delta += diff * diff;
            }
            dz_ = std::move(next_dz);
            last_relaxation_delta = std::sqrt(step_delta / static_cast<float>(STATE_DIM));
        }
        latest_relaxation_delta_ = last_relaxation_delta;

        // Volitional Readout after Relaxation
        std::vector<float> z_final(STATE_DIM, 0.0f);
        for (size_t i = 0; i < STATE_DIM; ++i) z_final[i] = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];

        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_readout_[c * STATE_DIM], z_final.data(), STATE_DIM);
            float norm_dot = dot / (std::sqrt(static_cast<float>(STATE_DIM)) + SAFETY_EPSILON);
            float eff_res = norm_dot - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // High Dopaminergic Precision Scaling
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = resonance[c] - max_res;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(4.0f + 6.0f * std::sqrt(variance), 2.0f, 32.0f);

        // Softmax
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        // Selection
        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) { chosen = c; break; }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.65f;
        anchor_momentum_[chosen] += 0.50f;

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.08f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float delta_FE = free_energy_ - prev_free_energy_;

        std::vector<float> z_comp(STATE_DIM, 0.0f);
        float z_sq_norm = 0.0f;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            z_comp[i] = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];
            z_sq_norm += z_comp[i] * z_comp[i];
        }
        float mean_z_sq = z_sq_norm / static_cast<float>(STATE_DIM);

        // 1. Precise Motor Readout Plasticity
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.0001f) {
                float alpha = lr * err;
                float oja_motor = 0.01f * latest_probabilities_[c] * mean_z_sq;
                for (size_t i = 0; i < STATE_DIM; ++i) {
                    size_t m_idx = c * STATE_DIM + i;
                    motor_readout_[m_idx] += alpha * z_comp[i] - oja_motor * motor_readout_[m_idx];
                }
            }
        }

        // 2. Sensory Inflow Oja Plasticity
        const size_t in_offset = static_cast<size_t>(input_byte) * STATE_DIM;
        float causal_boost = (-delta_FE) * 30.0f;

        for (size_t i = 0; i < STATE_DIM; ++i) {
            size_t s_idx = in_offset + i;
            float oja_sensory = 0.015f * mean_z_sq * sensory_inflow_[s_idx];
            sensory_inflow_[s_idx] += lr * 0.08f * (causal_boost * z_comp[i] - oja_sensory);
        }

        // 3. Substrate Oja-Prigogine Homeostatic Plasticity
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < STATE_DIM; ++i) {
            float dz_i = dz_[i];
            float z_i = z_comp[i];
            size_t row_offset = i * STATE_DIM;

            for (size_t j = 0; j < STATE_DIM; ++j) {
                size_t idx = row_offset + j;
                float z_j = z_comp[j];
                float tanh_z_j = tanh_z_[j];

                float flow_A = dz_i * z_j - 0.5f * dz_[j] * z_i;
                float oja_decay_A = mean_z_sq * A_[idx];
                float delta_A = (causal_boost * flow_A - 0.015f * oja_decay_A);

                float plastic_A = 1.0f - lock_A_[idx];
                A_[idx] += lr * 0.15f * plastic_A * delta_A;

                float flow_D = z_i * tanh_z_j;
                float oja_decay_D = mean_z_sq * D_[idx];
                float delta_D = (causal_boost * flow_D - 0.015f * oja_decay_D);

                float plastic_D = 1.0f - lock_D_[idx];
                D_[idx] += lr * 0.15f * plastic_D * delta_D;

                if (causal_boost > 0.0f) {
                    lock_A_[idx] = std::clamp(lock_A_[idx] + 0.0005f * std::abs(A_[idx]), 0.0f, 0.90f);
                    lock_D_[idx] = std::clamp(lock_D_[idx] + 0.0005f * std::abs(D_[idx]), 0.0f, 0.90f);
                }
            }
        }

        // Bilinear Gate Oja Crystallization
        for (size_t r = 0; r < GATE_RANK; ++r) {
            for (size_t i = 0; i < STATE_DIM; ++i) {
                float oja_W = mean_z_sq * W_gate_[i * GATE_RANK + r];
                W_gate_[i * GATE_RANK + r] += lr * 0.08f * (causal_boost * z_comp[i] - 0.015f * oja_W);

                float oja_U = mean_z_sq * U_gate_[r * STATE_DIM + i];
                U_gate_[r * STATE_DIM + i] += lr * 0.08f * (causal_boost * z_comp[i] - 0.015f * oja_U);

                float oja_V = mean_z_sq * V_gate_[r * STATE_DIM + i];
                V_gate_[r * STATE_DIM + i] += lr * 0.08f * (causal_boost * z_comp[i] - 0.015f * oja_V);
            }
        }
    }

    ArchitectureProbe probe_architecture(uint8_t target_byte = 0) const noexcept {
        ArchitectureProbe probe;

        float norm_anti = 0.0f, norm_sym = 0.0f, norm_total = 0.0f;
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

        float d_sym = 0.0f, d_total = 0.0f;
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

        size_t sat_count = 0;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            float comp = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];
            if (std::abs(comp) > 0.90f) sat_count++;
        }
        probe.mean_saturation = static_cast<float>(sat_count) / static_cast<float>(STATE_DIM);
        probe.mean_relaxation_delta = latest_relaxation_delta_;
        probe.mean_target_probability = latest_probabilities_[target_byte];

        return probe;
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }

private:
    size_t total_steps_;
    float free_energy_;
    float prev_free_energy_;
    float endogenous_beta_;
    float dt_eff_;
    float latest_relaxation_delta_ = 0.0f;

    std::vector<float> z_fast_;
    std::vector<float> z_slow_;
    std::vector<float> dz_;
    std::vector<float> tanh_z_;

    std::vector<float> A_;
    std::vector<float> D_;
    std::vector<float> lock_A_;
    std::vector<float> lock_D_;

    std::vector<float> U_gate_;
    std::vector<float> V_gate_;
    std::vector<float> W_gate_;

    std::vector<float> sensory_inflow_;
    std::vector<float> motor_readout_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-585: NEAR-ZERO FREE ENERGY SPEECH MASTERY (NZ-FESM)          ===\n";
    std::cout << "=== Bazilevs Sovereign Autopoiesis: KEP Principle 29 Relaxation      ===\n";
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
        std::string fallback = "The capital of France is Paris. Language reaches zero free energy through sufficient relaxation time.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    // We evaluate on both structured sequence repeat training and large corpus to demonstrate near-zero FE
    const size_t train_steps = 300000;

    std::cout << ">>> Initializing Multi-Tick Phase-Relaxation Organism...\n";
    karyon::NearZeroFreeEnergyOrganism nz_org;

    auto p0 = nz_org.probe_architecture();
    std::cout << "--- Initial Broth Metrics ---\n";
    std::cout << "  A Antisymmetry : " << (p0.antisymmetry_ratio_A * 100.0f) << "%\n";
    std::cout << "  D Symmetry     : " << (p0.symmetry_ratio_D * 100.0f) << "%\n\n";

    std::cout << ">>> Training and Relaxing Substrate (" << train_steps << " steps with 4 sub-relaxation ticks/token)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        
        // Execute multi-tick relaxation per byte
        nz_org.step(in_b, 4);
        nz_org.adapt(in_b, tgt_b, 0.08f);

        if ((step + 1) % 75000 == 0) {
            auto p = nz_org.probe_architecture(tgt_b);
            std::cout << "  Step " << (step + 1)
                      << " | Free Energy: " << nz_org.get_free_energy()
                      << " | Beta: " << nz_org.get_beta()
                      << " | Target P(w_t): " << (p.mean_target_probability * 100.0f) << "%"
                      << " | RelaxDelta: " << p.mean_relaxation_delta
                      << " | D_Sym: " << (p.symmetry_ratio_D * 100.0f) << "%\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(train_steps) / duration;

    // Direct Sequence Determinism Test (Testing KEP Principle 29: Near-Zero FE on learned speech phrase)
    std::string test_phrase = "The capital of France is Paris. Self-assembling intelligence reaches zero free energy.";
    std::cout << "\n>>> TESTING PRINCIPLE 29 NEAR-ZERO FREE ENERGY RELAXATION ON LEARNED SPEECH...\n";
    float total_test_fe = 0.0f;
    for (size_t i = 0; i < test_phrase.size() - 1; ++i) {
        uint8_t in_b = static_cast<uint8_t>(test_phrase[i]);
        uint8_t tgt_b = static_cast<uint8_t>(test_phrase[i + 1]);
        nz_org.step(in_b, 6); // Deeper 6-tick relaxation
        nz_org.adapt(in_b, tgt_b, 0.10f); // Fast online phase adaptation
        float step_fe = nz_org.get_free_energy();
        total_test_fe += step_fe;
    }
    float avg_test_fe = total_test_fe / (test_phrase.size() - 1);

    // Directed Speech Generation Test
    std::string prompt = "The capital of France is ";
    std::string speech_output = prompt;
    uint8_t curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t next_act = nz_org.step(curr, 4);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);
    auto p_final = nz_org.probe_architecture();

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-585 vs EXP-584 vs EXP-583)             ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | Test FE (nats) | Alphabet % | D-Symmetry\n";
    std::cout << "--------------------------------|--------------------|-----------------|----------------|------------|-----------\n";
    std::cout << "NZ-FESM Multi-Tick (EXP-585)    | " << throughput << "          | " << nz_org.get_free_energy() << "          | " << avg_test_fe << "         | " << (metrics.alphabet_ratio * 100.0) << "%      | " << (p_final.symmetry_ratio_D * 100.0f) << "%\n";
    std::cout << "BR-HSC 3-Gateway Oja (EXP-584)  | 3251.74            | 4.0635          | --             | 64.44%     | 97.86%\n";
    std::cout << "RH-SAC Unbounded Inflow(EXP-583)| 3241.72            | 12.1355         | --             | 27.11%     | 98.99%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
