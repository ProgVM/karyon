// ============================================================================
// EXP-584: BOUNDED RESONANT HOMEOSTATIC SELF-ASSEMBLING CONTINUUM (BR-HSC)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 27, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
// Hypothesis:
// In EXP-583, the dual-timescale self-assembling substrate achieved a record
// Free Energy of 3.8210 nats at step 150,000, but experienced late-stage
// bifurcation at step 250,000+ due to un-normalized sensory inflow growth.
//
// In EXP-584, we implement complete 3-Gateway Oja-Prigogine Homeostatic Normalization:
// 1. Sensory Inflow Oja Decay:
//    dIn_{c, i} = lr * 0.05 * (causal_boost * z_i - 0.02 * mean_z_sq * In_{c, i})
// 2. Internal Substrate Oja Decay for A, D, and Bilinear forms.
// 3. Competitive Motor Readout Oja Decay.
//
// Expectation:
// Prevents late-stage runaway, maintaining Free Energy <= 3.5 nats perpetually
// across 300,000 steps while preserving 98%+ Hopfield D-symmetry and 0% saturation.
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
    float bilinear_norm = 0.0f;
    float mean_saturation = 0.0f;
    float sensory_norm = 0.0f;
    float motor_norm = 0.0f;
};

class BoundedResonantHomeostaticOrganism {
public:
    BoundedResonantHomeostaticOrganism() noexcept {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        endogenous_beta_ = 4.5f;
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

        std::mt19937 rng(5840);
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

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * STATE_DIM;

        // 1. Kinetic energy & Endogenous Time Dilation
        float total_kinetic = 0.0f;
        for (size_t i = 0; i < STATE_DIM; ++i) total_kinetic += dz_[i] * dz_[i];
        float mean_kinetic = total_kinetic / static_cast<float>(STATE_DIM);
        dt_eff_ = std::clamp(0.040f / (1.0f + 2.0f * mean_kinetic), 0.010f, 0.080f);

        std::vector<float> z_comp(STATE_DIM, 0.0f);
        for (size_t i = 0; i < STATE_DIM; ++i) {
            z_comp[i] = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];
            tanh_z_[i] = std::tanh(z_comp[i]);
        }

        // 2. High-Speed AVX2 Bilinear Gate Contraction
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

        // 3. AVX2 Dynamic Integration:
        std::vector<float> next_dz(STATE_DIM, 0.0f);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < STATE_DIM; ++i) {
            float linear_flow = avx2_dot_product(&A_[i * STATE_DIM], z_comp.data(), STATE_DIM);
            float sat_flow = avx2_dot_product(&D_[i * STATE_DIM], tanh_z_.data(), STATE_DIM);
            float sensory_flow = sensory_inflow_[in_offset + i];

            float total_accel = linear_flow + sat_flow + bilinear_inflow[i] + sensory_flow;

            next_dz[i] = (dz_[i] + total_accel * dt_eff_) * 0.985f;
            z_fast_[i] = std::tanh(z_fast_[i] + next_dz[i] * dt_eff_);
            z_slow_[i] = 0.96f * z_slow_[i] + 0.04f * z_fast_[i];
        }
        dz_ = std::move(next_dz);

        // 4. Volitional Readout
        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_readout_[c * STATE_DIM], z_comp.data(), STATE_DIM);
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
        endogenous_beta_ = std::clamp(1.5f + 2.5f * std::sqrt(variance), 1.0f, 16.0f); // Bounded beta

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

        std::vector<float> z_comp(STATE_DIM, 0.0f);
        float z_sq_norm = 0.0f;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            z_comp[i] = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];
            z_sq_norm += z_comp[i] * z_comp[i];
        }
        float mean_z_sq = z_sq_norm / static_cast<float>(STATE_DIM);

        // 1. Motor Readout Oja Plasticity
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.0005f) {
                float alpha = lr * err;
                float oja_motor = 0.02f * latest_probabilities_[c] * mean_z_sq;
                for (size_t i = 0; i < STATE_DIM; ++i) {
                    size_t m_idx = c * STATE_DIM + i;
                    motor_readout_[m_idx] += alpha * z_comp[i] - oja_motor * motor_readout_[m_idx];
                }
            }
        }

        // 2. Sensory Inflow Oja Plasticity (Preventing late runaway!)
        const size_t in_offset = static_cast<size_t>(input_byte) * STATE_DIM;
        float causal_boost = (-delta_FE) * 20.0f;

        for (size_t i = 0; i < STATE_DIM; ++i) {
            size_t s_idx = in_offset + i;
            float oja_sensory = 0.02f * mean_z_sq * sensory_inflow_[s_idx];
            sensory_inflow_[s_idx] += lr * 0.05f * (causal_boost * z_comp[i] - oja_sensory);
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
                float delta_A = (causal_boost * flow_A - 0.02f * oja_decay_A);

                float plastic_A = 1.0f - lock_A_[idx];
                A_[idx] += lr * 0.1f * plastic_A * delta_A;

                float flow_D = z_i * tanh_z_j;
                float oja_decay_D = mean_z_sq * D_[idx];
                float delta_D = (causal_boost * flow_D - 0.02f * oja_decay_D);

                float plastic_D = 1.0f - lock_D_[idx];
                D_[idx] += lr * 0.1f * plastic_D * delta_D;

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
                W_gate_[i * GATE_RANK + r] += lr * 0.05f * (causal_boost * z_comp[i] - 0.02f * oja_W);

                float oja_U = mean_z_sq * U_gate_[r * STATE_DIM + i];
                U_gate_[r * STATE_DIM + i] += lr * 0.05f * (causal_boost * z_comp[i] - 0.02f * oja_U);

                float oja_V = mean_z_sq * V_gate_[r * STATE_DIM + i];
                V_gate_[r * STATE_DIM + i] += lr * 0.05f * (causal_boost * z_comp[i] - 0.02f * oja_V);
            }
        }
    }

    ArchitectureProbe probe_architecture() const noexcept {
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

        float w_gate_sum = 0.0f;
        for (float w : W_gate_) w_gate_sum += w * w;
        probe.bilinear_norm = std::sqrt(w_gate_sum) / static_cast<float>(GATE_RANK);

        size_t sat_count = 0;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            float comp = 0.70f * z_fast_[i] + 0.30f * z_slow_[i];
            if (std::abs(comp) > 0.90f) sat_count++;
        }
        probe.mean_saturation = static_cast<float>(sat_count) / static_cast<float>(STATE_DIM);

        float s_sum = 0.0f;
        for (float w : sensory_inflow_) s_sum += w * w;
        probe.sensory_norm = std::sqrt(s_sum) / static_cast<float>(CHANNEL_IO_DIM);

        float m_sum = 0.0f;
        for (float w : motor_readout_) m_sum += w * w;
        probe.motor_norm = std::sqrt(m_sum) / static_cast<float>(CHANNEL_IO_DIM);

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
    std::cout << "=== EXP-584: BOUNDED RESONANT HOMEOSTATIC CONTINUUM (BR-HSC)         ===\n";
    std::cout << "=== Bazilevs Sovereign Autopoiesis: 3-Gateway Oja Stability          ===\n";
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
        std::string fallback = "The capital of France is Paris. Self-assembling intelligence stabilizes via complete Oja homeostatic balance.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000;

    std::cout << ">>> Initializing 3-Gateway Bounded Homeostatic Organism...\n";
    karyon::BoundedResonantHomeostaticOrganism br_org;

    auto p0 = br_org.probe_architecture();
    std::cout << "--- Initial Broth Metrics ---\n";
    std::cout << "  A Antisymmetry : " << (p0.antisymmetry_ratio_A * 100.0f) << "%\n";
    std::cout << "  D Symmetry     : " << (p0.symmetry_ratio_D * 100.0f) << "%\n";
    std::cout << "  Sensory Norm   : " << p0.sensory_norm << "\n\n";

    std::cout << ">>> Training and Crystallizing Substrate (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        br_org.step(in_b);
        br_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 75000 == 0) {
            auto p = br_org.probe_architecture();
            std::cout << "  Step " << (step + 1)
                      << " | Free Energy: " << br_org.get_free_energy()
                      << " | Beta: " << br_org.get_beta()
                      << " | dt_eff: " << br_org.get_dt_eff()
                      << " | A_Anti: " << (p.antisymmetry_ratio_A * 100.0f) << "%"
                      << " | D_Sym: " << (p.symmetry_ratio_D * 100.0f) << "%"
                      << " | SensoryNorm: " << p.sensory_norm << "\n";
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
        size_t next_act = br_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);
    auto p_final = br_org.probe_architecture();

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-584 vs EXP-583 vs EXP-582)             ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | FE @ 150k | Alphabet % | Saturation\n";
    std::cout << "--------------------------------|--------------------|-----------------|-----------|------------|-----------\n";
    std::cout << "BR-HSC 3-Gateway Oja (EXP-584)  | " << throughput << "          | " << br_org.get_free_energy() << "          | --        | " << (metrics.alphabet_ratio * 100.0) << "%      | " << (p_final.mean_saturation * 100.0f) << "%\n";
    std::cout << "RH-SAC Unbounded Inflow(EXP-583)| 3241.72            | 12.1355         | 3.8210    | 27.11%     | 0.00%\n";
    std::cout << "NH-SAPS Homeostatic (EXP-582)   | 3312.51            | 6.1354          | 4.0454    | 36.00%     | 0.00%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
