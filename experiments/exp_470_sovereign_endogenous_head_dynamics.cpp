// experiments/exp_470_sovereign_endogenous_head_dynamics.cpp
// EXP-470 v2: Sovereign Endogenous Homeostatic Head Dynamics (Ashby Viability & Free-Energy Dispersion)
// Eradicates all hardcoded decay grids while preventing collapse to theta = -3.0.
// Uses diversity repulsion (spectral coverage) so heads endogenously partition the time horizon.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstring>

struct TestResult {
    std::string modality;
    std::string architecture;
    float initial_loss;
    float final_loss;
    float loss_delta;
    float accuracy;
    double steps_per_sec;
    size_t total_steps;
};

class ResonanceEngineBase {
public:
    size_t dim = 258;
    size_t num_heads = 4;
    size_t dim_k = 16;
    size_t dim_v = 16;
    size_t dim_emb = 64;
    float beta_scale = 12.0f;
    float eta_mem = 0.45f;
    float mu_emb = 0.05f;

    std::vector<float> byte_emb;
    std::vector<float> S_memory;

    ResonanceEngineBase() {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        byte_emb.resize(dim * dim_emb);
        for (size_t b = 0; b < dim; ++b) {
            float* eb = byte_emb.data() + b * dim_emb;
            for (size_t d = 0; d < dim_emb; ++d) eb[d] = dist(gen);
            normalize(eb, dim_emb);
        }
        S_memory.resize(num_heads * dim_k * dim_v, 0.0f);
    }

    virtual ~ResonanceEngineBase() = default;

    virtual void reset() {
        std::fill(S_memory.begin(), S_memory.end(), 0.0f);
    }

    void normalize(float* ptr, size_t n) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) sum += ptr[i] * ptr[i];
        float norm = std::sqrt(sum);
        if (norm > 1e-6f) {
            float inv = 1.0f / norm;
            for (size_t i = 0; i < n; ++i) ptr[i] *= inv;
        }
    }

    virtual std::pair<float, bool> step(uint8_t byte_in, uint8_t byte_target) = 0;
};

// 1. Handcoded Fixed Power-Law Baseline
class FixedPowerLawEngine : public ResonanceEngineBase {
public:
    std::pair<float, bool> step(uint8_t byte_in, uint8_t byte_target) override {
        static const float alphas[4] = {0.40f, 0.75f, 0.92f, 0.99f};
        float* e_in = byte_emb.data() + byte_in * dim_emb;

        std::vector<float> k(num_heads * dim_k);
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < dim_k; ++i) k[h * dim_k + i] = e_in[h * dim_k + i];
            normalize(k.data() + h * dim_k, dim_k);
        }

        std::vector<float> h_field(dim_emb, 0.0f);
        for (size_t h = 0; h < num_heads; ++h) {
            const float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            float* hh = h_field.data() + h * dim_v;
            for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                    sum += kh[k_idx] * s_head[k_idx * dim_v + v_idx];
                }
                hh[v_idx] = sum;
            }
        }

        for (size_t i = 0; i < dim_emb; ++i) h_field[i] = 0.70f * h_field[i] + 0.30f * e_in[i];

        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim_emb; ++i) sum_sq += h_field[i] * h_field[i];
        float inv_h_norm = 1.0f / (std::sqrt(sum_sq) + 1e-4f);

        std::vector<float> logits(dim, 0.0f);
        float max_l = -1e9f;
        for (size_t b = 0; b < dim; ++b) {
            const float* eb = byte_emb.data() + b * dim_emb;
            float dot = 0.0f;
            for (size_t d = 0; d < dim_emb; ++d) dot += h_field[d] * eb[d];
            float l = beta_scale * (dot * inv_h_norm);
            logits[b] = l;
            if (l > max_l) max_l = l;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(dim);
        for (size_t b = 0; b < dim; ++b) {
            probs[b] = std::exp(logits[b] - max_l);
            sum_exp += probs[b];
        }
        float inv_sum = 1.0f / sum_exp;
        uint8_t best_byte = 0;
        float max_p = 0.0f;
        for (size_t b = 0; b < dim; ++b) {
            probs[b] *= inv_sum;
            if (probs[b] > max_p) {
                max_p = probs[b];
                best_byte = static_cast<uint8_t>(b);
            }
        }

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));
        bool is_correct = (best_byte == byte_target);

        float* e_target = byte_emb.data() + byte_target * dim_emb;
        std::vector<float> error_wave(dim_emb);
        for (size_t d = 0; d < dim_emb; ++d) error_wave[d] = e_target[d] - h_field[d];

        for (size_t h = 0; h < num_heads; ++h) {
            float alpha = alphas[h];
            float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            const float* err_h = error_wave.data() + h * dim_v;

            for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                float kval = kh[k_idx];
                float* s_row = s_head + k_idx * dim_v;
                for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                    s_row[v_idx] = alpha * s_row[v_idx] + eta_mem * (kval * err_h[v_idx]);
                }
            }
        }

        for (size_t d = 0; d < dim_emb; ++d) e_target[d] += mu_emb * h_field[d];
        normalize(e_target, dim_emb);

        return {loss, is_correct};
    }
};

// 2. Sovereign Endogenous Dynamic Head Engine v2 (Ashby Self-Organizing Spectrum)
// Dynamics:
// 1. Each head has an intrinsic timescale coordinate theta_h.
// 2. Heads repel each other in timescale space: F_repulse = sum_{j != h} sign(theta_h - theta_j) / (|theta_h - theta_j| + eps).
// 3. Free energy adaptation: When the collective surprise is high, fast heads take charge;
//    when prediction stabilizes, long-range heads consolidate the state.
class SovereignEndogenousEngineV2 : public ResonanceEngineBase {
public:
    std::vector<float> theta_heads;
    float eta_theta = 0.01f;

    SovereignEndogenousEngineV2() : ResonanceEngineBase() {
        reset();
    }

    void reset() override {
        ResonanceEngineBase::reset();
        // Symmetrically initialized across scale coordinates (-1.5 to +2.5) without hardcoding decay constants
        theta_heads.resize(num_heads);
        for (size_t h = 0; h < num_heads; ++h) {
            theta_heads[h] = -1.5f + (4.0f * h) / (num_heads - 1);
        }
    }

    std::pair<float, bool> step(uint8_t byte_in, uint8_t byte_target) override {
        float* e_in = byte_emb.data() + byte_in * dim_emb;

        std::vector<float> k(num_heads * dim_k);
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < dim_k; ++i) k[h * dim_k + i] = e_in[h * dim_k + i];
            normalize(k.data() + h * dim_k, dim_k);
        }

        std::vector<float> h_field(dim_emb, 0.0f);
        std::vector<float> head_outputs(num_heads * dim_v, 0.0f);

        for (size_t h = 0; h < num_heads; ++h) {
            const float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            float* hh = head_outputs.data() + h * dim_v;
            for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                    sum += kh[k_idx] * s_head[k_idx * dim_v + v_idx];
                }
                hh[v_idx] = sum;
                h_field[h * dim_v + v_idx] = sum;
            }
        }

        for (size_t i = 0; i < dim_emb; ++i) h_field[i] = 0.70f * h_field[i] + 0.30f * e_in[i];

        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim_emb; ++i) sum_sq += h_field[i] * h_field[i];
        float inv_h_norm = 1.0f / (std::sqrt(sum_sq) + 1e-4f);

        std::vector<float> logits(dim, 0.0f);
        float max_l = -1e9f;
        for (size_t b = 0; b < dim; ++b) {
            const float* eb = byte_emb.data() + b * dim_emb;
            float dot = 0.0f;
            for (size_t d = 0; d < dim_emb; ++d) dot += h_field[d] * eb[d];
            float l = beta_scale * (dot * inv_h_norm);
            logits[b] = l;
            if (l > max_l) max_l = l;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(dim);
        for (size_t b = 0; b < dim; ++b) {
            probs[b] = std::exp(logits[b] - max_l);
            sum_exp += probs[b];
        }
        float inv_sum = 1.0f / sum_exp;
        uint8_t best_byte = 0;
        float max_p = 0.0f;
        for (size_t b = 0; b < dim; ++b) {
            probs[b] *= inv_sum;
            if (probs[b] > max_p) {
                max_p = probs[b];
                best_byte = static_cast<uint8_t>(b);
            }
        }

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));
        bool is_correct = (best_byte == byte_target);

        float* e_target = byte_emb.data() + byte_target * dim_emb;
        std::vector<float> error_wave(dim_emb);
        for (size_t d = 0; d < dim_emb; ++d) error_wave[d] = e_target[d] - h_field[d];

        // Autonomous Multi-Scale Homeostasis
        // 1. Compute repulsion to keep timescale channels dispersed across spectrum
        std::vector<float> d_thetas(num_heads, 0.0f);
        for (size_t i = 0; i < num_heads; ++i) {
            for (size_t j = 0; j < num_heads; ++j) {
                if (i == j) continue;
                float diff = theta_heads[i] - theta_heads[j];
                float sgn = (diff > 0.0f) ? 1.0f : -1.0f;
                float dist = std::abs(diff);
                d_thetas[i] += 0.05f * sgn / (dist * dist + 0.1f);
            }
        }

        for (size_t h = 0; h < num_heads; ++h) {
            // Head-specific alignment with target byte
            const float* target_h = e_target + h * dim_v;
            const float* out_h = head_outputs.data() + h * dim_v;
            float alignment = 0.0f;
            for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                alignment += out_h[v_idx] * target_h[v_idx];
            }

            // High alignment allows deeper retention; repulse keeps channels distinct
            float update = eta_theta * (alignment * 0.1f + d_thetas[h]);
            theta_heads[h] = std::clamp(theta_heads[h] + update, -2.5f, 5.0f);

            float alpha_h = 1.0f / (1.0f + std::exp(-theta_heads[h]));
            alpha_h = std::clamp(alpha_h, 0.15f, 0.999f);

            // Matrix Memory Relaxation
            float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            const float* err_h = error_wave.data() + h * dim_v;

            for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                float kval = kh[k_idx];
                float* s_row = s_head + k_idx * dim_v;
                for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                    s_row[v_idx] = alpha_h * s_row[v_idx] + eta_mem * (kval * err_h[v_idx]);
                }
            }
        }

        // Embedding Plasticity
        for (size_t d = 0; d < dim_emb; ++d) e_target[d] += mu_emb * h_field[d];
        normalize(e_target, dim_emb);

        return {loss, is_correct};
    }
};

TestResult evaluate(ResonanceEngineBase& engine, const std::string& arch_name,
                     const std::string& mod_name, const std::vector<uint8_t>& data) {
    engine.reset();
    size_t n = data.size() - 1;
    size_t q_size = n / 4;

    float q1_loss = 0.0f;
    float q4_loss = 0.0f;
    size_t correct = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; ++i) {
        auto [loss, is_corr] = engine.step(data[i], data[i + 1]);
        if (i < q_size) q1_loss += loss;
        if (i >= n - q_size) q4_loss += loss;
        if (is_corr) correct++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    q1_loss /= q_size;
    q4_loss /= (n - (n - q_size));

    TestResult res;
    res.modality = mod_name;
    res.architecture = arch_name;
    res.initial_loss = q1_loss;
    res.final_loss = q4_loss;
    res.loss_delta = q4_loss - q1_loss;
    res.accuracy = (100.0f * correct) / n;
    res.steps_per_sec = n / (elapsed_sec + 1e-6);
    res.total_steps = n;
    return res;
}

int main() {
    std::cout << "=== EXP-470 v2: Sovereign Self-Organizing Head Dynamics vs Fixed Power-Law Baseline ===" << std::endl;

    std::ifstream f_cpp("karyon_native_core.cpp", std::ios::binary);
    std::vector<uint8_t> data_cpp((std::istreambuf_iterator<char>(f_cpp)),
                                   std::istreambuf_iterator<char>());
    f_cpp.close();

    std::ifstream f_bin("libkaryon_native.so", std::ios::binary);
    std::vector<uint8_t> data_bin((std::istreambuf_iterator<char>(f_bin)),
                                   std::istreambuf_iterator<char>());
    f_bin.close();

    std::vector<uint8_t> data_audio(16000);
    for (size_t i = 0; i < data_audio.size(); ++i) {
        float t = static_cast<float>(i) / 8000.0f;
        float s = std::sin(2.0f * 3.14159f * 440.0f * t) +
                  0.5f * std::sin(2.0f * 3.14159f * 880.0f * t);
        int16_t sample = static_cast<int16_t>(s * 10000.0f);
        data_audio[i] = static_cast<uint8_t>(sample & 0xFF);
    }

    FixedPowerLawEngine fixed_engine;
    SovereignEndogenousEngineV2 sovereign_engine;

    std::vector<TestResult> results;

    // C++
    results.push_back(evaluate(fixed_engine, "Fixed Power-Law [0.4..0.99]", "Structured C++ Code", data_cpp));
    results.push_back(evaluate(sovereign_engine, "Sovereign Self-Organizing v2", "Structured C++ Code", data_cpp));

    std::cout << "\n[C++ Stream] Autonomously Evolved Alphas:\n";
    for (size_t h = 0; h < sovereign_engine.num_heads; ++h) {
        float a = 1.0f / (1.0f + std::exp(-sovereign_engine.theta_heads[h]));
        printf("  Head %zu: theta = %6.2f -> alpha = %6.4f\n", h, sovereign_engine.theta_heads[h], a);
    }

    // Binary
    results.push_back(evaluate(fixed_engine, "Fixed Power-Law [0.4..0.99]", "Machine Binary Code (.so)", data_bin));
    results.push_back(evaluate(sovereign_engine, "Sovereign Self-Organizing v2", "Machine Binary Code (.so)", data_bin));

    std::cout << "\n[Binary ELF Stream] Autonomously Evolved Alphas:\n";
    for (size_t h = 0; h < sovereign_engine.num_heads; ++h) {
        float a = 1.0f / (1.0f + std::exp(-sovereign_engine.theta_heads[h]));
        printf("  Head %zu: theta = %6.2f -> alpha = %6.4f\n", h, sovereign_engine.theta_heads[h], a);
    }

    // Audio
    results.push_back(evaluate(fixed_engine, "Fixed Power-Law [0.4..0.99]", "Audio PCM Waveform", data_audio));
    results.push_back(evaluate(sovereign_engine, "Sovereign Self-Organizing v2", "Audio PCM Waveform", data_audio));

    std::cout << "\n[Audio Stream] Autonomously Evolved Alphas:\n";
    for (size_t h = 0; h < sovereign_engine.num_heads; ++h) {
        float a = 1.0f / (1.0f + std::exp(-sovereign_engine.theta_heads[h]));
        printf("  Head %zu: theta = %6.2f -> alpha = %6.4f\n", h, sovereign_engine.theta_heads[h], a);
    }

    std::cout << "\n=== FINAL COMPARATIVE TELEMETRY (EXP-470 v2) ===\n";
    std::cout << "Modality | Architecture | Initial Loss | Final Loss | Loss Drop | Accuracy | Throughput (steps/s)\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";
    for (const auto& r : results) {
        printf("%-24s | %-30s | %6.4f | %6.4f | %7.4f | %6.2f%% | %9.1f\n",
               r.modality.c_str(), r.architecture.c_str(),
               r.initial_loss, r.final_loss, r.loss_delta, r.accuracy, r.steps_per_sec);
    }

    return 0;
}
