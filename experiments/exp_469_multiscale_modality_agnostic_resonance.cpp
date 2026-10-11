// experiments/exp_469_multiscale_modality_agnostic_resonance.cpp
// EXP-469: Power-Law Multi-Scale Horizon Resonance vs Fixed-Decay Baseline across Diverse Modalities
// Modalities evaluated in Single-Pass (N=1, Zero-Epoch):
// 1. Machine Executable Code (.so binary opcodes)
// 2. Structured C++ Source Code (karyon_native_core.cpp)
// 3. Audio/Signal Byte Waveform (Simulated raw byte audio stream)

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

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

    void reset() {
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

    virtual float get_head_alpha(size_t head_idx) const = 0;

    std::pair<float, bool> step(uint8_t byte_in, uint8_t byte_target) {
        float* e_in = byte_emb.data() + byte_in * dim_emb;

        // Keys
        std::vector<float> k(num_heads * dim_k);
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < dim_k; ++i) {
                k[h * dim_k + i] = e_in[h * dim_k + i];
            }
            normalize(k.data() + h * dim_k, dim_k);
        }

        // Associative Field Retrieval
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

        for (size_t i = 0; i < dim_emb; ++i) {
            h_field[i] = 0.70f * h_field[i] + 0.30f * e_in[i];
        }

        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim_emb; ++i) sum_sq += h_field[i] * h_field[i];
        float inv_h_norm = 1.0f / (std::sqrt(sum_sq) + 1e-4f);

        // Cosine Projection onto Coordinate Manifold
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

        // Error wave
        float* e_target = byte_emb.data() + byte_target * dim_emb;
        std::vector<float> error_wave(dim_emb);
        for (size_t d = 0; d < dim_emb; ++d) {
            error_wave[d] = e_target[d] - h_field[d];
        }

        // Multi-Scale Head Relaxation
        for (size_t h = 0; h < num_heads; ++h) {
            float alpha = get_head_alpha(h);
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

        // Embedding Plasticity
        for (size_t d = 0; d < dim_emb; ++d) {
            e_target[d] += mu_emb * h_field[d];
        }
        normalize(e_target, dim_emb);

        return {loss, is_correct};
    }
};

// Architecture A: Static Single-Decay Baseline (alpha = 0.92 for all heads)
class FixedBaselineEngine : public ResonanceEngineBase {
public:
    float get_head_alpha(size_t) const override {
        return 0.92f;
    }
};

// Architecture B: Power-Law Multi-Scale Spectrum (Scale-Free Horizon)
// Head 0: Ultra-Fast (alpha=0.40, half-life ~ 1.2 bytes: local byte/opcode transitions)
// Head 1: Fast (alpha=0.75, half-life ~ 4.0 bytes: word / integer / instruction bounds)
// Head 2: Medium (alpha=0.92, half-life ~ 12.0 bytes: syntactical expressions / blocks)
// Head 3: Deep (alpha=0.99, half-life ~ 100+ bytes: global context & structure)
class MultiScalePowerLawEngine : public ResonanceEngineBase {
public:
    float get_head_alpha(size_t head_idx) const override {
        static const float alphas[4] = {0.40f, 0.75f, 0.92f, 0.99f};
        return alphas[head_idx % 4];
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
    std::cout << "=== EXP-469: Multi-Scale Power-Law Horizon vs Fixed-Decay Baseline ===" << std::endl;

    // 1. Modality: Structured C++ Source Code
    std::ifstream f_cpp("karyon_native_core.cpp", std::ios::binary);
    std::vector<uint8_t> data_cpp((std::istreambuf_iterator<char>(f_cpp)),
                                   std::istreambuf_iterator<char>());
    f_cpp.close();

    // 2. Modality: Machine Binary Code (ELF Shared Object .so)
    std::ifstream f_bin("libkaryon_native.so", std::ios::binary);
    std::vector<uint8_t> data_bin((std::istreambuf_iterator<char>(f_bin)),
                                   std::istreambuf_iterator<char>());
    f_bin.close();

    // 3. Modality: Audio/Signal Byte Waveform (Simulated 16-bit PCM Audio Wave)
    std::vector<uint8_t> data_audio(16000);
    for (size_t i = 0; i < data_audio.size(); ++i) {
        float t = static_cast<float>(i) / 8000.0f;
        // Harmonic superpositions (fundamental + harmonics)
        float s = std::sin(2.0f * 3.14159f * 440.0f * t) +
                  0.5f * std::sin(2.0f * 3.14159f * 880.0f * t) +
                  0.25f * std::sin(2.0f * 3.14159f * 1320.0f * t);
        int16_t sample = static_cast<int16_t>(s * 10000.0f);
        data_audio[i] = static_cast<uint8_t>(sample & 0xFF);
    }

    FixedBaselineEngine fixed_engine;
    MultiScalePowerLawEngine multiscale_engine;

    std::vector<TestResult> results;

    // Run C++ Code Benchmark
    results.push_back(evaluate(fixed_engine, "Fixed-Decay (0.92)", "Structured C++ Code", data_cpp));
    results.push_back(evaluate(multiscale_engine, "Multi-Scale Power-Law", "Structured C++ Code", data_cpp));

    // Run Machine Binary Code Benchmark
    results.push_back(evaluate(fixed_engine, "Fixed-Decay (0.92)", "Machine Binary Code (.so)", data_bin));
    results.push_back(evaluate(multiscale_engine, "Multi-Scale Power-Law", "Machine Binary Code (.so)", data_bin));

    // Run Audio Signal Benchmark
    results.push_back(evaluate(fixed_engine, "Fixed-Decay (0.92)", "Audio PCM Waveform", data_audio));
    results.push_back(evaluate(multiscale_engine, "Multi-Scale Power-Law", "Audio PCM Waveform", data_audio));

    std::cout << "\n=== DETAILED EMPIRICAL RESULTS (SINGLE-PASS N=1) ===\n";
    std::cout << "Modality | Architecture | Initial Loss | Final Loss | Loss Drop | Accuracy | Throughput (steps/s)\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";
    for (const auto& r : results) {
        printf("%-24s | %-20s | %6.4f | %6.4f | %7.4f | %6.2f%% | %9.1f\n",
               r.modality.c_str(), r.architecture.c_str(),
               r.initial_loss, r.final_loss, r.loss_delta, r.accuracy, r.steps_per_sec);
    }

    // Write JSON telemetry for KEP
    std::ofstream out_json("experiments/exp_469_telemetry.json");
    out_json << "{\n  \"experiment\": \"EXP-469\",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out_json << "    {\n"
                 << "      \"modality\": \"" << r.modality << "\",\n"
                 << "      \"architecture\": \"" << r.architecture << "\",\n"
                 << "      \"initial_loss\": " << r.initial_loss << ",\n"
                 << "      \"final_loss\": " << r.final_loss << ",\n"
                 << "      \"loss_delta\": " << r.loss_delta << ",\n"
                 << "      \"accuracy\": " << r.accuracy << ",\n"
                 << "      \"steps_per_sec\": " << r.steps_per_sec << "\n"
                 << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out_json << "  ]\n}\n";
    out_json.close();

    return 0;
}
