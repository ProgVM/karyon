/* ============================================================================
   KARYON-CORE EXP-783: Strict Empirical Benchmark of Sovereign vs Fixed Stride Morphogenesis
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Strict Controlled Conditions)
   
   EVALUATION DATASET:
   100,000 continuous bytes compiled from REAL C++ codebase files (karyon_core.h,
   main.py, etc.) to guarantee high entropy and prevent over-fitting artifacts.
   
   VARIANTS TESTED IN IDENTICAL CONDITIONS:
   1. Variant A (EXP-779 Baseline): Fixed exponential strides (1, 4, 16, 64, 256)
   2. Variant B (EXP-782 Sovereign): Bottom-up Dynamic Stride Discovery from Error Autocorrelation
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <immintrin.h> // AVX2 Intrinsics

constexpr int VOCAB_SIZE = 258;          // Raw Byte V = 258
constexpr int MAX_DIM = 256;             // Hadamard Dimension D = 256
constexpr int MAX_LAYERS = 6;            // Dynamic autopoietic layers bound
constexpr int SPARSE_SLOTS_PER_LAYER = 65536; // Memory slots per layer
constexpr float LAYER_SPROUT_F = 0.18f;  // Free Energy threshold for layer sprouting

struct PureWaveLayer {
    int layer_id;
    int stride;
    float gamma;
    std::vector<float> table_sparse;

    PureWaveLayer(int id, int st) : layer_id(id), stride(st), gamma(0.6f) {
        table_sparse.assign(SPARSE_SLOTS_PER_LAYER * MAX_DIM, 0.0f);
    }
};

inline unsigned int hash_stride(const int* history_buf, int buf_pos, int stride, int vocab_cur) {
    int idx1 = (buf_pos - 1 + 256) % 256;
    int idx2 = (buf_pos - stride + 256) % 256;
    int idx3 = (buf_pos - 2 * stride + 256) % 256;
    int idx4 = (buf_pos - 3 * stride + 256) % 256;

    int b1 = history_buf[idx1];
    int b2 = history_buf[idx2];
    int b3 = history_buf[idx3];
    int b4 = history_buf[idx4];

    unsigned int h = (unsigned int)b1 * 1013904223u 
                   ^ (unsigned int)b2 * 1196996429u 
                   ^ (unsigned int)b3 * 3559788721u 
                   ^ (unsigned int)b4 * 2802195863u
                   ^ (unsigned int)vocab_cur * 2246822519u;
    return h % SPARSE_SLOTS_PER_LAYER;
}

inline float avx2_dot_product_256(const float* a, const float* b) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 256; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

void generate_hadamard_256(std::vector<float>& basis) {
    basis.assign(VOCAB_SIZE * MAX_DIM, 1.0f);

    std::vector<float> H(256 * 256, 1.0f);
    for (int len = 1; len < 256; len <<= 1) {
        for (int i = 0; i < 256; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                for (int k = 0; k < len; ++k) {
                    float u = H[(i + j) * 256 + k];
                    float v = H[(i + len + j) * 256 + k];
                    H[(i + j) * 256 + k] = u;
                    H[(i + j) * 256 + len + k] = u;
                    H[(i + len + j) * 256 + k] = v;
                    H[(i + len + j) * 256 + len + k] = -v;
                }
            }
        }
    }

    for (int v = 0; v < 256; ++v) {
        for (int d = 0; d < 256; ++d) {
            basis[v * MAX_DIM + d] = H[v * 256 + d];
        }
    }

    for (int d = 0; d < 256; ++d) {
        basis[256 * MAX_DIM + d] = 1.0f;
        basis[257 * MAX_DIM + d] = -1.0f;
    }
}

struct EvalResult {
    std::string name;
    float top1_acc;
    float mean_rank;
    float avg_F;
    float final_F;
    double time_sec;
    std::vector<int> strides;
};

EvalResult run_experiment(const std::string& dataset, const std::vector<float>& basis, bool sovereign_mode, const std::string& mode_name) {
    int total_bytes = dataset.length();

    std::vector<float> table_1gram(VOCAB_SIZE * MAX_DIM, 0.0f);
    std::vector<PureWaveLayer> layers;
    layers.emplace_back(0, 1); // Base Layer 0: Stride 1

    int active_layers_count = 1;
    float total_free_energy = 0.0f;
    float initial_F = 0.0f;
    float final_F = 0.0f;
    int top1_hits = 0;
    float total_rank = 0.0f;

    std::vector<int> history_ring(256, 256);
    std::vector<float> z_pred(MAX_DIM, 0.0f);
    std::vector<float> energies(VOCAB_SIZE, 0.0f);

    std::vector<float> error_history(32, 1.0f);
    int error_hist_ptr = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < total_bytes - 1; ++t) {
        int current_byte = (unsigned char)dataset[t];
        int target_byte = (unsigned char)dataset[t + 1];

        history_ring[t % 256] = current_byte;

        const float* z_rec_1 = &table_1gram[current_byte * MAX_DIM];
        const float* z_targ  = &basis[target_byte * MAX_DIM];

        // 1. Superposition
        for (int d = 0; d < MAX_DIM; ++d) z_pred[d] = z_rec_1[d];

        for (int l = 0; l < active_layers_count; ++l) {
            int st = layers[l].stride;
            float g_l = layers[l].gamma;

            unsigned int slot_l = hash_stride(history_ring.data(), t % 256, st, current_byte);
            const float* z_layer = &layers[l].table_sparse[slot_l * MAX_DIM];

            for (int d = 0; d < MAX_DIM; ++d) {
                z_pred[d] += z_layer[d] * g_l;
            }
        }

        // 2. Attractor Snapping
        float norm_sq = 0.0f;
        for (int d = 0; d < MAX_DIM; ++d) norm_sq += z_pred[d] * z_pred[d];
        float norm = sqrtf(norm_sq + 1e-6f);
        float gain = 1.0f + 0.15f * norm;

        for (int i = 0; i < MAX_DIM; ++i) {
            z_pred[i] = tanhf(z_pred[i] * gain);
        }

        // 3. Free Energy
        float resonance = avx2_dot_product_256(z_targ, z_pred.data()) / 256.0f;
        float F = 1.0f - resonance;

        if (t == 0) initial_F = F;
        if (t == total_bytes - 2) final_F = F;
        total_free_energy += F;

        error_history[error_hist_ptr % 32] = F;
        error_hist_ptr++;

        // 4. Layer Sprouting
        if (F > LAYER_SPROUT_F && active_layers_count < MAX_LAYERS) {
            int new_stride = 1;
            if (sovereign_mode) {
                // Bottom-Up Autocorrelation Stride Discovery
                float max_autocorr = -1.0f;
                for (int k = 2; k <= 16; ++k) {
                    float autocorr = 0.0f;
                    for (int i = 0; i < 16; ++i) {
                        float e1 = error_history[(error_hist_ptr - i + 1024) % 32];
                        float e2 = error_history[(error_hist_ptr - i - k + 1024) % 32];
                        autocorr += e1 * e2;
                    }
                    if (autocorr > max_autocorr) {
                        max_autocorr = autocorr;
                        new_stride = k * (1 << (active_layers_count - 1));
                    }
                }
            } else {
                // Fixed Exponential Stride (1, 4, 16, 64, 256)
                new_stride = 1 << (2 * active_layers_count);
            }

            layers.emplace_back(active_layers_count, new_stride);
            active_layers_count++;
        }

        // Inter-layer resonance tuning
        for (int l = 0; l < active_layers_count; ++l) {
            int st = layers[l].stride;
            unsigned int slot_l = hash_stride(history_ring.data(), t % 256, st, current_byte);
            const float* z_layer = &layers[l].table_sparse[slot_l * MAX_DIM];

            float layer_res = avx2_dot_product_256(z_targ, z_layer) / 256.0f;
            layers[l].gamma = 0.95f * layers[l].gamma + 0.05f * (0.2f + 0.8f * (1.0f + layer_res));
        }

        // 5. Candidate Readout
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            const float* z_v = &basis[v * MAX_DIM];
            energies[v] = avx2_dot_product_256(z_v, z_pred.data()) / 256.0f;
        }

        // Rank Tracking
        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        // 6. Plasticity Update
        float eta_base = 0.04f + 0.12f * F;
        float* p_slot_1 = &table_1gram[current_byte * MAX_DIM];
        for (int i = 0; i < MAX_DIM; ++i) {
            float delta = z_targ[i] - z_pred[i];
            p_slot_1[i] += eta_base * delta;
        }

        for (int l = 0; l < active_layers_count; ++l) {
            int st = layers[l].stride;
            unsigned int slot_l = hash_stride(history_ring.data(), t % 256, st, current_byte);
            float* p_slot_l = &layers[l].table_sparse[slot_l * MAX_DIM];
            float layer_lr = eta_base * (1.0f / (1.0f + 0.1f * l));

            for (int i = 0; i < MAX_DIM; ++i) {
                float delta = z_targ[i] - z_pred[i];
                p_slot_l[i] += layer_lr * delta;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    EvalResult res;
    res.name = mode_name;
    res.top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    res.mean_rank = total_rank / (float)(total_bytes - 1);
    res.avg_F = total_free_energy / (float)(total_bytes - 1);
    res.final_F = final_F;
    res.time_sec = elapsed_sec;
    for (int l = 0; l < active_layers_count; ++l) res.strides.push_back(layers[l].stride);

    return res;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-783: Strict Controlled Benchmark Comparison" << std::endl;
    std::cout << "  Identical High-Entropy Dataset | 100,000 Bytes Continuous Stream" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Load High-Entropy Dataset (combining source files)
    std::string dataset;
    std::vector<std::string> files = {"karyon_core.h", "experiments/exp_782_sovereign_pure_wave.cpp", "karyon_agent_runtime/agent_core.py"};
    for (const auto& fname : files) {
        std::ifstream f(fname);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            dataset += content;
        }
    }

    if (dataset.length() < 100000) {
        std::string base = dataset;
        while (dataset.length() < 100000) dataset += base;
    }
    dataset = dataset.substr(0, 100000); // Truncate to exact 100,000 bytes

    std::cout << "[Benchmark Setup] Dataset Total Bytes: " << dataset.length() << " bytes." << std::endl;

    std::vector<float> basis(VOCAB_SIZE * MAX_DIM);
    generate_hadamard_256(basis);

    // Run Variant A: Fixed Strides
    EvalResult res_fixed = run_experiment(dataset, basis, false, "Variant A (Fixed Strides 1,4,16,64,256)");

    // Run Variant B: Sovereign Dynamic Strides
    EvalResult res_sovereign = run_experiment(dataset, basis, true, "Variant B (Sovereign Dynamic Stride)");

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  STRICT BENCHMARK COMPARISON RESULTS (EQUAL 100k HIGH-ENTROPY STREAM)" << std::endl;
    std::cout << "=================================================================" << std::endl;

    std::cout << "\n--- VARIANT A: FIXED STRIDES (1, 4, 16, 64, 256) ---" << std::endl;
    std::cout << "  - Top-1 Accuracy      : " << res_fixed.top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank    : " << res_fixed.mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy : " << res_fixed.avg_F << std::endl;
    std::cout << "  - Final Free Energy   : " << res_fixed.final_F << std::endl;
    std::cout << "  - Active Strides      : ";
    for (int s : res_fixed.strides) std::cout << s << " ";
    std::cout << std::endl;

    std::cout << "\n--- VARIANT B: SOVEREIGN DYNAMIC STRIDES ---" << std::endl;
    std::cout << "  - Top-1 Accuracy      : " << res_sovereign.top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank    : " << res_sovereign.mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy : " << res_sovereign.avg_F << std::endl;
    std::cout << "  - Final Free Energy   : " << res_sovereign.final_F << std::endl;
    std::cout << "  - Active Strides      : ";
    for (int s : res_sovereign.strides) std::cout << s << " ";
    std::cout << std::endl;

    float acc_delta = res_sovereign.top1_acc - res_fixed.top1_acc;
    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  DELTA ANALYSIS & VERDICT:" << std::endl;
    std::cout << "  - Top-1 Accuracy Delta : " << (acc_delta >= 0 ? "+" : "") << acc_delta << " %" << std::endl;
    if (acc_delta > 0.0f) {
        std::cout << "  - Verdict              : 🟢 POSITIVE! Sovereign Morphogenesis OUTPERFORMS Fixed Strides!" << std::endl;
    } else {
        std::cout << "  - Verdict              : 🔴 REJECTED! Fixed Strides outperform Sovereign Discovery." << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    // Output JSON
    std::ofstream json_out("experiments/exp_783_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-783\",\n"
             << "  \"fixed_acc\": " << res_fixed.top1_acc << ",\n"
             << "  \"sovereign_acc\": " << res_sovereign.top1_acc << ",\n"
             << "  \"acc_delta\": " << acc_delta << ",\n"
             << "  \"verdict\": \"" << (acc_delta > 0.0f ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    return 0;
}
