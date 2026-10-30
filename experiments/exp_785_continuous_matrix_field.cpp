/* ============================================================================
   KARYON-CORE EXP-785: Continuous Complex Tensor Field Operator with Spatial Superposition
   ============================================================================
   HYPOTHESIS:
   To eliminate discrete hash slots WITHOUT collapsing memory capacity (as happened in 18 KB test),
   we expand the continuous state field into a CONTINUOUS MATRIX OPERATOR H_l in R^(K x D)
   where context x_t acts as a continuous projection kernel into the wave field:
   H_l(t+1) = (1 - gamma) * H_l(t) + eta * x_t (x_t)^T
   This gives infinite continuous state overlap with ZERO discrete buckets, retaining full 256x256 capacity!
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <immintrin.h>

constexpr int VOCAB_SIZE = 258;
constexpr int MAX_DIM = 256;
constexpr int MAX_LAYERS = 6;
constexpr float LAYER_SPROUT_F = 0.18f;

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

struct ContinuousMatrixFieldModel {
    struct MatrixLayer {
        int stride;
        float gamma;
        std::vector<float> H_field; // 256 x 256 Continuous Tensor Operator (256 KB per layer)

        MatrixLayer(int st) : stride(st), gamma(0.6f) {
            H_field.assign(MAX_DIM * MAX_DIM, 0.0f);
        }
    };

    std::vector<float> table_1gram;
    std::vector<MatrixLayer> layers;
    int active_layers;

    ContinuousMatrixFieldModel() : active_layers(1) {
        table_1gram.assign(VOCAB_SIZE * MAX_DIM, 0.0f);
        layers.emplace_back(1);
    }
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-785: Continuous Matrix Field Operator Benchmark" << std::endl;
    std::cout << "  Continuous Outer Product Field H_l in R^(256x256) | Zero Hash Tables" << std::endl;
    std::cout << "=================================================================" << std::endl;

    std::string dataset;
    std::vector<std::string> files = {"karyon_core.h", "experiments/exp_783_strict_benchmark_comparison.cpp", "karyon_agent_runtime/agent_core.py"};
    for (const auto& fname : files) {
        std::ifstream f(fname);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            dataset += content;
        }
    }
    while (dataset.length() < 100000) dataset += dataset;
    dataset = dataset.substr(0, 100000);

    int total_bytes = dataset.length();
    std::vector<float> basis(VOCAB_SIZE * MAX_DIM);
    generate_hadamard_256(basis);

    ContinuousMatrixFieldModel model;

    float total_free_energy = 0.0f;
    float final_F = 0.0f;
    int top1_hits = 0;
    float total_rank = 0.0f;

    std::vector<float> z_pred(MAX_DIM, 0.0f);
    std::vector<float> energies(VOCAB_SIZE, 0.0f);
    std::vector<float> ctx_wave(MAX_DIM, 0.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < total_bytes - 1; ++t) {
        int current_byte = (unsigned char)dataset[t];
        int target_byte = (unsigned char)dataset[t + 1];

        const float* z_rec_1 = &model.table_1gram[current_byte * MAX_DIM];
        const float* z_targ  = &basis[target_byte * MAX_DIM];

        for (int d = 0; d < MAX_DIM; ++d) z_pred[d] = z_rec_1[d];

        // Read from Continuous Matrix Field
        for (int l = 0; l < model.active_layers; ++l) {
            auto& layer = model.layers[l];
            float g_l = layer.gamma;

            // Project current_byte basis through continuous field H_l: z_layer = H_l * z_curr
            const float* z_curr = &basis[current_byte * MAX_DIM];
            for (int r = 0; r < MAX_DIM; ++r) {
                float sum = avx2_dot_product_256(&layer.H_field[r * MAX_DIM], z_curr) / 256.0f;
                z_pred[r] += g_l * sum;
            }
        }

        // Attractor Snapping
        float norm_sq = 0.0f;
        for (int d = 0; d < MAX_DIM; ++d) norm_sq += z_pred[d] * z_pred[d];
        float gain = 1.0f + 0.15f * sqrtf(norm_sq + 1e-6f);
        for (int i = 0; i < MAX_DIM; ++i) z_pred[i] = tanhf(z_pred[i] * gain);

        // Free Energy Measurement
        float resonance = avx2_dot_product_256(z_targ, z_pred.data()) / 256.0f;
        float F = 1.0f - resonance;
        if (t == total_bytes - 2) final_F = F;
        total_free_energy += F;

        // Layer Sprouting
        if (F > LAYER_SPROUT_F && model.active_layers < MAX_LAYERS) {
            int new_stride = 1 << (2 * model.active_layers);
            model.layers.emplace_back(new_stride);
            model.active_layers++;
        }

        // Candidate Readout
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            energies[v] = avx2_dot_product_256(&basis[v * MAX_DIM], z_pred.data()) / 256.0f;
        }

        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        // CONTINUOUS TENSOR PLASTICITY (Outer Product Outer Update)
        float eta_base = 0.04f + 0.12f * F;
        float* p_slot_1 = &model.table_1gram[current_byte * MAX_DIM];
        for (int i = 0; i < MAX_DIM; ++i) p_slot_1[i] += eta_base * (z_targ[i] - z_pred[i]);

        const float* z_curr = &basis[current_byte * MAX_DIM];
        for (int l = 0; l < model.active_layers; ++l) {
            auto& layer = model.layers[l];
            float layer_lr = eta_base * (1.0f / (1.0f + 0.1f * l));

            // Outer product update: dH_l(i, j) = lr * (z_targ[i] - z_pred[i]) * z_curr[j]
            for (int r = 0; r < MAX_DIM; ++r) {
                float err_r = z_targ[r] - z_pred[r];
                float* h_row = &layer.H_field[r * MAX_DIM];
                for (int c = 0; c < MAX_DIM; ++c) {
                    h_row[c] += layer_lr * err_r * z_curr[c];
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    float top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    float mean_rank = total_rank / (float)(total_bytes - 1);
    float avg_F = total_free_energy / (float)(total_bytes - 1);
    size_t mem_bytes = model.layers.size() * MAX_DIM * MAX_DIM * sizeof(float); // 256 KB per layer!

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  CONTINUOUS MATRIX FIELD TELEMETRY RESULTS" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Top-1 Accuracy      : " << top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank    : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy   : " << final_F << std::endl;
    std::cout << "  - Total VRAM / Memory : " << (mem_bytes / 1024.0f) << " KB (vs 384 MB Hash Tables!)" << std::endl;
    std::cout << "  - Compute Speed       : " << (100000.0 / elapsed_sec) << " bytes/sec" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
