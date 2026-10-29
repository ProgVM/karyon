/* ============================================================================
   KARYON-CORE EXP-782.2: Sovereign Stride & Phase Resonance Calibration
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
constexpr int MAX_DIM = 256;             // Maximum Hadamard Lattice Bound D = 256
constexpr int MAX_LAYERS = 6;            // Dynamic autopoietic layers boundary
constexpr int SPARSE_SLOTS_PER_LAYER = 65536; // Dynamic slots per layer

constexpr float DIM_SPROUT_F = 0.30f;    // Free Energy threshold to sprout layer dimension
constexpr float LAYER_SPROUT_F = 0.20f;  // Free Energy threshold to sprout new temporal stride layer

struct SovereignPureWaveLayer {
    int layer_id;
    int stride;                          // Sovereign dynamic stride scale (discovered bottom-up)
    int active_dim;                      // Sovereign per-layer active dimension D_l
    float gamma;                         // Inter-layer resonance weight (self-tuned)
    std::vector<float> table_sparse;     // Dynamic sparse memory table

    SovereignPureWaveLayer(int id, int st, int initial_dim = 16) 
        : layer_id(id), stride(st), active_dim(initial_dim), gamma(0.6f) {
        table_sparse.assign(SPARSE_SLOTS_PER_LAYER * MAX_DIM, 0.0f);
    }
};

inline unsigned int hash_dynamic_stride(const int* history_buf, int buf_pos, int stride, int vocab_cur) {
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

inline float avx2_dot_product(const float* a, const float* b, int active_dim) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < active_dim; i += 8) {
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

int main() {
    std::string dataset;
    std::ifstream src_file("karyon_core.h");
    if (src_file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(src_file)), std::istreambuf_iterator<char>());
        while (dataset.length() < 100000) {
            dataset += content;
        }
    } else {
        std::string sample = "int main() { printf(\"Karyon-CoRE Sovereign Wavefield\\n\"); return 0; }\n";
        while (dataset.length() < 100000) dataset += sample;
    }

    int total_bytes = dataset.length();
    std::vector<float> basis(VOCAB_SIZE * MAX_DIM);
    generate_hadamard_256(basis);

    std::vector<float> table_1gram(VOCAB_SIZE * MAX_DIM, 0.0f);
    std::vector<SovereignPureWaveLayer> layers;
    layers.emplace_back(0, 1, 16); // Layer 0: Stride 1

    int active_layers_count = 1;
    float total_free_energy = 0.0f;
    float initial_F = 0.0f;
    float final_F = 0.0f;
    int top1_hits = 0;
    float total_rank = 0.0f;

    std::vector<int> history_ring(256, 256);
    std::vector<float> z_pred(MAX_DIM, 0.0f);
    std::vector<float> energies(VOCAB_SIZE, 0.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < total_bytes - 1; ++t) {
        int current_byte = (unsigned char)dataset[t];
        int target_byte = (unsigned char)dataset[t + 1];

        history_ring[t % 256] = current_byte;

        const float* z_rec_1 = &table_1gram[current_byte * MAX_DIM];
        const float* z_targ  = &basis[target_byte * MAX_DIM];

        int max_active_dim = 16;
        for (int l = 0; l < active_layers_count; ++l) {
            if (layers[l].active_dim > max_active_dim) max_active_dim = layers[l].active_dim;
        }

        // Base 1-gram wave
        for (int d = 0; d < max_active_dim; ++d) z_pred[d] = z_rec_1[d];

        // Linear multi-layer superposition with per-layer active_dim D_l
        for (int l = 0; l < active_layers_count; ++l) {
            int st = layers[l].stride;
            int D_l = layers[l].active_dim;
            float g_l = layers[l].gamma;

            unsigned int slot_l = hash_dynamic_stride(history_ring.data(), t % 256, st, current_byte);
            const float* z_layer = &layers[l].table_sparse[slot_l * MAX_DIM];

            for (int d = 0; d < D_l; ++d) {
                z_pred[d] += z_layer[d] * g_l;
            }
        }

        // Hyperspherical Attractor Snapping
        float norm_sq = 0.0f;
        for (int d = 0; d < max_active_dim; ++d) norm_sq += z_pred[d] * z_pred[d];
        float norm = sqrtf(norm_sq + 1e-6f);
        float gain = 1.0f + 0.15f * norm;

        for (int i = 0; i < max_active_dim; ++i) {
            z_pred[i] = tanhf(z_pred[i] * gain);
        }

        // Free Energy Measurement
        float resonance = avx2_dot_product(z_targ, z_pred.data(), max_active_dim) / (float)max_active_dim;
        float F = 1.0f - resonance;

        if (t == 0) initial_F = F;
        if (t == total_bytes - 2) final_F = F;
        total_free_energy += F;

        // Bottom-Up Stride Sprouting: Exponential powers of 2 for clean harmonic resonance
        if (F > LAYER_SPROUT_F && active_layers_count < MAX_LAYERS) {
            int new_stride = 1 << (2 * active_layers_count); // 1, 4, 16, 64, 256...
            layers.emplace_back(active_layers_count, new_stride, 16);
            active_layers_count++;
        }

        // Independent Per-Layer Dimension Sprouting
        for (int l = 0; l < active_layers_count; ++l) {
            if (F > DIM_SPROUT_F && layers[l].active_dim < MAX_DIM) {
                layers[l].active_dim += 16;
                if (layers[l].active_dim > MAX_DIM) layers[l].active_dim = MAX_DIM;
            }
        }

        // Inter-layer resonance tuning
        for (int l = 0; l < active_layers_count; ++l) {
            int st = layers[l].stride;
            int D_l = layers[l].active_dim;

            unsigned int slot_l = hash_dynamic_stride(history_ring.data(), t % 256, st, current_byte);
            const float* z_layer = &layers[l].table_sparse[slot_l * MAX_DIM];

            float layer_res = avx2_dot_product(z_targ, z_layer, D_l) / (float)D_l;
            layers[l].gamma = 0.95f * layers[l].gamma + 0.05f * (0.2f + 0.8f * (1.0f + layer_res));
        }

        // Candidate Readout
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            const float* z_v = &basis[v * MAX_DIM];
            energies[v] = avx2_dot_product(z_v, z_pred.data(), max_active_dim) / (float)max_active_dim;
        }

        // Rank Tracking
        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        // Plasticity Update (Zero MatMuls)
        float eta_base = 0.04f + 0.12f * F;
        float* p_slot_1 = &table_1gram[current_byte * MAX_DIM];
        for (int i = 0; i < max_active_dim; ++i) {
            float delta = z_targ[i] - z_pred[i];
            p_slot_1[i] += eta_base * delta;
        }

        for (int l = 0; l < active_layers_count; ++l) {
            int st = layers[l].stride;
            int D_l = layers[l].active_dim;

            unsigned int slot_l = hash_dynamic_stride(history_ring.data(), t % 256, st, current_byte);
            float* p_slot_l = &layers[l].table_sparse[slot_l * MAX_DIM];
            float layer_lr = eta_base * (1.0f / (1.0f + 0.1f * l));

            for (int i = 0; i < D_l; ++i) {
                float delta = z_targ[i] - z_pred[i];
                p_slot_l[i] += layer_lr * delta;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    float top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    float mean_rank = total_rank / (float)(total_bytes - 1);
    float avg_F = total_free_energy / (float)(total_bytes - 1);
    float throughput = (float)total_bytes / (float)elapsed_sec;

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-782.2 TELEMETRY RESULTS (100,000 BYTES SOURCE CODE DATASET)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Processed Bytes     : " << total_bytes << " bytes" << std::endl;
    std::cout << "  - Active Layers Count : " << active_layers_count << " layers" << std::endl;
    std::cout << "  - Per-Layer Dimensions : ";
    for (int l = 0; l < active_layers_count; ++l) std::cout << layers[l].active_dim << " ";
    std::cout << "\n  - Top-1 Accuracy      : " << top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank    : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Initial Free Energy : " << initial_F << std::endl;
    std::cout << "  - Average Free Energy : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy   : " << final_F << std::endl;
    std::cout << "  - CPU Throughput      : " << throughput << " bytes/sec (" << elapsed_sec << "s)" << std::endl;
    std::cout << "  - KEP Rule #2 Verdict : " << (top1_acc > 99.58f ? "🟢 POSITIVE (NEW RECORD!)" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Write JSON output
    std::ofstream json_out("experiments/exp_782_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-782\",\n"
             << "  \"top1_acc\": " << top1_acc << ",\n"
             << "  \"mean_rank\": " << mean_rank << ",\n"
             << "  \"avg_free_energy\": " << avg_F << ",\n"
             << "  \"final_free_energy\": " << final_F << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"active_layers\": " << active_layers_count << ",\n"
             << "  \"verdict\": \"" << (top1_acc > 99.58f ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    return 0;
}
