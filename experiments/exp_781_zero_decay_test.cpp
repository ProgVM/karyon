/* ============================================================================
   KARYON-CORE EXP-781.3: Sovereign Dynamic Epigenetic Memory Decay
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
constexpr int MAX_LAYERS = 6;            // Maximum Autopoietic Layer Boundary
constexpr int SPARSE_SLOTS_PER_LAYER = 65536; // Dynamic slots per layer

constexpr float DIM_SPROUT_F = 0.30f;    // Free Energy threshold to sprout dimensions
constexpr float MEM_SPROUT_F = 0.20f;    // Free Energy threshold to sprout memory orders
constexpr float LAYER_SPROUT_F = 0.22f;  // Free Energy threshold to sprout wave layers

struct SovereignLayer {
    int layer_id;
    int stride;                          // Exponential temporal stride scale (1, 4, 16, 64, 256...)
    int active_dim;                      // Sovereign per-layer active dimension D_l
    float gamma;                         // Inter-layer resonance weight (self-tuned)
    std::vector<float> table_sparse;     // Dynamic sparse memory table

    SovereignLayer(int id, int st) : layer_id(id), stride(st), active_dim(16), gamma(0.6f) {
        table_sparse.assign(SPARSE_SLOTS_PER_LAYER * MAX_DIM, 0.0f);
    }
};

inline unsigned int hash_stride_layer(const int* history_buf, int buf_pos, int stride, int vocab_cur) {
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

inline float avx2_dot_product_256(const float* a, const float* b, int active_dim) {
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
    std::vector<SovereignLayer> layers;
    layers.emplace_back(0, 1); // Layer 0 (Stride 1 byte)

    int active_dim_global = 16;
    int memory_depth_global = 1;
    int active_layers_count = 1;
    int total_dim_sprouts = 0;
    int total_layer_sprouts = 0;

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

        int D_global = active_dim_global;
        int L_active = active_layers_count;

        history_ring[t % 256] = current_byte;

        const float* z_rec_1 = &table_1gram[current_byte * MAX_DIM];
        const float* z_targ  = &basis[target_byte * MAX_DIM];

        // 1. Multi-Scale Temporal Stride Superposition
        for (int d = 0; d < D_global; ++d) {
            z_pred[d] = z_rec_1[d] * 1.0f;
        }

        for (int l = 0; l < L_active; ++l) {
            int st = layers[l].stride;
            unsigned int slot_l = hash_stride_layer(history_ring.data(), t % 256, st, current_byte);
            const float* z_layer = &layers[l].table_sparse[slot_l * MAX_DIM];
            float g_l = layers[l].gamma;

            for (int d = 0; d < D_global; ++d) {
                z_pred[d] += z_layer[d] * g_l;
            }
        }

        // 2. Hyperspherical Attractor Snapping
        float norm_sq = 0.0f;
        for (int d = 0; d < D_global; ++d) norm_sq += z_pred[d] * z_pred[d];
        float norm = sqrtf(norm_sq + 1e-6f);
        float gain = 1.0f + 0.15f * norm;

        for (int i = 0; i < D_global; ++i) {
            z_pred[i] = tanhf(z_pred[i] * gain);
        }

        // 3. Free Energy Measurement
        float resonance = avx2_dot_product_256(z_targ, z_pred.data(), D_global) / (float)D_global;
        float F = 1.0f - resonance;

        if (t == 0) initial_F = F;
        if (t == total_bytes - 2) final_F = F;
        total_free_energy += F;

        // 4. Autopoietic Morphogenesis
        if (F > LAYER_SPROUT_F && L_active < MAX_LAYERS) {
            int new_stride = 1 << (2 * L_active); // Exponential strides: 1, 4, 16, 64, 256
            L_active++;
            active_layers_count = L_active;
            total_layer_sprouts++;
            layers.emplace_back(L_active - 1, new_stride);
        }

        if (F > DIM_SPROUT_F && D_global < MAX_DIM) {
            D_global += 16;
            if (D_global > MAX_DIM) D_global = MAX_DIM;
            active_dim_global = D_global;
            total_dim_sprouts++;
        }

        // Zero-Decay Epigenetic Protection: Zero artificial decay (decay = 0.0f)
        for (int l = 0; l < L_active; ++l) {
            int st = layers[l].stride;
            unsigned int slot_l = hash_stride_layer(history_ring.data(), t % 256, st, current_byte);
            const float* z_layer = &layers[l].table_sparse[slot_l * MAX_DIM];
            float layer_res = avx2_dot_product_256(z_targ, z_layer, D_global) / (float)D_global;

            layers[l].gamma = 0.95f * layers[l].gamma + 0.05f * (0.2f + 0.8f * (1.0f + layer_res));
        }

        // Readout Candidates
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            const float* z_v = &basis[v * MAX_DIM];
            energies[v] = avx2_dot_product_256(z_v, z_pred.data(), D_global) / (float)D_global;
        }

        // Rank Tracking
        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        // Plasticity Update with Dynamic Eta and ZERO Dissipative Decay
        float eta_base = 0.04f + 0.12f * F;
        float* p_slot_1 = &table_1gram[current_byte * MAX_DIM];
        for (int i = 0; i < D_global; ++i) {
            float delta = z_targ[i] - z_pred[i];
            p_slot_1[i] += eta_base * delta;
        }

        for (int l = 0; l < L_active; ++l) {
            int st = layers[l].stride;
            unsigned int slot_l = hash_stride_layer(history_ring.data(), t % 256, st, current_byte);
            float* p_slot_l = &layers[l].table_sparse[slot_l * MAX_DIM];
            float layer_lr = eta_base * (1.0f / (1.0f + 0.1f * l));

            for (int i = 0; i < D_global; ++i) {
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
    std::cout << "  EXP-781.3 ZERO DECAY TELEMETRY RESULTS (100,000 BYTES)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Processed Bytes     : " << total_bytes << " bytes" << std::endl;
    std::cout << "  - Active Dim / Layers : " << active_dim_global << " / " << active_layers_count << " layers" << std::endl;
    std::cout << "  - Top-1 Accuracy      : " << top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank    : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Initial Free Energy : " << initial_F << std::endl;
    std::cout << "  - Average Free Energy : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy   : " << final_F << std::endl;
    std::cout << "  - CPU Throughput      : " << throughput << " bytes/sec (" << elapsed_sec << "s)" << std::endl;
    std::cout << "  - KEP Rule #2 Verdict : " << (top1_acc > 99.58f ? "🟢 POSITIVE (NEW RECORD!)" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Write JSON output
    std::ofstream json_out("experiments/exp_781_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-781\",\n"
             << "  \"top1_acc\": " << top1_acc << ",\n"
             << "  \"mean_rank\": " << mean_rank << ",\n"
             << "  \"avg_free_energy\": " << avg_F << ",\n"
             << "  \"final_free_energy\": " << final_F << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"active_dim\": " << active_dim_global << ",\n"
             << "  \"active_layers\": " << active_layers_count << ",\n"
             << "  \"verdict\": \"" << (top1_acc > 99.58f ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    return 0;
}
