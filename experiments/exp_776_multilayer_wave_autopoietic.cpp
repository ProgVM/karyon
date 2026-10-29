/* ============================================================================
   KARYON-CORE EXP-776: Multi-Layer Hierarchical Wave Coupling on AVX2 CPU
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Principles 1, 2, 27, 29 & 30 Compliant)
   
   OBJECTIVE:
   Build a 2-layer hierarchical wavefield organism on CPU.
   Layer 1 (Micro-Wavefield): Fast 5-order N-gram non-linear wave superposition.
   Layer 2 (Macro-Wavefield): Macro-scale temporal wave accumulator capturing syntax/structure.
   Inter-Layer Resonance: z_pred = z_1 + gamma * z_2, where gamma self-tunes bottom-up.
   
   ZERO MATMULS | ZERO SOFTMAX | ZERO DENSE WEIGHTS | ZERO BACKPROP
   Pure AVX2 256-bit SIMD Vectorization + OpenMP Multi-Core Threading
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <immintrin.h> // AVX2 Intrinsics
#include <omp.h>       // OpenMP Multi-threading

constexpr int VOCAB_SIZE = 258;          // Raw Byte V = 258
constexpr int MAX_DIM = 256;             // Maximum Hadamard Lattice Bound D = 256
constexpr int SPARSE_SLOTS_2GRAM = 65536; // Dynamic 2-gram slots
constexpr int SPARSE_SLOTS_3GRAM = 65536; // Dynamic 3-gram slots
constexpr int SPARSE_SLOTS_4GRAM = 65536; // Dynamic 4-gram slots
constexpr int SPARSE_SLOTS_5GRAM = 65536; // Dynamic 5-gram slots
constexpr int MACRO_SLOTS = 65536;        // Macro wavefield slots

constexpr float SPROUT_THRESHOLD = 0.30f; // Free Energy stress threshold for dimensional sprouting
constexpr float MEMORY_SPROUT_F = 0.20f;  // Free Energy threshold to sprout high-order memory traces
constexpr float PRUNE_DECAY = 0.0001f;   // Natural dissipative decay rate

struct AutopoieticOrganismCPU {
    int active_dim;           // Active physical dimensions (starts at 16)
    int memory_depth;         // Active memory order depth (1=1gram, up to 5)
    int total_dim_sprouts;    // Count of dimensional sprouting events
    int total_mem_sprouts;    // Count of memory slot sprouting events
    float endogenous_eta;     // Self-tuned plasticity rate
    float free_energy;        // Instantaneous Free Energy
    float alpha[5];           // Layer 1 endogenous phase-coupling weights
    float gamma;              // Inter-layer resonance coupling weight
};

// Fast Hash Functions
inline unsigned int hash_2gram(int b1, int b2) {
    unsigned int h = (unsigned int)b1 * 2654435761u ^ (unsigned int)b2 * 2246822519u;
    return h % SPARSE_SLOTS_2GRAM;
}

inline unsigned int hash_3gram(int b1, int b2, int b3) {
    unsigned int h = (unsigned int)b1 * 2654435761u ^ (unsigned int)b2 * 2246822519u ^ (unsigned int)b3 * 3266489917u;
    return h % SPARSE_SLOTS_3GRAM;
}

inline unsigned int hash_4gram(int b1, int b2, int b3, int b4) {
    unsigned int h = (unsigned int)b1 * 2654435761u ^ (unsigned int)b2 * 2246822519u ^ (unsigned int)b3 * 3266489917u ^ (unsigned int)b4 * 668265263u;
    return h % SPARSE_SLOTS_4GRAM;
}

inline unsigned int hash_5gram(int b1, int b2, int b3, int b4, int b5) {
    unsigned int h = (unsigned int)b1 * 2654435761u ^ (unsigned int)b2 * 2246822519u ^ (unsigned int)b3 * 3266489917u ^ (unsigned int)b4 * 668265263u ^ (unsigned int)b5 * 374761393u;
    return h % SPARSE_SLOTS_5GRAM;
}

inline unsigned int hash_macro(int b1, int b2, int b3, int b4) {
    unsigned int h = (unsigned int)b1 * 1013904223u ^ (unsigned int)b2 * 1196996429u ^ (unsigned int)b3 * 3559788721u ^ (unsigned int)b4 * 2802195863u;
    return h % MACRO_SLOTS;
}

// AVX2 Optimized Dot Product
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

// AVX2 Multi-Layer Hierarchical Wave Superposition (Layer 1 Micro + Layer 2 Macro)
inline float avx2_multilayer_wave_superposition(
    const float* z1, const float* z2, const float* z3, const float* z4, const float* z5,
    const float* z_macro, const float* alpha, float gamma, float* z_pred, int active_dim, int memory_depth
) {
    __m256 v_a1 = _mm256_set1_ps(alpha[0]);
    __m256 v_a2 = _mm256_set1_ps(alpha[1]);
    __m256 v_a3 = _mm256_set1_ps(alpha[2]);
    __m256 v_a4 = _mm256_set1_ps(alpha[3]);
    __m256 v_a5 = _mm256_set1_ps(alpha[4]);
    __m256 v_gamma = _mm256_set1_ps(gamma);
    __m256 v_harmonic_gain = _mm256_set1_ps(0.25f);

    __m256 norm_acc = _mm256_setzero_ps();

    for (int i = 0; i < active_dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(z1 + i);
        __m256 v_sum = _mm256_mul_ps(v1, v_a1);

        if (memory_depth >= 2) {
            __m256 v2 = _mm256_loadu_ps(z2 + i);
            v_sum = _mm256_fmadd_ps(v2, v_a2, v_sum);
            __m256 v_interf = _mm256_mul_ps(v1, v2);
            v_sum = _mm256_fmadd_ps(v_interf, v_harmonic_gain, v_sum);
        }

        if (memory_depth >= 3) {
            __m256 v3 = _mm256_loadu_ps(z3 + i);
            v_sum = _mm256_fmadd_ps(v3, v_a3, v_sum);
        }

        if (memory_depth >= 4) {
            __m256 v4 = _mm256_loadu_ps(z4 + i);
            v_sum = _mm256_fmadd_ps(v4, v_a4, v_sum);
        }

        if (memory_depth >= 5) {
            __m256 v5 = _mm256_loadu_ps(z5 + i);
            v_sum = _mm256_fmadd_ps(v5, v_a5, v_sum);
        }

        // Layer 2 Inter-Layer Wave Coupling: + gamma * z_macro
        __m256 vm = _mm256_loadu_ps(z_macro + i);
        v_sum = _mm256_fmadd_ps(vm, v_gamma, v_sum);

        _mm256_storeu_ps(z_pred + i, v_sum);
        norm_acc = _mm256_fmadd_ps(v_sum, v_sum, norm_acc);
    }

    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, norm_acc);
    return sqrtf(tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + 1e-6f);
}

// Generate Hadamard Matrix H_256
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
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-776: Multi-Layer Hierarchical Wave Coupling" << std::endl;
    std::cout << "  2-Layer Inter-Layer Wave Resonator on AVX2 CPU" << std::endl;
    std::cout << "  0 MatMuls | 0 Softmax | 0 Weights | Self-Tuning Inter-Layer Gamma" << std::endl;
    std::cout << "=================================================================" << std::endl;

    std::string text_corpus = 
        "#include <iostream>\n"
        "#include <vector>\n"
        "#include <string>\n"
        "#include <cmath>\n"
        "// Multi-Layer Continuous Hierarchical Substrate in Karyon-CoRE\n"
        "template<typename T>\n"
        "class HierarchicalWaveOrganism {\n"
        "public:\n"
        "    T micro_state;\n"
        "    T macro_state;\n"
        "    float free_energy;\n"
        "    HierarchicalWaveOrganism() : micro_state(0), macro_state(0), free_energy(1.0f) {}\n"
        "    void update_hierarchical(T input) {\n"
        "        micro_state += input;\n"
        "        macro_state += micro_state * 2;\n"
        "        free_energy *= 0.85f;\n"
        "        std::cout << \"Multi-Layer State: Micro=\" << micro_state << \" | Macro=\" << macro_state << \" | F=\" << free_energy << std::endl;\n"
        "    }\n"
        "};\n"
        "int main() {\n"
        "    HierarchicalWaveOrganism<int> Karyon;\n"
        "    for (int i = 0; i < 1000; ++i) {\n"
        "        Karyon.update_hierarchical(i);\n"
        "    }\n"
        "    return 0;\n"
        "}\n";

    constexpr int TOTAL_BYTES = 100000; // 100 KB continuous stream
    std::string full_stream = "";
    while (full_stream.length() < TOTAL_BYTES) {
        full_stream += text_corpus;
    }
    full_stream = full_stream.substr(0, TOTAL_BYTES);

    // Generate Hadamard Basis
    std::vector<float> basis(VOCAB_SIZE * MAX_DIM);
    generate_hadamard_256(basis);

    // Allocate Flat CPU Memory Arrays for Layer 1 & Layer 2
    std::vector<float> table_1gram(VOCAB_SIZE * MAX_DIM, 0.0f);
    std::vector<float> table_2gram_sparse(SPARSE_SLOTS_2GRAM * MAX_DIM, 0.0f);
    std::vector<float> table_3gram_sparse(SPARSE_SLOTS_3GRAM * MAX_DIM, 0.0f);
    std::vector<float> table_4gram_sparse(SPARSE_SLOTS_4GRAM * MAX_DIM, 0.0f);
    std::vector<float> table_5gram_sparse(SPARSE_SLOTS_5GRAM * MAX_DIM, 0.0f);
    std::vector<float> table_macro_sparse(MACRO_SLOTS * MAX_DIM, 0.0f);

    AutopoieticOrganismCPU organism = {16, 1, 0, 0, 0.10f, 1.0f, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f}, 0.5f};

    std::cout << "\n[Karyon CPU] Starting 100,000 Byte Multi-Layer Wave Horizon..." << std::endl;

    float total_free_energy = 0.0f;
    float initial_F = 0.0f;
    float final_F = 0.0f;
    int top1_hits = 0;
    float total_rank = 0.0f;

    int history[5] = {256, 256, 256, 256, 256};

    std::vector<float> z_pred(MAX_DIM, 0.0f);
    std::vector<float> energies(VOCAB_SIZE, 0.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < TOTAL_BYTES - 1; ++t) {
        int current_byte = (unsigned char)full_stream[t];
        int target_byte = (unsigned char)full_stream[t + 1];

        int D_active = organism.active_dim;
        int M_depth  = organism.memory_depth;

        // Hash Lookups
        unsigned int slot_2 = hash_2gram(history[3], current_byte);
        unsigned int slot_3 = hash_3gram(history[2], history[3], current_byte);
        unsigned int slot_4 = hash_4gram(history[1], history[2], history[3], current_byte);
        unsigned int slot_5 = hash_5gram(history[0], history[1], history[2], history[3], current_byte);
        unsigned int slot_m = hash_macro(history[0], history[1], history[2], current_byte);

        const float* z_rec_1 = &table_1gram[current_byte * MAX_DIM];
        const float* z_rec_2 = &table_2gram_sparse[slot_2 * MAX_DIM];
        const float* z_rec_3 = &table_3gram_sparse[slot_3 * MAX_DIM];
        const float* z_rec_4 = &table_4gram_sparse[slot_4 * MAX_DIM];
        const float* z_rec_5 = &table_5gram_sparse[slot_5 * MAX_DIM];
        const float* z_macro = &table_macro_sparse[slot_m * MAX_DIM];
        const float* z_targ  = &basis[target_byte * MAX_DIM];

        // 1. Multi-Layer Hierarchical Wave Superposition (Micro L1 + Macro L2)
        float norm = avx2_multilayer_wave_superposition(
            z_rec_1, z_rec_2, z_rec_3, z_rec_4, z_rec_5,
            z_macro, organism.alpha, organism.gamma, z_pred.data(), D_active, M_depth
        );

        // 2. Attractor Snapping
        float gain = 1.0f + 0.15f * norm;
        for (int i = 0; i < D_active; ++i) {
            z_pred[i] = tanhf(z_pred[i] * gain);
        }

        // 3. Free Energy Measurement
        float resonance = avx2_dot_product_256(z_targ, z_pred.data(), D_active) / (float)D_active;
        float F = 1.0f - resonance;
        organism.free_energy = F;

        if (t == 0) initial_F = F;
        if (t == TOTAL_BYTES - 2) final_F = F;
        total_free_energy += F;

        // 4. AUTOPOIETICAL MORPHOGENESIS & INTER-LAYER RESONANCE TUNING
        if (F > SPROUT_THRESHOLD && D_active < MAX_DIM) {
            D_active += 16;
            if (D_active > MAX_DIM) D_active = MAX_DIM;
            organism.active_dim = D_active;
            organism.total_dim_sprouts++;
        }

        if (F > MEMORY_SPROUT_F && M_depth < 5) {
            M_depth++;
            organism.memory_depth = M_depth;
            organism.total_mem_sprouts++;
        }

        float eta_t = 0.04f + 0.12f * F;
        organism.endogenous_eta = eta_t;

        // Inter-Layer Coupling Weight Gamma Tuning (Bottom-Up Self-Regulation)
        float macro_res = avx2_dot_product_256(z_targ, z_macro, D_active) / (float)D_active;
        organism.gamma = 0.95f * organism.gamma + 0.05f * (0.2f + 0.8f * (1.0f + macro_res));

        // Bottom-Up Dynamic Phase Coupling Update
        if (M_depth >= 1) organism.alpha[0] = 0.9f * organism.alpha[0] + 0.1f * (1.0f + avx2_dot_product_256(z_targ, z_rec_1, D_active) / (float)D_active);
        if (M_depth >= 2) organism.alpha[1] = 0.9f * organism.alpha[1] + 0.1f * (1.0f + 1.5f * avx2_dot_product_256(z_targ, z_rec_2, D_active) / (float)D_active);
        if (M_depth >= 3) organism.alpha[2] = 0.9f * organism.alpha[2] + 0.1f * (1.0f + 2.0f * avx2_dot_product_256(z_targ, z_rec_3, D_active) / (float)D_active);
        if (M_depth >= 4) organism.alpha[3] = 0.9f * organism.alpha[3] + 0.1f * (1.0f + 2.5f * avx2_dot_product_256(z_targ, z_rec_4, D_active) / (float)D_active);
        if (M_depth >= 5) organism.alpha[4] = 0.9f * organism.alpha[4] + 0.1f * (1.0f + 3.0f * avx2_dot_product_256(z_targ, z_rec_5, D_active) / (float)D_active);

        // 5. Readout Energies for Candidates
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            const float* z_v = &basis[v * MAX_DIM];
            energies[v] = avx2_dot_product_256(z_v, z_pred.data(), D_active) / (float)D_active;
        }

        // Accuracy & Rank Tracking
        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        // 6. Direct Memory Plasticity Update (Layer 1 Micro + Layer 2 Macro)
        float* p_slot_1 = &table_1gram[current_byte * MAX_DIM];
        float* p_slot_2 = &table_2gram_sparse[slot_2 * MAX_DIM];
        float* p_slot_3 = &table_3gram_sparse[slot_3 * MAX_DIM];
        float* p_slot_4 = &table_4gram_sparse[slot_4 * MAX_DIM];
        float* p_slot_5 = &table_5gram_sparse[slot_5 * MAX_DIM];
        float* p_slot_m = &table_macro_sparse[slot_m * MAX_DIM];

        for (int i = 0; i < D_active; ++i) {
            float delta = z_targ[i] - z_pred[i];
            p_slot_1[i] = (1.0f - PRUNE_DECAY) * p_slot_1[i] + eta_t * delta;

            if (M_depth >= 2) p_slot_2[i] = (1.0f - PRUNE_DECAY) * p_slot_2[i] + eta_t * delta;
            if (M_depth >= 3) p_slot_3[i] = (1.0f - PRUNE_DECAY) * p_slot_3[i] + eta_t * delta;
            if (M_depth >= 4) p_slot_4[i] = (1.0f - PRUNE_DECAY) * p_slot_4[i] + eta_t * delta;
            if (M_depth >= 5) p_slot_5[i] = (1.0f - PRUNE_DECAY) * p_slot_5[i] + eta_t * delta;

            // Layer 2 Macro Plasticity Update
            p_slot_m[i] = (1.0f - PRUNE_DECAY) * p_slot_m[i] + (eta_t * 0.75f) * delta;
        }

        // Shift History
        history[0] = history[1];
        history[1] = history[2];
        history[2] = history[3];
        history[3] = current_byte;

        if ((t + 1) % 20000 == 0 || t == TOTAL_BYTES - 2) {
            float running_avg_F = total_free_energy / (t + 1);
            float top1_acc = 100.0f * (float)top1_hits / (float)(t + 1);
            float avg_rank = total_rank / (float)(t + 1);

            std::cout << "Step t=" << (t + 1) << "/" << TOTAL_BYTES 
                      << " | Dim D = " << organism.active_dim << "/256"
                      << " | Mem Depth = " << organism.memory_depth << "/5"
                      << " | Gamma (L1-L2) = " << organism.gamma
                      << " | Free Energy F = " << organism.free_energy
                      << " | Top-1 Acc = " << top1_acc << "%"
                      << " | Avg Rank = " << avg_rank << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    float avg_F = total_free_energy / (TOTAL_BYTES - 1);
    float final_top1 = 100.0f * (float)top1_hits / (float)(TOTAL_BYTES - 1);
    float mean_rank = total_rank / (float)(TOTAL_BYTES - 1);
    float throughput = TOTAL_BYTES / elapsed_sec;

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-776 MULTI-LAYER HIERARCHICAL WAVE TELEMETRY RESULTS" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Processed Bytes         : " << TOTAL_BYTES << " bytes (100 KB)" << std::endl;
    std::cout << "  Final Active Dim        : " << organism.active_dim << " / 256" << std::endl;
    std::cout << "  Final Memory Depth      : " << organism.memory_depth << " (5th Order)" << std::endl;
    std::cout << "  Inter-Layer Gamma       : " << organism.gamma << std::endl;
    std::cout << "  Dimensional Sprouts     : " << organism.total_dim_sprouts << std::endl;
    std::cout << "  Memory Order Sprouts    : " << organism.total_mem_sprouts << std::endl;
    std::cout << "  Initial Free Energy F   : " << initial_F << std::endl;
    std::cout << "  Final Free Energy F     : " << final_F << std::endl;
    std::cout << "  Average Free Energy F   : " << avg_F << std::endl;
    std::cout << "  Top-1 Accuracy          : " << final_top1 << " %" << std::endl;
    std::cout << "  Mean Target Rank        : " << mean_rank << " / 258" << std::endl;
    std::cout << "  Total Execution Time    : " << elapsed_sec << " seconds" << std::endl;
    std::cout << "  CPU Throughput          : " << throughput << " bytes/sec" << std::endl;
    std::cout << "  Verdict                 : " << (avg_F < initial_F && final_top1 > 90.0f ? "🟢 POSITIVE" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Output JSON result
    std::ofstream json_out("experiments/exp_776_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-776\",\n"
             << "  \"processed_bytes\": " << TOTAL_BYTES << ",\n"
             << "  \"final_active_dim\": " << organism.active_dim << ",\n"
             << "  \"final_memory_depth\": " << organism.memory_depth << ",\n"
             << "  \"final_gamma\": " << organism.gamma << ",\n"
             << "  \"total_dim_sprouts\": " << organism.total_dim_sprouts << ",\n"
             << "  \"total_mem_sprouts\": " << organism.total_mem_sprouts << ",\n"
             << "  \"initial_free_energy\": " << initial_F << ",\n"
             << "  \"final_free_energy\": " << final_F << ",\n"
             << "  \"avg_free_energy\": " << avg_F << ",\n"
             << "  \"top1_accuracy_percent\": " << final_top1 << ",\n"
             << "  \"mean_target_rank\": " << mean_rank << ",\n"
             << "  \"execution_time_sec\": " << elapsed_sec << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"verdict\": \"" << (avg_F < initial_F && final_top1 > 90.0f ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    return 0;
}
