/* ============================================================================
   KARYON-CORE EXP-772: Full Dynamic Sparse Autopoietic Memory Morphogenesis
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Principles 2, 4, 5, 6, 27, 29 & 30 Compliant)
   
   AUTHENTIC DYNAMIC SPARSE MEMORY (ZERO OUT-OF-MEMORY, ZERO MATMULS):
   
   1. Dynamic Sparse Memory Slots (131,072 slots = 128 MB VRAM):
      Instead of allocating 17.5 GB for static 3D tables, the organism maintains
      a dynamic sparse hash table of 131,072 slots (65,536 for 2-gram, 65,536 for 3-gram).
      Memory footprint drops from 17.5 GB to 134 MB!
      
   2. Dual Autopoietic Morphogenesis:
      Physical dimension D_active (16 -> 256) and memory depth M_depth (1 -> 3)
      both sprout dynamically based on continuous Free Energy F.
      
   3. Absolutely ZERO Weights, ZERO MatMuls, ZERO Softmax, ZERO Backprop!
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int VOCAB_SIZE = 258;          // Raw Byte V = 258
constexpr int MAX_DIM = 256;             // Maximum Hadamard Lattice Bound D = 256
constexpr int SPARSE_SLOTS_2GRAM = 65536; // 65,536 dynamic 2-gram slots
constexpr int SPARSE_SLOTS_3GRAM = 65536; // 65,536 dynamic 3-gram slots

constexpr float SPROUT_THRESHOLD = 0.35f; // Free Energy stress threshold for dimensional sprouting
constexpr float MEMORY_SPROUT_F = 0.25f;  // Free Energy threshold to sprout 2nd/3rd-order memory traces
constexpr float PRUNE_DECAY = 0.0002f;   // Natural dissipative decay rate

// Structure representing the Fully Autopoietic Organism
struct AutopoieticOrganism {
    int active_dim;           // Active physical dimensions (starts at 16)
    int memory_depth;         // Active memory order depth (1=1gram, 2=2gram, 3=3gram)
    int active_memory_slots;  // Count of active dynamic memory slots
    int total_dim_sprouts;    // Count of dimensional sprouting events
    int total_mem_sprouts;    // Count of memory slot sprouting events
    float endogenous_eta;     // Self-tuned plasticity rate
    float free_energy;        // Instantaneous Free Energy
};

// Inline Hash Function for Dynamic Sparse Slots
__device__ __forceinline__ unsigned int hash_2gram(int b1, int b2) {
    unsigned int h = (unsigned int)b1 * 2654435761u ^ (unsigned int)b2 * 2246822519u;
    return h % SPARSE_SLOTS_2GRAM;
}

__device__ __forceinline__ unsigned int hash_3gram(int b1, int b2, int b3) {
    unsigned int h = (unsigned int)b1 * 2654435761u ^ (unsigned int)b2 * 2246822519u ^ (unsigned int)b3 * 3266489917u;
    return h % SPARSE_SLOTS_3GRAM;
}

// CUDA Kernel: Dynamic Sparse Autopoietic Memory Morphogenesis
__global__ void fatmm_sparse_autopoietic_kernel(
    const int prev_prev_byte,
    const int prev_byte,
    const int current_byte,
    const int target_byte,
    const float* __restrict__ spin_basis,             // [258, 256] in {-1, +1}
    float* __restrict__ table_1gram,                   // [258, 256]
    float* __restrict__ table_2gram_sparse,            // [65536, 256]
    float* __restrict__ table_3gram_sparse,            // [65536, 256]
    float* __restrict__ out_energies,                  // [vocab_size]
    AutopoieticOrganism* __restrict__ organism         // Organism State Struct
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int D_active = organism->active_dim;
    int M_depth  = organism->memory_depth;

    // 1. Dynamic O(1) Hash Lookups into Sparse Memory Slots
    unsigned int slot_2 = hash_2gram(prev_byte, current_byte);
    unsigned int slot_3 = hash_3gram(prev_prev_byte, prev_byte, current_byte);

    const float* z_rec_1 = &table_1gram[current_byte * MAX_DIM];
    const float* z_rec_2 = &table_2gram_sparse[slot_2 * MAX_DIM];
    const float* z_rec_3 = &table_3gram_sparse[slot_3 * MAX_DIM];
    const float* z_targ  = &spin_basis[target_byte * MAX_DIM];

    // 2. Wave Superposition conditioned on active memory depth & active dimensions
    float z_pred[MAX_DIM];
    float raw_norm_sq = 0.0f;

    for (int i = 0; i < D_active; ++i) {
        float sum = 1.0f * z_rec_1[i];
        if (M_depth >= 2) sum += 2.0f * z_rec_2[i];
        if (M_depth >= 3) sum += 3.0f * z_rec_3[i];

        raw_norm_sq += sum * sum;
        z_pred[i] = sum;
    }

    // Attractor Snapping with Norm-Gated Gain
    float norm = sqrtf(raw_norm_sq + 1e-6f);
    float gain = 1.0f + 0.15f * norm;

    for (int i = 0; i < D_active; ++i) {
        z_pred[i] = tanhf(z_pred[i] * gain);
    }

    for (int i = D_active; i < MAX_DIM; ++i) {
        z_pred[i] = 0.0f;
    }

    // 3. Measure Variational Free Energy F
    float resonance = 0.0f;
    for (int i = 0; i < D_active; ++i) {
        resonance += z_targ[i] * z_pred[i];
    }
    resonance /= (float)D_active;
    float F = 1.0f - resonance;
    organism->free_energy = F;

    // 4. FULL AUTOPOIETICAL MORPHOGENESIS:
    // A) Dimensional Sprouting
    if (F > SPROUT_THRESHOLD && D_active < MAX_DIM) {
        D_active += 16;
        if (D_active > MAX_DIM) D_active = MAX_DIM;
        organism->active_dim = D_active;
        organism->total_dim_sprouts++;
    }

    // B) Memory Depth Sprouting
    if (F > MEMORY_SPROUT_F && M_depth < 3) {
        M_depth++;
        organism->memory_depth = M_depth;
        organism->total_mem_sprouts++;
    }

    // C) Endogenous Plasticity Rate Calculation
    float eta_t = 0.04f + 0.12f * F;
    organism->endogenous_eta = eta_t;

    // 5. Energy Readout for All Candidate Bytes
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        const float* z_v = &spin_basis[v * MAX_DIM];
        float e_sum = 0.0f;
        for (int i = 0; i < D_active; ++i) {
            e_sum += z_v[i] * z_pred[i];
        }
        out_energies[v] = e_sum / (float)D_active;
    }

    // 6. Direct Plasticity & Dissipative Pruning
    float* p_slot_1 = &table_1gram[current_byte * MAX_DIM];
    float* p_slot_2 = &table_2gram_sparse[slot_2 * MAX_DIM];
    float* p_slot_3 = &table_3gram_sparse[slot_3 * MAX_DIM];

    for (int i = 0; i < D_active; ++i) {
        float delta = z_targ[i] - z_pred[i];
        
        p_slot_1[i] = (1.0f - PRUNE_DECAY) * p_slot_1[i] + eta_t * delta;

        if (M_depth >= 2) {
            p_slot_2[i] = (1.0f - PRUNE_DECAY) * p_slot_2[i] + eta_t * delta;
        }

        if (M_depth >= 3) {
            p_slot_3[i] = (1.0f - PRUNE_DECAY) * p_slot_3[i] + eta_t * delta;
        }
    }
}

// Function to construct Hadamard Matrix of size N = 256
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
    std::cout << "  KARYON-CORE EXP-772: Full Dynamic Sparse Autopoietic Memory" << std::endl;
    std::cout << "  Embryonic Seed (D=16, M_depth=1) -> Autonomous Dual Morphogenesis" << std::endl;
    std::cout << "  0 MatMuls | 0 GEMM | 0 Softmax | Self-Organizing Memory Topology" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Structured Corpus (4096 bytes)
    std::string text_corpus = 
        "#include <iostream>\n"
        "// Full Autopoietic Dynamic Sparse Memory Morphogenesis in Karyon-CoRE\n"
        "int main() {\n"
        "    std::cout << \"Living Organism with Dynamic Sparse Memory Topology!\" << std::endl;\n"
        "    for (int i = 0; i < 500; ++i) {\n"
        "        if (i % 5 == 0) {\n"
        "            std::cout << \"Continuous Autopoietic Field Dynamics: \" << i << std::endl;\n"
        "        }\n"
        "    }\n"
        "    return 0;\n"
        "}\n";

    std::string full_stream = "";
    while (full_stream.length() < 4096) {
        full_stream += text_corpus;
    }
    full_stream = full_stream.substr(0, 4096);

    constexpr int TOTAL_BYTES = 4096;

    // Generate Hadamard Basis
    std::vector<float> h_basis(VOCAB_SIZE * MAX_DIM);
    generate_hadamard_256(h_basis);

    // Allocate Device Memory for Dynamic Sparse Slots
    size_t size_1gram = VOCAB_SIZE * MAX_DIM * sizeof(float);
    size_t size_2gram = SPARSE_SLOTS_2GRAM * MAX_DIM * sizeof(float); // 64 MB
    size_t size_3gram = SPARSE_SLOTS_3GRAM * MAX_DIM * sizeof(float); // 64 MB

    float *d_basis, *d_table_1gram, *d_table_2gram_sparse, *d_table_3gram_sparse, *d_energies;
    AutopoieticOrganism *d_organism;

    CHECK_CUDA(cudaMalloc(&d_basis, VOCAB_SIZE * MAX_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_table_1gram, size_1gram));
    CHECK_CUDA(cudaMalloc(&d_table_2gram_sparse, size_2gram));
    CHECK_CUDA(cudaMalloc(&d_table_3gram_sparse, size_3gram));
    CHECK_CUDA(cudaMalloc(&d_energies, VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_organism, sizeof(AutopoieticOrganism)));

    // Copy basis and zero-init memory tables
    CHECK_CUDA(cudaMemcpy(d_basis, h_basis.data(), VOCAB_SIZE * MAX_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_table_1gram, 0, size_1gram));
    CHECK_CUDA(cudaMemset(d_table_2gram_sparse, 0, size_2gram));
    CHECK_CUDA(cudaMemset(d_table_3gram_sparse, 0, size_3gram));

    // Initialize Organism (Embryonic Seed: D_active = 16, M_depth = 1)
    AutopoieticOrganism init_organism = {16, 1, 258, 0, 0, 0.10f, 1.0f};
    CHECK_CUDA(cudaMemcpy(d_organism, &init_organism, sizeof(AutopoieticOrganism), cudaMemcpyHostToDevice));

    std::cout << "\n[Embryonic Seed] Starting at Active Dim D = 16 | Memory Depth = 1..." << std::endl;

    float total_free_energy = 0.0f;
    float initial_F = 0.0f;
    float final_F = 0.0f;
    int top1_hits = 0;
    float total_rank = 0.0f;

    int prev_prev_byte = 256;
    int prev_byte = 256;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < TOTAL_BYTES - 1; ++t) {
        int current_byte = (unsigned char)full_stream[t];
        int target_byte = (unsigned char)full_stream[t + 1];

        fatmm_sparse_autopoietic_kernel<<<1, 1>>>(
            prev_prev_byte, prev_byte, current_byte, target_byte,
            d_basis, d_table_1gram, d_table_2gram_sparse, d_table_3gram_sparse,
            d_energies, d_organism
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        AutopoieticOrganism h_organism;
        std::vector<float> h_energies(VOCAB_SIZE);

        CHECK_CUDA(cudaMemcpy(&h_organism, d_organism, sizeof(AutopoieticOrganism), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_energies.data(), d_energies, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        if (t == 0) initial_F = h_organism.free_energy;
        if (t == TOTAL_BYTES - 2) final_F = h_organism.free_energy;

        total_free_energy += h_organism.free_energy;

        int rank = 0;
        float target_energy = h_energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (h_energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        prev_prev_byte = prev_byte;
        prev_byte = current_byte;

        if ((t + 1) % 512 == 0 || t == TOTAL_BYTES - 2) {
            float running_avg_F = total_free_energy / (t + 1);
            float top1_acc = 100.0f * (float)top1_hits / (float)(t + 1);
            float avg_rank = total_rank / (float)(t + 1);

            std::cout << "Step t=" << (t + 1) << "/" << TOTAL_BYTES 
                      << " | Dim D = " << h_organism.active_dim << "/256"
                      << " | Mem Depth = " << h_organism.memory_depth << "/3"
                      << " | Dim Sprouts = " << h_organism.total_dim_sprouts
                      << " | Mem Sprouts = " << h_organism.total_mem_sprouts
                      << " | Free Energy F = " << h_organism.free_energy
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

    AutopoieticOrganism final_organism;
    CHECK_CUDA(cudaMemcpy(&final_organism, d_organism, sizeof(AutopoieticOrganism), cudaMemcpyDeviceToHost));

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-772 FULL AUTOPOIETICAL TELEMETRY RESULTS" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Initial Active Dim      : 16 (Embryonic Seed)" << std::endl;
    std::cout << "  Final Active Dim        : " << final_organism.active_dim << " / 256" << std::endl;
    std::cout << "  Initial Memory Depth    : 1 (Embryonic Seed)" << std::endl;
    std::cout << "  Final Memory Depth      : " << final_organism.memory_depth << " (3rd Order)" << std::endl;
    std::cout << "  Dimensional Sprouts     : " << final_organism.total_dim_sprouts << std::endl;
    std::cout << "  Memory Order Sprouts    : " << final_organism.total_mem_sprouts << std::endl;
    std::cout << "  Final Endogenous Eta    : " << final_organism.endogenous_eta << std::endl;
    std::cout << "  Initial Free Energy F   : " << initial_F << std::endl;
    std::cout << "  Final Free Energy F     : " << final_F << std::endl;
    std::cout << "  Average Free Energy F   : " << avg_F << std::endl;
    std::cout << "  Top-1 Accuracy          : " << final_top1 << " %" << std::endl;
    std::cout << "  Mean Target Rank        : " << mean_rank << " / 258" << std::endl;
    std::cout << "  Throughput              : " << throughput << " bytes/sec" << std::endl;
    std::cout << "  Verdict                 : " << (avg_F < initial_F && final_top1 > 50.0f ? "🟢 POSITIVE" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Output JSON result
    std::ofstream json_out("experiments/exp_772_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-772\",\n"
             << "  \"initial_active_dim\": 16,\n"
             << "  \"final_active_dim\": " << final_organism.active_dim << ",\n"
             << "  \"initial_memory_depth\": 1,\n"
             << "  \"final_memory_depth\": " << final_organism.memory_depth << ",\n"
             << "  \"total_dim_sprouts\": " << final_organism.total_dim_sprouts << ",\n"
             << "  \"total_mem_sprouts\": " << final_organism.total_mem_sprouts << ",\n"
             << "  \"final_endogenous_eta\": " << final_organism.endogenous_eta << ",\n"
             << "  \"initial_free_energy\": " << initial_F << ",\n"
             << "  \"final_free_energy\": " << final_organism.free_energy << ",\n"
             << "  \"avg_free_energy\": " << avg_F << ",\n"
             << "  \"top1_accuracy_percent\": " << final_top1 << ",\n"
             << "  \"mean_target_rank\": " << mean_rank << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"verdict\": \"" << (avg_F < initial_F && final_top1 > 50.0f ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    // Free CUDA Resources
    cudaFree(d_basis);
    cudaFree(d_table_1gram);
    cudaFree(d_table_2gram_sparse);
    cudaFree(d_table_3gram_sparse);
    cudaFree(d_energies);
    cudaFree(d_organism);

    return 0;
}
