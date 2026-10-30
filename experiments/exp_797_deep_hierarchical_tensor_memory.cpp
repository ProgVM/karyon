/* ============================================================================
   KARYON-CORE EXP-797: Deep Hierarchical Autopoietic Tensor Memory
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Principles 2, 3, 4, 5, 6, 27, 29, 30, 31 Compliant)

   SOVEREIGN BAZILEVS DIRECTIVE:
   "Well, that's still kind of weak..."

   WHY EXP-796 WAS WEAK & HOW EXP-797 REVOLUTIONIZES STORAGE:
   In EXP-796, single-layer outer-product matrices decaying uniformly could only hold
   short-term 2-3 byte traces. True intelligence requires:
   1. Multi-Time-Scale Hierarchical Memory Waves:
      - Fast Phase Wave (tau_1 ~ 2-4 bytes): Instantaneous context
      - Medium Attractor Wave (tau_2 ~ 16-32 bytes): Local syntactic constructs / identifiers
      - Deep Macro Wave (tau_3 ~ 128-512 bytes): Global scope / document structure
   2. Resonant Dynamic Amplification:
      When an incoming byte matches an attractor in deep macro-memory, constructive wave
      interference massively amplifies the correct target state, driving Free Energy to near ZERO!

   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <memory>
#include <immintrin.h>

constexpr int VOCAB_SIZE = 258;          // Raw Byte V = 258
constexpr int MAX_DIM = 256;             // Continuous Substrate Dimension D = 256

inline float avx2_dot_product(const float* a, const float* b, int dim) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
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

// ----------------------------------------------------------------------------
// MULTI-SCALE HIERARCHICAL STORAGE FIELD UNIT
// ----------------------------------------------------------------------------
struct HierarchicalStorageUnit {
    int id;
    int dim;
    
    // Multi-Scale Memory Wave Matrices
    std::vector<float> M_fast;   // Fast Memory (tau ~ 0.80)
    std::vector<float> M_med;    // Medium Memory (tau ~ 0.98)
    std::vector<float> M_deep;   // Deep Macro Memory (tau ~ 0.999)

    // Dynamic State Traces
    std::vector<float> trace_fast;
    std::vector<float> trace_med;
    std::vector<float> trace_deep;

    float coupling_fast;
    float coupling_med;
    float coupling_deep;

    float energy_level;
    float causal_credit;

    HierarchicalStorageUnit(int unit_id, int d) 
        : id(unit_id), dim(d), coupling_fast(0.6f), coupling_med(0.8f), coupling_deep(1.2f), energy_level(1.0f), causal_credit(0.0f) {
        M_fast.assign(dim * dim, 0.0f);
        M_med.assign(dim * dim, 0.0f);
        M_deep.assign(dim * dim, 0.0f);

        trace_fast.assign(dim, 0.0f);
        trace_med.assign(dim, 0.0f);
        trace_deep.assign(dim, 0.0f);
    }

    void process(const float* current_phi, std::vector<float>& response_acc) {
        // Update traces across time-scales
        for (int i = 0; i < dim; ++i) {
            trace_fast[i] = trace_fast[i] * 0.80f + current_phi[i] * 0.20f;
            trace_med[i]  = trace_med[i]  * 0.98f + current_phi[i] * 0.02f;
            trace_deep[i] = trace_deep[i] * 0.999f + current_phi[i] * 0.001f;
        }

        // Resonant Wave Readout
        for (int i = 0; i < dim; ++i) {
            float r_fast = avx2_dot_product(&M_fast[i * dim], trace_fast.data(), dim) / (float)dim;
            float r_med  = avx2_dot_product(&M_med[i * dim],  trace_med.data(),  dim) / (float)dim;
            float r_deep = avx2_dot_product(&M_deep[i * dim], trace_deep.data(), dim) / (float)dim;

            float combined_read = r_fast * coupling_fast + r_med * coupling_med + r_deep * coupling_deep;
            response_acc[i] += combined_read * energy_level;
        }
    }

    void evolve(const float* current_phi, const float* error_vec, float free_energy, float delta_F) {
        causal_credit = 0.95f * causal_credit + 0.05f * delta_F;

        if (delta_F > 0.0f) {
            energy_level = std::min(4.0f, energy_level + 0.02f * delta_F);
        } else {
            energy_level -= 0.002f;
        }

        float eta = 0.02f + 0.08f * std::clamp(free_energy, 0.0f, 1.0f);

        // Dynamically adjust scale couplings based on local F reduction
        coupling_fast = std::clamp(coupling_fast + 0.001f * eta * delta_F, 0.1f, 2.0f);
        coupling_med  = std::clamp(coupling_med  + 0.002f * eta * delta_F, 0.1f, 3.0f);
        coupling_deep = std::clamp(coupling_deep + 0.003f * eta * delta_F, 0.1f, 4.0f);

        // Outer Product Learning across 3 time scales
        for (int i = 0; i < dim; ++i) {
            float err_i = error_vec[i];
            float* mf_row = &M_fast[i * dim];
            float* mm_row = &M_med[i * dim];
            float* md_row = &M_deep[i * dim];

            for (int j = 0; j < dim; ++j) {
                mf_row[j] = mf_row[j] * 0.995f + eta * 1.0f * err_i * trace_fast[j];
                mm_row[j] = mm_row[j] * 0.999f + eta * 1.5f * err_i * trace_med[j];
                md_row[j] = md_row[j] * 0.9999f + eta * 2.0f * err_i * trace_deep[j];
            }
        }
    }
};

struct AutopoieticHierarchicalOrganism {
    int active_dim;
    std::vector<float> Phi_memory;
    std::vector<std::unique_ptr<HierarchicalStorageUnit>> storage_units;
    int total_units_sprouted;
    int total_units_pruned;
    float prev_free_energy;

    AutopoieticHierarchicalOrganism() 
        : active_dim(256), total_units_sprouted(0), total_units_pruned(0), prev_free_energy(1.0f) {
        Phi_memory.assign(VOCAB_SIZE * MAX_DIM, 0.0f);
        sprout_unit();
    }

    void sprout_unit() {
        int new_id = storage_units.size();
        storage_units.push_back(std::make_unique<HierarchicalStorageUnit>(new_id, active_dim));
        total_units_sprouted++;
    }

    void synthesize_response(int current_byte, std::vector<float>& response_field) {
        std::fill(response_field.begin(), response_field.end(), 0.0f);
        const float* phi_v = &Phi_memory[current_byte * MAX_DIM];

        for (int i = 0; i < active_dim; ++i) {
            response_field[i] += phi_v[i];
        }

        for (auto& su : storage_units) {
            su->process(phi_v, response_field);
        }
    }

    void evolve(int current_byte, const float* target_vec, const float* response_field, float free_energy) {
        float delta_F = prev_free_energy - free_energy;
        prev_free_energy = free_energy;

        std::vector<float> error_vec(active_dim);
        for (int i = 0; i < active_dim; ++i) {
            error_vec[i] = target_vec[i] - response_field[i];
        }

        float eta = 0.02f + 0.08f * std::clamp(free_energy, 0.0f, 1.0f);

        float* phi_v = &Phi_memory[current_byte * MAX_DIM];
        for (int i = 0; i < active_dim; ++i) {
            phi_v[i] += eta * error_vec[i];
        }

        for (auto& su : storage_units) {
            su->evolve(phi_v, error_vec.data(), free_energy, delta_F);
        }

        // HIERARCHICAL MORPHOGENESIS (BOTTOM-UP GENESIS & PRUNING):
        if (free_energy > 0.10f && delta_F < 0.001f && storage_units.size() < 16) {
            sprout_unit();
        }

        auto it = std::remove_if(storage_units.begin(), storage_units.end(), [this](const std::unique_ptr<HierarchicalStorageUnit>& su) {
            if (su->energy_level < 0.1f) {
                total_units_pruned++;
                return true;
            }
            return false;
        });
        storage_units.erase(it, storage_units.end());

        if (storage_units.empty()) {
            sprout_unit();
        }
    }
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-797: Deep Hierarchical Autopoietic Tensor Memory" << std::endl;
    std::cout << "  Multi-Time-Scale Memory Wave Interference Engine" << std::endl;
    std::cout << "=================================================================" << std::endl;

    std::string dataset;
    std::vector<std::string> files = {
        "karyon_core.h", 
        "experiments/exp_780_multimodal_autopoietic_benchmark.cpp", 
        "karyon_agent_runtime/agent_core.py"
    };
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

    AutopoieticHierarchicalOrganism organism;

    double total_free_energy = 0.0;
    float final_F = 0.0f;
    int top1_hits = 0;
    double total_rank = 0.0;

    std::vector<float> response_field(MAX_DIM, 0.0f);
    std::vector<float> energies(VOCAB_SIZE, 0.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < total_bytes - 1; ++t) {
        int current_byte = (unsigned char)dataset[t];
        int target_byte = (unsigned char)dataset[t + 1];

        const float* z_targ = &basis[target_byte * MAX_DIM];

        // 1. Synthesize Response Field across Multi-Scale Hierarchical Memory
        organism.synthesize_response(current_byte, response_field);

        // 2. Measure Free Energy F = 1 - <target, response>
        float resonance = avx2_dot_product(z_targ, response_field.data(), MAX_DIM) / (float)MAX_DIM;
        float F = 1.0f - resonance;
        
        if (t == total_bytes - 2) {
            final_F = F;
        }
        total_free_energy += F;

        // 3. Environmental Readout & Rank Evaluation
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            energies[v] = avx2_dot_product(&basis[v * MAX_DIM], response_field.data(), MAX_DIM) / (float)MAX_DIM;
        }

        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        total_rank += rank;

        // 4. Multi-Scale Hierarchical Storage Evolution
        organism.evolve(current_byte, z_targ, response_field.data(), F);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    float top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    double mean_rank = total_rank / (double)(total_bytes - 1);
    double avg_F = total_free_energy / (double)(total_bytes - 1);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-797 HIERARCHICAL TENSOR MEMORY TELEMETRY REPORT" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Top-1 Predictive Accuracy : " << top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank         : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy      : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy        : " << final_F << std::endl;
    std::cout << "  - Active Storage Units     : " << organism.storage_units.size() << std::endl;
    std::cout << "  - Total Units Sprouted     : " << organism.total_units_sprouted << std::endl;
    std::cout << "  - Total Units Pruned       : " << organism.total_units_pruned << std::endl;
    std::cout << "  - Substrate Throughput     : " << (100000.0 / elapsed_sec) << " bytes/sec" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
