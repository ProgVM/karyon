/* ============================================================================
   KARYON-CORE EXP-796: Continuous Field Matrix Memory Genesis (KEP v30.0)
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Principles 2, 3, 4, 5, 6, 27, 29, 30, 31 Compliant)

   SOVEREIGN BAZILEVS DIRECTIVE:
   "The current autopoietic system doesn't know how to create storage structures."

   CONTINUOUS FIELD MATRIX MORPHOGENESIS:
   Instead of hardcoding discrete slot arrays or memory tables, the continuous physical
   substrate itself generates a dynamic rank-1 outer product field update:
      dM/dt = -gamma * M + eta * (error_t x trace_t)
   where trace_t is an endogenous continuous physical trace of previous states.

   When local Free Energy F rises due to context ambiguity, the substrate autonomously
   sprouts continuous non-local memory coupling matrices M_k.

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

// AVX2 SIMD Dot Product
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
// DYNAMIC STORAGE MATRIX FIELD (Continuous Storage Structure)
// ----------------------------------------------------------------------------
struct ContinuousStorageMatrix {
    int id;
    int dim;
    std::vector<float> M;       // Dynamic Memory Matrix R^(D x D)
    std::vector<float> trace;   // Continuous Physical State Trace R^D
    float decay_rate;           // Endogenous Memory Retention Parameter
    float coupling_alpha;       // Coupling Strength into main field
    float energy_level;         // Liveness / Survival Energy
    float causal_credit;

    ContinuousStorageMatrix(int mat_id, int d) 
        : id(mat_id), dim(d), decay_rate(0.90f), coupling_alpha(0.50f), energy_level(1.0f), causal_credit(0.0f) {
        M.assign(dim * dim, 0.0f);
        trace.assign(dim, 0.0f);
    }

    // Continuous Read & Write Stream
    void process(const float* current_phi, std::vector<float>& response_acc) {
        // 1. Update continuous physical state trace
        for (int i = 0; i < dim; ++i) {
            trace[i] = trace[i] * decay_rate + current_phi[i] * (1.0f - decay_rate);
        }

        // 2. Readout from continuous memory matrix M
        for (int i = 0; i < dim; ++i) {
            float m_read = avx2_dot_product(&M[i * dim], trace.data(), dim) / (float)dim;
            response_acc[i] += m_read * coupling_alpha * energy_level;
        }
    }

    // Outer-Product Memory Matrix Evolution
    void evolve(const float* current_phi, const float* error_vec, float free_energy, float delta_F) {
        causal_credit = 0.95f * causal_credit + 0.05f * delta_F;

        if (delta_F > 0.0f) {
            energy_level = std::min(2.5f, energy_level + 0.015f * delta_F);
        } else {
            energy_level -= 0.003f; // Physical dissipation
        }

        float eta = 0.01f + 0.04f * std::clamp(free_energy, 0.0f, 1.0f);

        // Self-adjust retention and coupling parameters based on local F reduction
        decay_rate = std::clamp(decay_rate + 0.001f * eta * delta_F, 0.60f, 0.98f);
        coupling_alpha = std::clamp(coupling_alpha + 0.002f * eta * delta_F, 0.10f, 1.50f);

        // Rank-1 Outer Product Matrix Update: dM_ij = eta * error_i * trace_j
        for (int i = 0; i < dim; ++i) {
            float err_i = error_vec[i];
            float* m_row = &M[i * dim];
            for (int j = 0; j < dim; ++j) {
                m_row[j] = m_row[j] * 0.999f + eta * err_i * trace[j];
            }
        }
    }
};

struct AutopoieticMatrixStorageOrganism {
    int active_dim;
    std::vector<float> Phi_memory; // Instantaneous State Field V x D
    std::vector<std::unique_ptr<ContinuousStorageMatrix>> storage_matrices;
    int total_matrices_sprouted;
    int total_matrices_pruned;
    float prev_free_energy;

    AutopoieticMatrixStorageOrganism() 
        : active_dim(256), total_matrices_sprouted(0), total_matrices_pruned(0), prev_free_energy(1.0f) {
        Phi_memory.assign(VOCAB_SIZE * MAX_DIM, 0.0f);
        sprout_matrix();
    }

    void sprout_matrix() {
        int new_id = storage_matrices.size();
        storage_matrices.push_back(std::make_unique<ContinuousStorageMatrix>(new_id, active_dim));
        total_matrices_sprouted++;
    }

    void synthesize_response(int current_byte, std::vector<float>& response_field) {
        std::fill(response_field.begin(), response_field.end(), 0.0f);
        const float* phi_v = &Phi_memory[current_byte * MAX_DIM];

        for (int i = 0; i < active_dim; ++i) {
            response_field[i] += phi_v[i];
        }

        for (auto& sm : storage_matrices) {
            sm->process(phi_v, response_field);
        }
    }

    void evolve(int current_byte, const float* target_vec, const float* response_field, float free_energy) {
        float delta_F = prev_free_energy - free_energy;
        prev_free_energy = free_energy;

        std::vector<float> error_vec(active_dim);
        for (int i = 0; i < active_dim; ++i) {
            error_vec[i] = target_vec[i] - response_field[i];
        }

        float eta = 0.02f + 0.05f * std::clamp(free_energy, 0.0f, 1.0f);

        float* phi_v = &Phi_memory[current_byte * MAX_DIM];
        for (int i = 0; i < active_dim; ++i) {
            phi_v[i] += eta * error_vec[i];
        }

        for (auto& sm : storage_matrices) {
            sm->evolve(phi_v, error_vec.data(), free_energy, delta_F);
        }

        // CONTINUOUS STORAGE MORPHOGENESIS:
        // Sprout if Free Energy remains high and unreduced
        if (free_energy > 0.15f && delta_F < 0.001f && storage_matrices.size() < 12) {
            sprout_matrix();
        }

        // Prune if energy level drops below threshold (causally useless)
        auto it = std::remove_if(storage_matrices.begin(), storage_matrices.end(), [this](const std::unique_ptr<ContinuousStorageMatrix>& sm) {
            if (sm->energy_level < 0.1f) {
                total_matrices_pruned++;
                return true;
            }
            return false;
        });
        storage_matrices.erase(it, storage_matrices.end());

        if (storage_matrices.empty()) {
            sprout_matrix();
        }
    }
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-796: Continuous Field Matrix Memory Engine" << std::endl;
    std::cout << "  KEP v30.0 Sovereign Master | Outer-Product Field Morphogenesis" << std::endl;
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

    AutopoieticMatrixStorageOrganism organism;

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

        // 1. Synthesize Physical Field Response with Continuous Outer-Product Memory
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

        // 4. Pure Autopoietic Matrix Storage Morphogenesis Evolution
        organism.evolve(current_byte, z_targ, response_field.data(), F);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    float top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    double mean_rank = total_rank / (double)(total_bytes - 1);
    double avg_F = total_free_energy / (double)(total_bytes - 1);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-796 CONTINUOUS MATRIX STORAGE TELEMETRY REPORT" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Top-1 Predictive Accuracy : " << top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank         : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy      : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy        : " << final_F << std::endl;
    std::cout << "  - Active Storage Matrices  : " << organism.storage_matrices.size() << std::endl;
    std::cout << "  - Total Matrices Sprouted  : " << organism.total_matrices_sprouted << std::endl;
    std::cout << "  - Total Matrices Pruned    : " << organism.total_matrices_pruned << std::endl;
    std::cout << "  - Substrate Throughput     : " << (100000.0 / elapsed_sec) << " bytes/sec" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
