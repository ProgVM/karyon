/* ============================================================================
   KARYON-CORE EXP-792B: Stable Universal Autopoietic Continuum Engine
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

struct ContinuumUnit {
    int id;
    int dim;
    std::vector<float> W;
    float c1, c3;
    float tau;
    std::vector<float> h;
    float causal_credit;
    float energy_level;

    ContinuumUnit(int unit_id, int d) 
        : id(unit_id), dim(d), c1(1.0f), c3(-0.01f), tau(1.0f), causal_credit(0.0f), energy_level(1.0f) {
        W.assign(dim * dim, 0.0f);
        h.assign(dim, 0.0f);
        for (int i = 0; i < dim; ++i) {
            W[i * dim + i] = 1.0f;
        }
    }

    inline float apply_dynamic_nonlinearity(float x) const {
        return c1 * x + c3 * x * x * x;
    }

    void propagate(const float* input_vec, std::vector<float>& output_acc) {
        for (int i = 0; i < dim; ++i) {
            float linear_in = avx2_dot_product(&W[i * dim], input_vec, dim) / (float)dim;
            h[i] = h[i] * 0.9f + linear_in * 0.1f;
            float active_val = apply_dynamic_nonlinearity(h[i]);
            output_acc[i] += active_val * energy_level;
        }
    }

    void evolve(const float* input_vec, const float* error_vec, float free_energy, float delta_F) {
        causal_credit = 0.95f * causal_credit + 0.05f * delta_F;
        
        if (delta_F > 0.0f) {
            energy_level = std::min(2.0f, energy_level + 0.01f * delta_F);
        } else {
            energy_level -= 0.002f;
        }

        float eta = 0.01f + 0.04f * std::clamp(free_energy, 0.0f, 1.0f);

        c1 = std::clamp(c1 + 0.0001f * eta * delta_F, 0.5f, 2.0f);
        c3 = std::clamp(c3 + 0.0001f * eta * delta_F, -0.1f, 0.0f);

        for (int i = 0; i < dim; ++i) {
            float err_i = error_vec[i];
            float* w_row = &W[i * dim];
            for (int j = 0; j < dim; ++j) {
                w_row[j] += eta * err_i * input_vec[j];
            }
        }
    }
};

struct AutopoieticContinuumOrganism {
    int active_dim;
    std::vector<float> Phi_memory;
    std::vector<std::unique_ptr<ContinuumUnit>> units;
    int total_units_sprouted;
    int total_units_pruned;
    float prev_free_energy;

    AutopoieticContinuumOrganism() 
        : active_dim(256), total_units_sprouted(0), total_units_pruned(0), prev_free_energy(1.0f) {
        Phi_memory.assign(VOCAB_SIZE * MAX_DIM, 0.0f);
        sprout_unit();
    }

    void sprout_unit() {
        int new_id = units.size();
        units.push_back(std::make_unique<ContinuumUnit>(new_id, active_dim));
        total_units_sprouted++;
    }

    void synthesize_response(int current_byte, std::vector<float>& response_field) {
        std::fill(response_field.begin(), response_field.end(), 0.0f);
        const float* phi_v = &Phi_memory[current_byte * MAX_DIM];

        for (int i = 0; i < active_dim; ++i) {
            response_field[i] += phi_v[i];
        }

        for (auto& unit : units) {
            unit->propagate(phi_v, response_field);
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

        for (auto& unit : units) {
            unit->evolve(phi_v, error_vec.data(), free_energy, delta_F);
        }

        // Sprouting & Pruning driven purely by local Free Energy & Causal Credit
        if (free_energy > 0.20f && delta_F < 0.001f && units.size() < 12) {
            sprout_unit();
        }

        auto it = std::remove_if(units.begin(), units.end(), [this](const std::unique_ptr<ContinuumUnit>& u) {
            if (u->energy_level < 0.1f) {
                total_units_pruned++;
                return true;
            }
            return false;
        });
        units.erase(it, units.end());

        if (units.empty()) {
            sprout_unit();
        }
    }
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-792B: Stable Autopoietic Continuum Engine" << std::endl;
    std::cout << "  Zero Hardcoded Details | Pure Local Free Energy Morphogenesis" << std::endl;
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

    AutopoieticContinuumOrganism organism;

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

        organism.synthesize_response(current_byte, response_field);

        float resonance = avx2_dot_product(z_targ, response_field.data(), MAX_DIM) / (float)MAX_DIM;
        float F = 1.0f - resonance;
        if (t == total_bytes - 2) final_F = F;
        total_free_energy += F;

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

        organism.evolve(current_byte, z_targ, response_field.data(), F);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    float top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    double mean_rank = total_rank / (double)(total_bytes - 1);
    double avg_F = total_free_energy / (double)(total_bytes - 1);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-792B AUTOPOIETIC CONTINUUM TELEMETRY REPORT" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Top-1 Predictive Accuracy : " << top1_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank         : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy      : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy        : " << final_F << std::endl;
    std::cout << "  - Active Continuum Units   : " << organism.units.size() << std::endl;
    std::cout << "  - Total Units Sprouted     : " << organism.total_units_sprouted << std::endl;
    std::cout << "  - Total Units Pruned       : " << organism.total_units_pruned << std::endl;
    std::cout << "  - Substrate Throughput     : " << (100000.0 / elapsed_sec) << " bytes/sec" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
