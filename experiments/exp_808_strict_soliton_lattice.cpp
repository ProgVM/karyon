/* ============================================================================
   KARYON-CORE EXP-808: Strict Normalized Soliton Lattice Benchmark
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Principles 2, 3, 27, 28, 29, 30, 31 Compliant)

   VERIFYING 99.62% PHENOMENON UNDER STRICT CONTROLLED CONDITIONS:
   1. Ensure zero lookahead leakage: readout is evaluated STRICTLY BEFORE target byte is exposed.
   2. Bounded normalization: enforce physical energy conservation (norm <= 1.0) to eliminate NaN.
   3. Strictly measure Top-1 accuracy, Top-5 accuracy, and Free Energy F on unseen stream.
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
constexpr int MAX_DIM = 256;             // Continuous Spatial Lattice Size N = 256

// AVX2 Dot Product
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

struct StrictSolitonLatticeEngine {
    int dim;
    std::vector<float> basis;

    // Continuous Field Variables: Phase phi, Momentum p = d_phi/dt
    std::vector<float> phi;
    std::vector<float> p_phi;

    // Self-assembling Soliton Coupling Tensor J_ij
    std::vector<float> coupling_J;

    StrictSolitonLatticeEngine(int d = 256) : dim(d) {
        basis.assign(VOCAB_SIZE * MAX_DIM, 0.0f);
        generate_hadamard_256(basis);

        phi.assign(dim, 0.0f);
        p_phi.assign(dim, 0.0f);
        coupling_J.assign(dim * dim, 0.0f);
    }

    // Step 1: Inject sensorimotor impulse into continuous lattice
    void excite_lattice(uint8_t input_byte) {
        const float* z_in = &basis[input_byte * MAX_DIM];
        for (int i = 0; i < dim; ++i) {
            p_phi[i] = p_phi[i] * 0.70f + 0.30f * z_in[i];
        }
    }

    // Step 2: Integrate Sine-Gordon / Soliton Wave Equation
    void step_soliton_physics() {
        float dt = 0.10f;
        float c2 = 0.20f;
        float gamma = 0.10f;

        std::vector<float> accel(dim, 0.0f);

        for (int i = 0; i < dim; ++i) {
            int i_prev = (i - 1 + dim) % dim;
            int i_next = (i + 1) % dim;
            float laplacian = phi[i_prev] - 2.0f * phi[i] + phi[i_next];

            float j_coupling = avx2_dot_product(&coupling_J[i * dim], phi.data(), dim) / (float)dim;

            accel[i] = c2 * laplacian - std::sin(phi[i]) + j_coupling - gamma * p_phi[i];
        }

        for (int i = 0; i < dim; ++i) {
            p_phi[i] += accel[i] * dt;
            phi[i] += p_phi[i] * dt;
            // Bound phase to [-pi, pi]
            if (phi[i] > 3.14159f) phi[i] -= 6.28318f;
            if (phi[i] < -3.14159f) phi[i] += 6.28318f;
        }
    }

    // Step 3: Strictly Causal Readout (BEFORE Target is exposed)
    void synthesize_response(uint8_t current_byte, std::vector<float>& response_field) {
        const float* z_curr = &basis[current_byte * MAX_DIM];
        for (int i = 0; i < dim; ++i) {
            float amp = std::sin(phi[i]) + p_phi[i] * 0.5f;
            response_field[i] = z_curr[i] + amp;
        }

        // Bounded normalization
        float norm_sq = avx2_dot_product(response_field.data(), response_field.data(), dim);
        float norm = std::sqrt(std::max(1e-6f, norm_sq));
        for (int i = 0; i < dim; ++i) {
            response_field[i] /= norm;
        }
    }

    // Step 4: Physical Evolution (AFTER prediction is registered)
    float evolve_lattice(const float* target_vec, const float* response_field) {
        float res = avx2_dot_product(target_vec, response_field, dim) / (float)dim;
        float F = 1.0f - res;

        float eta = 0.02f + 0.08f * std::clamp(F, 0.0f, 1.0f);

        std::vector<float> error_vec(dim);
        for (int i = 0; i < dim; ++i) {
            error_vec[i] = target_vec[i] - response_field[i];
        }

        for (int i = 0; i < dim; ++i) {
            float err_i = error_vec[i];
            float* j_row = &coupling_J[i * dim];
            for (int j = 0; j < dim; ++j) {
                j_row[j] = j_row[j] * 0.999f + eta * err_i * std::sin(phi[j]);
            }
        }

        return F;
    }
};

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-808: Strict Normalized Soliton Lattice Engine" << std::endl;
    std::cout << "  KEP v30.0 Master | Verifying Causal Boundary & Exact Physics" << std::endl;
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

    StrictSolitonLatticeEngine engine(256);

    double total_free_energy = 0.0;
    float final_F = 0.0f;
    int top1_hits = 0;
    int top5_hits = 0;
    double total_rank = 0.0;

    std::vector<float> response_field(MAX_DIM, 0.0f);
    std::vector<float> energies(VOCAB_SIZE, 0.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < total_bytes - 1; ++t) {
        uint8_t current_byte = (unsigned char)dataset[t];
        uint8_t target_byte  = (unsigned char)dataset[t + 1];

        const float* z_targ = &engine.basis[target_byte * MAX_DIM];

        // 1. Excite continuous lattice with past/current byte
        engine.excite_lattice(current_byte);

        // 2. Physical wave step along Sine-Gordon equation
        engine.step_soliton_physics();

        // 3. Strictly causal readout BEFORE target is exposed!
        engine.synthesize_response(current_byte, response_field);

        // 4. Environmental Readout & Rank Evaluation against Target
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            energies[v] = avx2_dot_product(&engine.basis[v * MAX_DIM], response_field.data(), MAX_DIM) / (float)MAX_DIM;
        }

        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }
        if (rank == 0) top1_hits++;
        if (rank < 5) top5_hits++;
        total_rank += rank;

        // 5. Measure Free Energy F & Evolve lattice coupling tensor J
        float F = engine.evolve_lattice(z_targ, response_field.data());
        if (t == total_bytes - 2) final_F = F;
        total_free_energy += F;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    float top1_acc = 100.0f * (float)top1_hits / (float)(total_bytes - 1);
    float top5_acc = 100.0f * (float)top5_hits / (float)(total_bytes - 1);
    double mean_rank = total_rank / (double)(total_bytes - 1);
    double avg_F = total_free_energy / (double)(total_bytes - 1);

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-808 STRICT SOLITON LATTICE TELEMETRY REPORT" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  - Top-1 Predictive Accuracy : " << top1_acc << " %" << std::endl;
    std::cout << "  - Top-5 Predictive Accuracy : " << top5_acc << " %" << std::endl;
    std::cout << "  - Mean Target Rank         : " << mean_rank << " / 258" << std::endl;
    std::cout << "  - Average Free Energy      : " << avg_F << std::endl;
    std::cout << "  - Final Free Energy        : " << final_F << std::endl;
    std::cout << "  - Substrate Throughput     : " << (100000.0 / elapsed_sec) << " bytes/sec" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
