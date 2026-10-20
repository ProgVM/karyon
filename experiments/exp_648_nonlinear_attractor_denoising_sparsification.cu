// ======================================================================================
// EXP-648: Non-Linear Attractor Denoising & Orthogonal Sparsification (ND-OSP)
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// In EXP-647, absolute recall was only 10.82% under 85% noise floor, even though
// relative gain was +35%. To achieve true cognitive breakthrough, absolute fidelity must
// exceed 80-90% by fundamentally eradicating the noise floor before hop propagation.
//
// Key Mechanisms:
// 1. High-Dimensional Orthogonal Subspace (D = 256): Exponential drop in random overlap.
// 2. Iterative Attractor Relaxation (R = 4 iterations): Hopfield energy descent.
// 3. Non-Linear Power-Law Sparsification: v <- sign(v) * |v|^p (p=3.0) suppressing background noise.
// 4. Clean Inter-Hop Resonance Routing across 50 chains under 85% noise floor.
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - Cond 1: 64D Direct SVE (EXP-647 baseline)
// - Cond 2: 256D High-Dimensional Direct SVE
// - Cond 3: 256D + Power-Law Sparsification (p=3.0)
// - Cond 4: 256D + Iterative Attractor Denoising (R=4) + Sparsification (Full ND-OSP SVE)
// ======================================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int DIM_LOW          = 64;       // 64D Low-dim baseline
constexpr int DIM_HIGH         = 256;      // 256D High-dim orthogonal space
constexpr int NUM_CHAINS       = 50;       // 50 4-hop associative chains
constexpr int CHAIN_LEN        = 4;        // 4 hops (A -> B -> C -> D)
constexpr int RELAX_STEPS      = 4;        // 4 iterative attractor relaxation steps
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct NDOSP_Result {
    float fid_cond1_64d_sve;
    float fid_cond2_256d_sve;
    float fid_cond3_256d_sparse;
    float fid_cond4_full_ndosp;
    float net_absolute_jump;
    double execution_sec;
};

__device__ inline float dot_prod(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

__device__ inline void normalize_vec(float* v, int n) {
    float norm = sqrtf(dot_prod(v, v, n) + 1e-6f);
    for (int i = 0; i < n; ++i) v[i] /= norm;
}

__device__ inline void power_sparsify(float* v, int n, float p) {
    for (int i = 0; i < n; ++i) {
        float sign = (v[i] >= 0.0f) ? 1.0f : -1.0f;
        v[i] = sign * powf(fabsf(v[i]), p);
    }
    normalize_vec(v, n);
}

__global__ void ndosp_kernel(NDOSP_Result* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 chains in both 64D and 256D spaces
    float nodes_64[NUM_CHAINS][CHAIN_LEN][DIM_LOW];
    float nodes_256[NUM_CHAINS][CHAIN_LEN][DIM_HIGH];

    for (int c = 0; c < NUM_CHAINS; ++c) {
        for (int hop = 0; hop < CHAIN_LEN; ++hop) {
            for (int i = 0; i < DIM_LOW; ++i) nodes_64[c][hop][i] = curand_normal(&st);
            normalize_vec(nodes_64[c][hop], DIM_LOW);

            for (int i = 0; i < DIM_HIGH; ++i) nodes_256[c][hop][i] = curand_normal(&st);
            normalize_vec(nodes_256[c][hop], DIM_HIGH);
        }
    }

    float tot_c1 = 0.0f, tot_c2 = 0.0f, tot_c3 = 0.0f, tot_c4 = 0.0f;
    const float beta = 12.0f;

    for (int target_c = 0; target_c < NUM_CHAINS; ++target_c) {
        // Query cue corrupted by 85% noise floor
        float raw_cue_64[DIM_LOW];
        for (int i = 0; i < DIM_LOW; ++i) {
            raw_cue_64[i] = (curand_uniform(&st) < 0.15f) ? nodes_64[target_c][0][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(raw_cue_64, DIM_LOW);

        float raw_cue_256[DIM_HIGH];
        for (int i = 0; i < DIM_HIGH; ++i) {
            raw_cue_256[i] = (curand_uniform(&st) < 0.15f) ? nodes_256[target_c][0][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(raw_cue_256, DIM_HIGH);

        // -------------------------------------------------------------
        // Cond 1: 64D Direct SVE (EXP-647 baseline)
        // -------------------------------------------------------------
        {
            float state[DIM_LOW];
            for (int i = 0; i < DIM_LOW; ++i) state[i] = raw_cue_64[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = beta * dot_prod(state, nodes_64[c][hop], DIM_LOW);
                    if (l[c] > max_l) max_l = l[c];
                }
                float sum_e = 0.0f;
                for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                float next_state[DIM_LOW] = {0.0f};
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < DIM_LOW; ++i) next_state[i] += a * nodes_64[c][hop + 1][i];
                }
                normalize_vec(next_state, DIM_LOW);
                for (int i = 0; i < DIM_LOW; ++i) state[i] = next_state[i];
            }
            tot_c1 += fmaxf(0.0f, dot_prod(state, nodes_64[target_c][CHAIN_LEN - 1], DIM_LOW));
        }

        // -------------------------------------------------------------
        // Cond 2: 256D High-Dimensional Direct SVE
        // -------------------------------------------------------------
        {
            float state[DIM_HIGH];
            for (int i = 0; i < DIM_HIGH; ++i) state[i] = raw_cue_256[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = beta * dot_prod(state, nodes_256[c][hop], DIM_HIGH);
                    if (l[c] > max_l) max_l = l[c];
                }
                float sum_e = 0.0f;
                for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                float next_state[DIM_HIGH] = {0.0f};
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < DIM_HIGH; ++i) next_state[i] += a * nodes_256[c][hop + 1][i];
                }
                normalize_vec(next_state, DIM_HIGH);
                for (int i = 0; i < DIM_HIGH; ++i) state[i] = next_state[i];
            }
            tot_c2 += fmaxf(0.0f, dot_prod(state, nodes_256[target_c][CHAIN_LEN - 1], DIM_HIGH));
        }

        // -------------------------------------------------------------
        // Cond 3: 256D + Power-Law Sparsification (p = 3.0)
        // -------------------------------------------------------------
        {
            float state[DIM_HIGH];
            for (int i = 0; i < DIM_HIGH; ++i) state[i] = raw_cue_256[i];
            power_sparsify(state, DIM_HIGH, 3.0f);

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = beta * dot_prod(state, nodes_256[c][hop], DIM_HIGH);
                    if (l[c] > max_l) max_l = l[c];
                }
                float sum_e = 0.0f;
                for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                float next_state[DIM_HIGH] = {0.0f};
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < DIM_HIGH; ++i) next_state[i] += a * nodes_256[c][hop + 1][i];
                }
                power_sparsify(next_state, DIM_HIGH, 3.0f);
                for (int i = 0; i < DIM_HIGH; ++i) state[i] = next_state[i];
            }
            tot_c3 += fmaxf(0.0f, dot_prod(state, nodes_256[target_c][CHAIN_LEN - 1], DIM_HIGH));
        }

        // -------------------------------------------------------------
        // Cond 4: Full ND-OSP SVE (256D + Iterative Denoising R=4 + Sparsification)
        // -------------------------------------------------------------
        {
            float state[DIM_HIGH];
            for (int i = 0; i < DIM_HIGH; ++i) state[i] = raw_cue_256[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                // Iterative Hopfield Attractor Denoising Loop
                for (int r = 0; r < RELAX_STEPS; ++r) {
                    power_sparsify(state, DIM_HIGH, 2.5f);
                    float max_l = -1e9f;
                    float l[NUM_CHAINS];
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        l[c] = (beta * 1.5f) * dot_prod(state, nodes_256[c][hop], DIM_HIGH);
                        if (l[c] > max_l) max_l = l[c];
                    }
                    float sum_e = 0.0f;
                    for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                    float clean_state[DIM_HIGH] = {0.0f};
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                        for (int i = 0; i < DIM_HIGH; ++i) clean_state[i] += a * nodes_256[c][hop][i];
                    }
                    normalize_vec(clean_state, DIM_HIGH);
                    for (int i = 0; i < DIM_HIGH; ++i) state[i] = 0.30f * state[i] + 0.70f * clean_state[i];
                    normalize_vec(state, DIM_HIGH);
                }

                // High-precision Hop propagation from cleansed attractor
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = (beta * 2.0f) * dot_prod(state, nodes_256[c][hop], DIM_HIGH);
                    if (l[c] > max_l) max_l = l[c];
                }
                float sum_e = 0.0f;
                for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                float next_state[DIM_HIGH] = {0.0f};
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < DIM_HIGH; ++i) next_state[i] += a * nodes_256[c][hop + 1][i];
                }
                normalize_vec(next_state, DIM_HIGH);
                for (int i = 0; i < DIM_HIGH; ++i) state[i] = next_state[i];
            }
            tot_c4 += fmaxf(0.0f, dot_prod(state, nodes_256[target_c][CHAIN_LEN - 1], DIM_HIGH));
        }
    }

    results[idx].fid_cond1_64d_sve    = tot_c1 / NUM_CHAINS;
    results[idx].fid_cond2_256d_sve   = tot_c2 / NUM_CHAINS;
    results[idx].fid_cond3_256d_sparse= tot_c3 / NUM_CHAINS;
    results[idx].fid_cond4_full_ndosp = tot_c4 / NUM_CHAINS;
    results[idx].net_absolute_jump    = (tot_c4 - tot_c1) / NUM_CHAINS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-648: NON-LINEAR ATTRACTOR DENOISING & SPARSIFICATION (ND-OSP) ===\n"
              << "========================================================================\n"
              << "Core Breakthrough   : 256D Orthogonality + Iterative Denoising + Sparsification\n"
              << "Task Benchmark      : 4-Hop Associative Sequence Recall (85% Noise Floor)\n"
              << "Trials Evaluated    : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    NDOSP_Result* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(NDOSP_Result)));
    std::vector<NDOSP_Result> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    ndosp_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(NDOSP_Result), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.fid_cond1_64d_sve;
        c2   += r.fid_cond2_256d_sve;
        c3   += r.fid_cond3_256d_sparse;
        c4   += r.fid_cond4_full_ndosp;
        gain += r.net_absolute_jump;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL ARCHETYPE                 | ABSOLUTE RECALL | NET JUMP OVER 64D |\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. 64D Direct SVE Baseline (EXP-647)    | " << c1 << " (" << (c1*100.0f) << "%)  | +0.0000 (Baseline)|\n"
              << "| 2. 256D High-Dimensional Space          | " << c2 << " (" << (c2*100.0f) << "%)  | " << std::showpos << (c2 - c1) << std::noshowpos << "             |\n"
              << "| 3. 256D + Power-Law Sparsification      | " << c3 << " (" << (c3*100.0f) << "%)  | " << std::showpos << (c3 - c1) << std::noshowpos << "             |\n"
              << "| 4. Full ND-OSP SVE (Denoise R=4+Sparse) | " << c4 << " (" << (c4*100.0f) << "%)  | " << std::showpos << gain << std::noshowpos << "             |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 chains x 4 conditions): " << sec << " s\n"
              << "Evaluation Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
