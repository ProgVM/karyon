// ======================================================================================
// EXP-651: Exact Top-1 Attractor Basin Recall & Holographic Resonance (ETB-HR)
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// In EXP-650, we uncovered a critical metric ambiguity: measuring dot product (cosine similarity)
// bounded by signal-to-noise ratio (~0.10) instead of discrete Attractor Hit Rate (Top-1 Accuracy).
//
// In EXP-651, we benchmark the true discrete cognitive retrieval performance:
// When an organism traverses a 4-hop associative chain under 85% noise floor,
// does the final vector state correctly collapse into the exact target attractor basin
// among 50 competing candidate attractors?
//
// Metric: Top-1 Hit Rate (%) = (argmax_j dot(h_final, Target_j) == Target_true) * 100%
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - Cond 1: Standard Monolithic Sequential Hop
// - Cond 2: Synthetic Vector Engine (SVE Multi-Vector Bundle M=4)
// - Cond 3: SVE + Operadic Recombinant Slicing
// - Cond 4: Full SVE Holographic Attractor Engine with Recurrent Denoising
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

constexpr int DIM              = 64;       // 64D Vector Space
constexpr int NUM_CHAINS       = 50;       // 50 4-hop associative chains
constexpr int CHAIN_LEN        = 4;        // 4 hops (A -> B -> C -> D)
constexpr int BUNDLE_SIZE      = 4;        // M = 4 channels
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct ETB_Result {
    float acc_cond1_monolithic;
    float acc_cond2_sve_bundle;
    float acc_cond3_sve_recomb;
    float acc_cond4_holo_sve;
    float net_acc_gain;
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

__global__ void etb_kernel(ETB_Result* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 4-hop chains
    float nodes[NUM_CHAINS][CHAIN_LEN][DIM];
    for (int c = 0; c < NUM_CHAINS; ++c) {
        for (int hop = 0; hop < CHAIN_LEN; ++hop) {
            for (int i = 0; i < DIM; ++i) nodes[c][hop][i] = curand_normal(&st);
            normalize_vec(nodes[c][hop], DIM);
        }
    }

    int hits_c1 = 0, hits_c2 = 0, hits_c3 = 0, hits_c4 = 0;
    const float beta = 12.0f;

    for (int target_c = 0; target_c < NUM_CHAINS; ++target_c) {
        // Query cue is Node A with 85% noise floor
        float raw_cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            raw_cue[i] = (curand_uniform(&st) < 0.15f) ? nodes[target_c][0][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(raw_cue, DIM);

        // -------------------------------------------------------------
        // Cond 1: Standard Monolithic Sequential Hop
        // -------------------------------------------------------------
        {
            float state[DIM];
            for (int i = 0; i < DIM; ++i) state[i] = raw_cue[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = beta * dot_prod(state, nodes[c][hop], DIM);
                    if (l[c] > max_l) max_l = l[c];
                }
                float sum_e = 0.0f;
                for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                float next_state[DIM] = {0.0f};
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < DIM; ++i) next_state[i] += a * nodes[c][hop + 1][i];
                }
                normalize_vec(next_state, DIM);
                for (int i = 0; i < DIM; ++i) state[i] = next_state[i];
            }

            // Top-1 ArgMax Decision among 50 candidate attractors
            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float sim = dot_prod(state, nodes[c][CHAIN_LEN - 1], DIM);
                if (sim > best_sim) { best_sim = sim; pred_c = c; }
            }
            if (pred_c == target_c) hits_c1++;
        }

        // -------------------------------------------------------------
        // Cond 2: Synthetic Vector Engine (Multi-Vector Bundle M=4)
        // -------------------------------------------------------------
        {
            float bundle[BUNDLE_SIZE][DIM];
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) bundle[m][i] = raw_cue[i] + 0.10f * curand_normal(&st);
                normalize_vec(bundle[m], DIM);
            }

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float next_bundle[BUNDLE_SIZE][DIM] = {0.0f};
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    float max_l = -1e9f;
                    float l[NUM_CHAINS];
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        l[c] = beta * dot_prod(bundle[m], nodes[c][hop], DIM);
                        if (l[c] > max_l) max_l = l[c];
                    }
                    float sum_e = 0.0f;
                    for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                        for (int i = 0; i < DIM; ++i) next_bundle[m][i] += a * nodes[c][hop + 1][i];
                    }
                    normalize_vec(next_bundle[m], DIM);
                }
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    for (int i = 0; i < DIM; ++i) bundle[m][i] = next_bundle[m][i];
                }
            }

            float consensus[DIM] = {0.0f};
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) consensus[i] += bundle[m][i];
            }
            normalize_vec(consensus, DIM);

            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float sim = dot_prod(consensus, nodes[c][CHAIN_LEN - 1], DIM);
                if (sim > best_sim) { best_sim = sim; pred_c = c; }
            }
            if (pred_c == target_c) hits_c2++;
        }

        // -------------------------------------------------------------
        // Cond 3: SVE + Operadic Recombinant Slicing (32D Sub-Slices)
        // -------------------------------------------------------------
        {
            float bundle[BUNDLE_SIZE][DIM];
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) bundle[m][i] = raw_cue[i] + 0.10f * curand_normal(&st);
                normalize_vec(bundle[m], DIM);
            }

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float next_bundle[BUNDLE_SIZE][DIM] = {0.0f};
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    float max_l1 = -1e9f, max_l2 = -1e9f;
                    float l1[NUM_CHAINS], l2[NUM_CHAINS];
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        l1[c] = beta * dot_prod(bundle[m], nodes[c][hop], 32);
                        l2[c] = beta * dot_prod(bundle[m] + 32, nodes[c][hop] + 32, 32);
                        if (l1[c] > max_l1) max_l1 = l1[c];
                        if (l2[c] > max_l2) max_l2 = l2[c];
                    }
                    float sum_e1 = 0.0f, sum_e2 = 0.0f;
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        sum_e1 += expf(l1[c] - max_l1);
                        sum_e2 += expf(l2[c] - max_l2);
                    }
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        float a1 = expf(l1[c] - max_l1) / (sum_e1 + 1e-6f);
                        float a2 = expf(l2[c] - max_l2) / (sum_e2 + 1e-6f);
                        for (int i = 0; i < 32; ++i) next_bundle[m][i] += a1 * nodes[c][hop + 1][i];
                        for (int i = 32; i < DIM; ++i) next_bundle[m][i] += a2 * nodes[c][hop + 1][i];
                    }
                    normalize_vec(next_bundle[m], DIM);
                }
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    for (int i = 0; i < DIM; ++i) bundle[m][i] = next_bundle[m][i];
                }
            }

            float consensus[DIM] = {0.0f};
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) consensus[i] += bundle[m][i];
            }
            normalize_vec(consensus, DIM);

            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float sim = dot_prod(consensus, nodes[c][CHAIN_LEN - 1], DIM);
                if (sim > best_sim) { best_sim = sim; pred_c = c; }
            }
            if (pred_c == target_c) hits_c3++;
        }

        // -------------------------------------------------------------
        // Cond 4: Full SVE Holographic Attractor Engine
        // Bundle + Recombinant Slicing + Consensus Attractor Locking
        // -------------------------------------------------------------
        {
            float bundle[BUNDLE_SIZE][DIM];
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) bundle[m][i] = raw_cue[i] + 0.10f * curand_normal(&st);
                normalize_vec(bundle[m], DIM);
            }

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                // Consensus anchor across channels before hop transition
                float consensus[DIM] = {0.0f};
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    for (int i = 0; i < DIM; ++i) consensus[i] += bundle[m][i];
                }
                normalize_vec(consensus, DIM);

                // Denoising hop projection
                float next_bundle[BUNDLE_SIZE][DIM] = {0.0f};
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    float max_l = -1e9f;
                    float l[NUM_CHAINS];
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        float s_sim = dot_prod(bundle[m], nodes[c][hop], DIM);
                        float c_sim = dot_prod(consensus, nodes[c][hop], DIM);
                        l[c] = (beta * 1.5f) * (0.50f * s_sim + 0.50f * c_sim);
                        if (l[c] > max_l) max_l = l[c];
                    }
                    float sum_e = 0.0f;
                    for (int c = 0; c < NUM_CHAINS; ++c) sum_e += expf(l[c] - max_l);
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        float a = expf(l[c] - max_l) / (sum_e + 1e-6f);
                        for (int i = 0; i < DIM; ++i) next_bundle[m][i] += a * nodes[c][hop + 1][i];
                    }
                    normalize_vec(next_bundle[m], DIM);
                }
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    for (int i = 0; i < DIM; ++i) bundle[m][i] = next_bundle[m][i];
                }
            }

            // Final Global Consensus Readout
            float final_rep[DIM] = {0.0f};
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) final_rep[i] += bundle[m][i];
            }
            normalize_vec(final_rep, DIM);

            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float sim = dot_prod(final_rep, nodes[c][CHAIN_LEN - 1], DIM);
                if (sim > best_sim) { best_sim = sim; pred_c = c; }
            }
            if (pred_c == target_c) hits_c4++;
        }
    }

    results[idx].acc_cond1_monolithic = (float)hits_c1 / NUM_CHAINS;
    results[idx].acc_cond2_sve_bundle = (float)hits_c2 / NUM_CHAINS;
    results[idx].acc_cond3_sve_recomb = (float)hits_c3 / NUM_CHAINS;
    results[idx].acc_cond4_holo_sve   = (float)hits_c4 / NUM_CHAINS;
    results[idx].net_acc_gain         = ((float)hits_c4 - (float)hits_c1) / NUM_CHAINS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-651: EXACT TOP-1 ATTRACTOR BASIN RECALL BENCHMARK            ===\n"
              << "========================================================================\n"
              << "Target Objective    : Discrete Top-1 Classification Accuracy (1 out of 50)\n"
              << "Task Benchmark      : 4-Hop Associative Sequence Recall (85% Noise Floor)\n"
              << "Random Chance Floor : 1 / 50 = 2.00%\n"
              << "Trials Evaluated    : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    ETB_Result* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(ETB_Result)));
    std::vector<ETB_Result> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    etb_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(ETB_Result), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.acc_cond1_monolithic;
        c2   += r.acc_cond2_sve_bundle;
        c3   += r.acc_cond3_sve_recomb;
        c4   += r.acc_cond4_holo_sve;
        gain += r.net_acc_gain;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL ARCHETYPE                 | TOP-1 HIT ACCURACY | NET JUMP OVER BASE|\n"
              << "---------------------------------------------------------------------------------\n"
              << "| Random Guess Ceiling (1 / 50)           |  2.00%             | -                 |\n"
              << "| 1. Monolithic Vector Sequential Hop     | " << (c1 * 100.0f) << "%             | +0.00% (Baseline) |\n"
              << "| 2. SVE Multi-Vector Bundle (M=4)        | " << (c2 * 100.0f) << "%             | " << std::showpos << ((c2 - c1) * 100.0f) << "%           " << std::noshowpos << "|\n"
              << "| 3. SVE + Operadic Recombinant Slicing   | " << (c3 * 100.0f) << "%             | " << std::showpos << ((c3 - c1) * 100.0f) << "%           " << std::noshowpos << "|\n"
              << "| 4. Full SVE Holographic Attractor Engine| " << (c4 * 100.0f) << "%             | " << std::showpos << (gain * 100.0f) << "%           " << std::noshowpos << "|\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 chains x 4 conditions): " << sec << " s\n"
              << "Evaluation Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
