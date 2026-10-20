// ======================================================================================
// EXP-645: Unified Synthetic Vector Engine (SVE) Benchmark
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// Synthesize the 4 validated positive vector laws:
//   1. Multi-Vector Bundle Encoding (EXP-638)
//   2. Dynamic Fission & Fusion (EXP-639)
//   3. Operadic Recombinant Hypergraph Slicing (EXP-640)
//   4. Heterogeneous Clonal Swarm Search (EXP-641)
// into a unified Autonomous Synthetic Vector Engine (SVE).
//
// Task: Multi-Hop Associative Sequence Reasoning under Extreme 85% Noise Floor.
// Given noisy cues A -> B -> C -> D across 50 independent complex fact chains,
// evaluate associative recall and chain completion fidelity.
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - Cond 1: Monolithic Vector Baseline (Single 64D Vector + Standard Cosine Hop)
// - Cond 2: Dual Multi-Vector Bundle (EXP-638 only)
// - Cond 3: Bundle + Recombinant Slicing (EXP-638 + EXP-640)
// - Cond 4: Full SVE Engine (Bundle + Fission/Fusion + Hypergraph Slicing + Clonal Swarms)
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
constexpr int NUM_CHAINS       = 50;       // 50 4-hop associative chains (A -> B -> C -> D)
constexpr int CHAIN_LEN        = 4;        // 4 hops: A -> B -> C -> D
constexpr int NUM_CLONES       = 8;        // K = 8 mutational search clones
constexpr int BUNDLE_SIZE      = 4;        // M = 4 multi-vector bundle channels
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct SVEResult {
    float fid_cond1_monolithic;
    float fid_cond2_bundle;
    float fid_cond3_bundle_recomb;
    float fid_cond4_full_sve;
    float net_sve_gain;
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

__global__ void unified_sve_kernel(SVEResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 4-hop chains: Chain c: node[c][0] -> node[c][1] -> node[c][2] -> node[c][3]
    float nodes[NUM_CHAINS][CHAIN_LEN][DIM];
    for (int c = 0; c < NUM_CHAINS; ++c) {
        for (int hop = 0; hop < CHAIN_LEN; ++hop) {
            for (int i = 0; i < DIM; ++i) nodes[c][hop][i] = curand_normal(&st);
            normalize_vec(nodes[c][hop], DIM);
        }
    }

    float tot_c1 = 0.0f, tot_c2 = 0.0f, tot_c3 = 0.0f, tot_c4 = 0.0f;
    const float beta = 12.0f;

    for (int target_c = 0; target_c < NUM_CHAINS; ++target_c) {
        // Query cue is Node A (hop 0) corrupted by 85% noise floor.
        // Task: traverse 3 consecutive hops to recall Node D (hop 3).
        float cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            cue[i] = (curand_uniform(&st) < 0.15f) ? nodes[target_c][0][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(cue, DIM);

        // -------------------------------------------------------------
        // Cond 1: Monolithic Baseline (Single Vector Sequential Hop)
        // -------------------------------------------------------------
        {
            float curr_state[DIM];
            for (int i = 0; i < DIM; ++i) curr_state[i] = cue[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = beta * dot_prod(curr_state, nodes[c][hop], DIM);
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
                for (int i = 0; i < DIM; ++i) curr_state[i] = next_state[i];
            }
            tot_c1 += fmaxf(0.0f, dot_prod(curr_state, nodes[target_c][CHAIN_LEN - 1], DIM));
        }

        // -------------------------------------------------------------
        // Cond 2: Multi-Vector Bundle Encoding (M = 4 channels)
        // -------------------------------------------------------------
        {
            float bundle[BUNDLE_SIZE][DIM];
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) {
                    bundle[m][i] = cue[i] + 0.10f * curand_normal(&st);
                }
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

            // Consensus Readout across bundle channels
            float consensus[DIM] = {0.0f};
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) consensus[i] += bundle[m][i];
            }
            normalize_vec(consensus, DIM);
            tot_c2 += fmaxf(0.0f, dot_prod(consensus, nodes[target_c][CHAIN_LEN - 1], DIM));
        }

        // -------------------------------------------------------------
        // Cond 3: Bundle + Recombinant Slicing (EXP-638 + EXP-640)
        // -------------------------------------------------------------
        {
            float bundle[BUNDLE_SIZE][DIM];
            for (int m = 0; m < BUNDLE_SIZE; ++m) {
                for (int i = 0; i < DIM; ++i) bundle[m][i] = cue[i] + 0.10f * curand_normal(&st);
                normalize_vec(bundle[m], DIM);
            }

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float next_bundle[BUNDLE_SIZE][DIM] = {0.0f};
                for (int m = 0; m < BUNDLE_SIZE; ++m) {
                    // Recombinant sub-slice matching: split into 2 32D sub-slices
                    float max_l_s1 = -1e9f, max_l_s2 = -1e9f;
                    float l_s1[NUM_CHAINS], l_s2[NUM_CHAINS];
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        l_s1[c] = beta * dot_prod(bundle[m], nodes[c][hop], 32);
                        l_s2[c] = beta * dot_prod(bundle[m] + 32, nodes[c][hop] + 32, 32);
                        if (l_s1[c] > max_l_s1) max_l_s1 = l_s1[c];
                        if (l_s2[c] > max_l_s2) max_l_s2 = l_s2[c];
                    }
                    float sum_e1 = 0.0f, sum_e2 = 0.0f;
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        sum_e1 += expf(l_s1[c] - max_l_s1);
                        sum_e2 += expf(l_s2[c] - max_l_s2);
                    }
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        float a1 = expf(l_s1[c] - max_l_s1) / (sum_e1 + 1e-6f);
                        float a2 = expf(l_s2[c] - max_l_s2) / (sum_e2 + 1e-6f);
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
            tot_c3 += fmaxf(0.0f, dot_prod(consensus, nodes[target_c][CHAIN_LEN - 1], DIM));
        }

        // -------------------------------------------------------------
        // Cond 4: Full Synthetic Vector Engine (SVE)
        // Bundle + Fission/Fusion + Hypergraph Slicing + Clonal Swarms
        // -------------------------------------------------------------
        {
            // Spawn mutational clonal swarm K=8 across bundle channels
            float clones[NUM_CLONES][DIM];
            for (int k = 0; k < NUM_CLONES; ++k) {
                for (int i = 0; i < DIM; ++i) {
                    clones[k][i] = cue[i] + 0.15f * curand_normal(&st);
                }
                normalize_vec(clones[k], DIM);
            }

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float next_clones[NUM_CLONES][DIM] = {0.0f};
                float clone_conf[NUM_CLONES];

                for (int k = 0; k < NUM_CLONES; ++k) {
                    // Hypergraph sub-slice scoring
                    float max_l1 = -1e9f, max_l2 = -1e9f;
                    float l1[NUM_CHAINS], l2[NUM_CHAINS];
                    for (int c = 0; c < NUM_CHAINS; ++c) {
                        l1[c] = beta * dot_prod(clones[k], nodes[c][hop], 32);
                        l2[c] = beta * dot_prod(clones[k] + 32, nodes[c][hop] + 32, 32);
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
                        for (int i = 0; i < 32; ++i) next_clones[k][i] += a1 * nodes[c][hop + 1][i];
                        for (int i = 32; i < DIM; ++i) next_clones[k][i] += a2 * nodes[c][hop + 1][i];
                    }
                    normalize_vec(next_clones[k], DIM);

                    // Dynamic Fission/Fusion Confidence evaluation
                    clone_conf[k] = 0.5f * (expf(max_l1 / beta) + expf(max_l2 / beta));
                }

                // Mutational Reseeding: Best clones absorb & guide weaker clones (Fusion)
                float max_conf = -1e9f;
                int best_k = 0;
                for (int k = 0; k < NUM_CLONES; ++k) {
                    if (clone_conf[k] > max_conf) { max_conf = clone_conf[k]; best_k = k; }
                }

                for (int k = 0; k < NUM_CLONES; ++k) {
                    if (k != best_k && clone_conf[k] < 0.8f * max_conf) {
                        // Fission-mutate towards best clone
                        for (int i = 0; i < DIM; ++i) {
                            clones[k][i] = 0.70f * next_clones[best_k][i] + 0.30f * next_clones[k][i] + 0.05f * curand_normal(&st);
                        }
                    } else {
                        for (int i = 0; i < DIM; ++i) clones[k][i] = next_clones[k][i];
                    }
                    normalize_vec(clones[k], DIM);
                }
            }

            // Consensus Readout across top-performing clones
            float final_rep[DIM] = {0.0f};
            for (int k = 0; k < NUM_CLONES; ++k) {
                for (int i = 0; i < DIM; ++i) final_rep[i] += clones[k][i];
            }
            normalize_vec(final_rep, DIM);
            tot_c4 += fmaxf(0.0f, dot_prod(final_rep, nodes[target_c][CHAIN_LEN - 1], DIM));
        }
    }

    results[idx].fid_cond1_monolithic   = tot_c1 / NUM_CHAINS;
    results[idx].fid_cond2_bundle       = tot_c2 / NUM_CHAINS;
    results[idx].fid_cond3_bundle_recomb= tot_c3 / NUM_CHAINS;
    results[idx].fid_cond4_full_sve     = tot_c4 / NUM_CHAINS;
    results[idx].net_sve_gain           = (tot_c4 - tot_c1) / NUM_CHAINS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-645: UNIFIED SYNTHETIC VECTOR ENGINE (SVE) BENCHMARK         ===\n"
              << "========================================================================\n"
              << "Synthesis           : Bundle Encoding + Fission/Fusion + Hypergraph + Clones\n"
              << "Task Benchmark      : 4-Hop Multi-Step Associative Reasoning (A->B->C->D)\n"
              << "Stress Condition    : 50 Complex Chains, 85% Extreme Noise Floor\n"
              << "Trials Evaluated    : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    SVEResult* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(SVEResult)));
    std::vector<SVEResult> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    unified_sve_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(SVEResult), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.fid_cond1_monolithic;
        c2   += r.fid_cond2_bundle;
        c3   += r.fid_cond3_bundle_recomb;
        c4   += r.fid_cond4_full_sve;
        gain += r.net_sve_gain;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL SYSTEM                    | 4-HOP RECALL FID| NET GAIN OVER BASE|\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Monolithic Vector Baseline           | " << c1 << "          | +0.0000 (Baseline)|\n"
              << "| 2. Multi-Vector Bundle (M=4)            | " << c2 << "          | " << std::showpos << (c2 - c1) << std::noshowpos << "             |\n"
              << "| 3. Bundle + Recombinant Slicing         | " << c3 << "          | " << std::showpos << (c3 - c1) << std::noshowpos << "             |\n"
              << "| 4. Full Synthetic Vector Engine (SVE)   | " << c4 << "          | " << std::showpos << gain << std::noshowpos << "             |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 chains x 4 conditions): " << sec << " s\n"
              << "Evaluation Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
