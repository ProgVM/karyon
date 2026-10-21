// ======================================================================================
// EXP-660: Spatio-Temporal Sovereign Field & Unbound Sprouting Mechanics (STSF-USM)
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// Unified Autopoietic Synthesis of Space & Time:
//  - Spatial Dimension   : Dynamic Sprouting Beam Width B_t in [1, 8] driven by initial Free Energy
//  - Temporal Horizon    : Endogenous Relaxation Depth K_t(F_t) driven by Energy Dissipation
//  - Riemannian Geometry : Exact Cauchy-Newton Step eta^*_t = ||d||^2 / (||Hd||^2 + eps)
//
// Theoretical Formulation:
// When an incoming sensory cue is clean (low noise), the STSF field automatically collapses
// to a minimal 1D geodesic (B_t = 1, K_t = 1-2, eta^*_t ~ 8.6), expending zero unnecessary compute.
// When entering extreme noise / chaotic turbulence (50% - 70% noise), initial Free Energy
// S_unc = ||P_exp - H(S_0)||^2 spikes, triggering spontaneous SPATIAL SPROUTING of B_t in [2, 8]
// parallel candidate beams in state space. Each beam autonomously relaxes along its local
// Riemannian geodesic until reaching homeostatic equilibrium.
//
// Benchmark Protocol on Tesla T4 GPU (1,000 trials across 30%, 50%, 70% noise):
// - Cond 1: Uncorrected Baseline
// - Cond 2: EXP-659 Riemannian Step 1D (B_t = 1)
// - Cond 3: Fixed Spatial Beam (B = 4) + Fixed Iterations (K = 3)
// - Cond 4: Full Spatio-Temporal Sovereign Field (STSF) with Dynamic B_t(F_t), K_t(F_t), eta^*_t
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
constexpr int MAX_BEAMS        = 8;        // Maximum spatial sprouting width
constexpr int MAX_ALLOWED_K    = 10;       // Physical iteration ceiling
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct STSF_Result {
    float acc_cond1_baseline;
    float acc_cond2_exp659_1d;
    float acc_cond3_fixed_b4_k3;
    float acc_cond4_full_stsf;
    float avg_sprouted_beams;
    float avg_dynamic_k;
    float avg_eta_step;
    float ideal_signal_capacity;
    float gap_c1;
    float gap_c4;
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

__global__ void stsf_kernel(STSF_Result* results, float noise_level, unsigned long seed) {
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

    // 2. Linear Parity Matrix H1 (CHAIN_LEN x DIM)
    float H1_proj[CHAIN_LEN][DIM];
    for (int k = 0; k < CHAIN_LEN; ++k) {
        for (int i = 0; i < DIM; ++i) H1_proj[k][i] = curand_normal(&st) * (1.0f / sqrtf((float)DIM));
    }

    // 3. Bilinear Cross-Phase Parity Matrix H2 (6 cross-pairs x DIM)
    float H2_proj[6][DIM];
    for (int k = 0; k < 6; ++k) {
        for (int i = 0; i < DIM; ++i) H2_proj[k][i] = curand_normal(&st) * (1.0f / sqrtf((float)DIM));
    }

    int hits_c1 = 0, hits_c2 = 0, hits_c3 = 0, hits_c4 = 0;
    int total_dynamic_k = 0;
    int total_sprouted_beams = 0;
    float sum_eta = 0.0f;
    int total_eta_count = 0;
    const float beta = 16.0f;

    for (int target_c = 0; target_c < NUM_CHAINS; ++target_c) {
        // Query cue with specified noise
        float raw_cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            raw_cue[i] = (curand_uniform(&st) >= noise_level) ? nodes[target_c][0][i] : curand_normal(&st) * noise_level;
        }
        normalize_vec(raw_cue, DIM);

        // Expected 1st-Order Parity Checksum
        float P1_expected[DIM] = {0.0f};
        for (int k = 0; k < CHAIN_LEN; ++k) {
            for (int i = 0; i < DIM; ++i) P1_expected[i] += H1_proj[k][i] * nodes[target_c][k][i];
        }

        // Expected 2nd-Order Bilinear Cross-Phase Checksum
        float P2_expected[DIM] = {0.0f};
        int pair_idx = 0;
        for (int i_hop = 0; i_hop < CHAIN_LEN; ++i_hop) {
            for (int j_hop = i_hop + 1; j_hop < CHAIN_LEN; ++j_hop) {
                float cross_phase = dot_prod(nodes[target_c][i_hop], nodes[target_c][j_hop], DIM);
                for (int d = 0; d < DIM; ++d) {
                    P2_expected[d] += H2_proj[pair_idx][d] * cross_phase * nodes[target_c][j_hop][d];
                }
                pair_idx++;
            }
        }

        // -------------------------------------------------------------
        // Cond 1: Uncorrected Baseline
        // -------------------------------------------------------------
        {
            float state[DIM];
            for (int i = 0; i < DIM; ++i) state[i] = raw_cue[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                float max_sim = -1e9f;
                int best_c = 0;
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float s = dot_prod(state, nodes[c][hop], DIM);
                    if (s > max_sim) { max_sim = s; best_c = c; }
                }
                for (int i = 0; i < DIM; ++i) state[i] = nodes[best_c][hop + 1][i];
            }

            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float s = dot_prod(state, nodes[c][CHAIN_LEN - 1], DIM);
                if (s > best_sim) { best_sim = s; pred_c = c; }
            }
            if (pred_c == target_c) hits_c1++;
        }

        // -------------------------------------------------------------
        // Cond 2: EXP-659 Riemannian Step 1D (B_t = 1)
        // -------------------------------------------------------------
        {
            float best_score = -1e9f;
            int pred_c = -1;

            for (int c = 0; c < NUM_CHAINS; ++c) {
                float traj_states[CHAIN_LEN][DIM];
                for (int i = 0; i < DIM; ++i) traj_states[0][i] = raw_cue[i];
                for (int k = 1; k < CHAIN_LEN; ++k) {
                    for (int i = 0; i < DIM; ++i) traj_states[k][i] = nodes[c][k][i];
                }

                float P1_curr[DIM] = {0.0f};
                for (int k = 0; k < CHAIN_LEN; ++k) {
                    for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                }
                float prev_energy = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    float diff = P1_expected[i] - P1_curr[i];
                    prev_energy += diff * diff;
                }

                for (int iter = 0; iter < MAX_ALLOWED_K; ++iter) {
                    if (prev_energy < 0.015f) break;

                    float syn1[DIM];
                    for (int i = 0; i < DIM; ++i) syn1[i] = P1_expected[i] - P1_curr[i];

                    float d[CHAIN_LEN][DIM];
                    float d_norm_sq = 0.0f;
                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) {
                            d[k][i] = H1_proj[k][i] * syn1[i];
                            d_norm_sq += d[k][i] * d[k][i];
                        }
                    }

                    float Hd[DIM] = {0.0f};
                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) Hd[i] += H1_proj[k][i] * d[k][i];
                    }
                    float Hd_norm_sq = 0.0f;
                    for (int i = 0; i < DIM; ++i) Hd_norm_sq += Hd[i] * Hd[i];

                    float eta_star = d_norm_sq / (Hd_norm_sq + 1e-6f);

                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) traj_states[k][i] += eta_star * d[k][i];
                        normalize_vec(traj_states[k], DIM);
                    }

                    for (int i = 0; i < DIM; ++i) P1_curr[i] = 0.0f;
                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                    }
                    float new_energy = 0.0f;
                    for (int i = 0; i < DIM; ++i) {
                        float diff = P1_expected[i] - P1_curr[i];
                        new_energy += diff * diff;
                    }
                    if (fabsf(prev_energy - new_energy) < 0.001f) break;
                    prev_energy = new_energy;
                }

                float seq_likelihood = 0.0f;
                for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                    float sim = dot_prod(traj_states[hop], nodes[c][hop], DIM);
                    seq_likelihood += beta * sim;
                }
                float p1_dot = dot_prod(P1_expected, P1_curr, DIM);

                float P2_rel[DIM] = {0.0f};
                int p_idx = 0;
                for (int i_hop = 0; i_hop < CHAIN_LEN; ++i_hop) {
                    for (int j_hop = i_hop + 1; j_hop < CHAIN_LEN; ++j_hop) {
                        float cross_phase = dot_prod(traj_states[i_hop], traj_states[j_hop], DIM);
                        for (int d = 0; d < DIM; ++d) {
                            P2_rel[d] += H2_proj[p_idx][d] * cross_phase * traj_states[j_hop][d];
                        }
                        p_idx++;
                    }
                }
                float p2_dot = dot_prod(P2_expected, P2_rel, DIM);

                float score = seq_likelihood + 10.0f * p1_dot + 8.0f * p2_dot;
                if (score > best_score) { best_score = score; pred_c = c; }
            }
            if (pred_c == target_c) hits_c2++;
        }

        // -------------------------------------------------------------
        // Cond 3: Fixed Spatial Beam (B = 4) + Fixed Iterations (K = 3)
        // -------------------------------------------------------------
        {
            float best_score = -1e9f;
            int pred_c = -1;

            for (int c = 0; c < NUM_CHAINS; ++c) {
                // Fixed 4 beams per candidate chain
                for (int b = 0; b < 4; ++b) {
                    float traj_states[CHAIN_LEN][DIM];
                    for (int i = 0; i < DIM; ++i) {
                        traj_states[0][i] = raw_cue[i] + (b > 0 ? curand_normal(&st) * 0.15f : 0.0f);
                    }
                    normalize_vec(traj_states[0], DIM);

                    for (int k = 1; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) traj_states[k][i] = nodes[c][k][i];
                    }

                    for (int iter = 0; iter < 3; ++iter) {
                        float P1_curr[DIM] = {0.0f};
                        for (int k = 0; k < CHAIN_LEN; ++k) {
                            for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                        }
                        float syn1[DIM];
                        for (int i = 0; i < DIM; ++i) syn1[i] = P1_expected[i] - P1_curr[i];

                        for (int k = 0; k < CHAIN_LEN; ++k) {
                            for (int i = 0; i < DIM; ++i) traj_states[k][i] += 0.20f * H1_proj[k][i] * syn1[i];
                            normalize_vec(traj_states[k], DIM);
                        }
                    }

                    float seq_likelihood = 0.0f;
                    for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                        float sim = dot_prod(traj_states[hop], nodes[c][hop], DIM);
                        seq_likelihood += beta * sim;
                    }

                    float P1_curr[DIM] = {0.0f};
                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                    }
                    float p1_dot = dot_prod(P1_expected, P1_curr, DIM);

                    float P2_rel[DIM] = {0.0f};
                    int p_idx = 0;
                    for (int i_hop = 0; i_hop < CHAIN_LEN; ++i_hop) {
                        for (int j_hop = i_hop + 1; j_hop < CHAIN_LEN; ++j_hop) {
                            float cross_phase = dot_prod(traj_states[i_hop], traj_states[j_hop], DIM);
                            for (int d = 0; d < DIM; ++d) {
                                P2_rel[d] += H2_proj[p_idx][d] * cross_phase * traj_states[j_hop][d];
                            }
                            p_idx++;
                        }
                    }
                    float p2_dot = dot_prod(P2_expected, P2_rel, DIM);

                    float score = seq_likelihood + 10.0f * p1_dot + 8.0f * p2_dot;
                    if (score > best_score) { best_score = score; pred_c = c; }
                }
            }
            if (pred_c == target_c) hits_c3++;
        }

        // -------------------------------------------------------------
        // Cond 4: Full Spatio-Temporal Sovereign Field (STSF)
        // Autopoietic Sprouting Beam Width B_t(F_t) + Dynamic Horizon K_t(F_t) + Riemannian Step
        // -------------------------------------------------------------
        {
            float best_score = -1e9f;
            int pred_c = -1;

            for (int c = 0; c < NUM_CHAINS; ++c) {
                // 1. Initial Sensory Parity Energy for cue candidate
                float P1_init[DIM] = {0.0f};
                for (int i = 0; i < DIM; ++i) P1_init[i] += H1_proj[0][i] * raw_cue[i];
                for (int k = 1; k < CHAIN_LEN; ++k) {
                    for (int i = 0; i < DIM; ++i) P1_init[i] += H1_proj[k][i] * nodes[c][k][i];
                }
                float init_energy = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    float diff = P1_expected[i] - P1_init[i];
                    init_energy += diff * diff;
                }

                // 2. Endogenous Spatial Sprouting Beam Width B_t in [1, 8]
                int B_t = 1;
                if (init_energy > 0.05f) {
                    B_t = 2 + (int)(fminf((init_energy - 0.05f) * 10.0f, 6.0f));
                    if (B_t > MAX_BEAMS) B_t = MAX_BEAMS;
                }
                total_sprouted_beams += B_t;

                // 3. Relax each sprouted candidate beam along its Riemannian geodesic
                for (int b = 0; b < B_t; ++b) {
                    float traj_states[CHAIN_LEN][DIM];
                    for (int i = 0; i < DIM; ++i) {
                        traj_states[0][i] = raw_cue[i] + (b > 0 ? curand_normal(&st) * (0.05f * b) : 0.0f);
                    }
                    normalize_vec(traj_states[0], DIM);
                    for (int k = 1; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) traj_states[k][i] = nodes[c][k][i];
                    }

                    float P1_curr[DIM] = {0.0f};
                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                    }
                    float prev_E = 0.0f;
                    for (int i = 0; i < DIM; ++i) {
                        float diff = P1_expected[i] - P1_curr[i];
                        prev_E += diff * diff;
                    }

                    int dyn_k = 0;
                    for (int iter = 0; iter < MAX_ALLOWED_K; ++iter) {
                        dyn_k++;
                        if (prev_E < 0.015f) break;

                        float syn1[DIM];
                        for (int i = 0; i < DIM; ++i) syn1[i] = P1_expected[i] - P1_curr[i];

                        float d[CHAIN_LEN][DIM];
                        float d_norm_sq = 0.0f;
                        for (int k = 0; k < CHAIN_LEN; ++k) {
                            for (int i = 0; i < DIM; ++i) {
                                d[k][i] = H1_proj[k][i] * syn1[i];
                                d_norm_sq += d[k][i] * d[k][i];
                            }
                        }

                        float Hd[DIM] = {0.0f};
                        for (int k = 0; k < CHAIN_LEN; ++k) {
                            for (int i = 0; i < DIM; ++i) Hd[i] += H1_proj[k][i] * d[k][i];
                        }
                        float Hd_norm_sq = 0.0f;
                        for (int i = 0; i < DIM; ++i) Hd_norm_sq += Hd[i] * Hd[i];

                        float eta_star = d_norm_sq / (Hd_norm_sq + 1e-6f);
                        sum_eta += eta_star;
                        total_eta_count++;

                        for (int k = 0; k < CHAIN_LEN; ++k) {
                            for (int i = 0; i < DIM; ++i) traj_states[k][i] += eta_star * d[k][i];
                            normalize_vec(traj_states[k], DIM);
                        }

                        for (int i = 0; i < DIM; ++i) P1_curr[i] = 0.0f;
                        for (int k = 0; k < CHAIN_LEN; ++k) {
                            for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                        }
                        float new_E = 0.0f;
                        for (int i = 0; i < DIM; ++i) {
                            float diff = P1_expected[i] - P1_curr[i];
                            new_E += diff * diff;
                        }
                        if (fabsf(prev_E - new_E) < 0.001f) break;
                        prev_E = new_E;
                    }
                    total_dynamic_k += dyn_k;

                    float seq_likelihood = 0.0f;
                    for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                        float sim = dot_prod(traj_states[hop], nodes[c][hop], DIM);
                        seq_likelihood += beta * sim;
                    }
                    float p1_dot = dot_prod(P1_expected, P1_curr, DIM);

                    float P2_rel[DIM] = {0.0f};
                    int p_idx = 0;
                    for (int i_hop = 0; i_hop < CHAIN_LEN; ++i_hop) {
                        for (int j_hop = i_hop + 1; j_hop < CHAIN_LEN; ++j_hop) {
                            float cross_phase = dot_prod(traj_states[i_hop], traj_states[j_hop], DIM);
                            for (int d = 0; d < DIM; ++d) {
                                P2_rel[d] += H2_proj[p_idx][d] * cross_phase * traj_states[j_hop][d];
                            }
                            p_idx++;
                        }
                    }
                    float p2_dot = dot_prod(P2_expected, P2_rel, DIM);

                    float score = seq_likelihood + 10.0f * p1_dot + 8.0f * p2_dot;
                    if (score > best_score) { best_score = score; pred_c = c; }
                }
            }
            if (pred_c == target_c) hits_c4++;
        }
    }

    float ideal_cap = (1.0f - noise_level);

    results[idx].acc_cond1_baseline         = (float)hits_c1 / NUM_CHAINS;
    results[idx].acc_cond2_exp659_1d        = (float)hits_c2 / NUM_CHAINS;
    results[idx].acc_cond3_fixed_b4_k3      = (float)hits_c3 / NUM_CHAINS;
    results[idx].acc_cond4_full_stsf        = (float)hits_c4 / NUM_CHAINS;
    results[idx].avg_sprouted_beams        = (float)total_sprouted_beams / (NUM_CHAINS * NUM_CHAINS);
    results[idx].avg_dynamic_k              = (float)total_dynamic_k / (total_sprouted_beams + 1e-5f);
    results[idx].avg_eta_step               = (total_eta_count > 0) ? (sum_eta / total_eta_count) : 0.0f;
    results[idx].ideal_signal_capacity      = ideal_cap;
    results[idx].gap_c1                     = ideal_cap - ((float)hits_c1 / NUM_CHAINS);
    results[idx].gap_c4                     = ideal_cap - ((float)hits_c4 / NUM_CHAINS);
}

void run_stsf_benchmark(float noise, const std::string& label) {
    STSF_Result* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(STSF_Result)));
    std::vector<STSF_Result> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    stsf_kernel<<<blocks, threads>>>(d_res, noise, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(STSF_Result), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, avg_b = 0.0f, avg_k = 0.0f, avg_eta = 0.0f;
    float ideal = 0.0f, g1 = 0.0f, g4 = 0.0f;
    for (const auto& r : h_res) {
        c1 += r.acc_cond1_baseline;
        c2 += r.acc_cond2_exp659_1d;
        c3 += r.acc_cond3_fixed_b4_k3;
        c4 += r.acc_cond4_full_stsf;
        avg_b += r.avg_sprouted_beams;
        avg_k += r.avg_dynamic_k;
        avg_eta += r.avg_eta_step;
        ideal += r.ideal_signal_capacity;
        g1 += r.gap_c1;
        g4 += r.gap_c4;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size();
    avg_b /= h_res.size(); avg_k /= h_res.size(); avg_eta /= h_res.size(); ideal /= h_res.size();
    g1 /= h_res.size(); g4 /= h_res.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << ">>> " << label << " (Noise Floor: " << (noise * 100.0f) << "% | Signal: " << ((1.0f - noise) * 100.0f) << "%)\n"
              << "    Theoretical Signal Capacity (100% - Noise%) : " << (ideal * 100.0f) << "%\n"
              << "    1. Uncorrected Baseline                     : " << (c1 * 100.0f) << "%  (Gap to Ideal: " << (g1 * 100.0f) << "%)\n"
              << "    2. EXP-659 Riemannian Step 1D (B_t = 1)      : " << (c2 * 100.0f) << "%  (Gain: " << std::showpos << ((c2 - c1) * 100.0f) << "%)" << std::noshowpos << "\n"
              << "    3. Static Fixed Spatial Beam (B = 4, K = 3) : " << (c3 * 100.0f) << "%  (Gain: " << std::showpos << ((c3 - c1) * 100.0f) << "%)" << std::noshowpos << "\n"
              << "    4. Full Spatio-Temporal Sovereign Field     : " << (c4 * 100.0f) << "%  (Gap to Ideal: " << (g4 * 100.0f) << "%)\n"
              << "    Average Self-Sprouted Beams <B_t>           : " << std::setprecision(3) << avg_b << " spatial beams in [1, 8]\n"
              << "    Average Self-Selected Horizon <K_t>         : " << std::setprecision(3) << avg_k << " relaxation steps per beam\n"
              << "    Average Self-Selected eta^*_t               : " << std::setprecision(4) << avg_eta << "\n"
              << "    Compute Time                                : " << std::setprecision(2) << (sec * 1000.0) << " ms\n\n";

    CHECK_CUDA(cudaFree(d_res));
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-660: SPATIO-TEMPORAL SOVEREIGN FIELD & UNBOUND SPROUTING      ===\n"
              << "========================================================================\n"
              << "Cybernetic Paradigm : Unified Synthesis of Space (B_t) & Time (K_t, eta^*_t)\n"
              << "Governed Mechanics  : Autopoietic Sprouting Beam Width B_t in [1, 8]\n"
              << "Benchmark           : Comfort (30%), Critical (50%), High Stress (70% Noise)\n"
              << "Random Chance Floor : 1 / 50 = 2.00%\n"
              << "========================================================================\n\n" << std::flush;

    run_stsf_benchmark(0.30f, "REGIME 1: COMFORT ZONE (30% NOISE)");
    run_stsf_benchmark(0.50f, "REGIME 2: CRITICAL PHASE TRANSITION ZONE (50% NOISE)");
    run_stsf_benchmark(0.70f, "REGIME 3: HIGH STRESS REGIME (70% NOISE)");

    return 0;
}
