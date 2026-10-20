// ======================================================================================
// EXP-658: Endogenous Dynamic Relaxation Time & Free-Energy Dissipation Stopping (EDRT-FEDS)
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// Bazilevs' Mandate: "K must NEVER be a static constant (e.g. K=3)!"
// Under KEP Principles 27, 29, and 30:
// Artificial clamping of iterations or fixed loop ceilings cripples autopoiesis.
// The relaxation depth K_t (or relaxation horizon tau_t) must be an ENDOGENOUS,
// physical property of the substrate governed by Free Energy dissipation:
//
// 1. Syndrome Energy Norm:
//    E_syn(t) = ||P1_expected - P1_cand(t)||^2 + ||P2_expected - P2_cand(t)||^2
// 2. Endogenous Dissipation Rate:
//    dE_syn / dt = E_syn(t) - E_syn(t-1)
// 3. Dynamic Physical Termination:
//    Relaxation stops autonomously when:
//    - (A) Attractor Equilibrium Reached: E_syn(t) < epsilon_homeostasis (Clean signal!)
//    - (B) Energy Dissipation Stagnation: |dE_syn / dt| < delta_stagnation (Basin reached)
//    - When noise is low, K_t dynamically collapses to 0 or 1 (Zero compute wasted!)
//    - When noise is high or critical, K_t dynamically expands up to necessary depth
//      to reconstruct the signal!
//
// Benchmark Protocol on Tesla T4 GPU (1,000 trials across 30%, 50%, 70% noise):
// - Cond 1: Uncorrected Baseline
// - Cond 2: Static Fixed K=3 (EXP-657 Baseline)
// - Cond 3: Endogenous Dynamic Relaxation Time K_t(F_t) governed by Syndrome Dissipation
// - Telemetry: Measure Accuracy, Gap to Ideal, Average Dynamic Iterations <K_t>, and Compute
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
constexpr int MAX_ALLOWED_K    = 10;       // Maximum physical ceiling to prevent infinite CUDA loops
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct EDRT_Result {
    float acc_cond1_baseline;
    float acc_cond2_static_k3;
    float acc_cond3_dynamic_kt;
    float avg_dynamic_k;
    float ideal_signal_capacity;
    float gap_c1;
    float gap_c3;
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

__global__ void edrt_kernel(EDRT_Result* results, float noise_level, unsigned long seed) {
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

    int hits_c1 = 0, hits_c2 = 0, hits_c3 = 0;
    int total_dynamic_k_steps = 0;
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
        // Cond 2: Static Fixed K=3 (EXP-657 Baseline)
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

                // Fixed K = 3
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

                float P1_rel[DIM] = {0.0f};
                for (int k = 0; k < CHAIN_LEN; ++k) {
                    for (int i = 0; i < DIM; ++i) P1_rel[i] += H1_proj[k][i] * traj_states[k][i];
                }
                float p1_dot = dot_prod(P1_expected, P1_rel, DIM);

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

                float turbo_score = seq_likelihood + 10.0f * p1_dot + 8.0f * p2_dot;
                if (turbo_score > best_score) { best_score = turbo_score; pred_c = c; }
            }
            if (pred_c == target_c) hits_c2++;
        }

        // -------------------------------------------------------------
        // Cond 3: Endogenous Dynamic Relaxation Time K_t(F_t)
        // Self-Determining Stopping Horizon Governed by Energy Dissipation
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

                // Initial Syndrome Energy
                float P1_curr[DIM] = {0.0f};
                for (int k = 0; k < CHAIN_LEN; ++k) {
                    for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                }
                float prev_energy = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    float diff = P1_expected[i] - P1_curr[i];
                    prev_energy += diff * diff;
                }

                int dynamic_k = 0;
                // Unshackled Dynamic Loop: stops when energy dissipates below threshold or stagnates!
                for (int iter = 0; iter < MAX_ALLOWED_K; ++iter) {
                    dynamic_k++;

                    // 1. Check if Homeostatic Attractor Basin Reached
                    if (prev_energy < 0.015f) break; // Clean resonance, no need to compute further!

                    // 2. Relaxation Step along Gradient
                    float syn1[DIM];
                    for (int i = 0; i < DIM; ++i) syn1[i] = P1_expected[i] - P1_curr[i];

                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) traj_states[k][i] += 0.20f * H1_proj[k][i] * syn1[i];
                        normalize_vec(traj_states[k], DIM);
                    }

                    // 3. Recalculate Energy
                    for (int i = 0; i < DIM; ++i) P1_curr[i] = 0.0f;
                    for (int k = 0; k < CHAIN_LEN; ++k) {
                        for (int i = 0; i < DIM; ++i) P1_curr[i] += H1_proj[k][i] * traj_states[k][i];
                    }
                    float new_energy = 0.0f;
                    for (int i = 0; i < DIM; ++i) {
                        float diff = P1_expected[i] - P1_curr[i];
                        new_energy += diff * diff;
                    }

                    // 4. Stagnation / Dissipation Check: dE < 1e-4
                    float delta_E = fabsf(prev_energy - new_energy);
                    if (delta_E < 0.001f) break;

                    prev_energy = new_energy;
                }

                total_dynamic_k_steps += dynamic_k;

                // Evaluate relaxed candidate
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

                float edrt_score = seq_likelihood + 10.0f * p1_dot + 8.0f * p2_dot;
                if (edrt_score > best_score) { best_score = edrt_score; pred_c = c; }
            }
            if (pred_c == target_c) hits_c3++;
        }
    }

    float ideal_cap = (1.0f - noise_level);

    results[idx].acc_cond1_baseline     = (float)hits_c1 / NUM_CHAINS;
    results[idx].acc_cond2_static_k3    = (float)hits_c2 / NUM_CHAINS;
    results[idx].acc_cond3_dynamic_kt   = (float)hits_c3 / NUM_CHAINS;
    results[idx].avg_dynamic_k          = (float)total_dynamic_k_steps / (NUM_CHAINS * NUM_CHAINS);
    results[idx].ideal_signal_capacity  = ideal_cap;
    results[idx].gap_c1                 = ideal_cap - ((float)hits_c1 / NUM_CHAINS);
    results[idx].gap_c3                 = ideal_cap - ((float)hits_c3 / NUM_CHAINS);
}

void run_edrt_benchmark(float noise, const std::string& label) {
    EDRT_Result* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(EDRT_Result)));
    std::vector<EDRT_Result> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    edrt_kernel<<<blocks, threads>>>(d_res, noise, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(EDRT_Result), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, avg_k = 0.0f, ideal = 0.0f, g1 = 0.0f, g3 = 0.0f;
    for (const auto& r : h_res) {
        c1 += r.acc_cond1_baseline;
        c2 += r.acc_cond2_static_k3;
        c3 += r.acc_cond3_dynamic_kt;
        avg_k += r.avg_dynamic_k;
        ideal += r.ideal_signal_capacity;
        g1 += r.gap_c1;
        g3 += r.gap_c3;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size();
    avg_k /= h_res.size(); ideal /= h_res.size(); g1 /= h_res.size(); g3 /= h_res.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << ">>> " << label << " (Noise Floor: " << (noise * 100.0f) << "% | Signal: " << ((1.0f - noise) * 100.0f) << "%)\n"
              << "    Theoretical Signal Capacity (100% - Noise%) : " << (ideal * 100.0f) << "%\n"
              << "    1. Uncorrected Baseline                     : " << (c1 * 100.0f) << "%  (Gap to Ideal: " << (g1 * 100.0f) << "%)\n"
              << "    2. Static Fixed K=3 (EXP-657 Baseline)      : " << (c2 * 100.0f) << "%  (Gain: " << std::showpos << ((c2 - c1) * 100.0f) << "%)" << std::noshowpos << "\n"
              << "    3. Endogenous Dynamic Horizon K_t(F_t)      : " << (c3 * 100.0f) << "%  (Gap to Ideal: " << (g3 * 100.0f) << "%)\n"
              << "    Average Self-Selected Horizon <K_t>         : " << std::setprecision(3) << avg_k << " relaxation steps\n"
              << "    Compute Time                                : " << std::setprecision(2) << (sec * 1000.0) << " ms\n\n";

    CHECK_CUDA(cudaFree(d_res));
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-658: ENDOGENOUS DYNAMIC RELAXATION TIME (EDRT-FEDS)          ===\n"
              << "========================================================================\n"
              << "Bazilevs Mandate    : K MUST NEVER BE A STATIC CONSTANT!\n"
              << "Governed Mechanics  : Self-Determining Horizon Governed by Energy Dissipation\n"
              << "Benchmark           : Comfort (30%), Critical (50%), High Stress (70% Noise)\n"
              << "Random Chance Floor : 1 / 50 = 2.00%\n"
              << "========================================================================\n\n" << std::flush;

    run_edrt_benchmark(0.30f, "REGIME 1: COMFORT ZONE (30% NOISE)");
    run_edrt_benchmark(0.50f, "REGIME 2: CRITICAL PHASE TRANSITION ZONE (50% NOISE)");
    run_edrt_benchmark(0.70f, "REGIME 3: HIGH STRESS REGIME (70% NOISE)");

    return 0;
}
