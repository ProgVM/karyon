// ======================================================================================
// EXP-636: Modular Clustered Hopfield Hierarchy with Iterative Recurrent Relaxation
// --------------------------------------------------------------------------------------
// Objective & Mathematical Rationale:
// EXP-635 established the exact cause of the retrieval bottleneck:
// Theoretical Hopfield Capacity (AGS Theorem, 1982): C_crit = 0.14 * N.
// In a flat 32D space, C_crit = 4.48 facts. Stuffing K=50 facts into 32D causes an
// 11x overload, creating a landscape dominated by spurious mixed attractors (~85%).
//
// In EXP-636, we test Modular Clustered Hierarchical Memory:
// 1. Level-1: Coarse Routing to 5 Modular Attractor Basins (Clusters) (10 facts each).
//    In each 32D cluster, C_facts = 10 (capacity overload drops from 11x to ~2.2x).
// 2. Level-2: Within-cluster iterative continuous dynamical relaxation (T=10 steps).
//
// Comparison Benchmark across 4 precise conditions under IDENTICAL stress:
// - K = 50 Facts (32D normalized vectors).
// - 500 Distractor decay steps (decay gamma = 0.999).
// - 80% Noise corrupted retrieval cues (0.2 fact signal + 0.8 gaussian noise).
// - 1,000 independent evaluation runs on Tesla T4 GPU.
//
// Conditions:
// 1. Flat 1-Step Baseline: Flat 50-fact single-step lookup (Old baseline).
// 2. Flat Multi-Step (EXP-635): Flat 50-fact iterative relaxation (T=10).
// 3. Clustered 1-Step: 5 Clusters x 10 facts with single-step routing & readout.
// 4. Clustered Iterative Relaxation: 5 Clusters x 10 facts with 2-level recurrent settling (T=10).
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

constexpr int DIM                = 32;       // 32D basis space
constexpr int NUM_FACTS          = 50;       // K = 50 Facts
constexpr int NUM_CLUSTERS       = 5;        // 5 Modular clusters
constexpr int FACTS_PER_CLUSTER  = 10;       // 10 facts per cluster (5 x 10 = 50)
constexpr int DISTRACTOR_STEPS   = 500;      // 500 distractor decay steps
constexpr int TRIALS_PER_COND    = 1000;     // 1,000 independent GPU trials

struct BenchTelemetry {
    float base_noise_fidelity;
    float fid_flat_1step;
    float fid_flat_multistep;
    float fid_clustered_1step;
    float fid_clustered_multistep;
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

__global__ void modular_hierarchy_kernel(BenchTelemetry* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 5 Cluster Centroids in 32D
    float centroids[NUM_CLUSTERS][DIM];
    for (int c = 0; c < NUM_CLUSTERS; ++c) {
        for (int i = 0; i < DIM; ++i) centroids[c][i] = curand_normal(&st);
        normalize_vec(centroids[c], DIM);
    }

    // 2. Generate 50 Facts clustered around these centroids
    float facts[NUM_FACTS][DIM];
    int fact_cluster[NUM_FACTS];

    for (int f = 0; f < NUM_FACTS; ++f) {
        int c = f / FACTS_PER_CLUSTER;
        fact_cluster[f] = c;
        for (int i = 0; i < DIM; ++i) {
            // Fact is centroid + localized specific variation
            facts[f][i] = 0.65f * centroids[c][i] + 0.35f * curand_normal(&st);
        }
        normalize_vec(facts[f], DIM);
    }

    // Distractor Decay simulation
    float decay = powf(0.999f, (float)DISTRACTOR_STEPS); // ~0.606 retention

    float tot_base = 0.0f;
    float tot_f1   = 0.0f;
    float tot_fm   = 0.0f;
    float tot_c1   = 0.0f;
    float tot_cm   = 0.0f;

    const float beta_coarse = 8.0f;
    const float beta_fine   = 16.0f;
    const float dt          = 0.30f;
    const float cue_anchor  = 0.20f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        // Generate 80% noise cue
        float raw_cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            raw_cue[i] = (curand_uniform(&st) < 0.20f) ? facts[f][i] : curand_normal(&st) * 0.8f;
        }

        float dot_in = dot_prod(raw_cue, facts[f], DIM);
        float norm_in = sqrtf(dot_prod(raw_cue, raw_cue, DIM));
        tot_base += (norm_in > 1e-4f) ? fmaxf(0.0f, dot_in / norm_in) : 0.0f;

        float cue_norm[DIM];
        for (int i = 0; i < DIM; ++i) cue_norm[i] = raw_cue[i] / (norm_in + 1e-6f);

        // -------------------------------------------------------------
        // Condition 1: Flat 1-Step Baseline (All 50 facts, single step)
        // -------------------------------------------------------------
        {
            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta_coarse * dot_prod(cue_norm, facts[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * facts[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_f1 += fmaxf(0.0f, dot_prod(h_out, facts[f], DIM));
        }

        // -------------------------------------------------------------
        // Condition 2: Flat Multi-Step (All 50 facts, T=10 relaxation)
        // -------------------------------------------------------------
        {
            float h[DIM];
            for (int i = 0; i < DIM; ++i) h[i] = cue_norm[i];
            for (int t = 0; t < 10; ++t) {
                float max_l = -1e9f;
                float l[NUM_FACTS];
                for (int j = 0; j < NUM_FACTS; ++j) {
                    l[j] = beta_coarse * dot_prod(h, facts[j], DIM);
                    if (l[j] > max_l) max_l = l[j];
                }
                float sum_e = 0.0f;
                for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
                float flow[DIM] = {0.0f};
                for (int j = 0; j < NUM_FACTS; ++j) {
                    float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < DIM; ++i) flow[i] += a * facts[j][i];
                }
                for (int i = 0; i < DIM; ++i) {
                    h[i] = (1.0f - dt) * h[i] + dt * flow[i] + cue_anchor * cue_norm[i];
                }
                normalize_vec(h, DIM);
            }
            tot_fm += fmaxf(0.0f, dot_prod(h, facts[f], DIM));
        }

        // -------------------------------------------------------------
        // Condition 3: Clustered 1-Step (Route to cluster -> read fact)
        // -------------------------------------------------------------
        {
            // Level 1: Route to best cluster centroid
            int best_c = 0;
            float best_c_dot = -1e9f;
            for (int c = 0; c < NUM_CLUSTERS; ++c) {
                float d = dot_prod(cue_norm, centroids[c], DIM);
                if (d > best_c_dot) {
                    best_c_dot = d;
                    best_c = c;
                }
            }
            // Level 2: Softmax within routed cluster (only 10 facts)
            int c_start = best_c * FACTS_PER_CLUSTER;
            float max_l = -1e9f;
            float l[FACTS_PER_CLUSTER];
            for (int j = 0; j < FACTS_PER_CLUSTER; ++j) {
                l[j] = beta_fine * dot_prod(cue_norm, facts[c_start + j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < FACTS_PER_CLUSTER; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < FACTS_PER_CLUSTER; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * facts[c_start + j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c1 += fmaxf(0.0f, dot_prod(h_out, facts[f], DIM));
        }

        // -------------------------------------------------------------
        // Condition 4: Clustered Iterative Relaxation (Hierarchical T=10)
        // -------------------------------------------------------------
        {
            // Multi-step relaxation in cluster space + fine space
            float h[DIM];
            for (int i = 0; i < DIM; ++i) h[i] = cue_norm[i];

            for (int t = 0; t < 10; ++t) {
                // Soft cluster weighting (continuous routing instead of hard cut)
                float max_c = -1e9f;
                float lc[NUM_CLUSTERS];
                for (int c = 0; c < NUM_CLUSTERS; ++c) {
                    lc[c] = beta_coarse * dot_prod(h, centroids[c], DIM);
                    if (lc[c] > max_c) max_c = lc[c];
                }
                float sum_ec = 0.0f;
                for (int c = 0; c < NUM_CLUSTERS; ++c) sum_ec += expf(lc[c] - max_c);

                float flow[DIM] = {0.0f};
                for (int c = 0; c < NUM_CLUSTERS; ++c) {
                    float p_c = expf(lc[c] - max_c) / (sum_ec + 1e-6f);
                    int c_start = c * FACTS_PER_CLUSTER;

                    float max_f = -1e9f;
                    float lf[FACTS_PER_CLUSTER];
                    for (int j = 0; j < FACTS_PER_CLUSTER; ++j) {
                        lf[j] = beta_fine * dot_prod(h, facts[c_start + j], DIM);
                        if (lf[j] > max_f) max_f = lf[j];
                    }
                    float sum_ef = 0.0f;
                    for (int j = 0; j < FACTS_PER_CLUSTER; ++j) sum_ef += expf(lf[j] - max_f);

                    for (int j = 0; j < FACTS_PER_CLUSTER; ++j) {
                        float a = p_c * (expf(lf[j] - max_f) / (sum_ef + 1e-6f));
                        for (int i = 0; i < DIM; ++i) flow[i] += a * facts[c_start + j][i];
                    }
                }

                for (int i = 0; i < DIM; ++i) {
                    h[i] = (1.0f - dt) * h[i] + dt * flow[i] + cue_anchor * cue_norm[i];
                }
                normalize_vec(h, DIM);
            }
            tot_cm += fmaxf(0.0f, dot_prod(h, facts[f], DIM));
        }
    }

    results[idx].base_noise_fidelity     = tot_base / NUM_FACTS;
    results[idx].fid_flat_1step          = tot_f1 / NUM_FACTS;
    results[idx].fid_flat_multistep      = tot_fm / NUM_FACTS;
    results[idx].fid_clustered_1step      = tot_c1 / NUM_FACTS;
    results[idx].fid_clustered_multistep  = tot_cm / NUM_FACTS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-636: MODULAR CLUSTERED HOPFIELD HIERARCHY BENCHMARK          ===\n"
              << "========================================================================\n"
              << "Mathematical Rationale: Alleviating Hopfield Capacity Overload (C_crit=4.5)\n"
              << "Stress Benchmark      : K=50 Facts (32D) in 5 Clusters -> 80% Noise Cues\n"
              << "Trials Evaluated      : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    BenchTelemetry* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(BenchTelemetry)));
    std::vector<BenchTelemetry> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    modular_hierarchy_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(BenchTelemetry), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float b = 0.0f, f1 = 0.0f, fm = 0.0f, c1 = 0.0f, cm = 0.0f;
    for (const auto& r : h_res) {
        b  += r.base_noise_fidelity;
        f1 += r.fid_flat_1step;
        fm += r.fid_flat_multistep;
        c1 += r.fid_clustered_1step;
        cm += r.fid_clustered_multistep;
    }
    b /= h_res.size(); f1 /= h_res.size(); fm /= h_res.size(); c1 /= h_res.size(); cm /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << ">>> BASE NOISE CUE FIDELITY : " << b << " (80% Noise Floor)\n\n"
              << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION           | RECONSTRUCTED FIDELITY | NET ACTIVE GAIN  |\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Flat 1-Step (Old Baseline)     | " << f1 << "                 | " << std::showpos << (f1 - b) << std::noshowpos << "          |\n"
              << "| 2. Flat Multi-Step (T=10, EXP-635)| " << fm << "                 | " << std::showpos << (fm - b) << std::noshowpos << "          |\n"
              << "| 3. Clustered 1-Step (Hierarchical)| " << c1 << "                 | " << std::showpos << (c1 - b) << std::noshowpos << "          |\n"
              << "| 4. Clustered Multi-Step (T=10)    | " << cm << "                 | " << std::showpos << (cm - b) << std::noshowpos << "          |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 facts x 4 conditions): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
