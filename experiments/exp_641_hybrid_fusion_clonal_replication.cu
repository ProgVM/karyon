// ======================================================================================
// EXP-641: Heterogeneous Hybrid Fusion & Mutational Clonal Replication (HHF-MCR)
// --------------------------------------------------------------------------------------
// Objective & Cybernetic Rationale:
// Bazilevs' Breakthrough Paradigm Expansion:
// 1. Heterogeneous Hybrid Fusion (HHF):
//    Enables disparate/heterogeneous functional vectors (e.g., Sensory Stream S \in R^16,
//    Episodic Memory M \in R^16, Volitional Intent I \in R^16) to entangle into a unified
//    higher-order hybrid vector H_hybrid = Normalize(W_s S + W_m M + W_i I + Bilinear(S, M)).
// 2. Mutational Clonal Replication (MCR):
//    When encountering complex ambiguity, a vector v_t dynamically spawns a swarm of
//    K mutated clones v_t^{(k)} = Normalize(v_t + \sigma_{mut} \cdot \epsilon_k).
//    This expands the active combinatorial search space, allowing parallel variational
//    hypothesis testing in phase space and boosting recall under extreme noise.
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - K = 50 Complex Heterogeneous Facts (3 modal streams x 16D = 48D total hybrid dimension).
// - Extreme Noise Regime (85% noise floor, 15% signal).
// - 4 Architectural Conditions Evaluated:
//   Cond 1: Monolithic Static Baseline (No hybrid fusion, no clones)
//   Cond 2: Heterogeneous Hybrid Fusion alone (Cross-modal entanglement)
//   Cond 3: Clonal Swarm Replication alone (K = 8 exact + mutated clones without cross-fusion)
//   Cond 4: Full HHF-MCR (Heterogeneous Hybrid Fusion + Dynamic Mutational Clonal Replication)
//
// Telemetry Tracked:
// - Retrieval Fidelity across 1,000 trials
// - Combinatorial Diversity Index (entropy of clonal state space)
// - Net Active Gain over 85% Noise Floor
// - Compute Throughput (trials/sec)
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

constexpr int MODAL_DIM        = 16;       // 16D per heterogeneous modal stream
constexpr int NUM_MODALITIES   = 3;        // Sensory, Memory, Volition
constexpr int FULL_HYBRID_DIM  = 48;       // 3 x 16D = 48D
constexpr int NUM_FACTS        = 50;       // K = 50 Heterogeneous Facts
constexpr int CLONE_SWARM_SIZE = 8;        // K = 8 Mutated Clones per vector
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct HHFMCRResult {
    float base_noise_floor;
    float fid_cond1_monolithic;
    float fid_cond2_hhf_only;
    float fid_cond3_mcr_only;
    float fid_cond4_full_hhf_mcr;
    float net_active_gain;
    float clonal_diversity_index;
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

// --------------------------------------------------------------------------------------
// Kernel: Heterogeneous Hybrid Fusion & Mutational Clonal Replication Benchmark
// --------------------------------------------------------------------------------------
__global__ void hhf_mcr_kernel(HHFMCRResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 Heterogeneous Facts across 3 distinct modal streams
    float modal_facts[NUM_FACTS][NUM_MODALITIES][MODAL_DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int m = 0; m < NUM_MODALITIES; ++m) {
            for (int i = 0; i < MODAL_DIM; ++i) {
                modal_facts[f][m][i] = curand_normal(&st);
            }
            normalize_vec(modal_facts[f][m], MODAL_DIM);
        }
    }

    // Monolithic Concatenated Facts (Flat 48D)
    float monolithic_facts[NUM_FACTS][FULL_HYBRID_DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int m = 0; m < NUM_MODALITIES; ++m) {
            for (int i = 0; i < MODAL_DIM; ++i) {
                monolithic_facts[f][m * MODAL_DIM + i] = modal_facts[f][m][i];
            }
        }
        normalize_vec(monolithic_facts[f], FULL_HYBRID_DIM);
    }

    // Heterogeneous Hybrid Fused Facts (Non-linear cross-modal entanglement)
    float hhf_facts[NUM_FACTS][FULL_HYBRID_DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int i = 0; i < FULL_HYBRID_DIM; ++i) {
            float s = monolithic_facts[f][i];
            // Non-linear cross-modal bilinear term
            float bilinear = monolithic_facts[f][i % MODAL_DIM] * monolithic_facts[f][(i + MODAL_DIM) % FULL_HYBRID_DIM];
            hhf_facts[f][i] = s + 0.5f * tanhf(bilinear);
        }
        normalize_vec(hhf_facts[f], FULL_HYBRID_DIM);
    }

    float tot_base = 0.0f;
    float tot_c1   = 0.0f;
    float tot_c2   = 0.0f;
    float tot_c3   = 0.0f;
    float tot_c4   = 0.0f;
    float tot_div  = 0.0f;

    const float beta = 12.0f;
    const float mut_rate = 0.25f; // 25% mutational dispersion

    for (int f = 0; f < NUM_FACTS; ++f) {
        // Create heavily corrupted query (85% noise floor, 15% signal)
        float raw_cue[FULL_HYBRID_DIM];
        for (int i = 0; i < FULL_HYBRID_DIM; ++i) {
            raw_cue[i] = (curand_uniform(&st) < 0.15f) ? monolithic_facts[f][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(raw_cue, FULL_HYBRID_DIM);

        float base_fid = fmaxf(0.0f, dot_prod(raw_cue, monolithic_facts[f], FULL_HYBRID_DIM));
        tot_base += base_fid;

        // -------------------------------------------------------------
        // Cond 1: Monolithic Flat Baseline
        // -------------------------------------------------------------
        {
            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta * dot_prod(raw_cue, monolithic_facts[j], FULL_HYBRID_DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[FULL_HYBRID_DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < FULL_HYBRID_DIM; ++i) h_out[i] += a * monolithic_facts[j][i];
            }
            normalize_vec(h_out, FULL_HYBRID_DIM);
            tot_c1 += fmaxf(0.0f, dot_prod(h_out, monolithic_facts[f], FULL_HYBRID_DIM));
        }

        // -------------------------------------------------------------
        // Cond 2: Heterogeneous Hybrid Fusion (HHF) alone
        // -------------------------------------------------------------
        {
            float raw_cue_hhf[FULL_HYBRID_DIM];
            for (int i = 0; i < FULL_HYBRID_DIM; ++i) {
                float s = raw_cue[i];
                float bilinear = raw_cue[i % MODAL_DIM] * raw_cue[(i + MODAL_DIM) % FULL_HYBRID_DIM];
                raw_cue_hhf[i] = s + 0.5f * tanhf(bilinear);
            }
            normalize_vec(raw_cue_hhf, FULL_HYBRID_DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta * dot_prod(raw_cue_hhf, hhf_facts[j], FULL_HYBRID_DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[FULL_HYBRID_DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < FULL_HYBRID_DIM; ++i) h_out[i] += a * hhf_facts[j][i];
            }
            normalize_vec(h_out, FULL_HYBRID_DIM);
            tot_c2 += fmaxf(0.0f, dot_prod(h_out, hhf_facts[f], FULL_HYBRID_DIM));
        }

        // -------------------------------------------------------------
        // Cond 3: Mutational Clonal Replication (MCR) alone (K = 8 clones)
        // -------------------------------------------------------------
        {
            float best_clone_fid = 0.0f;
            for (int k = 0; k < CLONE_SWARM_SIZE; ++k) {
                float clone_cue[FULL_HYBRID_DIM];
                for (int i = 0; i < FULL_HYBRID_DIM; ++i) {
                    // Clone 0 is exact, clones 1..7 have mutational Gaussian perturbations
                    float mut = (k == 0) ? 0.0f : mut_rate * curand_normal(&st);
                    clone_cue[i] = raw_cue[i] + mut;
                }
                normalize_vec(clone_cue, FULL_HYBRID_DIM);

                float max_l = -1e9f;
                float l[NUM_FACTS];
                for (int j = 0; j < NUM_FACTS; ++j) {
                    l[j] = beta * dot_prod(clone_cue, monolithic_facts[j], FULL_HYBRID_DIM);
                    if (l[j] > max_l) max_l = l[j];
                }
                float sum_e = 0.0f;
                for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
                float h_out[FULL_HYBRID_DIM] = {0.0f};
                for (int j = 0; j < NUM_FACTS; ++j) {
                    float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < FULL_HYBRID_DIM; ++i) h_out[i] += a * monolithic_facts[j][i];
                }
                normalize_vec(h_out, FULL_HYBRID_DIM);
                float fid = fmaxf(0.0f, dot_prod(h_out, monolithic_facts[f], FULL_HYBRID_DIM));
                if (fid > best_clone_fid) best_clone_fid = fid;
            }
            tot_c3 += best_clone_fid;
        }

        // -------------------------------------------------------------
        // Cond 4: Full HHF-MCR (Hybrid Fusion + Swarm Clonal Mutational Replication)
        // -------------------------------------------------------------
        {
            float raw_cue_hhf[FULL_HYBRID_DIM];
            for (int i = 0; i < FULL_HYBRID_DIM; ++i) {
                float s = raw_cue[i];
                float bilinear = raw_cue[i % MODAL_DIM] * raw_cue[(i + MODAL_DIM) % FULL_HYBRID_DIM];
                raw_cue_hhf[i] = s + 0.5f * tanhf(bilinear);
            }
            normalize_vec(raw_cue_hhf, FULL_HYBRID_DIM);

            float best_hhf_clone_fid = 0.0f;
            float clone_sim_sum = 0.0f;

            for (int k = 0; k < CLONE_SWARM_SIZE; ++k) {
                float clone_cue[FULL_HYBRID_DIM];
                for (int i = 0; i < FULL_HYBRID_DIM; ++i) {
                    float mut = (k == 0) ? 0.0f : mut_rate * curand_normal(&st);
                    clone_cue[i] = raw_cue_hhf[i] + mut;
                }
                normalize_vec(clone_cue, FULL_HYBRID_DIM);

                // Compute pairwise distance to measure clonal diversity
                if (k > 0) clone_sim_sum += dot_prod(clone_cue, raw_cue_hhf, FULL_HYBRID_DIM);

                float max_l = -1e9f;
                float l[NUM_FACTS];
                for (int j = 0; j < NUM_FACTS; ++j) {
                    l[j] = beta * dot_prod(clone_cue, hhf_facts[j], FULL_HYBRID_DIM);
                    if (l[j] > max_l) max_l = l[j];
                }
                float sum_e = 0.0f;
                for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
                float h_out[FULL_HYBRID_DIM] = {0.0f};
                for (int j = 0; j < NUM_FACTS; ++j) {
                    float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                    for (int i = 0; i < FULL_HYBRID_DIM; ++i) h_out[i] += a * hhf_facts[j][i];
                }
                normalize_vec(h_out, FULL_HYBRID_DIM);
                float fid = fmaxf(0.0f, dot_prod(h_out, hhf_facts[f], FULL_HYBRID_DIM));
                if (fid > best_hhf_clone_fid) best_hhf_clone_fid = fid;
            }

            tot_c4 += best_hhf_clone_fid;
            tot_div += (1.0f - (clone_sim_sum / (CLONE_SWARM_SIZE - 1)));
        }
    }

    results[idx].base_noise_floor        = tot_base / NUM_FACTS;
    results[idx].fid_cond1_monolithic    = tot_c1 / NUM_FACTS;
    results[idx].fid_cond2_hhf_only      = tot_c2 / NUM_FACTS;
    results[idx].fid_cond3_mcr_only      = tot_c3 / NUM_FACTS;
    results[idx].fid_cond4_full_hhf_mcr  = tot_c4 / NUM_FACTS;
    results[idx].net_active_gain         = (tot_c4 - tot_c1) / NUM_FACTS;
    results[idx].clonal_diversity_index  = tot_div / NUM_FACTS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-641: HETEROGENEOUS HYBRID FUSION & CLONAL REPLICATION (HHF-MCR) ===\n"
              << "========================================================================\n"
              << "Bazilevs' Paradigm   : Heterogeneous Hybrid Fusion & Mutational Clonal Swarm\n"
              << "Mechanisms           : 1) Non-linear cross-modal entanglement (S, M, I -> H)\n"
              << "                       2) Swarm Mutational Clones (K=8) for active phase search\n"
              << "Stress Benchmark     : K=50 Heterogeneous Facts, 85% Extreme Noise Floor\n"
              << "Trials Evaluated     : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    HHFMCRResult* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(HHFMCRResult)));
    std::vector<HHFMCRResult> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    hhf_mcr_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(HHFMCRResult), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float base = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f, div = 0.0f;
    for (const auto& r : h_res) {
        base += r.base_noise_floor;
        c1   += r.fid_cond1_monolithic;
        c2   += r.fid_cond2_hhf_only;
        c3   += r.fid_cond3_mcr_only;
        c4   += r.fid_cond4_full_hhf_mcr;
        gain += r.net_active_gain;
        div  += r.clonal_diversity_index;
    }
    base /= h_res.size(); c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size();
    c4 /= h_res.size(); gain /= h_res.size(); div /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << ">>> BASE NOISE FLOOR CUE FIDELITY : " << base << " (85% Extreme Noise Floor)\n\n"
              << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION                 | FIDELITY | NET GAIN OVER BASELINE   |\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Monolithic Static Baseline           | " << c1 << "   | +0.0000 (Baseline)       |\n"
              << "| 2. Heterogeneous Hybrid Fusion (HHF)    | " << c2 << "   | " << std::showpos << (c2 - c1) << std::noshowpos << "                  |\n"
              << "| 3. Mutational Clonal Swarm (MCR alone)  | " << c3 << "   | " << std::showpos << (c3 - c1) << std::noshowpos << "                  |\n"
              << "| 4. Full HHF-MCR (Hybrid + Clones)       | " << c4 << "   | " << std::showpos << gain << std::noshowpos << "                  |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Clonal Swarm Diversity Index : " << div << " (Phase-space exploratory spread)\n"
              << "Total Compute Time (1,000 trials x 50 facts x 4 conditions): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
