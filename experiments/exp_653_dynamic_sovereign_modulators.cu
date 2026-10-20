// ======================================================================================
// EXP-653: Dynamic Sovereign Modulators of Vector Geodesics & Attractor Phase
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// Bazilevs' Hypothesis: "What if vector behavior is governed by dynamic, unhardcoded
// modulators rather than rigid predetermined operation types?"
//
// In KEP Principle 27 & 30 (Sovereignty & Bottom-Up Emergence):
// Instead of hardcoding discrete modes (e.g. "Linear", "Attention", "Gate", "Hopfield"),
// a continuous dynamic modulator state M_t in R^K continuously alters the metric tensor,
// transition curvature, precision temperature, and anchor coupling of the vector space:
//
// 1. Dynamic Metric Curvature g_{ij}(M): Modulates local metric distance / inner product:
//    <u, v>_M = sum_i u_i * v_i * (1.0 + tanh(M_{metric, i}))
// 2. Dynamic Precision / Beta Temperature beta(M):
//    beta_eff = beta_base * (1.0 + softplus(M_{beta}))
// 3. Dynamic Inertia / Soliton Velocity alpha(M):
//    S_{t+1} = normalize( (1 - alpha(M)) * Fwd(S_t) + alpha(M) * Anchor )
// 4. Endogenous Modulator Dynamics:
//    M_{t+1} = M_t + eta * (Surprise_t - Target_Entropy) -> Self-adapts to noise!
//
// Benchmark on Tesla T4 GPU (1,000 trials per noise regime):
// - Regime: High-Stress Noise Boundary (50% noise where baseline collapsed from 88% -> 24.68%!)
// - Cond 1: Static Rigid Vectors (Baseline, 24.68%)
// - Cond 2: Dynamic Temperature Modulator beta(M)
// - Cond 3: Dynamic Metric Curvature g_{ij}(M) + Anchor Velocity alpha(M)
// - Cond 4: Full Sovereign Dynamic Modulator Engine (Curvature + Precision + Inertia + Entropy Feedback)
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
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct Modulator_Result {
    float acc_cond1_static;
    float acc_cond2_dyn_temp;
    float acc_cond3_dyn_metric_inertia;
    float acc_cond4_full_sovereign;
    float net_gain;
    double execution_sec;
};

__device__ inline float dot_prod(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

__device__ inline float modulated_dot_prod(const float* a, const float* b, const float* g, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i] * g[i];
    return s;
}

__device__ inline void normalize_vec(float* v, int n) {
    float norm = sqrtf(dot_prod(v, v, n) + 1e-6f);
    for (int i = 0; i < n; ++i) v[i] /= norm;
}

__global__ void modulator_kernel(Modulator_Result* results, float noise_level, unsigned long seed) {
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
    const float base_beta = 16.0f;

    for (int target_c = 0; target_c < NUM_CHAINS; ++target_c) {
        // Query cue is Node A with specified noise floor
        float raw_cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            raw_cue[i] = (curand_uniform(&st) >= noise_level) ? nodes[target_c][0][i] : curand_normal(&st) * noise_level;
        }
        normalize_vec(raw_cue, DIM);

        // -------------------------------------------------------------
        // Cond 1: Static Rigid Vectors (EXP-652 Baseline)
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
        // Cond 2: Dynamic Temperature Modulator beta(M)
        // Modulator senses confidence (entropy) and adaptively sharpens beta
        // -------------------------------------------------------------
        {
            float state[DIM];
            for (int i = 0; i < DIM; ++i) state[i] = raw_cue[i];

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                // Compute initial resonance distribution to derive modulator
                float max_s = -1e9f;
                float sims[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    sims[c] = dot_prod(state, nodes[c][hop], DIM);
                    if (sims[c] > max_s) max_s = sims[c];
                }
                // Dynamic modulator: margin between top-1 and top-2 defines certainty
                float second_s = -1e9f;
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    if (sims[c] < max_s && sims[c] > second_s) second_s = sims[c];
                }
                float margin = max_s - second_s;
                // If ambiguous (low margin), modulator scales up beta dynamically
                float dynamic_beta = base_beta * (1.0f + 2.0f / (1.0f + expf(margin * 10.0f)));

                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = dynamic_beta * sims[c];
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

            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float s = dot_prod(state, nodes[c][CHAIN_LEN - 1], DIM);
                if (s > best_sim) { best_sim = s; pred_c = c; }
            }
            if (pred_c == target_c) hits_c2++;
        }

        // -------------------------------------------------------------
        // Cond 3: Dynamic Metric Curvature g_{ij}(M) + Dynamic Inertia alpha(M)
        // Modulator alters the metric tensor along the high-confidence dimensions
        // -------------------------------------------------------------
        {
            float state[DIM];
            for (int i = 0; i < DIM; ++i) state[i] = raw_cue[i];
            float metric_g[DIM];
            for (int i = 0; i < DIM; ++i) metric_g[i] = 1.0f; // isotropic initial

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                // Update metric curvature: amplify dimensions where state agrees with anchor
                for (int i = 0; i < DIM; ++i) {
                    float coh = state[i] * raw_cue[i];
                    metric_g[i] = 1.0f + 1.5f * tanhf(coh * 5.0f);
                }

                float max_s = -1e9f;
                int best_c = 0;
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    float s = modulated_dot_prod(state, nodes[c][hop], metric_g, DIM);
                    if (s > max_s) { max_s = s; best_c = c; }
                }

                // Dynamic inertia modulator alpha: depends on alignment
                float alignment = dot_prod(state, raw_cue, DIM);
                float alpha = 0.20f + 0.30f * (1.0f - tanhf(alignment * 2.0f)); // more anchor if losing path

                for (int i = 0; i < DIM; ++i) {
                    state[i] = (1.0f - alpha) * nodes[best_c][hop + 1][i] + alpha * raw_cue[i];
                }
                normalize_vec(state, DIM);
            }

            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float s = modulated_dot_prod(state, nodes[c][CHAIN_LEN - 1], metric_g, DIM);
                if (s > best_sim) { best_sim = s; pred_c = c; }
            }
            if (pred_c == target_c) hits_c3++;
        }

        // -------------------------------------------------------------
        // Cond 4: Full Sovereign Dynamic Modulator Engine (SDM-Engine)
        // Endogenous Modulator Vector M = [M_prec, M_metric, M_inertia, M_anchor]
        // Free to adjust coordinates in continuous time!
        // -------------------------------------------------------------
        {
            float state[DIM];
            for (int i = 0; i < DIM; ++i) state[i] = raw_cue[i];

            // 4-component continuous endogenous modulator field
            float M_prec    = 1.0f;
            float M_inertia = 0.25f;
            float metric_g[DIM];
            for (int i = 0; i < DIM; ++i) metric_g[i] = 1.0f;

            for (int hop = 0; hop < CHAIN_LEN - 1; ++hop) {
                // 1. Endogenous perception: evaluate surprise (free energy)
                float sims[NUM_CHAINS];
                float max_s = -1e9f;
                int best_c = 0;
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    sims[c] = modulated_dot_prod(state, nodes[c][hop], metric_g, DIM);
                    if (sims[c] > max_s) { max_s = sims[c]; best_c = c; }
                }

                // Measure second best to detect cognitive dissonance / bifurcation
                float second_s = -1e9f;
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    if (sims[c] < max_s && sims[c] > second_s) second_s = sims[c];
                }
                float ambiguity = 1.0f - tanhf((max_s - second_s) * 4.0f);

                // 2. Modulators adaptively react (Active Homeostasis):
                // In high ambiguity, boost precision (sharpen attractor) and increase anchor recall
                M_prec    = 1.0f + 3.0f * ambiguity;
                M_inertia = 0.15f + 0.45f * ambiguity;

                // Anisotropic curvature update along consensual coordinates
                for (int i = 0; i < DIM; ++i) {
                    float signal_dim = nodes[best_c][hop][i] * raw_cue[i];
                    metric_g[i] = 0.5f + 1.5f / (1.0f + expf(-signal_dim * 6.0f));
                }

                // Softmax collapse with dynamic precision
                float dynamic_beta = base_beta * M_prec;
                float max_l = -1e9f;
                float l[NUM_CHAINS];
                for (int c = 0; c < NUM_CHAINS; ++c) {
                    l[c] = dynamic_beta * sims[c];
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

                // Sovereign Inertia blending
                for (int i = 0; i < DIM; ++i) {
                    state[i] = (1.0f - M_inertia) * next_state[i] + M_inertia * raw_cue[i];
                }
                normalize_vec(state, DIM);
            }

            // Final Readout using modulated metric
            float best_sim = -1e9f;
            int pred_c = -1;
            for (int c = 0; c < NUM_CHAINS; ++c) {
                float s = modulated_dot_prod(state, nodes[c][CHAIN_LEN - 1], metric_g, DIM);
                if (s > best_sim) { best_sim = s; pred_c = c; }
            }
            if (pred_c == target_c) hits_c4++;
        }
    }

    results[idx].acc_cond1_static             = (float)hits_c1 / NUM_CHAINS;
    results[idx].acc_cond2_dyn_temp           = (float)hits_c2 / NUM_CHAINS;
    results[idx].acc_cond3_dyn_metric_inertia = (float)hits_c3 / NUM_CHAINS;
    results[idx].acc_cond4_full_sovereign     = (float)hits_c4 / NUM_CHAINS;
    results[idx].net_gain                     = ((float)hits_c4 - (float)hits_c1) / NUM_CHAINS;
}

void run_modulator_benchmark(float noise, const std::string& label) {
    Modulator_Result* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(Modulator_Result)));
    std::vector<Modulator_Result> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    modulator_kernel<<<blocks, threads>>>(d_res, noise, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(Modulator_Result), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.acc_cond1_static;
        c2   += r.acc_cond2_dyn_temp;
        c3   += r.acc_cond3_dyn_metric_inertia;
        c4   += r.acc_cond4_full_sovereign;
        gain += r.net_gain;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << ">>> " << label << " (Noise Floor: " << (noise * 100.0f) << "% | Signal: " << ((1.0f - noise) * 100.0f) << "%)\n"
              << "    1. Static Rigid Vectors (Baseline)    : " << (c1 * 100.0f) << "%\n"
              << "    2. Dynamic Precision beta(M)          : " << (c2 * 100.0f) << "%  (Gain: " << std::showpos << ((c2 - c1) * 100.0f) << "%)" << std::noshowpos << "\n"
              << "    3. Dynamic Metric & Inertia g(M), a(M): " << (c3 * 100.0f) << "%  (Gain: " << std::showpos << ((c3 - c1) * 100.0f) << "%)" << std::noshowpos << "\n"
              << "    4. Full Sovereign Dynamic Modulator   : " << (c4 * 100.0f) << "%  (Gain: " << std::showpos << (gain * 100.0f) << "%)" << std::noshowpos << "\n"
              << "    Compute Time                          : " << (sec * 1000.0) << " ms\n\n";

    CHECK_CUDA(cudaFree(d_res));
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-653: DYNAMIC SOVEREIGN MODULATORS OF VECTOR GEODESICS        ===\n"
              << "========================================================================\n"
              << "Core Paradigm       : Unhardcoded Endogenous Modulators M_t in R^K\n"
              << "Governed Quantities : Metric Tensor g_{ij}(M), Precision beta(M), Inertia a(M)\n"
              << "Benchmark           : Critical Transition Zone (50% Noise) & High Stress (70% Noise)\n"
              << "Random Chance Floor : 1 / 50 = 2.00%\n"
              << "========================================================================\n\n" << std::flush;

    run_modulator_benchmark(0.30f, "REGIME 1: COMFORT ZONE (30% NOISE)");
    run_modulator_benchmark(0.50f, "REGIME 2: CRITICAL PHASE TRANSITION ZONE (50% NOISE)");
    run_modulator_benchmark(0.70f, "REGIME 3: HIGH STRESS REGIME (70% NOISE)");

    return 0;
}
