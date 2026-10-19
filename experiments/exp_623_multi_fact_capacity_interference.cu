// ======================================================================================
// EXP-623: Multi-Fact Capacity & Interference Stress Test (MF-CIST)
// --------------------------------------------------------------------------------------
// Crucial Scientific Probe:
// In EXP-622, a single anchor fact was tested. Under a single fact, a linear matrix
// M = x \otimes x is just a rank-1 projection, yielding a trivial 1.0 cosine match.
// BUT IN REAL REASONING / LANGUAGE, THOUSANDS OF DISTINCT FACTS ARE INTERLEAVED!
// When 100 DIFFERENT FACTS are written:
// 1. Linear Leaky Matrix suffers catastrophic SEMANTIC BLEED (M = \sum x_i \otimes x_i -> White Noise).
//    Capacity is bound by Field Dimension D=32 (Rank bottleneck).
// 2. Non-Euclidean Metric Curvature (NE-PMMF) maintains LOCALIZED GEODESIC WELLS.
//    Different facts occupy distinct topological basins without linear superposition interference!
// Let us test N_FACTS = 50 distinct facts interleaved with 5,000 distractor steps.
// ======================================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
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

constexpr int FIELD_DIM          = 32;     // Dimension of the continuous phase field \psi
constexpr int METRIC_BASINS      = 64;     // Metric curvature centers
constexpr int NUM_FACTS          = 50;     // 50 DISTINCT FACTS (Exceeds Linear Dimension 32!)
constexpr int DISTRACTOR_STEPS   = 5000;   // Chaotic stream
constexpr int NUM_TRIALS         = 10;     // Independent trials
constexpr float G_COUPLING       = 0.25f;  // Gravitational Christoffel deflection
constexpr float METRIC_ETA       = 0.50f;  // Metric indentation rate

struct MetricWell {
    float center[FIELD_DIM];
    float curvature;
    float metric_cov[FIELD_DIM];
    uint32_t lifetime_hits;
};

struct PhaseFieldSubstrate {
    float psi[FIELD_DIM];
    float dpsi[FIELD_DIM];
    MetricWell wells[METRIC_BASINS];
    int active_wells;
};

struct LeakyLinearMemoryBaseline {
    float M[FIELD_DIM][FIELD_DIM];
    float decay_alpha;
};

__device__ inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

__device__ inline float euclidean_dist_sq(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

__device__ void step_phase_metric_field(PhaseFieldSubstrate& sub, const float* external_input, float fe_flux) {
    for (int i = 0; i < FIELD_DIM; ++i) {
        sub.dpsi[i] = 0.80f * sub.dpsi[i] + 0.20f * external_input[i];
    }

    float deflection[FIELD_DIM] = {0.0f};
    for (int w = 0; w < sub.active_wells; ++w) {
        float dist_sq = euclidean_dist_sq(sub.psi, sub.wells[w].center, FIELD_DIM);
        float proximity = 1.0f / (1.0f + 0.5f * dist_sq);
        float force_mag = sub.wells[w].curvature * proximity * proximity * G_COUPLING;

        for (int i = 0; i < FIELD_DIM; ++i) {
            float grad = (sub.wells[w].center[i] - sub.psi[i]);
            deflection[i] += force_mag * grad * sub.wells[w].metric_cov[i];
        }
    }

    for (int i = 0; i < FIELD_DIM; ++i) {
        sub.psi[i] = tanhf(sub.psi[i] + sub.dpsi[i] + deflection[i]);
    }

    if (fe_flux > 0.5f) {
        int best_w = -1;
        float min_d = 1e9f;
        for (int w = 0; w < sub.active_wells; ++w) {
            float d = euclidean_dist_sq(sub.psi, sub.wells[w].center, FIELD_DIM);
            if (d < min_d) {
                min_d = d;
                best_w = w;
            }
        }

        if (best_w >= 0 && min_d < 1.0f) {
            sub.wells[best_w].curvature += METRIC_ETA * fe_flux;
        } else if (sub.active_wells < METRIC_BASINS) {
            int new_w = sub.active_wells++;
            sub.wells[new_w].curvature = 2.0f + METRIC_ETA * fe_flux;
            for (int i = 0; i < FIELD_DIM; ++i) {
                sub.wells[new_w].center[i] = sub.psi[i];
                sub.wells[new_w].metric_cov[i] = 1.0f;
            }
        }
    }
}

__device__ void step_leaky_baseline(LeakyLinearMemoryBaseline& base, const float* input, float fe_flux, float* recall_out) {
    for (int i = 0; i < FIELD_DIM; ++i) {
        for (int j = 0; j < FIELD_DIM; ++j) {
            base.M[i][j] = base.decay_alpha * base.M[i][j] + (fe_flux > 0.5f ? (input[i] * input[j]) : 0.0f);
        }
    }
    for (int i = 0; i < FIELD_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < FIELD_DIM; ++j) {
            sum += base.M[i][j] * input[j];
        }
        recall_out[i] = tanhf(sum);
    }
}

__global__ void run_multi_fact_capacity_kernel(float* out_metric_fidelity, float* out_leaky_fidelity) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState rand_state;
    curand_init(2026, 0, 0, &rand_state);

    float avg_metric_fid = 0.0f;
    float avg_leaky_fid  = 0.0f;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        PhaseFieldSubstrate substrate;
        substrate.active_wells = 0;
        for (int i = 0; i < FIELD_DIM; ++i) {
            substrate.psi[i] = 0.0f;
            substrate.dpsi[i] = 0.0f;
        }

        LeakyLinearMemoryBaseline leaky;
        leaky.decay_alpha = 0.9998f;
        for (int i = 0; i < FIELD_DIM; ++i) {
            for (int j = 0; j < FIELD_DIM; ++j) leaky.M[i][j] = 0.0f;
        }

        // GENERATE 50 DISTINCT FACTS (Random unit vectors)
        float facts[NUM_FACTS][FIELD_DIM];
        for (int f = 0; f < NUM_FACTS; ++f) {
            float norm = 0.0f;
            for (int i = 0; i < FIELD_DIM; ++i) {
                facts[f][i] = curand_normal(&rand_state);
                norm += facts[f][i] * facts[f][i];
            }
            norm = sqrtf(norm);
            for (int i = 0; i < FIELD_DIM; ++i) facts[f][i] /= norm;
        }

        // 1. SEQUENTIALLY ENCODE ALL 50 FACTS (High surprise \Delta F_t = 1.0)
        for (int f = 0; f < NUM_FACTS; ++f) {
            for (int s = 0; s < 3; ++s) {
                step_phase_metric_field(substrate, facts[f], 1.0f);
                float dummy[FIELD_DIM];
                step_leaky_baseline(leaky, facts[f], 1.0f, dummy);
            }
        }

        // 2. CHAOTIC DISTRACTOR STREAM (5,000 steps)
        for (int step = 0; step < DISTRACTOR_STEPS; ++step) {
            float distractor[FIELD_DIM];
            for (int i = 0; i < FIELD_DIM; ++i) distractor[i] = curand_normal(&rand_state) * 0.2f;
            step_phase_metric_field(substrate, distractor, 0.02f);
            float dummy[FIELD_DIM];
            step_leaky_baseline(leaky, distractor, 0.02f, dummy);
        }

        // 3. RETRIEVAL PROBE ACROSS ALL 50 FACTS (With 50% noisy cues)
        float trial_metric_sum = 0.0f;
        float trial_leaky_sum  = 0.0f;

        for (int f = 0; f < NUM_FACTS; ++f) {
            float cue[FIELD_DIM];
            for (int i = 0; i < FIELD_DIM; ++i) {
                cue[i] = (curand_uniform(&rand_state) < 0.50f) ? facts[f][i] : curand_normal(&rand_state) * 0.5f;
            }

            // Resonant Induction in Metric Substrate
            for (int s = 0; s < 5; ++s) {
                step_phase_metric_field(substrate, cue, 0.05f);
            }

            float dot_m = dot_product(substrate.psi, facts[f], FIELD_DIM);
            float n_sub = sqrtf(dot_product(substrate.psi, substrate.psi, FIELD_DIM));
            float n_f   = sqrtf(dot_product(facts[f], facts[f], FIELD_DIM));
            float fid_m = (n_sub > 1e-4f) ? (dot_m / (n_sub * n_f)) : 0.0f;

            // Leaky Linear Read
            float leaky_out[FIELD_DIM];
            step_leaky_baseline(leaky, cue, 0.05f, leaky_out);
            float dot_l = dot_product(leaky_out, facts[f], FIELD_DIM);
            float n_l   = sqrtf(dot_product(leaky_out, leaky_out, FIELD_DIM));
            float fid_l = (n_l > 1e-4f) ? (dot_l / (n_l * n_f)) : 0.0f;

            trial_metric_sum += fid_m;
            trial_leaky_sum  += fid_l;
        }

        avg_metric_fid += (trial_metric_sum / NUM_FACTS);
        avg_leaky_fid  += (trial_leaky_sum  / NUM_FACTS);
    }

    *out_metric_fidelity = avg_metric_fid / NUM_TRIALS;
    *out_leaky_fidelity  = avg_leaky_fid  / NUM_TRIALS;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-623: MULTI-FACT CAPACITY & INTERFERENCE STRESS TEST (MF-CIST) ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Field Dimension (D)  : " << FIELD_DIM << "\n";
    std::cout << "Interleaved Facts (K): " << NUM_FACTS << " (K > D: Exceeds Linear Capacity!)\n";
    std::cout << "Distractor Stream    : " << DISTRACTOR_STEPS << " steps of chaotic noise\n";
    std::cout << "Query Condition      : 50% Noisy Cue per Fact across all " << NUM_FACTS << " Facts\n\n";

    float* d_mf = nullptr;
    float* d_lf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_mf, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lf, sizeof(float)));

    auto t_start = std::chrono::high_resolution_clock::now();
    run_multi_fact_capacity_kernel<<<1, 1>>>(d_mf, d_lf);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    float h_mf = 0.0f, h_lf = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_mf, d_mf, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_lf, d_lf, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "========================================================================\n";
    std::cout << ">>> BENCHMARK RESULTS (Across " << NUM_TRIALS << " Trials, " << NUM_FACTS << " Facts Each):\n";
    std::cout << ">>> Execution Runtime              : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << ">>> 1. NON-EUCLIDEAN METRIC CURVATURE (NE-PMMF):\n";
    std::cout << "    - Multi-Fact Recall Fidelity   : " << std::fixed << std::setprecision(4) << h_mf << " (Cosine)\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << ">>> 2. LINEAR LEAKY MATRIX (SSD / Mamba / Outer Product):\n";
    std::cout << "    - Multi-Fact Recall Fidelity   : " << std::fixed << std::setprecision(4) << h_lf << " (Cosine)\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << ">>> FIDELITY ADVANTAGE UNDER CAPACITY OVERLOAD (K > D):\n";
    std::cout << "    Delta (Metric - Linear)        : " 
              << ((h_mf - h_lf) >= 0 ? "+" : "") << std::fixed << std::setprecision(4) << (h_mf - h_lf) << "\n";
    std::cout << "========================================================================\n";

    CHECK_CUDA(cudaFree(d_mf));
    CHECK_CUDA(cudaFree(d_lf));
    return 0;
}
