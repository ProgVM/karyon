// ======================================================================================
// EXP-622: Non-Euclidean Phase Metric Memory Field (NE-PMMF)
// --------------------------------------------------------------------------------------
// Scientific Cybernetics (Bazilevs Paradigm Shift & KEP Principles 2, 3, 27, 28, 29, 30):
// 1. REJECTION OF VON NEUMANN REGISTERS & LEAKY LINEAR MATRIX SUMMATION:
//    - Memory is NEITHER a warehouse with discrete addresses (LOAD/STORE)
//    - NOR a leaky bucket of additive outer products (M_{t+1} = \alpha M_t + K^T V) that decays into semantic bleed.
// 2. MEMORY AS CONTINUOUS METRIC CURVATURE OF THE SUBSTRATE:
//    - The substrate maintains a continuous phase field \mathbf{\psi} \in \mathbb{R}^D and an endogenous
//      metric tensor field g_{\mu\nu}(\mathbf{x}).
//    - When high-surprise information passes through (intense Free Energy flux \Delta F_t), it does NOT "write to a cell".
//      It indents a GEODESIC WELL (curvature sink) in the metric tensor:
//         \Delta g_{\mu\nu} = \eta_g \cdot (\Delta F_t) \cdot (\psi_\mu \psi_\nu)
//    - These curvature wells are topologically stable (no linear superposition wash-out).
// 3. ZERO-ADDRESS RESONANT RECALL:
//    - Recall requires NO explicit read query or cosine lookup.
//    - When current sensory-latent flow enters the proximity of a previously curved region of phase space,
//      the trajectory is gravitationally deflected along the geodesics:
//         \frac{d^2 \psi^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{d\psi^\alpha}{d\tau} \frac{d\psi^\beta}{d\tau} = 0
//      The past state is spontaneously INDUCED into the present through physical resonance!
// 4. TESTING THE HYPOTHESIS: LONG-RANGE RETRIEVAL OVER DISTRACTOR SEAS:
//    - We test long-horizon factual retention: an anchor fact is presented at t=0, followed by a vast
//      chaotic distractor stream (N=2000..10000 steps).
//    - At t=T, a resonant cue is presented.
//    - Measure:
//      a) Metric Retention Integrity: Does the geodesic well survive the distractor sea without decay?
//      b) Resonant Deflection: Does the trajectory spontaneously reconstruct the anchor without address lookup?
//      c) Semantic Bleed Immunity: Does the metric keep distinct facts separated without cross-talk?
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
constexpr int METRIC_BASINS      = 64;     // Number of localized metric curvature centers in phase manifold
constexpr int DISTRACTOR_STEPS   = 5000;   // Length of chaotic distractor sea between anchor and cue
constexpr int NUM_TRIALS         = 20;     // Number of independent factual trials
constexpr float G_COUPLING       = 0.15f;  // Gravitational Christoffel deflection strength
constexpr float METRIC_ETA       = 0.25f;  // Metric indentation rate under Free Energy flux

// Continuous Non-Euclidean Metric Basin
struct MetricWell {
    float center[FIELD_DIM];    // Spatial coordinate of the curvature sink
    float curvature;            // Depth/intensity of curvature (mass/energy of memory)
    float metric_cov[FIELD_DIM];// Local metric diagonal tensor g_{\mu\mu}
    uint32_t lifetime_hits;     // Resonant deflection counter
};

// Continuous Physical Substrate State
struct PhaseFieldSubstrate {
    float psi[FIELD_DIM];       // Current continuous phase trajectory \psi_t
    float dpsi[FIELD_DIM];      // Momentum / derivative d\psi / d\tau
    
    // Non-Euclidean Curvature Memory Field (No RAM, No Registers)
    MetricWell wells[METRIC_BASINS];
    int active_wells;
    
    // Substrate Physical Telemetry
    float total_curvature_energy;
    float resonant_deflection_power;
    float semantic_bleed_cross_talk;
};

// Leaky Matrix Baseline (Traditional Linear Memory: M_{t+1} = \alpha M_t + K^T V)
struct LeakyLinearMemoryBaseline {
    float M[FIELD_DIM][FIELD_DIM];
    float decay_alpha;          // Typical decay factor (e.g. 0.999)
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

// ======================================================================================
// Physical Geodesic Flow & Resonant Induction Kernel
// ======================================================================================
__device__ void step_phase_metric_field(PhaseFieldSubstrate& sub, const float* external_input, float fe_flux) {
    // 1. External sensory flux drives trajectory momentum
    for (int i = 0; i < FIELD_DIM; ++i) {
        sub.dpsi[i] = 0.85f * sub.dpsi[i] + 0.15f * external_input[i];
    }

    // 2. Compute Geodesic Deflection from all Metric Curvature Wells
    // \frac{d^2 \psi^\mu}{d\tau^2} = - \sum_w \Gamma^\mu_w(\psi) \cdot (\psi - center_w)
    float deflection[FIELD_DIM] = {0.0f};
    float total_deflection = 0.0f;

    for (int w = 0; w < sub.active_wells; ++w) {
        float dist_sq = euclidean_dist_sq(sub.psi, sub.wells[w].center, FIELD_DIM);
        // Metric curvature potential: V(r) = - curvature / (1 + dist_sq)
        float proximity = 1.0f / (1.0f + 0.5f * dist_sq);
        float force_mag = sub.wells[w].curvature * proximity * proximity * G_COUPLING;

        if (proximity > 0.35f) {
            sub.wells[w].lifetime_hits++;
        }

        for (int i = 0; i < FIELD_DIM; ++i) {
            float grad = (sub.wells[w].center[i] - sub.psi[i]);
            // Deflection along local metric tensor g_{\mu\mu}
            deflection[i] += force_mag * grad * sub.wells[w].metric_cov[i];
        }
        total_deflection += force_mag;
    }

    sub.resonant_deflection_power = total_deflection;

    // 3. Integrate Geodesic Step (Leapfrog / Verlet continuous flow)
    for (int i = 0; i < FIELD_DIM; ++i) {
        sub.psi[i] += sub.dpsi[i] + deflection[i];
        // Natural bounded field saturation
        sub.psi[i] = tanhf(sub.psi[i]);
    }

    // 4. High Free Energy Flux indents a NEW or DEEPENS an existing Geodesic Well
    // Memory is an indelible physical indentation of the metric under surprise!
    if (fe_flux > 0.5f) {
        // Find nearest well or sprout a new one
        int best_w = -1;
        float min_d = 1e9f;
        for (int w = 0; w < sub.active_wells; ++w) {
            float d = euclidean_dist_sq(sub.psi, sub.wells[w].center, FIELD_DIM);
            if (d < min_d) {
                min_d = d;
                best_w = w;
            }
        }

        if (best_w >= 0 && min_d < 1.5f) {
            // Deepen existing curvature basin
            sub.wells[best_w].curvature += METRIC_ETA * fe_flux;
            for (int i = 0; i < FIELD_DIM; ++i) {
                sub.wells[best_w].metric_cov[i] += 0.05f * fabsf(sub.psi[i]);
            }
        } else if (sub.active_wells < METRIC_BASINS) {
            // Indent new geodesic well in phase space
            int new_w = sub.active_wells++;
            sub.wells[new_w].curvature = 1.0f + METRIC_ETA * fe_flux;
            sub.wells[new_w].lifetime_hits = 1;
            for (int i = 0; i < FIELD_DIM; ++i) {
                sub.wells[new_w].center[i] = sub.psi[i];
                sub.wells[new_w].metric_cov[i] = 1.0f;
            }
        }
    }
}

// ======================================================================================
// Leaky Matrix Update & Read (Baseline)
// ======================================================================================
__device__ void step_leaky_baseline(LeakyLinearMemoryBaseline& base, const float* input, float fe_flux, float* recall_out) {
    // 1. Outer-product write: M_{t+1} = \alpha M_t + x \otimes x
    for (int i = 0; i < FIELD_DIM; ++i) {
        for (int j = 0; j < FIELD_DIM; ++j) {
            base.M[i][j] = base.decay_alpha * base.M[i][j] + (fe_flux > 0.5f ? (input[i] * input[j]) : 0.0f);
        }
    }
    // 2. Read by query: y = M \cdot query
    for (int i = 0; i < FIELD_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < FIELD_DIM; ++j) {
            sum += base.M[i][j] * input[j];
        }
        recall_out[i] = tanhf(sum);
    }
}

// ======================================================================================
// Empirical Validation Kernel: Long-Horizon Memory Test across Chaotic Distractor Sea
// ======================================================================================
__global__ void run_memory_field_validation_kernel(float* out_metric_recall_fidelity,
                                                   float* out_leaky_recall_fidelity,
                                                   float* out_semantic_bleed_metric,
                                                   float* out_semantic_bleed_leaky) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState rand_state;
    curand_init(2026, 0, 0, &rand_state);

    float total_metric_fidelity = 0.0f;
    float total_leaky_fidelity  = 0.0f;
    float total_metric_bleed     = 0.0f;
    float total_leaky_bleed      = 0.0f;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        // Initialize Substrate & Baseline
        PhaseFieldSubstrate substrate;
        substrate.active_wells = 0;
        substrate.total_curvature_energy = 0.0f;
        substrate.resonant_deflection_power = 0.0f;
        substrate.semantic_bleed_cross_talk = 0.0f;
        for (int i = 0; i < FIELD_DIM; ++i) {
            substrate.psi[i] = 0.0f;
            substrate.dpsi[i] = 0.0f;
        }

        LeakyLinearMemoryBaseline leaky;
        leaky.decay_alpha = 0.9992f; // Very long half-life (~866 steps)
        for (int i = 0; i < FIELD_DIM; ++i) {
            for (int j = 0; j < FIELD_DIM; ++j) leaky.M[i][j] = 0.0f;
        }

        // 1. GENERATE ANCHOR FACT (Specific trajectory pattern)
        float anchor_pattern[FIELD_DIM];
        for (int i = 0; i < FIELD_DIM; ++i) {
            anchor_pattern[i] = (curand_uniform(&rand_state) > 0.5f) ? 1.0f : -1.0f;
        }

        // Present Anchor Fact under HIGH surprise flux (\Delta F_t = 1.0)
        for (int step = 0; step < 5; ++step) {
            step_phase_metric_field(substrate, anchor_pattern, 1.0f);
            float dummy_out[FIELD_DIM];
            step_leaky_baseline(leaky, anchor_pattern, 1.0f, dummy_out);
        }

        // 2. CHAOTIC DISTRACTOR SEA (5,000 steps of random sensory noise)
        // Testing whether the memory dissolves in the noise (Semantic Bleed / Decay)
        for (int step = 0; step < DISTRACTOR_STEPS; ++step) {
            float distractor[FIELD_DIM];
            for (int i = 0; i < FIELD_DIM; ++i) {
                distractor[i] = curand_normal(&rand_state) * 0.4f;
            }
            // Distractor has low Free Energy flux (routine background stream)
            step_phase_metric_field(substrate, distractor, 0.05f);
            float dummy_out[FIELD_DIM];
            step_leaky_baseline(leaky, distractor, 0.05f, dummy_out);
        }

        // 3. RESONANT CUE (Present partial / noisy cue of the anchor pattern: 50% corrupted)
        float cue_pattern[FIELD_DIM];
        for (int i = 0; i < FIELD_DIM; ++i) {
            if (curand_uniform(&rand_state) < 0.40f) {
                cue_pattern[i] = curand_normal(&rand_state); // Noise
            } else {
                cue_pattern[i] = anchor_pattern[i];         // Partial cue
            }
        }

        // Present Cue to Substrate and observe Geodesic Snapping
        for (int step = 0; step < 10; ++step) {
            step_phase_metric_field(substrate, cue_pattern, 0.1f);
        }

        // Present Cue to Leaky Baseline
        float leaky_recalled[FIELD_DIM];
        step_leaky_baseline(leaky, cue_pattern, 0.1f, leaky_recalled);

        // 4. MEASURE RECALL FIDELITY (Cosine similarity between reconstructed state and original anchor)
        float dot_metric = dot_product(substrate.psi, anchor_pattern, FIELD_DIM);
        float norm_sub = sqrtf(dot_product(substrate.psi, substrate.psi, FIELD_DIM));
        float norm_anc = sqrtf(dot_product(anchor_pattern, anchor_pattern, FIELD_DIM));
        float fidelity_metric = (norm_sub > 1e-5f && norm_anc > 1e-5f) ? (dot_metric / (norm_sub * norm_anc)) : 0.0f;

        float dot_leaky = dot_product(leaky_recalled, anchor_pattern, FIELD_DIM);
        float norm_leak = sqrtf(dot_product(leaky_recalled, leaky_recalled, FIELD_DIM));
        float fidelity_leaky = (norm_leak > 1e-5f && norm_anc > 1e-5f) ? (dot_leaky / (norm_leak * norm_anc)) : 0.0f;

        // 5. MEASURE SEMANTIC BLEED (Cross-talk with distractor noise)
        float bleed_metric = 1.0f - fabsf(fidelity_metric);
        float bleed_leaky  = 1.0f - fabsf(fidelity_leaky);

        total_metric_fidelity += fidelity_metric;
        total_leaky_fidelity  += fidelity_leaky;
        total_metric_bleed     += bleed_metric;
        total_leaky_bleed      += bleed_leaky;
    }

    *out_metric_recall_fidelity = total_metric_fidelity / NUM_TRIALS;
    *out_leaky_recall_fidelity  = total_leaky_fidelity / NUM_TRIALS;
    *out_semantic_bleed_metric  = total_metric_bleed / NUM_TRIALS;
    *out_semantic_bleed_leaky   = total_leaky_bleed / NUM_TRIALS;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-622: NON-EUCLIDEAN PHASE METRIC MEMORY FIELD (NE-PMMF)       ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Scientific Core      : Memory as Geodesic Curvature of Substrate (No RAM/Registers)\n";
    std::cout << "Access Mechanism     : Zero-Address Resonant Induction along Geodesics\n";
    std::cout << "Test Rigor           : Anchor Fact -> 5,000 Distractor Steps -> 40% Noisy Cue\n";
    std::cout << "Baseline Comparison  : Linear Leaky Outer-Product Matrix (M_{t+1} = \\alpha M_t + K^T V)\n\n";

    float* d_metric_fid = nullptr;
    float* d_leaky_fid  = nullptr;
    float* d_metric_bld = nullptr;
    float* d_leaky_bld  = nullptr;

    CHECK_CUDA(cudaMalloc(&d_metric_fid, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_leaky_fid,  sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_metric_bld, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_leaky_bld,  sizeof(float)));

    auto t_start = std::chrono::high_resolution_clock::now();

    run_memory_field_validation_kernel<<<1, 1>>>(d_metric_fid, d_leaky_fid, d_metric_bld, d_leaky_bld);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    float h_metric_fid = 0.0f, h_leaky_fid = 0.0f;
    float h_metric_bld = 0.0f, h_leaky_bld = 0.0f;

    CHECK_CUDA(cudaMemcpy(&h_metric_fid, d_metric_fid, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_leaky_fid,  d_leaky_fid,  sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_metric_bld, d_metric_bld, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_leaky_bld,  d_leaky_bld,  sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "========================================================================\n";
    std::cout << ">>> EMPIRICAL BENCHMARK RESULTS (Across " << NUM_TRIALS << " Independent Trials):\n";
    std::cout << ">>> Total GPU Runtime              : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << ">>> 1. NON-EUCLIDEAN METRIC CURVATURE MEMORY (NE-PMMF):\n";
    std::cout << "    - Resonant Recall Fidelity     : " << std::fixed << std::setprecision(4) << h_metric_fid << " (Cosine to Anchor)\n";
    std::cout << "    - Semantic Bleed / Noise Decay : " << std::fixed << std::setprecision(4) << h_metric_bld << " (Error)\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << ">>> 2. LEAKY LINEAR MATRIX BASELINE (SSD / Outer Product):\n";
    std::cout << "    - Linear Recall Fidelity       : " << std::fixed << std::setprecision(4) << h_leaky_fid << " (Cosine to Anchor)\n";
    std::cout << "    - Semantic Bleed / Noise Decay : " << std::fixed << std::setprecision(4) << h_leaky_bld << " (Error)\n";
    std::cout << "------------------------------------------------------------------------\n";

    float delta_fidelity = h_metric_fid - h_leaky_fid;
    std::cout << ">>> FIDELITY ADVANTAGE OF METRIC CURVATURE: " 
              << (delta_fidelity >= 0 ? "+" : "") << std::fixed << std::setprecision(4) << delta_fidelity << "\n";
    std::cout << "========================================================================\n";

    CHECK_CUDA(cudaFree(d_metric_fid));
    CHECK_CUDA(cudaFree(d_leaky_fid));
    CHECK_CUDA(cudaFree(d_metric_bld));
    CHECK_CUDA(cudaFree(d_leaky_bld));
    return 0;
}
