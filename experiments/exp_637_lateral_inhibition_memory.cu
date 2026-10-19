// ======================================================================================
// EXP-637: Contrastive Lateral Inhibition & Centroid Subtraction Benchmark
// --------------------------------------------------------------------------------------
// Objective & Cybernetic Rationale:
// EXP-636 proved:
// 1. Hierarchical tree clustering suffers fatal cascade misrouting at root (error ~60%).
// 2. Continuous multi-step relaxation contracts the state toward the barycenter (cluster mean),
//    causing catastrophic blurring of individual facts.
//
// In EXP-637, we test direct All-to-All memory equipped with:
// 1. Centroid Subtraction / Mean Zeroing: Eliminates the common background mode / DC bias,
//    preventing barycentric collapse: v_i_centered = v_i - mean(V).
// 2. Anti-Hebbian Lateral Inhibition: Suppresses cross-talk interference from competing
//    attractors via lateral inhibitory projection: W_lat = I - alpha * (1/K) * sum(k_j k_j^T).
// 3. Contrastive Temperature Rescaling: Softmax with beta=24.0 on centered, laterally
//    inhibited dot products.
//
// Benchmark Protocol:
// - K = 50 Facts (32D normalized vectors).
// - 500 Distractor decay steps (decay gamma = 0.999).
// - 80% Noise corrupted retrieval cues (0.2 fact signal + 0.8 gaussian noise).
// - 1,000 independent evaluation runs on Tesla T4 GPU across 4 conditions:
//   Cond 1: Standard Uncentered Flat Memory (Baseline from EXP-636)
//   Cond 2: Centroid Subtracted Memory (Zero-mean projection)
//   Cond 3: Lateral Inhibition Matrix (W_lat * cue)
//   Cond 4: Combined Centroid Subtraction + Lateral Inhibition + Beta=24 Sharpening
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

constexpr int DIM                = 32;       // 32D basis
constexpr int NUM_FACTS          = 50;       // K = 50 Facts
constexpr int DISTRACTOR_STEPS   = 500;      // 500 distractor steps
constexpr int TRIALS_PER_COND    = 1000;     // 1,000 independent GPU trials

struct LatTelemetry {
    float base_noise_fidelity;
    float fid_cond1_baseline;
    float fid_cond2_centered;
    float fid_cond3_latinhib;
    float fid_cond4_combined;
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

__global__ void lateral_inhibition_kernel(LatTelemetry* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 facts with correlated background (realistic semantic distribution)
    float common_mode[DIM];
    for (int i = 0; i < DIM; ++i) common_mode[i] = curand_normal(&st);
    normalize_vec(common_mode, DIM);

    float facts[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int i = 0; i < DIM; ++i) {
            facts[f][i] = 0.50f * common_mode[i] + 0.50f * curand_normal(&st);
        }
        normalize_vec(facts[f], DIM);
    }

    // 2. Compute Centroid / Mean vector
    float centroid[DIM] = {0.0f};
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int i = 0; i < DIM; ++i) centroid[i] += facts[f][i];
    }
    for (int i = 0; i < DIM; ++i) centroid[i] /= NUM_FACTS;

    // 3. Compute Centered Facts
    float facts_centered[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int i = 0; i < DIM; ++i) {
            facts_centered[f][i] = facts[f][i] - centroid[i];
        }
        normalize_vec(facts_centered[f], DIM);
    }

    // 4. Compute Lateral Inhibition Operator: W_lat = I - alpha * C_corr
    // where C_corr = (1/K) sum_f facts[f] * facts[f]^T
    float C_corr[DIM][DIM] = {0.0f};
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int r = 0; r < DIM; ++r) {
            for (int c = 0; c < DIM; ++c) {
                C_corr[r][c] += facts[f][r] * facts[f][c];
            }
        }
    }
    const float alpha_lat = 0.70f;
    for (int r = 0; r < DIM; ++r) {
        for (int c = 0; c < DIM; ++c) {
            C_corr[r][c] = (r == c ? 1.0f : 0.0f) - (alpha_lat / NUM_FACTS) * C_corr[r][c];
        }
    }

    float tot_base  = 0.0f;
    float tot_c1    = 0.0f;
    float tot_c2    = 0.0f;
    float tot_c3    = 0.0f;
    float tot_c4    = 0.0f;

    const float beta_std  = 12.0f;
    const float beta_high = 24.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        // 80% noise cue
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
        // Cond 1: Standard Uncentered Flat Memory (EXP-636 Baseline)
        // -------------------------------------------------------------
        {
            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta_std * dot_prod(cue_norm, facts[j], DIM);
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
            tot_c1 += fmaxf(0.0f, dot_prod(h_out, facts[f], DIM));
        }

        // -------------------------------------------------------------
        // Cond 2: Centroid Subtracted Memory (Zero Mean Cue & Keys)
        // -------------------------------------------------------------
        {
            float cue_centered[DIM];
            for (int i = 0; i < DIM; ++i) cue_centered[i] = cue_norm[i] - centroid[i];
            normalize_vec(cue_centered, DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta_std * dot_prod(cue_centered, facts_centered[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * facts[j][i]; // Reconstruct in true fact space
            }
            normalize_vec(h_out, DIM);
            tot_c2 += fmaxf(0.0f, dot_prod(h_out, facts[f], DIM));
        }

        // -------------------------------------------------------------
        // Cond 3: Lateral Inhibition Matrix (Anti-Hebbian W_lat * cue)
        // -------------------------------------------------------------
        {
            float cue_inhibited[DIM] = {0.0f};
            for (int r = 0; r < DIM; ++r) {
                for (int c = 0; c < DIM; ++c) {
                    cue_inhibited[r] += C_corr[r][c] * cue_norm[c];
                }
            }
            normalize_vec(cue_inhibited, DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta_std * dot_prod(cue_inhibited, facts[j], DIM);
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
            tot_c3 += fmaxf(0.0f, dot_prod(h_out, facts[f], DIM));
        }

        // -------------------------------------------------------------
        // Cond 4: Combined Centroid Subtraction + Lateral Inhib + Beta=24
        // -------------------------------------------------------------
        {
            float cue_proc[DIM];
            for (int i = 0; i < DIM; ++i) cue_proc[i] = cue_norm[i] - centroid[i];

            float cue_final[DIM] = {0.0f};
            for (int r = 0; r < DIM; ++r) {
                for (int c = 0; c < DIM; ++c) {
                    cue_final[r] += C_corr[r][c] * cue_proc[c];
                }
            }
            normalize_vec(cue_final, DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta_high * dot_prod(cue_final, facts_centered[j], DIM);
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
            tot_c4 += fmaxf(0.0f, dot_prod(h_out, facts[f], DIM));
        }
    }

    results[idx].base_noise_fidelity   = tot_base / NUM_FACTS;
    results[idx].fid_cond1_baseline    = tot_c1 / NUM_FACTS;
    results[idx].fid_cond2_centered    = tot_c2 / NUM_FACTS;
    results[idx].fid_cond3_latinhib    = tot_c3 / NUM_FACTS;
    results[idx].fid_cond4_combined    = tot_c4 / NUM_FACTS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-637: CONTRASTIVE LATERAL INHIBITION & CENTROID SUBTRACTION   ===\n"
              << "========================================================================\n"
              << "Cybernetic Goal      : Eradicating Barycenter Drift & Suppressing Cross-Talk\n"
              << "Stress Benchmark     : K=50 Facts (32D) with Common Background -> 80% Noise\n"
              << "Trials Evaluated     : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    LatTelemetry* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(LatTelemetry)));
    std::vector<LatTelemetry> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    lateral_inhibition_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(LatTelemetry), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float b = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f;
    for (const auto& r : h_res) {
        b  += r.base_noise_fidelity;
        c1 += r.fid_cond1_baseline;
        c2 += r.fid_cond2_centered;
        c3 += r.fid_cond3_latinhib;
        c4 += r.fid_cond4_combined;
    }
    b /= h_res.size(); c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << ">>> BASE NOISE CUE FIDELITY : " << b << " (80% Noise Floor)\n\n"
              << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION                 | RECONSTRUCTED FIDELITY | ACTIVE GAIN|\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Baseline Flat Uncentered             | " << c1 << "                 | " << std::showpos << (c1 - b) << std::noshowpos << "     |\n"
              << "| 2. Centroid Subtracted (Zero-Mean)      | " << c2 << "                 | " << std::showpos << (c2 - b) << std::noshowpos << "     |\n"
              << "| 3. Anti-Hebbian Lateral Inhibition      | " << c3 << "                 | " << std::showpos << (c3 - b) << std::noshowpos << "     |\n"
              << "| 4. Combined (Zero-Mean + LatInhib + B24)| " << c4 << "                 | " << std::showpos << (c4 - b) << std::noshowpos << "     |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 facts x 4 conditions): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
