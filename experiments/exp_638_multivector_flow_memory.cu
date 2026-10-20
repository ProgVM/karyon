// ======================================================================================
// EXP-638: Multi-Vector Tensor Wave Bundle (Multi-Vector Flow Memory) Benchmark
// --------------------------------------------------------------------------------------
// Objective & Mathematical Rationale:
// As established in EXP-637, a single 1D vector h in R^32 under 80% noise carries
// a Shannon mutual information capacity of only ~5.12 bits, making complete retrieval
// of 50 facts (requiring >100 bits) physically impossible in a single 1D channel.
//
// Bazilevs' Breakthrough Thesis:
// Transform Karyon's continuous state from a single 1D vector into a Multi-Vector Tensor
// Wave Bundle H_t = [h_t^{(1)}, h_t^{(2)}, ..., h_t^{(M)}] in R^{M x D}.
//
// Physical Noise Cancellation Dynamics:
// When a fact descriptor is distributed across M independent vector channels,
// incoherent noise across channels cancels out at a rate of O(sqrt(M)), while the coherent
// fact signal sums constructively at O(M).
// Thus, Effective SNR = M * SNR_single!
//
// Benchmark Protocol on Tesla T4 GPU (1,000 trials per condition):
// - K = 50 Facts (32D basis per vector channel).
// - 80% Noise corrupted retrieval cues (0.2 signal + 0.8 gaussian noise per channel).
// - 4 Precise Architectural Conditions:
//   Cond 1: Single Vector Baseline (M = 1 channel, D = 32)
//   Cond 2: 4-Vector Bundle (M = 4 channels, D = 32)
//   Cond 3: 8-Vector Bundle (M = 8 channels, D = 32)
//   Cond 4: 16-Vector Bundle (M = 16 channels, D = 32)
//
// Telemetry Tracked:
// - Base noise fidelity
// - Multi-channel reconstructed fidelity
// - Net active gain over noise
// - Semantic bleed error (1.0 - fidelity)
// - Compute latency (seconds / 1,000 trials)
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

constexpr int DIM                = 32;       // 32D basis per channel
constexpr int NUM_FACTS          = 50;       // K = 50 Facts
constexpr int TRIALS_PER_COND    = 1000;     // 1,000 independent GPU trials

struct MultiVecResult {
    float base_noise_fidelity;
    float fid_m1_single;
    float fid_m4_bundle;
    float fid_m8_bundle;
    float fid_m16_bundle;
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
// Kernel: Multi-Vector Tensor Wave Bundle Retrieval
// --------------------------------------------------------------------------------------
__global__ void multivector_flow_kernel(MultiVecResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // Generate max channel capacity: M_max = 16 channels x 32D
    constexpr int MAX_M = 16;
    float facts_multivec[NUM_FACTS][MAX_M][DIM];

    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int m = 0; m < MAX_M; ++m) {
            for (int i = 0; i < DIM; ++i) {
                facts_multivec[f][m][i] = curand_normal(&st);
            }
            normalize_vec(facts_multivec[f][m], DIM);
        }
    }

    float tot_base = 0.0f;
    float tot_m1   = 0.0f;
    float tot_m4   = 0.0f;
    float tot_m8   = 0.0f;
    float tot_m16  = 0.0f;

    const float beta = 12.0f;

    // Evaluate for each of the 50 facts
    for (int f = 0; f < NUM_FACTS; ++f) {
        // Generate multi-channel noisy cue (independent noise per channel)
        float cue_multivec[MAX_M][DIM];
        float base_fid_sum = 0.0f;

        for (int m = 0; m < MAX_M; ++m) {
            for (int i = 0; i < DIM; ++i) {
                cue_multivec[m][i] = (curand_uniform(&st) < 0.20f) ? facts_multivec[f][m][i] : curand_normal(&st) * 0.8f;
            }
            float norm_in = sqrtf(dot_prod(cue_multivec[m], cue_multivec[m], DIM));
            if (norm_in > 1e-4f) {
                base_fid_sum += fmaxf(0.0f, dot_prod(cue_multivec[m], facts_multivec[f][m], DIM) / norm_in);
            }
            normalize_vec(cue_multivec[m], DIM);
        }
        tot_base += base_fid_sum / MAX_M;

        // -------------------------------------------------------------
        // Helper Lambda: Multi-Channel Tensor Contraction Readout
        // -------------------------------------------------------------
        auto eval_m_channels = [&](int M_active) -> float {
            float max_logit = -1e9f;
            float logits[NUM_FACTS];

            for (int j = 0; j < NUM_FACTS; ++j) {
                float channel_sim_sum = 0.0f;
                for (int m = 0; m < M_active; ++m) {
                    channel_sim_sum += dot_prod(cue_multivec[m], facts_multivec[j][m], DIM);
                }
                // Normalize by sqrt(M_active) for stable temperature scaling
                logits[j] = (beta / sqrtf((float)M_active)) * channel_sim_sum;
                if (logits[j] > max_logit) max_logit = logits[j];
            }

            float sum_exp = 0.0f;
            float weights[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                weights[j] = expf(logits[j] - max_logit);
                sum_exp += weights[j];
            }

            // Reconstruct multi-channel target tensor
            float multi_rec_fid = 0.0f;
            for (int m = 0; m < M_active; ++m) {
                float h_out[DIM] = {0.0f};
                for (int j = 0; j < NUM_FACTS; ++j) {
                    float a = weights[j] / (sum_exp + 1e-6f);
                    for (int i = 0; i < DIM; ++i) {
                        h_out[i] += a * facts_multivec[j][m][i];
                    }
                }
                normalize_vec(h_out, DIM);
                multi_rec_fid += fmaxf(0.0f, dot_prod(h_out, facts_multivec[f][m], DIM));
            }
            return multi_rec_fid / M_active;
        };

        tot_m1  += eval_m_channels(1);
        tot_m4  += eval_m_channels(4);
        tot_m8  += eval_m_channels(8);
        tot_m16 += eval_m_channels(16);
    }

    results[idx].base_noise_fidelity = tot_base / NUM_FACTS;
    results[idx].fid_m1_single       = tot_m1 / NUM_FACTS;
    results[idx].fid_m4_bundle       = tot_m4 / NUM_FACTS;
    results[idx].fid_m8_bundle       = tot_m8 / NUM_FACTS;
    results[idx].fid_m16_bundle      = tot_m16 / NUM_FACTS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-638: MULTI-VECTOR TENSOR WAVE BUNDLE (MULTI-VECTOR FLOW)    ===\n"
              << "========================================================================\n"
              << "Bazilevs' Thesis     : Transforming Karyon's state into M-Vector Bundle H_t\n"
              << "Physical Dynamics    : Incoherent Noise Cancels O(sqrt M), Signal Accumulates O(M)\n"
              << "Stress Benchmark     : K=50 Facts across M = {1, 4, 8, 16} 32D Channels -> 80% Noise\n"
              << "Trials Evaluated     : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    MultiVecResult* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(MultiVecResult)));
    std::vector<MultiVecResult> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    multivector_flow_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(MultiVecResult), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float b = 0.0f, m1 = 0.0f, m4 = 0.0f, m8 = 0.0f, m16 = 0.0f;
    for (const auto& r : h_res) {
        b   += r.base_noise_fidelity;
        m1  += r.fid_m1_single;
        m4  += r.fid_m4_bundle;
        m8  += r.fid_m8_bundle;
        m16 += r.fid_m16_bundle;
    }
    b /= h_res.size(); m1 /= h_res.size(); m4 /= h_res.size(); m8 /= h_res.size(); m16 /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << ">>> BASE NOISE CUE FIDELITY : " << b << " (80% Noise Floor)\n\n"
              << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION                 | RECONSTRUCTED FIDELITY | NET ACTIVE GAIN  |\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Single Vector Baseline (M = 1)       | " << m1  << "                 | " << std::showpos << (m1 - b)  << std::noshowpos << "          |\n"
              << "| 2. 4-Vector Tensor Bundle (M = 4)       | " << m4  << "                 | " << std::showpos << (m4 - b)  << std::noshowpos << "          |\n"
              << "| 3. 8-Vector Tensor Bundle (M = 8)       | " << m8  << "                 | " << std::showpos << (m8 - b)  << std::noshowpos << "          |\n"
              << "| 4. 16-Vector Tensor Bundle (M = 16)     | " << m16 << "                 | " << std::showpos << (m16 - b) << std::noshowpos << "          |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 facts x 4 M-configs): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
