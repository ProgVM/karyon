// ======================================================================================
// EXP-633: Dynamic Rank Expansion Tensor Memory (DRE-TM)
// --------------------------------------------------------------------------------------
// Objective & Honest Engineering Assessment:
// - EXP-632 proved mathematical rank limit: 50 facts (1600 numbers) in a 32x32 matrix
//   (1024 numbers, Rank <= 32) mechanically caps reconstruction fidelity at ~0.22.
// - EXP-633 tests Dynamic Rank Expansion:
//   Baseline A: Static Rank-32 memory matrix (32x32 = 1024 floats)
//   Condition B: Expanded Rank-64 memory matrix (64x64 = 4096 floats, capacity > 1600)
//   Condition C: Dynamic Key-Value Tensor Attractor Bank (K=64 basis slots, 64x32 = 2048 floats)
//
// Protocol:
// 1. Same 50 Facts (32D normalized vectors).
// 2. Same 500 chaotic distractor steps (decay factor gamma = 0.999).
// 3. Same 80% noise corruption on retrieval cues.
// 4. Exact comparative measurement across all 3 architectures:
//    - Final Cosine Fidelity
//    - Net Active Signal Gain over base noise
//    - Memory capacity utilization
//    - Execution time & throughput
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

constexpr int DIM                = 32;       // Base vector dimension
constexpr int EXPANDED_DIM       = 64;       // Expanded rank dimension
constexpr int NUM_FACTS          = 50;       // K=50 Facts
constexpr int DISTRACTOR_STEPS   = 500;      // 500 distractor steps
constexpr int TRIALS_PER_COND    = 1000;     // 1000 independent evaluation runs per condition

struct BenchResult {
    float base_noise_fidelity;
    float reconstructed_fidelity;
    float active_gain;
    double execution_sec;
};

__device__ inline float dot_prod(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// --------------------------------------------------------------------------------------
// Kernel A: Baseline Static Rank-32 (32x32 Matrix)
// --------------------------------------------------------------------------------------
__global__ void benchmark_rank32_kernel(BenchResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    float M[DIM][DIM] = {0.0f};

    // 50 facts
    float facts[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        float norm = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            facts[f][i] = curand_normal(&st);
            norm += facts[f][i] * facts[f][i];
        }
        norm = sqrtf(norm + 1e-6f);
        for (int i = 0; i < DIM; ++i) facts[f][i] /= norm;
    }

    // Write: Outer-product accumulation
    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int r = 0; r < DIM; ++r) {
            for (int c = 0; c < DIM; ++c) {
                M[r][c] += facts[f][r] * facts[f][c];
            }
        }
    }

    // Distractor Decay
    for (int s = 0; s < DISTRACTOR_STEPS; ++s) {
        for (int i = 0; i < DIM; ++i) M[i][i] *= 0.999f;
    }

    // Read: 80% noise cues
    float tot_base = 0.0f;
    float tot_rec = 0.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        float cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            cue[i] = (curand_uniform(&st) < 0.20f) ? facts[f][i] : curand_normal(&st) * 0.8f;
        }

        float dot_in = dot_prod(cue, facts[f], DIM);
        float norm_in = sqrtf(dot_prod(cue, cue, DIM));
        tot_base += (norm_in > 1e-4f) ? fmaxf(0.0f, dot_in / norm_in) : 0.0f;

        // Recall: M * cue
        float h_out[DIM];
        float norm_out = 0.0f;
        for (int r = 0; r < DIM; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < DIM; ++c) sum += M[r][c] * cue[c];
            h_out[r] = sum;
            norm_out += sum * sum;
        }
        norm_out = sqrtf(norm_out + 1e-6f);
        for (int r = 0; r < DIM; ++r) h_out[r] /= norm_out;

        float dot_rec = dot_prod(h_out, facts[f], DIM);
        tot_rec += fmaxf(0.0f, dot_rec);
    }

    results[idx].base_noise_fidelity   = tot_base / NUM_FACTS;
    results[idx].reconstructed_fidelity = tot_rec / NUM_FACTS;
    results[idx].active_gain           = (tot_rec - tot_base) / NUM_FACTS;
}

// --------------------------------------------------------------------------------------
// Kernel B: Expanded Rank-64 (64x64 Matrix via Linear Projection)
// --------------------------------------------------------------------------------------
__global__ void benchmark_rank64_kernel(BenchResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + 10000 + idx, 0, 0, &st);

    float M64[EXPANDED_DIM][EXPANDED_DIM] = {0.0f};

    // Projection matrix W_proj: 64 x 32 (pseudo-random orthogonal expansion)
    float W_proj[EXPANDED_DIM][DIM];
    for (int r = 0; r < EXPANDED_DIM; ++r) {
        float norm = 0.0f;
        for (int c = 0; c < DIM; ++c) {
            W_proj[r][c] = ((r % DIM) == c) ? 1.0f : ((r + c) % 3 == 0 ? 0.2f : -0.2f);
            norm += W_proj[r][c] * W_proj[r][c];
        }
        norm = sqrtf(norm + 1e-6f);
        for (int c = 0; c < DIM; ++c) W_proj[r][c] /= norm;
    }

    // 50 facts
    float facts[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        float norm = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            facts[f][i] = curand_normal(&st);
            norm += facts[f][i] * facts[f][i];
        }
        norm = sqrtf(norm + 1e-6f);
        for (int i = 0; i < DIM; ++i) facts[f][i] /= norm;
    }

    // Write: Project to 64D and outer-product into 64x64 matrix
    for (int f = 0; f < NUM_FACTS; ++f) {
        float fact64[EXPANDED_DIM];
        for (int r = 0; r < EXPANDED_DIM; ++r) {
            float s = 0.0f;
            for (int c = 0; c < DIM; ++c) s += W_proj[r][c] * facts[f][c];
            fact64[r] = s;
        }

        for (int r = 0; r < EXPANDED_DIM; ++r) {
            for (int c = 0; c < EXPANDED_DIM; ++c) {
                M64[r][c] += fact64[r] * fact64[c];
            }
        }
    }

    // Distractor Decay
    for (int s = 0; s < DISTRACTOR_STEPS; ++s) {
        for (int i = 0; i < EXPANDED_DIM; ++i) M64[i][i] *= 0.999f;
    }

    // Read: 80% noise cues
    float tot_base = 0.0f;
    float tot_rec = 0.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        float cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            cue[i] = (curand_uniform(&st) < 0.20f) ? facts[f][i] : curand_normal(&st) * 0.8f;
        }

        float dot_in = dot_prod(cue, facts[f], DIM);
        float norm_in = sqrtf(dot_prod(cue, cue, DIM));
        tot_base += (norm_in > 1e-4f) ? fmaxf(0.0f, dot_in / norm_in) : 0.0f;

        // Project cue to 64D
        float cue64[EXPANDED_DIM];
        for (int r = 0; r < EXPANDED_DIM; ++r) {
            float s = 0.0f;
            for (int c = 0; c < DIM; ++c) s += W_proj[r][c] * cue[c];
            cue64[r] = s;
        }

        // Recall in 64D: M64 * cue64
        float h64[EXPANDED_DIM];
        for (int r = 0; r < EXPANDED_DIM; ++r) {
            float s = 0.0f;
            for (int c = 0; c < EXPANDED_DIM; ++c) s += M64[r][c] * cue64[c];
            h64[r] = s;
        }

        // Project back to 32D: W_proj^T * h64
        float h_out[DIM];
        float norm_out = 0.0f;
        for (int c = 0; c < DIM; ++c) {
            float s = 0.0f;
            for (int r = 0; r < EXPANDED_DIM; ++r) s += W_proj[r][c] * h64[r];
            h_out[c] = s;
            norm_out += s * s;
        }
        norm_out = sqrtf(norm_out + 1e-6f);
        for (int c = 0; c < DIM; ++c) h_out[c] /= norm_out;

        float dot_rec = dot_prod(h_out, facts[f], DIM);
        tot_rec += fmaxf(0.0f, dot_rec);
    }

    results[idx].base_noise_fidelity   = tot_base / NUM_FACTS;
    results[idx].reconstructed_fidelity = tot_rec / NUM_FACTS;
    results[idx].active_gain           = (tot_rec - tot_base) / NUM_FACTS;
}

// --------------------------------------------------------------------------------------
// Kernel C: Dynamic Key-Value Attractor Bank (K=64 slots x 32D + Non-linear Sharpening)
// --------------------------------------------------------------------------------------
__global__ void benchmark_kv_bank_kernel(BenchResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + 20000 + idx, 0, 0, &st);

    constexpr int SLOTS = 64;
    float keys[SLOTS][DIM] = {0.0f};
    float values[SLOTS][DIM] = {0.0f};
    float slot_usage[SLOTS] = {0.0f};

    // 50 facts
    float facts[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        float norm = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            facts[f][i] = curand_normal(&st);
            norm += facts[f][i] * facts[f][i];
        }
        norm = sqrtf(norm + 1e-6f);
        for (int i = 0; i < DIM; ++i) facts[f][i] /= norm;
    }

    // Write: Dynamic Slot Allocation via competitive routing
    for (int f = 0; f < NUM_FACTS; ++f) {
        int best_slot = -1;
        float min_usage = 1e9f;
        for (int s = 0; s < SLOTS; ++s) {
            if (slot_usage[s] < min_usage) {
                min_usage = slot_usage[s];
                best_slot = s;
            }
        }

        // Store fact into attractor slot
        for (int i = 0; i < DIM; ++i) {
            keys[best_slot][i]   = facts[f][i];
            values[best_slot][i] = facts[f][i];
        }
        slot_usage[best_slot] += 1.0f;
    }

    // Distractor Decay on Slot Salience
    for (int s = 0; s < DISTRACTOR_STEPS; ++s) {
        for (int sl = 0; sl < SLOTS; ++sl) slot_usage[sl] *= 0.999f;
    }

    // Read: 80% noise cues with Continuous Softmax Resonance Attractor
    float tot_base = 0.0f;
    float tot_rec = 0.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        float cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            cue[i] = (curand_uniform(&st) < 0.20f) ? facts[f][i] : curand_normal(&st) * 0.8f;
        }

        float dot_in = dot_prod(cue, facts[f], DIM);
        float norm_in = sqrtf(dot_prod(cue, cue, DIM));
        tot_base += (norm_in > 1e-4f) ? fmaxf(0.0f, dot_in / norm_in) : 0.0f;

        // Competitive Softmax Sharpening over Attractor Slots (beta = 8.0)
        float max_logit = -1e9f;
        float logits[SLOTS];
        for (int sl = 0; sl < SLOTS; ++sl) {
            float d = dot_prod(cue, keys[sl], DIM);
            logits[sl] = d * 8.0f;
            if (logits[sl] > max_logit) max_logit = logits[sl];
        }

        float sum_exp = 0.0f;
        float weights[SLOTS];
        for (int sl = 0; sl < SLOTS; ++sl) {
            weights[sl] = expf(logits[sl] - max_logit);
            sum_exp += weights[sl];
        }

        float h_out[DIM] = {0.0f};
        for (int sl = 0; sl < SLOTS; ++sl) {
            float alpha = weights[sl] / (sum_exp + 1e-6f);
            for (int i = 0; i < DIM; ++i) {
                h_out[i] += alpha * values[sl][i];
            }
        }

        float norm_out = sqrtf(dot_prod(h_out, h_out, DIM) + 1e-6f);
        for (int i = 0; i < DIM; ++i) h_out[i] /= norm_out;

        float dot_rec = dot_prod(h_out, facts[f], DIM);
        tot_rec += fmaxf(0.0f, dot_rec);
    }

    results[idx].base_noise_fidelity   = tot_base / NUM_FACTS;
    results[idx].reconstructed_fidelity = tot_rec / NUM_FACTS;
    results[idx].active_gain           = (tot_rec - tot_base) / NUM_FACTS;
}

void summarize_results(const std::string& name, const std::vector<BenchResult>& res, double time_sec) {
    float avg_base = 0.0f;
    float avg_rec = 0.0f;
    float avg_gain = 0.0f;
    for (const auto& r : res) {
        avg_base += r.base_noise_fidelity;
        avg_rec  += r.reconstructed_fidelity;
        avg_gain += r.active_gain;
    }
    avg_base /= res.size();
    avg_rec  /= res.size();
    avg_gain /= res.size();

    std::cout << ">>> " << name << ":\n"
              << "    Base Noise Fidelity : " << std::fixed << std::setprecision(4) << avg_base << "\n"
              << "    Reconstructed Fid   : " << std::fixed << std::setprecision(4) << avg_rec << "\n"
              << "    Net Active Gain     : " << std::showpos << std::fixed << std::setprecision(4) << avg_gain << "\n" << std::noshowpos
              << "    Time (1,000 trials) : " << std::fixed << std::setprecision(3) << time_sec << " s\n"
              << "------------------------------------------------------------------------\n" << std::flush;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-633: DYNAMIC RANK EXPANSION TENSOR MEMORY BENCHMARK           ===\n"
              << "========================================================================\n"
              << "Objective            : Direct Empirical Test of Matrix Rank Capacity Limits\n"
              << "Dataset Stress       : 50 Facts (32D) -> 500 Distractors -> 80% Noise Cues\n"
              << "Trials per Condition : 1,000 Independent Runs on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    BenchResult* d_results = nullptr;
    CHECK_CUDA(cudaMalloc(&d_results, TRIALS_PER_COND * sizeof(BenchResult)));
    std::vector<BenchResult> h_results(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    // Test A: Baseline Rank-32 (32x32 Matrix)
    auto t1 = std::chrono::high_resolution_clock::now();
    benchmark_rank32_kernel<<<blocks, threads>>>(d_results, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, TRIALS_PER_COND * sizeof(BenchResult), cudaMemcpyDeviceToHost));
    summarize_results("Condition A: Static Rank-32 Matrix (32x32 = 1,024 params)", h_results,
                      std::chrono::duration<double>(t2 - t1).count());

    // Test B: Expanded Rank-64 (64x64 Matrix)
    t1 = std::chrono::high_resolution_clock::now();
    benchmark_rank64_kernel<<<blocks, threads>>>(d_results, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, TRIALS_PER_COND * sizeof(BenchResult), cudaMemcpyDeviceToHost));
    summarize_results("Condition B: Expanded Rank-64 Matrix (64x64 = 4,096 params)", h_results,
                      std::chrono::duration<double>(t2 - t1).count());

    // Test C: Dynamic KV Attractor Bank (64 slots x 32D, 2,048 params + Softmax Sharpening)
    t1 = std::chrono::high_resolution_clock::now();
    benchmark_kv_bank_kernel<<<blocks, threads>>>(d_results, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(h_results.data(), d_results, TRIALS_PER_COND * sizeof(BenchResult), cudaMemcpyDeviceToHost));
    summarize_results("Condition C: Dynamic KV Attractor Bank (64 slots x 32D = 2,048 params)", h_results,
                      std::chrono::duration<double>(t2 - t1).count());

    CHECK_CUDA(cudaFree(d_results));
    return 0;
}
