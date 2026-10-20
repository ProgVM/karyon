// ======================================================================================
// EXP-643: Topological Vector Spin & Phase Rotation Dynamics (Vector Spin Mechanics)
// --------------------------------------------------------------------------------------
// Objective & Cybernetic Rationale:
// Bazilevs' Vector Field Paradigm Hypothesis B:
// Vectors in cognitive phase space possess intrinsic "Topological Spin / Phase Rotation Angle"
// theta_i \in [0, 2\pi) and Spin Frequency omega_i.
// When two vectors v_1, v_2 interact, their binding is governed by Lie Algebra Phase Coupling:
//   Binding(v_1, v_2) = R(\theta_1 - \theta_2) (v_1 \odot v_2)
// where R(\theta) is a continuous SO(2) 2D plane rotation matrix:
//   R(\theta) = [ cos(\theta)  -sin(\theta) ]
//               [ sin(\theta)   cos(\theta) ]
//
// Key Advantages Evaluated:
// 1. Orthogonal Role-Content Disambiguation:
//    Different spins allow the SAME semantic vector (e.g. "Cat") to act as Subject (\theta = 0),
//    Object (\theta = \pi/2), or Instrument (\theta = \pi) without destructive interference!
// 2. High-Dimensional Phase Resonance:
//    Vectors with aligned spins (\Delta \theta \to 0) constructively interfere, while
//    out-of-phase vectors (\Delta \theta \to \pi) destructive cancel out noise!
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - K = 50 Role-Fact Triplets (Subject, Action, Object) in 64D under 85% noise floor.
// - 4 Architectural Conditions Evaluated:
//   Cond 1: Flat Unrotated Vector Concatenation
//   Cond 2: Fixed Positional Embeddings (Static Addition)
//   Cond 3: Discrete Circular Shift (Permutation)
//   Cond 4: Continuous Lie-Algebraic SO(2) Topological Vector Spin Rotation R(\theta)
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
constexpr int NUM_FACTS        = 50;       // K = 50 Role-Fact Triplets
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct SpinResult {
    float fid_cond1_flat;
    float fid_cond2_pos_add;
    float fid_cond3_permute;
    float fid_cond4_vector_spin;
    float net_spin_gain;
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

// Continuous SO(2) 2D Plane Rotation on pairs of dimensions
__device__ inline void apply_vector_spin(const float* in, float* out, float theta, int dim) {
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    for (int i = 0; i < dim; i += 2) {
        float x = in[i];
        float y = in[i + 1];
        out[i]     = x * cos_t - y * sin_t;
        out[i + 1] = x * sin_t + y * cos_t;
    }
}

__global__ void vector_spin_kernel(SpinResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 Role-Fact Triplets: Subject S_f, Action A_f, Object O_f
    float subj[NUM_FACTS][DIM];
    float act[NUM_FACTS][DIM];
    float obj[NUM_FACTS][DIM];

    for (int f = 0; f < NUM_FACTS; ++f) {
        for (int i = 0; i < DIM; ++i) {
            subj[f][i] = curand_normal(&st);
            act[f][i]  = curand_normal(&st);
            obj[f][i]  = curand_normal(&st);
        }
        normalize_vec(subj[f], DIM);
        normalize_vec(act[f], DIM);
        normalize_vec(obj[f], DIM);
    }

    // Role Spin Angles: Subject \theta = 0, Action \theta = 2\pi/3, Object \theta = 4\pi/3
    const float spin_S = 0.0f;
    const float spin_A = 2.0f * 3.14159265f / 3.0f;
    const float spin_O = 4.0f * 3.14159265f / 3.0f;

    // Build Spin-Rotated Composite Memory Binding
    float spun_memory[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        float rot_S[DIM], rot_A[DIM], rot_O[DIM];
        apply_vector_spin(subj[f], rot_S, spin_S, DIM);
        apply_vector_spin(act[f],  rot_A, spin_A, DIM);
        apply_vector_spin(obj[f],  rot_O, spin_O, DIM);

        for (int i = 0; i < DIM; ++i) {
            spun_memory[f][i] = rot_S[i] + rot_A[i] + rot_O[i];
        }
        normalize_vec(spun_memory[f], DIM);
    }

    float tot_c1 = 0.0f, tot_c2 = 0.0f, tot_c3 = 0.0f, tot_c4 = 0.0f;
    const float beta = 12.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        // Query: "What is the Object of Action A_f under 85% noise?"
        // Target: Retrieve O_f using Action A_f rotated at spin_A
        float noisy_act_cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            noisy_act_cue[i] = (curand_uniform(&st) < 0.15f) ? act[f][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(noisy_act_cue, DIM);

        // -------------------------------------------------------------
        // Cond 1: Flat Unrotated Vector Matching
        // -------------------------------------------------------------
        {
            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta * dot_prod(noisy_act_cue, act[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * obj[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c1 += fmaxf(0.0f, dot_prod(h_out, obj[f], DIM));
        }

        // -------------------------------------------------------------
        // Cond 2: Fixed Positional Embedding Addition
        // -------------------------------------------------------------
        {
            float pos_cue[DIM];
            for (int i = 0; i < DIM; ++i) pos_cue[i] = noisy_act_cue[i] + 0.5f * sinf(i * 0.5f);
            normalize_vec(pos_cue, DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                float pos_act[DIM];
                for (int i = 0; i < DIM; ++i) pos_act[i] = act[j][i] + 0.5f * sinf(i * 0.5f);
                normalize_vec(pos_act, DIM);
                l[j] = beta * dot_prod(pos_cue, pos_act, DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * obj[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c2 += fmaxf(0.0f, dot_prod(h_out, obj[f], DIM));
        }

        // -------------------------------------------------------------
        // Cond 3: Discrete Circular Shift
        // -------------------------------------------------------------
        {
            float shift_cue[DIM];
            for (int i = 0; i < DIM; ++i) shift_cue[i] = noisy_act_cue[(i + 4) % DIM];
            normalize_vec(shift_cue, DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                float shift_act[DIM];
                for (int i = 0; i < DIM; ++i) shift_act[i] = act[j][i][(i + 4) % DIM];
                normalize_vec(shift_act, DIM);
                l[j] = beta * dot_prod(shift_cue, shift_act, DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * obj[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c3 += fmaxf(0.0f, dot_prod(h_out, obj[f], DIM));
        }

        // -------------------------------------------------------------
        // Cond 4: Topological Lie SO(2) Vector Spin Rotation
        // Query rotated at spin_A resonance directly queries spun_memory
        // -------------------------------------------------------------
        {
            float spun_cue[DIM];
            apply_vector_spin(noisy_act_cue, spun_cue, spin_A, DIM);
            normalize_vec(spun_cue, DIM);

            float max_l = -1e9f;
            float l[NUM_FACTS];
            for (int j = 0; j < NUM_FACTS; ++j) {
                l[j] = beta * dot_prod(spun_cue, spun_memory[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_FACTS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_FACTS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * obj[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c4 += fmaxf(0.0f, dot_prod(h_out, obj[f], DIM));
        }
    }

    results[idx].fid_cond1_flat        = tot_c1 / NUM_FACTS;
    results[idx].fid_cond2_pos_add      = tot_c2 / NUM_FACTS;
    results[idx].fid_cond3_permute      = tot_c3 / NUM_FACTS;
    results[idx].fid_cond4_vector_spin  = tot_c4 / NUM_FACTS;
    results[idx].net_spin_gain          = (tot_c4 - tot_c1) / NUM_FACTS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-643: TOPOLOGICAL VECTOR SPIN & PHASE ROTATION BENCHMARK        ===\n"
              << "========================================================================\n"
              << "Bazilevs' Hypothesis B: Lie SO(2) Vector Spin Mechanics (R(theta) * v)\n"
              << "Mechanism            : Continuous phase rotation enables orthogonal role binding\n"
              << "Stress Benchmark     : K=50 Role-Fact Triplets, 85% Extreme Noise Floor\n"
              << "Trials Evaluated     : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    SpinResult* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(SpinResult)));
    std::vector<SpinResult> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    vector_spin_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(SpinResult), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.fid_cond1_flat;
        c2   += r.fid_cond2_pos_add;
        c3   += r.fid_cond3_permute;
        c4   += r.fid_cond4_vector_spin;
        gain += r.net_spin_gain;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION                 | RECALL FIDELITY | NET GAIN OVER BASE|\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Flat Unrotated Vector Matching       | " << c1 << "          | +0.0000 (Baseline)|\n"
              << "| 2. Fixed Positional Embedding Addition  | " << c2 << "          | " << std::showpos << (c2 - c1) << std::noshowpos << "             |\n"
              << "| 3. Discrete Circular Permutation        | " << c3 << "          | " << std::showpos << (c3 - c1) << std::noshowpos << "             |\n"
              << "| 4. Topological Lie SO(2) Vector Spin    | " << c4 << "          | " << std::showpos << gain << std::noshowpos << "             |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 facts x 4 conditions): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
