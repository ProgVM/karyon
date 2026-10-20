// ======================================================================================
// EXP-642: Phase Gravitational Attractor Fields (Vector Gravity Mechanics)
// --------------------------------------------------------------------------------------
// Objective & Cybernetic Rationale:
// Bazilevs' Vector Field Paradigm Hypothesis A:
// Vectors in cognitive phase space possess intrinsic "semantic mass/charge" m_i = ||v_i||^2
// and generate continuous gravitational potential wells:
//   Phi(x) = - \sum_i \frac{G \cdot m_i}{||x - c_i||^2 + \epsilon}
// A query trajectory x_t is acted upon by gravitational force vectors:
//   F_grav(x_t) = \sum_i \frac{G \cdot m_i \cdot (c_i - x_t)}{(||x_t - c_i||^2 + \epsilon)^{3/2}}
// This curves the trajectory toward relevant semantic attractors without explicit matrix projections!
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - K = 50 Attractor Basins in 64D space under extreme 85% noise floor.
// - 4 Architectural Conditions Evaluated:
//   Cond 1: Standard Cosine Similarity Matching (Flat Linear Projection)
//   Cond 2: Static Euclidean Distance Nearest Neighbor
//   Cond 3: Discrete Hopfield Basin Snapping
//   Cond 4: Continuous Vector Gravitational Field Trajectory Integration (Euler SDE)
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
constexpr int NUM_ATTRACTORS   = 50;       // K = 50 Gravitational Attractors
constexpr int INTEGRATION_STEPS= 10;       // 10 Continuous gravitational drift steps
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct GravResult {
    float fid_cond1_cosine;
    float fid_cond2_euclidean;
    float fid_cond3_hopfield;
    float fid_cond4_gravitational_field;
    float net_gravitational_gain;
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

__global__ void vector_gravity_kernel(GravResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 Attractor Centers c_i with random semantic masses m_i
    float centers[NUM_ATTRACTORS][DIM];
    float masses[NUM_ATTRACTORS];

    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        for (int i = 0; i < DIM; ++i) centers[k][i] = curand_normal(&st);
        normalize_vec(centers[k], DIM);
        masses[k] = 1.0f + 0.5f * curand_uniform(&st); // Semantic mass m_i \in [1.0, 1.5]
    }

    float tot_c1 = 0.0f, tot_c2 = 0.0f, tot_c3 = 0.0f, tot_c4 = 0.0f;
    const float G_const = 0.85f;
    const float eps = 0.05f;

    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        // Create initial query x_0 corrupting target center k under 85% noise
        float raw_cue[DIM];
        for (int i = 0; i < DIM; ++i) {
            raw_cue[i] = (curand_uniform(&st) < 0.15f) ? centers[k][i] : curand_normal(&st) * 0.85f;
        }
        normalize_vec(raw_cue, DIM);

        // -------------------------------------------------------------
        // Cond 1: Cosine Similarity
        // -------------------------------------------------------------
        {
            float max_sim = -1e9f;
            int best_idx = -1;
            for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                float sim = dot_prod(raw_cue, centers[j], DIM);
                if (sim > max_sim) { max_sim = sim; best_idx = j; }
            }
            tot_c1 += (best_idx == k) ? 1.0f : 0.0f;
        }

        // -------------------------------------------------------------
        // Cond 2: Static Euclidean Distance
        // -------------------------------------------------------------
        {
            float min_dist = 1e9f;
            int best_idx = -1;
            for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                float dist_sq = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    float diff = raw_cue[i] - centers[j][i];
                    dist_sq += diff * diff;
                }
                if (dist_sq < min_dist) { min_dist = dist_sq; best_idx = j; }
            }
            tot_c2 += (best_idx == k) ? 1.0f : 0.0f;
        }

        // -------------------------------------------------------------
        // Cond 3: Discrete Hopfield Snapping
        // -------------------------------------------------------------
        {
            float beta = 12.0f;
            float max_l = -1e9f;
            float l[NUM_ATTRACTORS];
            for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                l[j] = beta * dot_prod(raw_cue, centers[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_ATTRACTORS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * centers[j][i];
            }
            normalize_vec(h_out, DIM);

            float max_sim = -1e9f;
            int best_idx = -1;
            for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                float sim = dot_prod(h_out, centers[j], DIM);
                if (sim > max_sim) { max_sim = sim; best_idx = j; }
            }
            tot_c3 += (best_idx == k) ? 1.0f : 0.0f;
        }

        // -------------------------------------------------------------
        // Cond 4: Continuous Vector Gravitational Field Trajectory Drift
        // x_{t+1} = Normalize(x_t + \Delta t \cdot F_grav(x_t))
        // -------------------------------------------------------------
        {
            float x_t[DIM];
            for (int i = 0; i < DIM; ++i) x_t[i] = raw_cue[i];

            for (int step = 0; step < INTEGRATION_STEPS; ++step) {
                float force[DIM] = {0.0f};

                for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                    float r_sq = 0.0f;
                    for (int i = 0; i < DIM; ++i) {
                        float diff = centers[j][i] - x_t[i];
                        r_sq += diff * diff;
                    }
                    float denom = powf(r_sq + eps, 1.5f);
                    float factor = (G_const * masses[j]) / (denom + 1e-6f);

                    for (int i = 0; i < DIM; ++i) {
                        force[i] += factor * (centers[j][i] - x_t[i]);
                    }
                }

                // Euler SDE integration step
                for (int i = 0; i < DIM; ++i) {
                    x_t[i] += 0.10f * force[i];
                }
                normalize_vec(x_t, DIM);
            }

            float max_sim = -1e9f;
            int best_idx = -1;
            for (int j = 0; j < NUM_ATTRACTORS; ++j) {
                float sim = dot_prod(x_t, centers[j], DIM);
                if (sim > max_sim) { max_sim = sim; best_idx = j; }
            }
            tot_c4 += (best_idx == k) ? 1.0f : 0.0f;
        }
    }

    results[idx].fid_cond1_cosine          = tot_c1 / NUM_ATTRACTORS;
    results[idx].fid_cond2_euclidean       = tot_c2 / NUM_ATTRACTORS;
    results[idx].fid_cond3_hopfield        = tot_c3 / NUM_ATTRACTORS;
    results[idx].fid_cond4_gravitational_field = tot_c4 / NUM_ATTRACTORS;
    results[idx].net_gravitational_gain    = (tot_c4 - tot_c1) / NUM_ATTRACTORS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-642: PHASE GRAVITATIONAL ATTRACTOR FIELDS BENCHMARK           ===\n"
              << "========================================================================\n"
              << "Bazilevs' Hypothesis A: Vector Gravity Mechanics (F_grav = G*m*(c - x)/r^3)\n"
              << "Stress Benchmark     : K=50 Attractor Centers in 64D, 85% Noise Floor\n"
              << "Trials Evaluated     : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    GravResult* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(GravResult)));
    std::vector<GravResult> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    vector_gravity_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(GravResult), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.fid_cond1_cosine;
        c2   += r.fid_cond2_euclidean;
        c3   += r.fid_cond3_hopfield;
        c4   += r.fid_cond4_gravitational_field;
        gain += r.net_gravitational_gain;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION                 | RECALL ACCURACY | NET GAIN OVER BASE|\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Flat Cosine Similarity               | " << c1 << "          | +0.0000 (Baseline)|\n"
              << "| 2. Static Euclidean Distance            | " << c2 << "          | " << std::showpos << (c2 - c1) << std::noshowpos << "             |\n"
              << "| 3. Discrete Hopfield Snapping           | " << c3 << "          | " << std::showpos << (c3 - c1) << std::noshowpos << "             |\n"
              << "| 4. Continuous Vector Gravitational Drift| " << c4 << "          | " << std::showpos << gain << std::noshowpos << "             |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 attractors x 4 conditions): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
