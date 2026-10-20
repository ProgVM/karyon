// ======================================================================================
// EXP-644: Vector Fluid Dynamics & Laminar Flow Mechanics (Vector Hydrodynamics)
// --------------------------------------------------------------------------------------
// Objective & Cybernetic Rationale:
// Bazilevs' Vector Field Paradigm Hypothesis C:
// Information propagation in cognitive phase space is modeled as a continuous Vector Fluid
// governed by Navier-Stokes style Hydrodynamics:
//   \frac{\partial v}{\partial t} + (v \cdot \nabla) v = -\nabla p + \nu \nabla^2 v + f_{ext}
//
// Key Hydrodynamic Operators Evaluated:
// 1. Laminar Flow Smoothing (\nu \nabla^2 v): Viscous dissipation damps high-frequency noise spikes.
// 2. Vorticity / Circulation (\nabla \times v): Preserves rotational memory vortices without decay.
// 3. Pressure Gradient (-\nabla p): Prevents state density explosion, maintaining unit volume.
//
// Benchmark Protocol on Tesla T4 GPU (1,000 independent trials):
// - K = 50 Continuous State Streams in 64D under 85% extreme noise floor.
// - 4 Architectural Conditions Evaluated:
//   Cond 1: Discrete Euler State Steps (Standard Recurrent Baseline)
//   Cond 2: Exponential Moving Average (EMA Decay)
//   Cond 3: Pure Viscous Diffusion Damping (\nu \nabla^2 v)
//   Cond 4: Full Vector Hydrodynamics (Laminar Viscous Smoothing + Pressure Gradient + Vorticity)
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
constexpr int NUM_STREAMS      = 50;       // K = 50 Continuous State Streams
constexpr int TIME_STEPS       = 20;       // 20 Hydrodynamic Fluid Flow Steps
constexpr int TRIALS_PER_COND  = 1000;     // 1,000 GPU trials

struct FluidResult {
    float fid_cond1_discrete_euler;
    float fid_cond2_ema_decay;
    float fid_cond3_viscous_diffusion;
    float fid_cond4_vector_hydrodynamics;
    float net_hydrodynamic_gain;
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

__global__ void vector_fluid_kernel(FluidResult* results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TRIALS_PER_COND) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // 1. Generate 50 Target Hydrodynamic Fluid Trajectories (Clean Signal Streams)
    float targets[NUM_STREAMS][DIM];
    for (int k = 0; k < NUM_STREAMS; ++k) {
        for (int i = 0; i < DIM; ++i) targets[k][i] = curand_normal(&st);
        normalize_vec(targets[k], DIM);
    }

    float tot_c1 = 0.0f, tot_c2 = 0.0f, tot_c3 = 0.0f, tot_c4 = 0.0f;
    const float beta = 12.0f;
    const float nu_viscosity = 0.15f; // Viscous diffusion coefficient

    for (int k = 0; k < NUM_STREAMS; ++k) {
        // Create 20 time steps of noisy sensory flow (85% noise floor)
        float noisy_flow[TIME_STEPS][DIM];
        for (int t = 0; t < TIME_STEPS; ++t) {
            for (int i = 0; i < DIM; ++i) {
                noisy_flow[t][i] = (curand_uniform(&st) < 0.15f) ? targets[k][i] : curand_normal(&st) * 0.85f;
            }
            normalize_vec(noisy_flow[t], DIM);
        }

        // -------------------------------------------------------------
        // Cond 1: Discrete Euler State Steps (Discrete RNN Baseline)
        // -------------------------------------------------------------
        {
            float state[DIM] = {0.0f};
            for (int t = 0; t < TIME_STEPS; ++t) {
                for (int i = 0; i < DIM; ++i) state[i] += 0.20f * noisy_flow[t][i];
            }
            normalize_vec(state, DIM);

            float max_l = -1e9f;
            float l[NUM_STREAMS];
            for (int j = 0; j < NUM_STREAMS; ++j) {
                l[j] = beta * dot_prod(state, targets[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_STREAMS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_STREAMS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * targets[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c1 += fmaxf(0.0f, dot_prod(h_out, targets[k], DIM));
        }

        // -------------------------------------------------------------
        // Cond 2: Exponential Moving Average (EMA Decay)
        // -------------------------------------------------------------
        {
            float state[DIM] = {0.0f};
            float alpha = 0.25f;
            for (int t = 0; t < TIME_STEPS; ++t) {
                for (int i = 0; i < DIM; ++i) state[i] = (1.0f - alpha) * state[i] + alpha * noisy_flow[t][i];
            }
            normalize_vec(state, DIM);

            float max_l = -1e9f;
            float l[NUM_STREAMS];
            for (int j = 0; j < NUM_STREAMS; ++j) {
                l[j] = beta * dot_prod(state, targets[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_STREAMS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_STREAMS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * targets[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c2 += fmaxf(0.0f, dot_prod(h_out, targets[k], DIM));
        }

        // -------------------------------------------------------------
        // Cond 3: Pure Viscous Diffusion Damping (\nu \nabla^2 v)
        // -------------------------------------------------------------
        {
            float state[DIM] = {0.0f};
            for (int t = 0; t < TIME_STEPS; ++t) {
                float laplacian[DIM];
                for (int i = 0; i < DIM; ++i) {
                    float left  = state[(i - 1 + DIM) % DIM];
                    float right = state[(i + 1) % DIM];
                    laplacian[i] = left - 2.0f * state[i] + right; // Discrete 1D Laplacian \nabla^2 v
                }
                for (int i = 0; i < DIM; ++i) {
                    state[i] += 0.20f * noisy_flow[t][i] + nu_viscosity * laplacian[i];
                }
            }
            normalize_vec(state, DIM);

            float max_l = -1e9f;
            float l[NUM_STREAMS];
            for (int j = 0; j < NUM_STREAMS; ++j) {
                l[j] = beta * dot_prod(state, targets[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_STREAMS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_STREAMS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * targets[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c3 += fmaxf(0.0f, dot_prod(h_out, targets[k], DIM));
        }

        // -------------------------------------------------------------
        // Cond 4: Full Vector Hydrodynamics (Laminar Viscous Diffusion + Pressure Gradient + Vorticity)
        // dv/dt = -\nabla p + \nu \nabla^2 v + \omega \times v + f_ext
        // -------------------------------------------------------------
        {
            float velocity[DIM] = {0.0f};
            float pressure[DIM] = {0.0f};

            for (int t = 0; t < TIME_STEPS; ++t) {
                // 1. Compute Pressure Gradient \nabla p
                float grad_p[DIM];
                for (int i = 0; i < DIM; ++i) {
                    float right = noisy_flow[t][(i + 1) % DIM];
                    float left  = noisy_flow[t][(i - 1 + DIM) % DIM];
                    grad_p[i]   = 0.5f * (right - left);
                }

                // 2. Compute Viscous Laplacian \nabla^2 v
                float laplacian[DIM];
                for (int i = 0; i < DIM; ++i) {
                    float left  = velocity[(i - 1 + DIM) % DIM];
                    float right = velocity[(i + 1) % DIM];
                    laplacian[i] = left - 2.0f * velocity[i] + right;
                }

                // 3. Compute Vorticity / Circulation term \omega \times v
                float vorticity[DIM];
                for (int i = 0; i < DIM; ++i) {
                    vorticity[i] = velocity[(i + 2) % DIM] - velocity[(i - 2 + DIM) % DIM];
                }

                // Integrate Navier-Stokes Fluid Update Step
                for (int i = 0; i < DIM; ++i) {
                    float dv = -0.10f * grad_p[i] + nu_viscosity * laplacian[i] + 0.05f * vorticity[i] + 0.25f * noisy_flow[t][i];
                    velocity[i] += dv;
                }
                normalize_vec(velocity, DIM);
            }

            float max_l = -1e9f;
            float l[NUM_STREAMS];
            for (int j = 0; j < NUM_STREAMS; ++j) {
                l[j] = beta * dot_prod(velocity, targets[j], DIM);
                if (l[j] > max_l) max_l = l[j];
            }
            float sum_e = 0.0f;
            for (int j = 0; j < NUM_STREAMS; ++j) sum_e += expf(l[j] - max_l);
            float h_out[DIM] = {0.0f};
            for (int j = 0; j < NUM_STREAMS; ++j) {
                float a = expf(l[j] - max_l) / (sum_e + 1e-6f);
                for (int i = 0; i < DIM; ++i) h_out[i] += a * targets[j][i];
            }
            normalize_vec(h_out, DIM);
            tot_c4 += fmaxf(0.0f, dot_prod(h_out, targets[k], DIM));
        }
    }

    results[idx].fid_cond1_discrete_euler     = tot_c1 / NUM_STREAMS;
    results[idx].fid_cond2_ema_decay          = tot_c2 / NUM_STREAMS;
    results[idx].fid_cond3_viscous_diffusion  = tot_c3 / NUM_STREAMS;
    results[idx].fid_cond4_vector_hydrodynamics = tot_c4 / NUM_STREAMS;
    results[idx].net_hydrodynamic_gain        = (tot_c4 - tot_c1) / NUM_STREAMS;
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-644: VECTOR FLUID DYNAMICS & LAMINAR FLOW BENCHMARK             ===\n"
              << "========================================================================\n"
              << "Bazilevs' Hypothesis C: Vector Hydrodynamics (Navier-Stokes Fluid Field)\n"
              << "Mechanisms           : 1) Viscous Damping (Laplacian smooths high-freq noise)\n"
              << "                       2) Pressure Gradient (-grad P prevents saturation)\n"
              << "                       3) Vorticity Conservation (rotational memory preservation)\n"
              << "Stress Benchmark     : K=50 State Streams, 85% Noise Floor, 20 Flow Steps\n"
              << "Trials Evaluated     : 1,000 Independent GPU Trials on Tesla T4\n"
              << "========================================================================\n\n" << std::flush;

    FluidResult* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, TRIALS_PER_COND * sizeof(FluidResult)));
    std::vector<FluidResult> h_res(TRIALS_PER_COND);

    int threads = 128;
    int blocks = (TRIALS_PER_COND + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    vector_fluid_kernel<<<blocks, threads>>>(d_res, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, TRIALS_PER_COND * sizeof(FluidResult), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, gain = 0.0f;
    for (const auto& r : h_res) {
        c1   += r.fid_cond1_discrete_euler;
        c2   += r.fid_cond2_ema_decay;
        c3   += r.fid_cond3_viscous_diffusion;
        c4   += r.fid_cond4_vector_hydrodynamics;
        gain += r.net_hydrodynamic_gain;
    }
    c1 /= h_res.size(); c2 /= h_res.size(); c3 /= h_res.size(); c4 /= h_res.size(); gain /= h_res.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "---------------------------------------------------------------------------------\n"
              << "| ARCHITECTURAL CONDITION                 | RECALL FIDELITY | NET GAIN OVER BASE|\n"
              << "---------------------------------------------------------------------------------\n"
              << "| 1. Discrete Euler RNN Steps             | " << c1 << "          | +0.0000 (Baseline)|\n"
              << "| 2. Exponential Moving Average (EMA)     | " << c2 << "          | " << std::showpos << (c2 - c1) << std::noshowpos << "             |\n"
              << "| 3. Pure Viscous Diffusion (\nu \nabla^2 v)| " << c3 << "          | " << std::showpos << (c3 - c1) << std::noshowpos << "             |\n"
              << "| 4. Full Vector Hydrodynamics (Navier)   | " << c4 << "          | " << std::showpos << gain << std::noshowpos << "             |\n"
              << "---------------------------------------------------------------------------------\n\n"
              << "Total Compute Time (1,000 trials x 50 streams x 4 conditions): " << sec << " s\n"
              << "Throughput: " << (TRIALS_PER_COND / sec) << " trials/s\n" << std::flush;

    CHECK_CUDA(cudaFree(d_res));
    return 0;
}
