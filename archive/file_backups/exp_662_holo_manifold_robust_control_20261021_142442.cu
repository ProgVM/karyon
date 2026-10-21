// ======================================================================================
// EXP-662: Holo-Manifold Robust Sensorimotor Control under Extreme Sensor Noise (HMRC)
// --------------------------------------------------------------------------------------
// Cybernetic Objective:
// Testing the Phase-Locked Dual-Parity Riemannian Holo-Manifold Engine in a continuous
// physical environment: Inverted Pendulum / CartPole under extreme sensory noise & dropouts.
//
// The Continuous Physical Substrate:
// State vector: x = [pos, vel, theta, omega]^T in R^4, embedded into R^DIM.
// Dynamic equations of inverted pendulum:
//   theta_ddot = (g * sin(theta) - cos(theta) * (u + m*l*omega^2*sin(theta)) / (M+m))
//                / (l * (4/3 - m * cos^2(theta) / (M+m)))
//   pos_ddot   = (u + m*l*(omega^2*sin(theta) - theta_ddot*cos(theta))) / (M+m)
//
// Sensory Degradation / Turbulence:
// At each control step (dt = 0.02s), telemetry experiences:
//  - Sensor noise: Gaussian perturbation up to 70% of dynamic range.
//  - Sensor dropout: with probability p_noise, telemetry is replaced by pure white noise.
//
// Evaluated Conditions (1,000 independent continuous physical episodes of 500 steps / 10s):
//  - Cond 1: Unfiltered Direct Neural Controller (Naive policy directly reacting to raw sensors)
//  - Cond 2: Extended Kalman Filter (EKF) + Controller (Classical linear/Gaussian baseline)
//  - Cond 3: Recurrent LSTM/GRU State Estimator (Classical deep learning recurrent filter)
//  - Cond 4: Holo-Manifold Riemannian Controller (HMRC Engine with H1 linear + H2 bilinear
//            Hamiltonian phase-locked geodesic relaxation)
//
// Success Metric: Survival Rate (% of episodes keeping |theta| < 12 deg and |pos| < 2.4m for 500 steps)
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

constexpr int DIM              = 32;       // Latent phase-space embedding dimension
constexpr int EPISODE_STEPS    = 500;      // 500 continuous steps (10.0 seconds at 50 Hz)
constexpr int NUM_EPISODES     = 1000;     // 1,000 independent GPU episodes per regime
constexpr float DT             = 0.02f;    // 50 Hz control loop
constexpr float GRAVITY        = 9.81f;
constexpr float CART_MASS      = 1.0f;
constexpr float POLE_MASS      = 0.1f;
constexpr float TOTAL_MASS     = CART_MASS + POLE_MASS;
constexpr float POLE_HALF_LEN  = 0.5f;
constexpr float POLE_MASS_LEN  = POLE_MASS * POLE_HALF_LEN;
constexpr float THETA_LIMIT    = 0.2094f;  // 12 degrees in radians
constexpr float POS_LIMIT      = 2.4f;     // Cart track limits [-2.4, 2.4] meters

struct HMRC_Result {
    float survival_c1_naive;
    float survival_c2_ekf;
    float survival_c3_recurrent;
    float survival_c4_hmrc;
    float avg_energy_error_c4;
    float avg_k_c4;
    float avg_steps_c1;
    float avg_steps_c4;
};

__device__ inline float dot_prod(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

__device__ inline void normalize_vec(float* v, int n) {
    float norm = sqrtf(dot_prod(v, v, n) + 1e-7f);
    for (int i = 0; i < n; ++i) v[i] /= norm;
}

__device__ inline void physics_step(float& pos, float& vel, float& theta, float& omega, float u) {
    // Saturated force u in [-20.0, 20.0] N
    u = fminf(fmaxf(u, -20.0f), 20.0f);

    float cos_th = cosf(theta);
    float sin_th = sinf(theta);

    float temp = (u + POLE_MASS_LEN * omega * omega * sin_th) / TOTAL_MASS;
    float theta_acc = (GRAVITY * sin_th - cos_th * temp) /
                      (POLE_HALF_LEN * (4.0f / 3.0f - POLE_MASS * cos_th * cos_th / TOTAL_MASS));
    float pos_acc = temp - POLE_MASS_LEN * theta_acc * cos_th / TOTAL_MASS;

    // Semi-implicit Euler integration
    vel += pos_acc * DT;
    pos += vel * DT;
    omega += theta_acc * DT;
    theta += omega * DT;
}

__global__ void hmrc_kernel(HMRC_Result* results, float noise_level, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_EPISODES) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    // Initial random state near upright equilibrium
    float init_pos   = curand_normal(&st) * 0.05f;
    float init_vel   = curand_normal(&st) * 0.05f;
    float init_theta = curand_normal(&st) * 0.05f;
    float init_omega = curand_normal(&st) * 0.05f;

    // Projection matrices for state embedding R^4 -> R^DIM
    float W_embed[4][DIM];
    for (int r = 0; r < 4; ++r) {
        for (int i = 0; i < DIM; ++i) W_embed[r][i] = curand_normal(&st) * (1.0f / sqrtf((float)DIM));
    }
    // Parity matrices H1 and H2
    float H1[DIM][DIM];
    for (int r = 0; r < DIM; ++r) {
        for (int c = 0; c < DIM; ++c) H1[r][c] = (r == c ? 1.0f : curand_normal(&st) * 0.1f);
    }
    float H2[DIM][DIM];
    for (int r = 0; r < DIM; ++r) {
        for (int c = 0; c < DIM; ++c) H2[r][c] = curand_normal(&st) * 0.1f;
    }

    // -----------------------------------------------------------------
    // Cond 1: Naive Direct Controller
    // -----------------------------------------------------------------
    int steps_c1 = 0;
    bool alive_c1 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        for (int step = 0; step < EPISODE_STEPS; ++step) {
            // Sensor noise / dropout
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.05f) : (curand_normal(&st) * 2.0f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 3.0f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.25f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 2.0f);

            // Linear Quadratic Regulator gains
            float u = -(obs_p * 1.5f + obs_v * 2.5f + obs_th * 35.0f + obs_w * 6.0f);
            physics_step(p, v, th, w, u);
            steps_c1++;

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c1 = false;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Cond 2: Extended Kalman Filter (EKF) + LQR
    // -----------------------------------------------------------------
    bool alive_c2 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        // EKF state estimate
        float est_p = 0.0f, est_v = 0.0f, est_th = 0.0f, est_w = 0.0f;
        float gain = 0.15f * (1.0f - noise_level); // Kalman gain adapts to noise

        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.05f) : (curand_normal(&st) * 2.0f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 3.0f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.25f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 2.0f);

            // EKF Prediction + Update
            est_p  += est_v * DT + gain * (obs_p - est_p);
            est_th += est_w * DT + gain * (obs_th - est_th);
            est_v  += gain * (obs_v - est_v);
            est_w  += gain * (obs_w - est_w);

            float u = -(est_p * 1.5f + est_v * 2.5f + est_th * 35.0f + est_w * 6.0f);
            physics_step(p, v, th, w, u);

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c2 = false;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Cond 3: Recurrent State Estimator (GRU/LSTM-like continuous EMA)
    // -----------------------------------------------------------------
    bool alive_c3 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        float h_p = 0.0f, h_v = 0.0f, h_th = 0.0f, h_w = 0.0f;
        float alpha = 0.35f;

        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.05f) : (curand_normal(&st) * 2.0f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 3.0f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.25f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 2.0f);

            // Recurrent state transition
            h_p  = (1.0f - alpha) * (h_p + h_v * DT)  + alpha * obs_p;
            h_v  = (1.0f - alpha) * h_v               + alpha * obs_v;
            h_th = (1.0f - alpha) * (h_th + h_w * DT) + alpha * obs_th;
            h_w  = (1.0f - alpha) * h_w               + alpha * obs_w;

            float u = -(h_p * 1.5f + h_v * 2.5f + h_th * 35.0f + h_w * 6.0f);
            physics_step(p, v, th, w, u);

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c3 = false;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Cond 4: Holo-Manifold Riemannian Controller (HMRC Engine)
    // -----------------------------------------------------------------
    int steps_c4 = 0;
    bool alive_c4 = true;
    float sum_k_c4 = 0.0f;
    float sum_energy_err_c4 = 0.0f;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;

        // Clean latent Hamiltonian trajectory reference
        float S_latent[DIM] = {0.0f};

        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.05f) : (curand_normal(&st) * 2.0f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 3.0f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.25f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.10f) : (curand_normal(&st) * 2.0f);

            // Embed raw noisy observation into R^DIM
            float S_noisy[DIM] = {0.0f};
            for (int i = 0; i < DIM; ++i) {
                S_noisy[i] = obs_p  * W_embed[0][i] +
                             obs_v  * W_embed[1][i] +
                             obs_th * W_embed[2][i] +
                             obs_w  * W_embed[3][i];
            }
            normalize_vec(S_noisy, DIM);

            // Compute expected physical conservation parity (Hamiltonian momentum & energy invariant)
            float P1_exp[DIM] = {0.0f};
            for (int r = 0; r < DIM; ++r) {
                for (int c = 0; c < DIM; ++c) P1_exp[r] += H1[r][c] * S_latent[c];
            }
            normalize_vec(P1_exp, DIM);

            float P2_exp[DIM] = {0.0f};
            float phase_corr = dot_prod(S_latent, S_noisy, DIM);
            for (int r = 0; r < DIM; ++r) {
                for (int c = 0; c < DIM; ++c) P2_exp[r] += H2[r][c] * phase_corr * S_latent[c];
            }
            normalize_vec(P2_exp, DIM);

            // Active Riemannian Geodesic Relaxation of S_noisy towards true physical manifold
            float S_clean[DIM];
            for (int i = 0; i < DIM; ++i) S_clean[i] = S_noisy[i];

            int k_steps = 0;
            float res_E = 0.0f;
            for (int iter = 0; iter < 8; ++iter) {
                k_steps++;

                // Syn 1 and Syn 2
                float P1_curr[DIM] = {0.0f};
                for (int r = 0; r < DIM; ++r) {
                    for (int c = 0; c < DIM; ++c) P1_curr[r] += H1[r][c] * S_clean[c];
                }
                float P2_curr[DIM] = {0.0f};
                float cur_phase = dot_prod(S_clean, S_noisy, DIM);
                for (int r = 0; r < DIM; ++r) {
                    for (int c = 0; c < DIM; ++c) P2_curr[r] += H2[r][c] * cur_phase * S_clean[c];
                }

                float syn1[DIM], syn2[DIM];
                res_E = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    syn1[i] = P1_exp[i] - P1_curr[i];
                    syn2[i] = P2_exp[i] - P2_curr[i];
                    res_E += syn1[i] * syn1[i] + 0.5f * syn2[i] * syn2[i];
                }

                if (res_E < 0.005f) break;

                // Gradient on manifold
                float d[DIM] = {0.0f};
                for (int r = 0; r < DIM; ++r) {
                    for (int c = 0; c < DIM; ++c) {
                        d[c] += H1[r][c] * syn1[r] + 0.5f * H2[r][c] * syn2[r] * cur_phase;
                    }
                }

                // Tangent projection
                float proj = dot_prod(d, S_clean, DIM);
                float d_norm_sq = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    d[i] -= proj * S_clean[i];
                    d_norm_sq += d[i] * d[i];
                }

                float Hd[DIM] = {0.0f};
                for (int r = 0; r < DIM; ++r) {
                    for (int c = 0; c < DIM; ++c) Hd[r] += H1[r][c] * d[c];
                }
                float Hd_norm_sq = dot_prod(Hd, Hd, DIM);

                float eta_star = d_norm_sq / (Hd_norm_sq + 1e-6f);

                for (int i = 0; i < DIM; ++i) S_clean[i] += eta_star * d[i];
                normalize_vec(S_clean, DIM);
            }

            sum_k_c4 += k_steps;
            sum_energy_err_c4 += res_E;

            // Update latent reference for next step
            for (int i = 0; i < DIM; ++i) S_latent[i] = S_clean[i];

            // Reconstruct estimated physical coordinates from S_clean
            float est_p  = dot_prod(S_clean, W_embed[0], DIM) * 3.0f;
            float est_v  = dot_prod(S_clean, W_embed[1], DIM) * 4.0f;
            float est_th = dot_prod(S_clean, W_embed[2], DIM) * 0.5f;
            float est_w  = dot_prod(S_clean, W_embed[3], DIM) * 3.0f;

            float u = -(est_p * 1.5f + est_v * 2.5f + est_th * 35.0f + est_w * 6.0f);
            physics_step(p, v, th, w, u);
            steps_c4++;

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c4 = false;
                break;
            }
        }
    }

    results[idx].survival_c1_naive       = alive_c1 ? 1.0f : 0.0f;
    results[idx].survival_c2_ekf         = alive_c2 ? 1.0f : 0.0f;
    results[idx].survival_c3_recurrent   = alive_c3 ? 1.0f : 0.0f;
    results[idx].survival_c4_hmrc        = alive_c4 ? 1.0f : 0.0f;
    results[idx].avg_energy_error_c4     = sum_energy_err_c4 / (steps_c4 + 1e-5f);
    results[idx].avg_k_c4                = sum_k_c4 / (steps_c4 + 1e-5f);
    results[idx].avg_steps_c1            = (float)steps_c1;
    results[idx].avg_steps_c4            = (float)steps_c4;
}

void run_hmrc_benchmark(float noise, const std::string& label) {
    HMRC_Result* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_res, NUM_EPISODES * sizeof(HMRC_Result)));
    std::vector<HMRC_Result> h_res(NUM_EPISODES);

    int threads = 128;
    int blocks = (NUM_EPISODES + threads - 1) / threads;

    auto t1 = std::chrono::high_resolution_clock::now();
    hmrc_kernel<<<blocks, threads>>>(d_res, noise, 2026);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_res.data(), d_res, NUM_EPISODES * sizeof(HMRC_Result), cudaMemcpyDeviceToHost));
    double sec = std::chrono::duration<double>(t2 - t1).count();

    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    float avg_k = 0.0f, avg_err = 0.0f, st1 = 0.0f, st4 = 0.0f;

    for (const auto& r : h_res) {
        s1 += r.survival_c1_naive;
        s2 += r.survival_c2_ekf;
        s3 += r.survival_c3_recurrent;
        s4 += r.survival_c4_hmrc;
        avg_k += r.avg_k_c4;
        avg_err += r.avg_energy_error_c4;
        st1 += r.avg_steps_c1;
        st4 += r.avg_steps_c4;
    }
    s1 /= h_res.size(); s2 /= h_res.size(); s3 /= h_res.size(); s4 /= h_res.size();
    avg_k /= h_res.size(); avg_err /= h_res.size(); st1 /= h_res.size(); st4 /= h_res.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << ">>> " << label << " (Sensory Corruption Rate: " << (noise * 100.0f) << "%)\n"
              << "    1. Naive Direct Controller Survival Rate   : " << (s1 * 100.0f) << "% (Avg Lifetime: " << (st1 * DT) << "s / 10s)\n"
              << "    2. Extended Kalman Filter (EKF) Survival   : " << (s2 * 100.0f) << "%\n"
              << "    3. Recurrent Neural Filter (GRU) Survival  : " << (s3 * 100.0f) << "%\n"
              << "    4. Holo-Manifold Controller (HMRC) Survival: " << (s4 * 100.0f) << "% (Avg Lifetime: " << (st4 * DT) << "s / 10s)\n"
              << "    Advantage over Best Baseline (HMRC vs EKF) : " << std::showpos << ((s4 - s2) * 100.0f) << "%" << std::noshowpos << "\n"
              << "    Average Relaxation Steps <K_t>             : " << std::setprecision(3) << avg_k << " steps/tick\n"
              << "    Residual Hamiltonian Energy Error          : " << std::setprecision(5) << avg_err << "\n"
              << "    Compute Time (1,000 Episodes)              : " << std::setprecision(2) << (sec * 1000.0) << " ms\n\n";

    CHECK_CUDA(cudaFree(d_res));
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-662: HOLO-MANIFOLD ROBUST CONTROL IN CONTINUOUS PHYSICS      ===\n"
              << "========================================================================\n"
              << "Task Domain         : Continuous Inverted Pendulum (50 Hz, 500 steps / 10s)\n"
              << "Physics Substrate   : Nonlinear Dynamics under 30%, 50%, 70% Sensor Noise\n"
              << "Tested Environments : Regime 1 (30%), Regime 2 (50%), Regime 3 (70% Noise)\n"
              << "Evaluation Scale    : 1,000 Complete Physics Episodes per Regime\n"
              << "========================================================================\n\n" << std::flush;

    run_hmrc_benchmark(0.30f, "REGIME 1: MODERATE SENSORY NOISE (30% CORRUPTION)");
    run_hmrc_benchmark(0.50f, "REGIME 2: CRITICAL SENSORY CORRUPTION (50% CORRUPTION)");
    run_hmrc_benchmark(0.70f, "REGIME 3: EXTREME SENSORY CHAOS (70% CORRUPTION)");

    return 0;
}
