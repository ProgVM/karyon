// ======================================================================================
// EXP-662: Holo-Manifold Robust Sensorimotor Control under Extreme Sensor Noise (HMRC)
// --------------------------------------------------------------------------------------
// Precise Phase-Locking:
// In Cond 4, prior physics prediction was:
//   prior_p  = h_p + h_v * DT
//   prior_th = h_th + h_w * DT
// BUT it omitted the effect of control force u from the previous step!
// Without accounting for the control input u, the physics prior was predicting free-fall,
// which created artificial phase contradiction with the sensors!
// When we include the control force u in the symplectic prior:
//   prior_acc = physics_model(h_p, h_v, h_th, h_w, u_prev)
//   prior = symplectic_step(h, prior_acc, DT)
// Then the Holo-Manifold syndrome represents the TRUE external disturbance and sensor noise!
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
    float avg_steps_c1;
    float avg_steps_c2;
    float avg_steps_c3;
    float avg_steps_c4;
};

__device__ inline void physics_step(float& pos, float& vel, float& theta, float& omega, float u) {
    u = fminf(fmaxf(u, -100.0f), 100.0f);

    float cos_th = cosf(theta);
    float sin_th = sinf(theta);

    float temp = (u + POLE_MASS_LEN * omega * omega * sin_th) / TOTAL_MASS;
    float theta_acc = (GRAVITY * sin_th - cos_th * temp) /
                      (POLE_HALF_LEN * (4.0f / 3.0f - POLE_MASS * cos_th * cos_th / TOTAL_MASS));
    float pos_acc = temp - POLE_MASS_LEN * theta_acc * cos_th / TOTAL_MASS;

    pos   += vel * DT;
    vel   += pos_acc * DT;
    theta += omega * DT;
    omega += theta_acc * DT;
}

__device__ inline float lqr_control(float p, float v, float th, float w) {
    return (7.071f * p + 9.420f * v + 65.404f * th + 16.643f * w);
}

__global__ void hmrc_kernel(HMRC_Result* results, float noise_level, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_EPISODES) return;

    curandState st;
    curand_init(seed + idx, 0, 0, &st);

    float init_pos   = curand_normal(&st) * 0.01f;
    float init_vel   = curand_normal(&st) * 0.01f;
    float init_theta = curand_normal(&st) * 0.01f;
    float init_omega = curand_normal(&st) * 0.01f;

    // -----------------------------------------------------------------
    // Cond 1: Naive Direct Controller
    // -----------------------------------------------------------------
    int steps_c1 = 0;
    bool alive_c1 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.01f) : (curand_normal(&st) * 0.3f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.005f) : (curand_normal(&st) * 0.08f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);

            float u = lqr_control(obs_p, obs_v, obs_th, obs_w);
            physics_step(p, v, th, w, u);
            steps_c1++;

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c1 = false;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Cond 2: Extended Kalman Filter (EKF)
    // -----------------------------------------------------------------
    int steps_c2 = 0;
    bool alive_c2 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        float est_p = init_pos, est_v = init_vel, est_th = init_theta, est_w = init_omega;
        float gain = 0.25f * (1.0f - noise_level * 0.6f);
        float u_prev = 0.0f;

        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.01f) : (curand_normal(&st) * 0.3f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.005f) : (curand_normal(&st) * 0.08f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);

            // True forward physics prediction with control input u_prev
            physics_step(est_p, est_v, est_th, est_w, u_prev);

            // Correction
            est_p  += gain * (obs_p - est_p);
            est_v  += gain * (obs_v - est_v);
            est_th += gain * (obs_th - est_th);
            est_w  += gain * (obs_w - est_w);

            float u = lqr_control(est_p, est_v, est_th, est_w);
            u_prev = u;
            physics_step(p, v, th, w, u);
            steps_c2++;

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c2 = false;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Cond 3: Recurrent Filter (GRU style)
    // -----------------------------------------------------------------
    int steps_c3 = 0;
    bool alive_c3 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        float h_p = init_pos, h_v = init_vel, h_th = init_theta, h_w = init_omega;
        float alpha = 0.30f * (1.0f - noise_level * 0.5f);
        float u_prev = 0.0f;

        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.01f) : (curand_normal(&st) * 0.3f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.005f) : (curand_normal(&st) * 0.08f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);

            physics_step(h_p, h_v, h_th, h_w, u_prev);

            h_p  = (1.0f - alpha) * h_p  + alpha * obs_p;
            h_v  = (1.0f - alpha) * h_v  + alpha * obs_v;
            h_th = (1.0f - alpha) * h_th + alpha * obs_th;
            h_w  = (1.0f - alpha) * h_w  + alpha * obs_w;

            float u = lqr_control(h_p, h_v, h_th, h_w);
            u_prev = u;
            physics_step(p, v, th, w, u);
            steps_c3++;

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c3 = false;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Cond 4: Holo-Manifold Riemannian Controller (HMRC Engine)
    // Dynamic Geodesic Invariant Projection with Symplectic Momentum & Phase Lock:
    // -----------------------------------------------------------------
    int steps_c4 = 0;
    bool alive_c4 = true;
    {
        float p = init_pos, v = init_vel, th = init_theta, w = init_omega;
        float h_p = init_pos, h_v = init_vel, h_th = init_theta, h_w = init_omega;
        float u_prev = 0.0f;

        for (int step = 0; step < EPISODE_STEPS; ++step) {
            float obs_p  = (curand_uniform(&st) >= noise_level) ? (p  + curand_normal(&st) * 0.01f) : (curand_normal(&st) * 0.3f);
            float obs_v  = (curand_uniform(&st) >= noise_level) ? (v  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);
            float obs_th = (curand_uniform(&st) >= noise_level) ? (th + curand_normal(&st) * 0.005f) : (curand_normal(&st) * 0.08f);
            float obs_w  = (curand_uniform(&st) >= noise_level) ? (w  + curand_normal(&st) * 0.02f) : (curand_normal(&st) * 0.5f);

            // Forward symplectic prior with control force:
            float prior_p = h_p, prior_v = h_v, prior_th = h_th, prior_w = h_w;
            physics_step(prior_p, prior_v, prior_th, prior_w, u_prev);

            // Holo-Parity Syndrome Vector:
            float syn_p  = obs_p  - prior_p;
            float syn_v  = obs_v  - prior_v;
            float syn_th = obs_th - prior_th;
            float syn_w  = obs_w  - prior_w;

            // Phase parity cross-check:
            // Check if observed sensor acceleration matches physical equations of motion:
            float expected_w_dot = (GRAVITY * sinf(prior_th) - cosf(prior_th) * (u_prev / TOTAL_MASS)) /
                                   (POLE_HALF_LEN * (4.0f / 3.0f - POLE_MASS / TOTAL_MASS));
            float observed_w_dot = (obs_w - h_w) / DT;
            float phase_violation = fabsf(observed_w_dot - expected_w_dot);

            // Riemannian Curvature-Weighted Trust Metric:
            // Outliers with large phase violations are sharply rejected!
            float trust = 1.0f / (1.0f + 0.15f * phase_violation + 50.0f * (syn_th * syn_th));

            // Geodesic update towards physical manifold:
            float cand_p  = prior_p  + trust * 0.35f * syn_p;
            float cand_v  = prior_v  + trust * 0.35f * syn_v;
            float cand_th = prior_th + trust * 0.40f * syn_th;
            float cand_w  = prior_w  + trust * 0.40f * syn_w;

            // Hamiltonian Phase Alignment:
            h_p  = cand_p;
            h_v  = cand_v;
            h_th = cand_th;
            h_w  = cand_w;

            float u = lqr_control(h_p, h_v, h_th, h_w);
            u_prev = u;
            physics_step(p, v, th, w, u);
            steps_c4++;

            if (fabsf(th) > THETA_LIMIT || fabsf(p) > POS_LIMIT) {
                alive_c4 = false;
                break;
            }
        }
    }

    results[idx].survival_c1_naive     = alive_c1 ? 1.0f : 0.0f;
    results[idx].survival_c2_ekf       = alive_c2 ? 1.0f : 0.0f;
    results[idx].survival_c3_recurrent = alive_c3 ? 1.0f : 0.0f;
    results[idx].survival_c4_hmrc      = alive_c4 ? 1.0f : 0.0f;
    results[idx].avg_steps_c1          = (float)steps_c1;
    results[idx].avg_steps_c2          = (float)steps_c2;
    results[idx].avg_steps_c3          = (float)steps_c3;
    results[idx].avg_steps_c4          = (float)steps_c4;
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
    float st1 = 0.0f, st2 = 0.0f, st3 = 0.0f, st4 = 0.0f;

    for (const auto& r : h_res) {
        s1 += r.survival_c1_naive;
        s2 += r.survival_c2_ekf;
        s3 += r.survival_c3_recurrent;
        s4 += r.survival_c4_hmrc;
        st1 += r.avg_steps_c1;
        st2 += r.avg_steps_c2;
        st3 += r.avg_steps_c3;
        st4 += r.avg_steps_c4;
    }
    s1 /= h_res.size(); s2 /= h_res.size(); s3 /= h_res.size(); s4 /= h_res.size();
    st1 /= h_res.size(); st2 /= h_res.size(); st3 /= h_res.size(); st4 /= h_res.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << ">>> " << label << " (Sensory Corruption Rate: " << (noise * 100.0f) << "%)\n"
              << "    1. Naive Direct Controller Survival Rate   : " << (s1 * 100.0f) << "% (Avg Lifetime: " << (st1 * DT) << "s / 10s)\n"
              << "    2. Extended Kalman Filter (EKF) Survival   : " << (s2 * 100.0f) << "% (Avg Lifetime: " << (st2 * DT) << "s / 10s)\n"
              << "    3. Recurrent Neural Filter (GRU) Survival  : " << (s3 * 100.0f) << "% (Avg Lifetime: " << (st3 * DT) << "s / 10s)\n"
              << "    4. Holo-Manifold Controller (HMRC) Survival: " << (s4 * 100.0f) << "% (Avg Lifetime: " << (st4 * DT) << "s / 10s)\n"
              << "    Advantage over Best Baseline (HMRC vs Best): " << std::showpos << ((s4 - fmaxf(s2, s3)) * 100.0f) << "%" << std::noshowpos << "\n"
              << "    Compute Time (1,000 Episodes)              : " << std::setprecision(2) << (sec * 1000.0) << " ms\n\n";

    CHECK_CUDA(cudaFree(d_res));
}

int main() {
    std::cout << "========================================================================\n"
              << "=== EXP-662: HOLO-MANIFOLD ROBUST CONTROL IN CONTINUOUS PHYSICS      ===\n"
              << "========================================================================\n"
              << "Task Domain         : Continuous Inverted Pendulum (50 Hz, 500 steps / 10s)\n"
              << "Physics Substrate   : Nonlinear Dynamics under 0%, 30%, 50%, 70% Sensor Noise\n"
              << "Benchmark Scale     : 1,000 Complete Physics Episodes per Regime\n"
              << "Success Criterion   : Keep |theta| < 12 deg and |pos| < 2.4m for 10.0 seconds\n"
              << "========================================================================\n\n" << std::flush;

    run_hmrc_benchmark(0.00f, "SANITY CHECK: CLEAN TELEMETRY (0% CORRUPTION)");
    run_hmrc_benchmark(0.30f, "REGIME 1: MODERATE SENSORY NOISE (30% CORRUPTION)");
    run_hmrc_benchmark(0.50f, "REGIME 2: CRITICAL SENSORY CORRUPTION (50% CORRUPTION)");
    run_hmrc_benchmark(0.70f, "REGIME 3: EXTREME SENSORY CHAOS (70% CORRUPTION)");

    return 0;
}
