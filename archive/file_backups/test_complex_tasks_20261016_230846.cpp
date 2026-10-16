// ============================================================================
// Complex Multi-Domain Sovereign Benchmark for Karyon-CoRE v28.0
// Evaluating Karyon on 4 Hard Non-Linear Dynamical Systems:
// 1. Extreme Double-Inverted Pendulum (Acrobot-Class Underactuated Control)
// 2. High-Frequency Stochastic Brownian Perturbation Inverted Pendulum
// 3. Multi-Frequency Non-Stationary Harmonic Tracking (Anti-Phase Cancellation)
// 4. Non-Linear Duffing Double-Well Chaotic Oscillator Trajectory Entrainment
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <cassert>
#include "karyon_core.h"

using namespace karyon;

// Helper: print divider
void print_header(const std::string& title) {
    std::cout << "\n========================================================================\n";
    std::cout << ">>> TASK: " << title << "\n";
    std::cout << "========================================================================\n";
}

// ----------------------------------------------------------------------------
// TASK 1: Extreme Inverted Pendulum Under Stochastic Brownian Noise & Wind Gusts
// Evaluates active disturbance rejection under continuous Wiener drift.
// ----------------------------------------------------------------------------
bool benchmark_noisy_stochastic_pendulum() {
    print_header("1. STOCHASTIC BROWNIAN INVERTED PENDULUM WITH GUSTS");
    SovereignContinuumCore core;

    float theta = 0.65f; // Large initial displacement (~37.2 degrees)
    float omega = 0.0f;
    const float dt = 0.01f;
    const size_t total_steps = 15000; // 150 seconds of continuous physical time
    const float g = 9.81f;
    const float L = 1.0f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise_dist(0.0f, 0.45f); // Continuous Gaussian wind force

    size_t upright_steps = 0;
    float cumulative_fe = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        // External intermittent stochastic gust
        float gust = (t > 4000 && t < 4500) ? 2.5f : 0.0f; // Sustained lateral wind blast
        float brownian = noise_dist(rng);

        float torque = core.perceive_and_act(theta, omega, dt);

        float alpha = (g / L) * std::sin(theta) - torque - 0.2f * omega + brownian + gust;
        omega += dt * alpha;
        theta += dt * omega;

        // Bounded upright zone: |theta| < 0.25 rad (~14 deg)
        if (std::abs(theta) < 0.25f) {
            upright_steps++;
        }
        cumulative_fe += core.get_free_energy();
    }

    float upright_pct = 100.0f * static_cast<float>(upright_steps) / total_steps;
    std::cout << "  -> Total Steps: " << total_steps << " (150s)\n";
    std::cout << "  -> Upright In-Zone Retention: " << std::fixed << std::setprecision(2) << upright_pct << "%\n";
    std::cout << "  -> Final Angular Deviation: " << std::abs(theta) << " rad\n";
    std::cout << "  -> Average Step Free Energy: " << (cumulative_fe / total_steps) << "\n";

    bool passed = upright_pct > 92.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Robust disturbance rejection!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 2: High-Frequency Multi-Harmonic Tracking & Phase Entrainment
// System must track non-stationary composite wave: y(t) = A1*sin(w1*t) + A2*cos(w2*t)
// ----------------------------------------------------------------------------
bool benchmark_multi_frequency_tracking() {
    print_header("2. NON-STATIONARY MULTI-FREQUENCY HARMONIC TRACKING");
    SovereignContinuumCore core;

    const float dt = 0.01f;
    const size_t total_steps = 10000;
    const float w1 = 1.5f;
    const float w2 = 4.2f;

    float tracking_error_sq_sum = 0.0f;
    float late_error_sq_sum = 0.0f;

    float plant_y = 0.0f;
    float plant_v = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        float time_sec = t * dt;
        float target = 0.8f * std::sin(w1 * time_sec) + 0.35f * std::cos(w2 * time_sec);
        float target_v = 0.8f * w1 * std::cos(w1 * time_sec) - 0.35f * w2 * std::sin(w2 * time_sec);

        float err = plant_y - target;
        float err_rate = plant_v - target_v;

        float control = core.perceive_and_act(err, err_rate, dt);

        // Dynamical plant: \ddot{y} + 0.5 \dot{y} = -control
        float plant_acc = -control - 0.5f * plant_v;
        plant_v += dt * plant_acc;
        plant_y += dt * plant_v;

        tracking_error_sq_sum += err * err;
        if (t >= 5000) {
            late_error_sq_sum += err * err;
        }
    }

    float overall_rmse = std::sqrt(tracking_error_sq_sum / total_steps);
    float late_rmse = std::sqrt(late_error_sq_sum / 5000);

    std::cout << "  -> Overall Tracking RMSE: " << overall_rmse << "\n";
    std::cout << "  -> Steady-State Late Phase RMSE: " << late_rmse << "\n";

    bool passed = late_rmse < 0.08f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Continuum modes matched non-stationary frequencies!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 3: Non-Linear Duffing Double-Well Chaotic Oscillator Entrainment
// Potential V(x) = -0.5*x^2 + 0.25*x^4. System has two stable wells at x = +/- 1
// Core must force and stabilize the state onto the unstable saddle point x = 0.
// ----------------------------------------------------------------------------
bool benchmark_duffing_double_well_saddle_stabilization() {
    print_header("3. DUFFING DOUBLE-WELL SADDLE POINT STABILIZATION (x* = 0)");
    SovereignContinuumCore core;

    const float dt = 0.005f;
    const size_t total_steps = 12000;

    // Start inside left deep potential well with high momentum
    float x = -1.2f;
    float v = 2.0f;
    const float delta_damp = 0.15f; // natural damping

    size_t saddle_retained_steps = 0;

    for (size_t t = 0; t < total_steps; ++t) {
        // Saddle point error is x - 0
        float err = x;
        float err_rate = v;

        float u = core.perceive_and_act(err, err_rate, dt);

        // Duffing equation: \ddot{x} + \delta \dot{x} - x + x^3 = -u
        // The term (-x + x^3) creates anti-damping near origin and double wells at +/- 1
        float restoring_force = x - x * x * x; // Destabilizing near origin!
        float acc = restoring_force - delta_damp * v - u;

        v += dt * acc;
        x += dt * v;

        if (std::abs(x) < 0.05f && std::abs(v) < 0.1f) {
            saddle_retained_steps++;
        }
    }

    float lock_pct = 100.0f * static_cast<float>(saddle_retained_steps) / total_steps;
    std::cout << "  -> Total Steps: " << total_steps << "\n";
    std::cout << "  -> Unstable Saddle Lock (within 0.05): " << std::fixed << std::setprecision(2) << lock_pct << "%\n";
    std::cout << "  -> Final State: x = " << x << ", v = " << v << "\n";
    std::cout << "  -> Final Free Energy: " << core.get_free_energy() << "\n";

    bool passed = std::abs(x) < 0.01f && lock_pct > 80.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Unstable chaotic saddle point locked!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 4: Lorenz Attractor Chaotic Trajectory Synchronization
// Core attempts to synchronize continuous Lie-modes to drive an external
// chaotic slave system into identical synchronization with master Lorenz attractor.
// ----------------------------------------------------------------------------
bool benchmark_lorenz_chaos_synchronization() {
    print_header("4. LORENZ CHAOTIC ATTRACTOR SYNCHRONIZATION");
    SovereignContinuumCore core_x;
    SovereignContinuumCore core_y;
    SovereignContinuumCore core_z;

    const float dt = 0.005f;
    const size_t total_steps = 15000;

    const float sigma = 10.0f;
    const float rho = 28.0f;
    const float beta = 8.0f / 3.0f;

    // Master Lorenz
    float mx = 1.0f, my = 1.0f, mz = 1.0f;

    // Slave Lorenz (different initial condition)
    float sx = 10.0f, sy = -5.0f, sz = 20.0f;

    float late_sync_error_sq = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        // Master dynamics
        float dmx = sigma * (my - mx);
        float dmy = mx * (rho - mz) - my;
        float dmz = mx * my - beta * mz;

        mx += dt * dmx;
        my += dt * dmy;
        mz += dt * dmz;

        // Synchronization error
        float ex = sx - mx;
        float ey = sy - my;
        float ez = sz - mz;

        float ux = core_x.perceive_and_act(ex, 0.0f, dt);
        float uy = core_y.perceive_and_act(ey, 0.0f, dt);
        float uz = core_z.perceive_and_act(ez, 0.0f, dt);

        // Slave dynamics with core action
        float dsx = sigma * (sy - sx) - ux;
        float dsy = sx * (rho - sz) - sy - uy;
        float dsz = sx * sy - beta * sz - uz;

        sx += dt * dsx;
        sy += dt * dsy;
        sz += dt * dsz;

        if (t >= 8000) {
            late_sync_error_sq += (ex * ex + ey * ey + ez * ez);
        }
    }

    float sync_rmse = std::sqrt(late_sync_error_sq / (total_steps - 8000));
    std::cout << "  -> Chaotic Trajectory Sync RMSE (Late Phase): " << sync_rmse << "\n";
    std::cout << "  -> Final Errors: ex=" << (sx - mx) << ", ey=" << (sy - my) << ", ez=" << (sz - mz) << "\n";

    bool passed = sync_rmse < 0.1f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Chaotic trajectories entrained into exact synchronization!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== KARYON-CORE V28.0 EXTREME COMPLEX DYNAMICS BENCHMARK SUITE =========\n";
    std::cout << "========================================================================\n";

    auto start_all = std::chrono::high_resolution_clock::now();

    bool t1 = benchmark_noisy_stochastic_pendulum();
    bool t2 = benchmark_multi_frequency_tracking();
    bool t3 = benchmark_duffing_double_well_saddle_stabilization();
    bool t4 = benchmark_lorenz_chaos_synchronization();

    auto end_all = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_all - start_all).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPLEX BENCHMARK SUMMARY ==========================================\n";
    std::cout << "========================================================================\n";
    std::cout << "  1. Stochastic Brownian Inverted Pendulum:      " << (t1 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  2. Multi-Frequency Non-Stationary Tracking:    " << (t2 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  3. Duffing Double-Well Saddle Stabilization:   " << (t3 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  4. Lorenz Chaotic Attractor Synchronization:   " << (t4 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  -> Total Benchmark Execution Time: " << total_sec << " seconds\n";
    std::cout << "========================================================================\n";

    int failed = (!t1) + (!t2) + (!t3) + (!t4);
    if (failed == 0) {
        std::cout << ">>> 100% COMPLEX BENCHMARKS PASSED! KARYON CORE V28.0 PROVEN ROBUST! <<<\n";
    } else {
        std::cout << ">>> " << failed << " TASKS FAILED! <<<\n";
    }
    return failed;
}
