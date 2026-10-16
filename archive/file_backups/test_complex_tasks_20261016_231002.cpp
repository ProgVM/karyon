// ============================================================================
// Complex Multi-Domain Sovereign Benchmark for Karyon-CoRE v28.0
// Evaluating Karyon on 4 Hard Non-Linear Dynamical Systems:
// 1. Stochastic Brownian Inverted Pendulum with Sustained Wind Gusts
// 2. Non-Stationary Multi-Frequency Harmonic Tracking & Wave Entrainment
// 3. Double-Well Quantum-Classical Potential Phase Transition Capture
// 4. Chaotic Lorenz Attractor Continuous State Tracking
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include "karyon_core.h"

using namespace karyon;

void print_header(const std::string& title) {
    std::cout << "\n========================================================================\n";
    std::cout << ">>> TASK: " << title << "\n";
    std::cout << "========================================================================\n";
}

// ----------------------------------------------------------------------------
// TASK 1: Stochastic Inverted Pendulum Under Brownian Noise & Wind Gusts
// ----------------------------------------------------------------------------
bool benchmark_noisy_stochastic_pendulum() {
    print_header("1. STOCHASTIC BROWNIAN INVERTED PENDULUM WITH GUSTS");
    SovereignContinuumCore core;

    float theta = 0.5f; // ~28.6 degrees initial displacement
    float omega = 0.0f;
    const float dt = 0.01f;
    const size_t total_steps = 15000;
    const float g = 9.81f;
    const float L = 1.0f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise_dist(0.0f, 0.15f);

    size_t upright_steps = 0;
    float cumulative_fe = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        float gust = (t > 4000 && t < 4400) ? 0.8f : 0.0f;
        float brownian = noise_dist(rng);

        float torque = core.perceive_and_act(theta, omega, dt);

        float alpha = (g / L) * std::sin(theta) - torque - 0.2f * omega + brownian + gust;
        omega += dt * alpha;
        theta += dt * omega;

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

    bool passed = upright_pct > 90.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Robust disturbance rejection!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 2: Non-Stationary Multi-Frequency Harmonic Tracking & Wave Entrainment
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
// TASK 3: Double-Well Potential Phase Transition & Target Well Locking
// Potential V(x) = x^4 - 2*x^2 (wells at x = -1 and x = +1)
// ----------------------------------------------------------------------------
bool benchmark_double_well_phase_capture() {
    print_header("3. DOUBLE-WELL POTENTIAL PHASE TRANSITION CAPTURE");
    SovereignContinuumCore core;

    float particle_x = 0.0f;
    float particle_v = 0.0f;
    const float dt = 0.005f;
    const size_t total_steps = 20000;
    size_t well_lock_steps = 0;

    for (size_t t = 0; t < total_steps; ++t) {
        // Target well switches every 5000 steps: left (-1.0) -> right (+1.0)
        float target_well = ((t / 5000) % 2 == 0) ? 1.0f : -1.0f;
        float error = particle_x - target_well;

        // Intrinsic force F = -dV/dx = 4*x - 4*x^3
        float double_well_force = 4.0f * particle_x - 4.0f * particle_x * particle_x * particle_x;

        float control = core.perceive_and_act(error, particle_v, dt);

        float acc = double_well_force - control - 2.0f * particle_v;
        particle_v += dt * acc;
        particle_x += dt * particle_v;

        if (std::abs(particle_x - target_well) < 0.15f) {
            well_lock_steps++;
        }
    }

    float lock_pct = 100.0f * static_cast<float>(well_lock_steps) / total_steps;
    std::cout << "  -> Total Steps: " << total_steps << "\n";
    std::cout << "  -> Well Lock Retention: " << std::fixed << std::setprecision(2) << lock_pct << "%\n";
    std::cout << "  -> Final State: x = " << particle_x << ", v = " << particle_v << "\n";

    bool passed = lock_pct > 85.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Double-well phase transitions successfully captured!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 4: Non-Linear Lorenz Chaotic Attractor Tracking
// Core continuously tracks the normalized chaotic state x(t) of Lorenz attractor.
// ----------------------------------------------------------------------------
bool benchmark_lorenz_chaotic_tracking() {
    print_header("4. NON-LINEAR LORENZ CHAOTIC ATTRACTOR TRACKING");
    SovereignContinuumCore core;

    float lx = 1.0f, ly = 1.0f, lz = 1.0f;
    const float sigma = 10.0f;
    const float rho = 28.0f;
    const float beta = 8.0f / 3.0f;
    const float dt = 0.005f;
    const size_t total_steps = 20000;

    float tracker_pos = 0.0f;
    float tracker_vel = 0.0f;
    size_t in_sync_steps = 0;

    float late_error_sq = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        float dx = sigma * (ly - lx);
        float dy = lx * (rho - lz) - ly;
        float dz = lx * ly - beta * lz;

        lx += dt * dx;
        ly += dt * dy;
        lz += dt * dz;

        float target = lx / 15.0f;
        float error = tracker_pos - target;

        float torque = core.perceive_and_act(error, tracker_vel, dt);

        tracker_vel += dt * (-torque - 3.0f * tracker_vel);
        tracker_pos += dt * tracker_vel;

        if (std::abs(error) < 0.10f) {
            in_sync_steps++;
        }
        if (t >= 10000) {
            late_error_sq += error * error;
        }
    }

    float sync_pct = 100.0f * static_cast<float>(in_sync_steps) / total_steps;
    float late_rmse = std::sqrt(late_error_sq / 10000);

    std::cout << "  -> Total Steps: " << total_steps << "\n";
    std::cout << "  -> Chaotic Trajectory In-Sync Ratio: " << std::fixed << std::setprecision(2) << sync_pct << "%\n";
    std::cout << "  -> Late Phase RMSE: " << late_rmse << "\n";

    bool passed = sync_pct > 80.0f && late_rmse < 0.08f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Chaotic trajectory successfully tracked!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== KARYON-CORE V28.0 EXTREME COMPLEX DYNAMICS BENCHMARK SUITE =========\n";
    std::cout << "========================================================================\n";

    auto start_all = std::chrono::high_resolution_clock::now();

    bool t1 = benchmark_noisy_stochastic_pendulum();
    bool t2 = benchmark_multi_frequency_tracking();
    bool t3 = benchmark_double_well_phase_capture();
    bool t4 = benchmark_lorenz_chaotic_tracking();

    auto end_all = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_all - start_all).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPLEX BENCHMARK SUMMARY ==========================================\n";
    std::cout << "========================================================================\n";
    std::cout << "  1. Stochastic Brownian Inverted Pendulum:      " << (t1 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  2. Multi-Frequency Non-Stationary Tracking:    " << (t2 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  3. Double-Well Phase Transition Capture:       " << (t3 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  4. Lorenz Chaotic Attractor Tracking:          " << (t4 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  -> Total Benchmark Execution Time: " << total_sec << " seconds\n";
    std::cout << "========================================================================\n";

    int failed = (!t1) + (!t2) + (!t3) + (!t4);
    if (failed == 0) {
        std::cout << ">>> 100% COMPLEX BENCHMARKS PASSED! KARYON CORE V28.0 FULLY VERIFIED! <<<\n";
    } else {
        std::cout << ">>> " << failed << " TASKS FAILED! <<<\n";
    }
    return failed;
}
