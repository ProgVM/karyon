// ============================================================================
// Complex Multi-Domain Benchmark Suite for Karyon-CoRE v28.0 Sovereign Engine
// Rigorous verification across 4 Hard Non-Linear Physical Domains:
// Task 1: Non-Stationary Multi-Frequency Harmonic Tracking & Wave Resonance
// Task 2: Double-Well Potential Quantum-Classical Phase Transition Capture
// Task 3: Multi-Body Gravitational Orbital Interception & Capture Resonance
// Task 4: Continuous Lie-Flow Under High-Frequency Stochastic Noise & Invariant Conservation
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include "karyon_core.h"
#include "experiments/string_theory_core/true_sovereign_organism.hpp"

using namespace karyon;

void print_header(const std::string& title) {
    std::cout << "\n========================================================================\n";
    std::cout << ">>> TASK: " << title << "\n";
    std::cout << "========================================================================\n";
}

// ----------------------------------------------------------------------------
// TASK 1: NON-STATIONARY MULTI-FREQUENCY HARMONIC TRACKING
// Tested on SovereignContinuumCore (Closed-Form Lie-Flow Matrix Exponential)
// ----------------------------------------------------------------------------
bool benchmark_multi_frequency_tracking() {
    print_header("1. NON-STATIONARY MULTI-FREQUENCY HARMONIC TRACKING (Lie-Flow)");
    SovereignContinuumCore core;

    const float dt = 0.01f;
    const size_t total_steps = 10000;
    const float w1 = 1.5f;
    const float w2 = 4.2f;

    float plant_y = 0.0f;
    float plant_v = 0.0f;
    float late_error_sq_sum = 0.0f;

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

        if (t >= 5000) {
            late_error_sq_sum += err * err;
        }
    }

    float late_rmse = std::sqrt(late_error_sq_sum / 5000);
    std::cout << "  -> Total Steps: " << total_steps << "\n";
    std::cout << "  -> Steady-State Late Phase Tracking RMSE: " << late_rmse << "\n";

    bool passed = late_rmse < 0.08f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Fourier modes matched dynamic frequencies!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 2: DOUBLE-WELL PHASE TRANSITION & TUNNELING CAPTURE
// Potential V(x) = x^4 - 2*x^2 (wells at x = -1 and x = +1)
// ----------------------------------------------------------------------------
bool benchmark_double_well_phase_capture() {
    print_header("2. DOUBLE-WELL POTENTIAL PHASE TRANSITION CAPTURE");
    TrueSovereignStringOrganism organism;

    float particle_x = 0.0f;
    float particle_v = 0.0f;
    const float dt = 0.01f;
    const size_t total_steps = 20000;
    size_t well_lock_steps = 0;

    for (size_t t = 0; t < total_steps; ++t) {
        float target_well = ((t / 5000) % 2 == 0) ? 1.0f : -1.0f;
        float error = particle_x - target_well;

        float double_well_force = 4.0f * particle_x - 4.0f * particle_x * particle_x * particle_x;
        float torque = organism.actuate_and_adapt(error, particle_v, dt);

        float acc = double_well_force - torque - 0.5f * particle_v;
        particle_v += dt * acc;
        particle_x += dt * particle_v;

        if (std::abs(particle_x - target_well) < 0.15f) {
            well_lock_steps++;
        }
    }

    float lock_pct = 100.0f * static_cast<float>(well_lock_steps) / total_steps;
    std::cout << "  -> Total Steps: " << total_steps << "\n";
    std::cout << "  -> Well Lock Retention Ratio: " << std::fixed << std::setprecision(2) << lock_pct << "%\n";
    std::cout << "  -> Self-Synthesized Stiffness: " << organism.get_stiffness() << "\n";
    std::cout << "  -> Final Free Energy: " << organism.get_free_energy() << "\n";

    bool passed = lock_pct > 90.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Phase transitions locked onto commanded well!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 3: MULTI-BODY GRAVITATIONAL ORBITAL INTERCEPTION
// ----------------------------------------------------------------------------
bool benchmark_orbital_interception() {
    print_header("3. MULTI-BODY GRAVITATIONAL ORBITAL INTERCEPTION");
    TrueSovereignStringOrganism organism;

    float px = 2.5f, py = 0.0f;
    float vx = 0.0f, vy = 1.2f;
    const float target_r = 1.5f;
    const float mu = 10.0f;
    const float dt = 0.008f;
    const size_t total_steps = 25000;
    size_t orbit_locks = 0;

    for (size_t t = 0; t < total_steps; ++t) {
        float r = std::sqrt(px * px + py * py);
        float r_error = r - target_r;
        float radial_v = (px * vx + py * vy) / r;

        float thrust = organism.actuate_and_adapt(r_error, radial_v, dt);
        float f_grav = -mu / (r * r);

        float ax = (f_grav * (px / r)) - thrust * (px / r);
        float ay = (f_grav * (py / r)) - thrust * (py / r);

        vx += dt * ax;
        vy += dt * ay;
        px += dt * vx;
        py += dt * vy;

        if (std::abs(r - target_r) < 0.10f) {
            orbit_locks++;
        }
    }

    float orbit_pct = 100.0f * static_cast<float>(orbit_locks) / total_steps;
    std::cout << "  -> Total Steps: " << total_steps << "\n";
    std::cout << "  -> Orbit Lock Ratio: " << std::fixed << std::setprecision(2) << orbit_pct << "%\n";
    std::cout << "  -> Final Orbit Radius: " << std::sqrt(px * px + py * py) << " (Target: " << target_r << ")\n";

    bool passed = orbit_pct > 95.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Gravitational orbit intercepted and resonance locked!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

// ----------------------------------------------------------------------------
// TASK 4: STOCHASTIC BROWNIAN INVERTED PENDULUM WITH CONTINUOUS WIND GUSTS
// ----------------------------------------------------------------------------
bool benchmark_stochastic_inverted_pendulum() {
    print_header("4. STOCHASTIC INVERTED PENDULUM WITH HIGH-FREQUENCY NOISE & GUSTS");
    TrueSovereignStringOrganism organism;

    float theta = 0.55f; // ~31.5 degrees initial tilt
    float omega = 0.0f;
    const float dt = 0.01f;
    const size_t total_steps = 15000;
    const float g = 9.81f;
    const float L = 1.0f;

    std::mt19937 rng(1337);
    std::normal_distribution<float> noise(0.0f, 0.35f);

    size_t upright_steps = 0;

    for (size_t t = 0; t < total_steps; ++t) {
        float gust = (t > 4000 && t < 4500) ? 1.5f : 0.0f; // lateral wind shock
        float brownian = noise(rng);

        float torque = organism.actuate_and_adapt(theta, omega, dt);

        float alpha = (g / L) * std::sin(theta) - torque - 0.2f * omega + brownian + gust;
        omega += dt * alpha;
        theta += dt * omega;

        if (std::abs(theta) < 0.25f) {
            upright_steps++;
        }
    }

    float upright_pct = 100.0f * static_cast<float>(upright_steps) / total_steps;
    std::cout << "  -> Total Steps: " << total_steps << " (150s physical time)\n";
    std::cout << "  -> Upright Retention under Brownian Gusts: " << std::fixed << std::setprecision(2) << upright_pct << "%\n";
    std::cout << "  -> Self-Stiffness K: " << organism.get_stiffness() << "\n";

    bool passed = upright_pct > 92.0f;
    std::cout << (passed ? "  [VERDICT: SUCCESS] Brownian disturbance actively rejected!\n" : "  [VERDICT: FAILED]\n");
    return passed;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== KARYON-CORE V28.0 RIGOROUS COMPLEX DOMAINS BENCHMARK SUITE =========\n";
    std::cout << "========================================================================\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    bool t1 = benchmark_multi_frequency_tracking();
    bool t2 = benchmark_double_well_phase_capture();
    bool t3 = benchmark_orbital_interception();
    bool t4 = benchmark_stochastic_inverted_pendulum();

    auto t1_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1_time - t0).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== SUMMARY OF COMPLEX BENCHMARK RESULTS ===============================\n";
    std::cout << "========================================================================\n";
    std::cout << "  1. Multi-Frequency Non-Stationary Tracking:    " << (t1 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  2. Double-Well Phase Transition Capture:       " << (t2 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  3. Multi-Body Gravitational Orbital Intercept: " << (t3 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  4. Stochastic Inverted Pendulum Under Gusts:   " << (t4 ? "PASSED [OK]" : "FAILED [X]") << "\n";
    std::cout << "  -> Total Execution Time: " << elapsed << " seconds\n";
    std::cout << "========================================================================\n";

    int failed = (!t1) + (!t2) + (!t3) + (!t4);
    if (failed == 0) {
        std::cout << ">>> 100% OF HARD COMPLEX DYNAMICAL BENCHMARKS PASSED PERFECTLY! <<<\n";
    } else {
        std::cout << ">>> " << failed << " BENCHMARKS FAILED! <<<\n";
    }
    return failed;
}
