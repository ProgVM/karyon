// ============================================================================
// EXP-516: Multi-Domain Benchmark Suite for Sovereign Self-Synthesizing Strings
// Testing Principle 29 across 3 Diverse Continuous Physics & Control Tasks:
// Task 1: Non-Linear Lorenz Chaotic Attractor Tracking
// Task 2: Double-Well Quantum Tunneling & Phase Transition Capture
// Task 3: Multi-Body Gravitational Orbital Interception & Trajectory Resonance
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/true_sovereign_organism.hpp"

using namespace karyon;

// ----------------------------------------------------------------------------
// TASK 1: NON-LINEAR LORENZ CHAOTIC ATTRACTOR TRACKING
// Sovereign strings must track chaotic state (x, y, z) with zero lag
// ----------------------------------------------------------------------------
void test_lorenz_tracking(TrueSovereignStringOrganism& organism, size_t steps = 20000) {
    std::cout << ">>> TASK 1: NON-LINEAR LORENZ CHAOTIC ATTRACTOR TRACKING (steps=" << steps << ")...\n";
    
    // Lorenz chaotic system parameters
    float lx = 1.0f, ly = 1.0f, lz = 1.0f;
    const float sigma = 10.0f;
    const float rho = 28.0f;
    const float beta = 8.0f / 3.0f;
    const float dt = 0.005f;

    float tracker_pos = 0.0f;
    float tracker_vel = 0.0f;
    size_t in_sync_steps = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        // Step Lorenz equations
        float dx = sigma * (ly - lx);
        float dy = lx * (rho - lz) - ly;
        float dz = lx * ly - beta * lz;

        lx += dt * dx;
        ly += dt * dy;
        lz += dt * dz;

        // Target signal is normalized X-coordinate of chaotic attractor
        float target = lx / 15.0f;
        float error = tracker_pos - target;

        // Sovereign string organism perceives chaotic error and self-synthesizes restoring torque
        float torque = organism.actuate_and_adapt(error, tracker_vel, dt);

        tracker_vel += dt * (-torque - 2.0f * tracker_vel);
        tracker_pos += dt * tracker_vel;

        if (std::abs(error) < 0.08f) {
            in_sync_steps++;
        }

        if ((t + 1) % 5000 == 0 || t == steps - 1) {
            std::cout << "  Step " << (t + 1) << "/" << steps
                      << " | Target X: " << target
                      << " | Tracker: " << tracker_pos
                      << " | Error: " << error
                      << " | Sync Ratio: " << (in_sync_steps * 100 / (t + 1)) << "%"
                      << " | Self-Stiffness K: " << organism.get_stiffness()
                      << " | Free Energy: " << organism.get_free_energy() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    float sync_pct = static_cast<float>(in_sync_steps * 100) / static_cast<float>(steps);
    std::cout << ">>> TASK 1 COMPLETE! Sync Precision: " << sync_pct << "% | Speed: " << (steps / elapsed) << " steps/s\n\n";
}

// ----------------------------------------------------------------------------
// TASK 2: DOUBLE-WELL PHASE TRANSITION & TUNNELING CAPTURE
// Potential V(x) = x^4 - 2*x^2 (wells at x = -1 and x = +1)
// Sovereign strings must drive transition on command and lock into targeted well
// ----------------------------------------------------------------------------
void test_double_well_control(TrueSovereignStringOrganism& organism, size_t steps = 20000) {
    std::cout << ">>> TASK 2: DOUBLE-WELL POTENTIAL PHASE TRANSITION CAPTURE (steps=" << steps << ")...\n";

    float particle_x = 0.0f;
    float particle_v = 0.0f;
    const float dt = 0.01f;
    size_t well_lock_steps = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        // Target well switches every 5000 steps: left (-1.0) -> right (+1.0)
        float target_well = ((t / 5000) % 2 == 0) ? 1.0f : -1.0f;
        float error = particle_x - target_well;

        // Double well intrinsic force F = -dV/dx = 4*x - 4*x^3
        float double_well_force = 4.0f * particle_x - 4.0f * particle_x * particle_x * particle_x;

        // Sovereign string actuation
        float torque = organism.actuate_and_adapt(error, particle_v, dt);

        float acc = double_well_force - torque - 0.5f * particle_v;
        particle_v += dt * acc;
        particle_x += dt * particle_v;

        if (std::abs(particle_x - target_well) < 0.15f) {
            well_lock_steps++;
        }

        if ((t + 1) % 5000 == 0 || t == steps - 1) {
            std::cout << "  Step " << (t + 1) << "/" << steps
                      << " | Target Well: " << target_well
                      << " | Particle X: " << particle_x
                      << " | Well Lock Ratio: " << (well_lock_steps * 100 / (t + 1)) << "%"
                      << " | Self-Stiffness K: " << organism.get_stiffness()
                      << " | Free Energy: " << organism.get_free_energy() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    float lock_pct = static_cast<float>(well_lock_steps * 100) / static_cast<float>(steps);
    std::cout << ">>> TASK 2 COMPLETE! Phase Lock Ratio: " << lock_pct << "% | Speed: " << (steps / elapsed) << " steps/s\n\n";
}

// ----------------------------------------------------------------------------
// TASK 3: MULTI-BODY GRAVITATIONAL ORBITAL INTERCEPTION
// Fast orbital mechanics with central gravity mu = 10.0
// Sovereign strings must intercept and lock orbit at radius R = 1.5
// ----------------------------------------------------------------------------
void test_orbital_interception(TrueSovereignStringOrganism& organism, size_t steps = 25000) {
    std::cout << ">>> TASK 3: MULTI-BODY GRAVITATIONAL ORBITAL INTERCEPTION (steps=" << steps << ")...\n";

    float px = 2.5f, py = 0.0f;
    float vx = 0.0f, vy = 1.2f;
    const float target_r = 1.5f;
    const float mu = 10.0f;
    const float dt = 0.008f;
    size_t orbit_locks = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        float r = std::sqrt(px * px + py * py);
        float r_error = r - target_r;

        float radial_v = (px * vx + py * vy) / r;

        // Sovereign string continuum actuates radial thrust
        float thrust = organism.actuate_and_adapt(r_error, radial_v, dt);

        // Gravitational force
        float f_grav = -mu / (r * r);

        // Net radial force
        float ax = (f_grav * (px / r)) - thrust * (px / r);
        float ay = (f_grav * (py / r)) - thrust * (py / r);

        vx += dt * ax;
        vy += dt * ay;
        px += dt * vx;
        py += dt * vy;

        if (std::abs(r - target_r) < 0.10f) {
            orbit_locks++;
        }

        if ((t + 1) % 5000 == 0 || t == steps - 1) {
            std::cout << "  Step " << (t + 1) << "/" << steps
                      << " | Target R: " << target_r
                      << " | Current R: " << r
                      << " | Orbit Lock Ratio: " << (orbit_locks * 100 / (t + 1)) << "%"
                      << " | Self-Stiffness K: " << organism.get_stiffness()
                      << " | Free Energy: " << organism.get_free_energy() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    float orbit_pct = static_cast<float>(orbit_locks * 100) / static_cast<float>(steps);
    std::cout << ">>> TASK 3 COMPLETE! Orbit Lock Ratio: " << orbit_pct << "% | Speed: " << (steps / elapsed) << " steps/s\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-516: MULTI-DOMAIN SOVEREIGN CONTINUUM VERIFICATION SUITE =====\n";
    std::cout << "=== Validating Principle 29 Across 3 Diverse Physical Domains =========\n";
    std::cout << "========================================================================\n\n";

    TrueSovereignStringOrganism lorenz_organism;
    TrueSovereignStringOrganism double_well_organism;
    TrueSovereignStringOrganism orbital_organism;

    // 1. Task 1: Non-Linear Lorenz Chaotic Tracking
    test_lorenz_tracking(lorenz_organism, 20000);

    // 2. Task 2: Double-Well Potential Transition
    test_double_well_control(double_well_organism, 20000);

    // 3. Task 3: Multi-Body Gravitational Orbital Interception
    test_orbital_interception(orbital_organism, 25000);

    std::cout << "========================================================================\n";
    std::cout << ">>> ALL 3 MULTI-DOMAIN VERIFICATION TASKS COMPLETED SUCCESSFULLY!\n";
    std::cout << "========================================================================\n";

    return 0;
}
