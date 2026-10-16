// ============================================================================
// EXP-502: Hierarchical Particle & Molecule Emergence Benchmark
// Karyon-CoRE v28.0 Sovereign Multi-Regime Evaluation
// Testing Zero-Weight String Substrate -> Hadrons -> Molecules
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/emergent_molecular_string_core.hpp"

using namespace karyon;

// 1. Regime A: Continuous Orbit Pursuit
void run_molecular_pursuit(EmergentMolecularStringCore& core, size_t steps = 10000, bool is_pass2 = false) {
    std::cout << ">>> Running Molecular Regime A: Orbit Pursuit (steps=" << steps 
              << (is_pass2 ? ", PASS 2 RETEST" : ", PASS 1") << ")...\n";
    const size_t S = 256;
    const size_t A = 4;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);
    std::vector<float> action(A, 0.0f);

    float px = 0.0f, py = 0.0f;
    float vx = 0.0f, vy = 0.0f;
    float angle = 0.0f;
    size_t catches = 0;

    auto encode_pos = [](std::vector<float>& buf, float x, float y) {
        std::fill(buf.begin(), buf.end(), 0.0f);
        int gx = std::clamp(static_cast<int>((x + 2.0f) * 3.75f), 0, 15);
        int gy = std::clamp(static_cast<int>((y + 2.0f) * 3.75f), 0, 15);
        buf[gy * 16 + gx] = 1.0f;
    };

    float total_fe = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        float tx = 1.2f * std::cos(angle);
        float ty = 1.2f * std::sin(angle);
        angle += 0.015f;

        encode_pos(curr_s, px, py);
        encode_pos(goal_s, tx, ty);

        float dist = std::sqrt((px - tx) * (px - tx) + (py - ty) * (py - ty));
        bool caught = (dist < 0.25f);
        if (caught) {
            catches++;
            angle += 1.2f; // Target shifts away upon interception
        }

        float valence = caught ? 1.0f : -0.1f * dist;

        auto telem = core.step(curr_s, goal_s, valence, action);
        total_fe += telem.free_energy;

        // Apply physical dynamics
        float ax = 4.0f * (action[0] - action[1]);
        float ay = 4.0f * (action[2] - action[3]);

        vx = 0.85f * vx + 0.15f * ax;
        vy = 0.85f * vy + 0.15f * ay;
        px += 0.05f * vx;
        py += 0.05f * vy;

        px = std::clamp(px, -2.0f, 2.0f);
        py = std::clamp(py, -2.0f, 2.0f);

        if (t % 2500 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Catches: " << catches 
                      << " | Dist: " << dist 
                      << " | Hadrons: " << telem.active_hadrons
                      << " | Mol: " << telem.active_molecules
                      << " | Binding E: " << telem.binding_energy_total
                      << " | FE: " << telem.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << ">>> Regime A Finished! Catches: " << catches 
              << " | Avg FE: " << (total_fe / steps) 
              << " | Throughput: " << (steps / elapsed) << " steps/s\n\n";
}

// 2. Regime B: Inverted Pendulum Balancing on String Substrate
void run_molecular_pendulum(EmergentMolecularStringCore& core, size_t steps = 15000) {
    std::cout << ">>> Running Molecular Regime B: Inverted Pendulum (steps=" << steps << ")...\n";
    const size_t S = 256;
    const size_t A = 4;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);
    std::vector<float> action(A, 0.0f);

    float theta = 0.15f; // Small tilt in radians
    float omega = 0.0f;
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;
    size_t balanced_ticks = 0;

    auto encode_state = [](std::vector<float>& fb, float th, float om) {
        std::fill(fb.begin(), fb.end(), 0.0f);
        int gx = std::clamp(static_cast<int>((th / 3.14159f + 1.0f) * 7.5f), 0, 15);
        int gy = std::clamp(static_cast<int>((om / 10.0f + 1.0f) * 7.5f), 0, 15);
        fb[gy * 16 + gx] = 1.0f;
    };

    encode_state(goal_s, 0.0f, 0.0f); // Goal: upright at rest

    for (size_t t = 0; t < steps; ++t) {
        encode_state(curr_s, theta, omega);

        bool is_upright = (std::abs(theta) < 0.35f);
        if (is_upright) balanced_ticks++;

        // Reset if fallen
        if (std::abs(theta) > 1.8f) {
            theta = (t % 2 == 0 ? 0.2f : -0.2f);
            omega = 0.0f;
        }

        float valence = is_upright ? 0.8f : -0.4f;

        auto telem = core.step(curr_s, goal_s, valence, action);

        // Pendulum physical step
        float torque = 12.0f * (action[0] - action[1]);
        float alpha = (g / L) * std::sin(theta) + torque - 0.2f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        if (t % 3750 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Theta: " << theta 
                      << " | Upright: " << (balanced_ticks * 100 / (t + 1)) << "%"
                      << " | Hadrons: " << telem.active_hadrons
                      << " | Binding E: " << telem.binding_energy_total
                      << " | FE: " << telem.free_energy << "\n";
        }
    }

    std::cout << ">>> Regime B Finished! Upright Ratio: " 
              << (balanced_ticks * 100 / steps) << "% (" << balanced_ticks << "/" << steps << ")\n\n";
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-502: HIERARCHICAL PARTICLE & MOLECULE EMERGENCE CORE ===\n";
    std::cout << "=== Strings -> Quarks -> Gluon Flux -> Hadrons -> Molecules ===\n";
    std::cout << "====================================================================\n\n";

    ParticleConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.num_quarks = 32;
    cfg.spatial_dim = 16;
    cfg.max_hadrons = 24;
    cfg.max_molecules = 8;
    cfg.fundamental_w0 = 0.2f;
    cfg.dt = 0.04f;

    EmergentMolecularStringCore core(cfg);

    // 1. Pass 1: Test Orbit Pursuit (Quarks learn to form hadrons)
    std::cout << "--- PHASE 1: ORBIT PURSUIT (HADRON NUCLEATION) ---\n";
    run_molecular_pursuit(core, 10000, false);

    // 2. Regime B: Inverted Pendulum (Quarks & Hadrons specialize without wiping)
    std::cout << "--- PHASE 2: INVERTED PENDULUM (MULTITASK COMPETENCE) ---\n";
    run_molecular_pendulum(core, 15000);

    // 3. Pass 2: Re-test Orbit Pursuit (Crucial check: did bound hadrons survive?)
    std::cout << "--- PHASE 3: ORBIT PURSUIT RE-TEST (ZERO CATASTROPHIC FORGETTING) ---\n";
    run_molecular_pursuit(core, 10000, true);

    return 0;
}
