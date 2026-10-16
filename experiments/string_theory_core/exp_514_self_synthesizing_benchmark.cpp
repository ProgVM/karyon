// ============================================================================
// EXP-514: Self-Synthesizing String Continuum Benchmark
// Zero Hardcoded Macro Laws: Rules are Synthesized by Vibrating Strings
// Continuum Field Stress Tensor T_mu_nu Drives Emergent Mechanical Torque
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/self_synthesizing_string_organism.hpp"

using namespace karyon;

// 1. Regime A: Continuous Orbit Pursuit on Self-Synthesized String Organism
void run_string_pursuit(SelfSynthesizingStringOrganism& organism, size_t steps = 10000, bool is_pass2 = false) {
    std::cout << ">>> Running Self-Synthesizing String Organism Regime A: Orbit Pursuit (steps=" << steps 
              << (is_pass2 ? ", PASS 2 RETEST" : ", PASS 1") << ")...\n";
    const size_t S = 256;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);

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
    float total_action = 0.0f;
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
            angle += 1.2f;
        }

        float valence = caught ? 1.0f : -0.1f * dist;

        auto state = organism.perceive_and_act(curr_s, goal_s, valence);
        total_fe += state.free_energy;
        total_action += state.total_string_action;

        float ax = 4.0f * (state.motor_efferent_actions[0] - state.motor_efferent_actions[1]);
        float ay = 4.0f * state.motor_efferent_actions[2];

        vx = 0.85f * vx + 0.15f * ax;
        vy = 0.85f * vy + 0.15f * ay;
        px += 0.05f * vx;
        py += 0.05f * vy;

        px = std::clamp(px, -2.0f, 2.0f);
        py = std::clamp(py, -2.0f, 2.0f);

        if (t % 2500 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Catches: " << catches 
                      << " | Dist: " << dist 
                      << " | Avg V: " << state.avg_emergent_potential
                      << " | Left Force: " << state.left_boundary_traction
                      << " | Right Force: " << state.right_boundary_traction
                      << " | Net Torque: " << state.net_physical_torque
                      << " | FE: " << state.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << ">>> Regime A Finished! Catches: " << catches 
              << " | Avg String Action: " << (total_action / steps)
              << " | Avg FE: " << (total_fe / steps) 
              << " | Throughput: " << (steps / elapsed) << " steps/s\n\n";
}

// 2. Regime B: Inverted Pendulum Balancing on Self-Synthesizing String Organism
void run_string_pendulum(SelfSynthesizingStringOrganism& organism, size_t steps = 15000) {
    std::cout << ">>> Running Self-Synthesizing String Organism Regime B: Inverted Pendulum (steps=" << steps << ")...\n";
    const size_t S = 256;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);

    float theta = 0.15f;
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

    encode_state(goal_s, 0.0f, 0.0f);

    for (size_t t = 0; t < steps; ++t) {
        encode_state(curr_s, theta, omega);

        bool is_upright = (std::abs(theta) < 0.35f);
        if (is_upright) balanced_ticks++;

        if (std::abs(theta) > 1.8f) {
            theta = (t % 2 == 0 ? 0.2f : -0.2f);
            omega = 0.0f;
        }

        float valence = is_upright ? 0.8f : -0.4f;

        auto state = organism.perceive_and_act(curr_s, goal_s, valence);

        // Torque is emergent from differential traction of the string stress tensor
        float torque = 14.0f * (state.motor_efferent_actions[0] - state.motor_efferent_actions[1]);
        float alpha = (g / L) * std::sin(theta) - torque - 0.2f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        if (t % 3750 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Theta: " << theta 
                      << " | Upright: " << (balanced_ticks * 100 / (t + 1)) << "%"
                      << " | Avg V: " << state.avg_emergent_potential
                      << " | Left Force: " << state.left_boundary_traction
                      << " | Right Force: " << state.right_boundary_traction
                      << " | Net Torque: " << state.net_physical_torque
                      << " | FE: " << state.free_energy << "\n";
        }
    }

    std::cout << ">>> Regime B Finished! Upright Ratio: " 
              << (balanced_ticks * 100 / steps) << "% (" << balanced_ticks << "/" << steps << ")\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-514: SELF-SYNTHESIZING STRING CONTINUUM BENCHMARK ===========\n";
    std::cout << "=== Zero Hand-Crafted Macro Laws: Rules Synthesized from Strings =====\n";
    std::cout << "========================================================================\n\n";

    StringContinuumConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.num_string_modes = 48;
    cfg.grid_nx = 4;
    cfg.grid_ny = 6;
    cfg.relaxation_substeps = 8;
    cfg.string_tension = 1.8f;
    cfg.string_coupling_beta = 0.02f;
    cfg.dt = 0.04f;

    SelfSynthesizingStringOrganism karyon_strings(cfg);

    // 1. Pass 1: Test Orbit Pursuit
    std::cout << "--- PHASE 1: ORBIT PURSUIT (STRING TENSOR INTERFERENCE) ---\n";
    run_string_pursuit(karyon_strings, 10000, false);

    // 2. Regime B: Inverted Pendulum
    std::cout << "--- PHASE 2: INVERTED PENDULUM (TRACTION-DRIVEN ANTAGONISTIC BALANCE) ---\n";
    run_string_pendulum(karyon_strings, 15000);

    // 3. Pass 2: Re-test Orbit Pursuit
    std::cout << "--- PHASE 3: ORBIT PURSUIT RE-TEST (STRING RESIDUAL MEMORY CHECK) ---\n";
    run_string_pursuit(karyon_strings, 10000, true);

    return 0;
}
