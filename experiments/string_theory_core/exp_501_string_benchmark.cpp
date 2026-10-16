// ============================================================================
// EXP-501: Multi-Dimensional Resonant String Core Benchmark
// Karyon-CoRE v28.0 Sovereign Benchmark
// Evaluating Zero-Weight String Substrate on Orbit Pursuit & Pendulum
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/resonant_string_core.hpp"

using namespace karyon;

// 1. Regime A: Continuous Orbit Pursuit on String Substrate
void test_string_pursuit(ResonantStringCore& core, size_t steps = 10000) {
    std::cout << ">>> Running String Regime A: Orbit Pursuit (steps=" << steps << ")...\n";
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
            angle += 1.2f;
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
                      << " | String Energy: " << telem.total_string_energy 
                      << " | FE: " << telem.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << ">>> Regime A Finished! Total Catches: " << catches 
              << " | Avg FE: " << (total_fe / steps) 
              << " | Speed: " << (steps / elapsed) << " steps/s\n\n";
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-501: RESONANT STRING SUBSTRATE (ZERO WEIGHT MATRICES) ===\n";
    std::cout << "=== Physics of String Modes, Spatial Fourier & FPU Coupling ===\n";
    std::cout << "====================================================================\n\n";

    StringConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.num_modes = 32;
    cfg.mode_spatial_dim = 16;
    cfg.base_frequency = 0.2f;
    cfg.damping_base = 0.01f;
    cfg.nonlinearity_beta = 0.05f;

    ResonantStringCore core(cfg);

    // Pass 1: Test Pursuit
    std::cout << "--- PASS 1: STRING PURSUIT EVALUATION ---\n";
    test_string_pursuit(core, 10000);

    // Pass 2: Cumulative re-test to verify zero catastrophic forgetting
    std::cout << "--- PASS 2: RETEST WITHOUT RESETTING STRING STATE ---\n";
    test_string_pursuit(core, 10000);

    return 0;
}
