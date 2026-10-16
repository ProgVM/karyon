// ============================================================================
// EXP-506: Multi-Scale Complex Cellular Organism Benchmark
// Realistic matter scaling: 128 Atoms/Ions -> 32 Molecules -> 2 Complex Cells
// With Temporal Relaxation (8 sub-ticks of physical relaxation per macro step)
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/multi_scale_complex_cellular_organism.hpp"

using namespace karyon;

// 1. Regime A: Continuous Orbit Pursuit on Multi-Scale KARYON Organism
void run_multiscale_karyon_pursuit(MultiScaleComplexOrganism& organism, size_t steps = 10000, bool is_pass2 = false) {
    std::cout << ">>> Running Multi-Scale KARYON Organism Regime A: Orbit Pursuit (steps=" << steps 
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
    size_t total_decays = 0;
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
        total_decays += state.meson_decays;

        float ax = 4.0f * (state.motor_efferent_actions[0] - state.motor_efferent_actions[1]);
        float ay = 4.0f * (state.motor_efferent_actions[2] - state.motor_efferent_actions[3]);

        vx = 0.85f * vx + 0.15f * ax;
        vy = 0.85f * vy + 0.15f * ay;
        px += 0.05f * vx;
        py += 0.05f * vy;

        px = std::clamp(px, -2.0f, 2.0f);
        py = std::clamp(py, -2.0f, 2.0f);

        if (t % 2500 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Catches: " << catches 
                      << " | Dist: " << dist 
                      << " | Atoms: " << state.total_atoms
                      << " | Mol: " << state.total_molecules
                      << " | Cells: " << state.total_cells
                      << " | V_sens: " << state.sensory_cell_vmem << "mV"
                      << " | Myo-Ca: " << state.myocyte_calcium
                      << " | Strain: " << state.myocyte_strain
                      << " | Coherence: " << state.tissue_coherence
                      << " | FE: " << state.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << ">>> Regime A Finished! Catches: " << catches 
              << " | Total Meson Decays: " << total_decays
              << " | Avg FE: " << (total_fe / steps) 
              << " | Throughput: " << (steps / elapsed) << " steps/s\n\n";
}

// 2. Regime B: Inverted Pendulum Balancing on Multi-Scale KARYON Organism
void run_multiscale_karyon_pendulum(MultiScaleComplexOrganism& organism, size_t steps = 15000) {
    std::cout << ">>> Running Multi-Scale KARYON Organism Regime B: Inverted Pendulum (steps=" << steps << ")...\n";
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

        float torque = 12.0f * (state.motor_efferent_actions[0] - state.motor_efferent_actions[1]);
        float alpha = (g / L) * std::sin(theta) + torque - 0.2f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        if (t % 3750 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Theta: " << theta 
                      << " | Upright: " << (balanced_ticks * 100 / (t + 1)) << "%"
                      << " | V_sens: " << state.sensory_cell_vmem << "mV"
                      << " | Myo-Ca: " << state.myocyte_calcium
                      << " | Strain: " << state.myocyte_strain
                      << " | Coherence: " << state.tissue_coherence
                      << " | FE: " << state.free_energy << "\n";
        }
    }

    std::cout << ">>> Regime B Finished! Upright Ratio: " 
              << (balanced_ticks * 100 / steps) << "% (" << balanced_ticks << "/" << steps << ")\n\n";
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-506: MULTI-SCALE COMPLEX CELLULAR ORGANISM BENCHMARK ===\n";
    std::cout << "=== 128 Atoms/Ions -> 32 Molecules -> 2 Deep Cells -> Tissue -> KARYON ===\n";
    std::cout << "=== With 8 Sub-Ticks of Temporal Physical Relaxation ===\n";
    std::cout << "====================================================================\n\n";

    MultiScaleConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.num_quarks = 64;
    cfg.spatial_dim = 16;
    cfg.max_hadrons = 64;
    cfg.max_atoms = 128;       // 128 Atoms & Free Ions
    cfg.max_molecules = 32;    // 32 Molecular Species
    cfg.max_cells = 2;         // 2 Deep Complex Cells
    cfg.max_tissues = 1;       // 1 Neuromuscular Tissue
    cfg.relaxation_substeps = 8; // 8 Physical relaxation sub-steps
    cfg.dt = 0.04f;

    MultiScaleComplexOrganism karyon_multiscale(cfg);

    // 1. Pass 1: Test Orbit Pursuit
    std::cout << "--- PHASE 1: ORBIT PURSUIT (RELAXATION DYNAMICS & ORGANISM GROWTH) ---\n";
    run_multiscale_karyon_pursuit(karyon_multiscale, 10000, false);

    // 2. Regime B: Inverted Pendulum
    std::cout << "--- PHASE 2: INVERTED PENDULUM (EXCITATION-CONTRACTION ADAPTATION) ---\n";
    run_multiscale_karyon_pendulum(karyon_multiscale, 15000);

    // 3. Pass 2: Re-test Orbit Pursuit
    std::cout << "--- PHASE 3: ORBIT PURSUIT RE-TEST (SUB-SCALE MEMORY RETENTION) ---\n";
    run_multiscale_karyon_pursuit(karyon_multiscale, 10000, true);

    return 0;
}
