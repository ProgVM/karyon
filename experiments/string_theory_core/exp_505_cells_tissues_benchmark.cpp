// ============================================================================
// EXP-505: Complete Hierarchical Physics Core: Cells & Tissues Organism
// Karyon-CoRE v28.0 Sovereign Multi-Regime Evaluation
// Testing: Photons -> Quarks -> Hadrons -> Atoms/Ions -> Molecules -> 
//          Single Cells -> Tissue Aggregates -> Living KARYON Organism
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/discrete_cells_and_tissues_organism.hpp"

using namespace karyon;

// 1. Regime A: Continuous Orbit Pursuit on Complete KARYON Organism
void run_complete_karyon_pursuit(CompleteKaryonOrganism& organism, size_t steps = 10000, bool is_pass2 = false) {
    std::cout << ">>> Running Complete KARYON Organism Regime A: Orbit Pursuit (steps=" << steps 
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
            angle += 1.2f; // Target jumps
        }

        float valence = caught ? 1.0f : -0.1f * dist;

        auto state = organism.perceive_and_act(curr_s, goal_s, valence);
        total_fe += state.free_energy;
        total_decays += state.meson_decays;

        // Apply efferent motor actions driven by Muscle Fascicle Tissue
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
                      << " | Cells: " << state.total_discrete_cells
                      << " | Tissues: " << state.total_tissue_aggregates
                      << " | Coherence: " << state.tissue_coherence_factor
                      << " | V_mem: " << state.avg_membrane_potential << "mV"
                      << " | Organism Mass: " << state.total_organism_mass
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

// 2. Regime B: Inverted Pendulum Balancing on Complete KARYON Organism
void run_complete_karyon_pendulum(CompleteKaryonOrganism& organism, size_t steps = 15000) {
    std::cout << ">>> Running Complete KARYON Organism Regime B: Inverted Pendulum (steps=" << steps << ")...\n";
    const size_t S = 256;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);

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

        auto state = organism.perceive_and_act(curr_s, goal_s, valence);

        // Pendulum physical step driven by Muscle Fascicle Tissue Organ
        float torque = 12.0f * (state.motor_efferent_actions[0] - state.motor_efferent_actions[1]);
        float alpha = (g / L) * std::sin(theta) + torque - 0.2f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        if (t % 3750 == 0 || t == steps - 1) {
            std::cout << "  Step " << t << " | Theta: " << theta 
                      << " | Upright: " << (balanced_ticks * 100 / (t + 1)) << "%"
                      << " | Atoms: " << state.total_atoms
                      << " | Mol: " << state.total_molecules
                      << " | Cells: " << state.total_discrete_cells
                      << " | Tissues: " << state.total_tissue_aggregates
                      << " | Coherence: " << state.tissue_coherence_factor
                      << " | FE: " << state.free_energy << "\n";
        }
    }

    std::cout << ">>> Regime B Finished! Upright Ratio: " 
              << (balanced_ticks * 100 / steps) << "% (" << balanced_ticks << "/" << steps << ")\n\n";
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-505: COMPLETE HIERARCHICAL PHYSICS CORE BENCHMARK ===\n";
    std::cout << "=== Photons -> Quarks -> Hadrons -> Atoms/Ions -> Molecules -> ===\n";
    std::cout << "=== Single Cells -> Tissue Aggregates -> Living KARYON Organism ===\n";
    std::cout << "====================================================================\n\n";

    OrganismConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.num_quarks = 32;
    cfg.spatial_dim = 16;
    cfg.max_hadrons = 32;
    cfg.max_atoms = 16;
    cfg.max_molecules = 8;
    cfg.max_cells = 12;
    cfg.max_tissues = 3;
    cfg.dt = 0.04f;

    CompleteKaryonOrganism karyon_organism(cfg);

    // 1. Pass 1: Test Orbit Pursuit
    std::cout << "--- PHASE 1: ORBIT PURSUIT (CELLULAR & TISSUE NUCLEOSYNTHESIS) ---\n";
    run_complete_karyon_pursuit(karyon_organism, 10000, false);

    // 2. Regime B: Inverted Pendulum
    std::cout << "--- PHASE 2: INVERTED PENDULUM (TISSUE COHERENCE ADAPTATION) ---\n";
    run_complete_karyon_pendulum(karyon_organism, 15000);

    // 3. Pass 2: Re-test Orbit Pursuit
    std::cout << "--- PHASE 3: ORBIT PURSUIT RE-TEST (ORGANISM LIFE-CYCLE MEMORY) ---\n";
    run_complete_karyon_pursuit(karyon_organism, 10000, true);

    return 0;
}
