// experiments/exp_486_autopoietic_morphogenesis_sprouting.cpp
// EXP-486: Autopoietic In-Flight Morphogenesis & Net2Net Smooth Sprouting
// Standard: KEP v20.0 Sovereign Master (Principles 2, 5, 6, 27)

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=== EXP-486: Autopoietic In-Flight Morphogenesis & Smooth Sprouting (Core v7.2) ===\n";
    std::cout << "Substrate: Karyon Native Core v7.2 Sovereign\n";
    std::cout << "Initial Primitives: 2 | Sovereign Ceiling: 6\n";
    std::cout << "Sprouting Metric: Free Energy Stress + Noradrenaline Arousal\n";
    std::cout << "Identity Preservation: Net2Net Zero-Shock (birth_gain = 0.0 -> 1.0)\n\n";

    karyon::CoreConfig cfg;
    cfg.state_dim = 256;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 3;
    cfg.plasticity_rate = 0.02f;
    cfg.sprouting_fe_threshold = 0.50f;
    cfg.sprouting_cooldown_ticks = 2000;

    karyon::KaryonSovereignCore core(cfg);

    size_t total_cycles = 15000;
    std::vector<float> state_curr(256, 0.0f);
    std::vector<float> state_target(256, 0.0f);
    std::vector<float> state_next(256, 0.0f);
    std::vector<float> action(4, 0.0f);

    std::mt19937 rng(1337);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    size_t initial_count = core.get_primitives_count();
    size_t total_sprouts = 0;
    float cumulative_fe = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 1; t <= total_cycles; ++t) {
        // Synthesize dynamic continuous environment waves
        float phase = static_cast<float>(t) * 0.02f;
        for (size_t d = 0; d < 256; ++d) {
            state_curr[d] = std::sin(phase + static_cast<float>(d) * 0.05f) + noise(rng);
            state_target[d] = std::sin(phase + 0.1f + static_cast<float>(d) * 0.05f);
            state_next[d] = 0.85f * state_curr[d] + 0.15f * state_target[d] + noise(rng) * 0.1f;
        }

        // 1. Analytical adjoint action decision
        core.decide_continuous_action(state_curr, state_target, action);

        // 2. Continuous environmental closed-loop step
        auto telem = core.step_environment(state_curr, state_target, state_next, action);
        cumulative_fe += telem.free_energy;

        if (telem.event_sprouted) {
            total_sprouts++;
            std::cout << "[⚡ SPORE SPROUT EVENT] Tick " << std::setw(5) << t 
                      << " | High Stress FE: " << std::fixed << std::setprecision(4) << telem.free_energy
                      << " | NA: " << std::fixed << std::setprecision(3) << telem.noradrenaline_arousal
                      << " | New Super-Primitive Added! Total Active: " << telem.active_primitives_count
                      << "\n";
        }

        if (t % 3000 == 0 || t == total_cycles) {
            float avg_fe = cumulative_fe / 3000.0f;
            std::cout << "Cycle " << std::setw(5) << t << "/" << total_cycles
                      << " | Active Primitives: " << core.get_primitives_count()
                      << " | Window FE: " << std::fixed << std::setprecision(4) << avg_fe
                      << " | NA: " << std::fixed << std::setprecision(3) << core.get_noradrenaline()
                      << " | Epigenetic Lock: " << std::fixed << std::setprecision(3) << core.get_average_epigenetic_lock()
                      << "\n";
            cumulative_fe = 0.0f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(total_cycles) / (elapsed_sec + 1e-6);

    std::cout << "\n=== FINAL TELEMETRY (EXP-486 MORPHOGENESIS v7.2) ===\n";
    std::cout << "Initial Primitives Count: " << initial_count << "\n";
    std::cout << "Final Evolved Primitives: " << core.get_primitives_count() << "\n";
    std::cout << "Total Endogenous Sprouts: " << total_sprouts << "\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " Cycles/Sec\n";
    std::cout << "Elapsed Time: " << std::fixed << std::setprecision(3) << elapsed_sec << " s\n";

    return 0;
}
