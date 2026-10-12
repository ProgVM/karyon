// experiments/exp_487_laminar_morphogenesis_speedup.cpp
// EXP-487: Laminar Field Compression & AVX2-FMA Plasticity Verification
// Standard: KEP v20.0 Sovereign Master (Principles 2, 5, 6, 27)

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=== EXP-487: Laminar Field Compression & High-Speed Morphogenesis (Core v7.3) ===\n";
    std::cout << "Sensory Dim: 256 (16x16 Raw Pixels) | Latent Dim: 32 (Compressed Substrate)\n";
    std::cout << "Plasticity Engine: 100% AVX2-FMA SIMD (_mm256_fmadd_ps)\n";
    std::cout << "Sprouting Metric: Endogenous Free Energy Stress\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 256;
    cfg.latent_dim = 32;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 3;
    cfg.plasticity_rate = 0.02f;
    cfg.sprouting_fe_threshold = 0.08f;
    cfg.sprouting_cooldown_ticks = 1500;

    karyon::KaryonSovereignCore core(cfg);

    size_t total_cycles = 20000;
    std::vector<float> sensory_curr(256, 0.0f);
    std::vector<float> sensory_target(256, 0.0f);
    std::vector<float> sensory_next(256, 0.0f);
    std::vector<float> action(4, 0.0f);

    size_t initial_count = core.get_primitives_count();
    size_t total_sprouts = 0;
    float cumulative_fe = 0.0f;

    // Structured coherent moving optical field (not pure noise!)
    float agent_pos_x = 8.0f;
    float agent_pos_y = 8.0f;
    float target_pos_x = 12.0f;
    float target_pos_y = 12.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 1; t <= total_cycles; ++t) {
        // Move target smoothly
        target_pos_x = 8.0f + 5.0f * std::cos(static_cast<float>(t) * 0.005f);
        target_pos_y = 8.0f + 5.0f * std::sin(static_cast<float>(t) * 0.005f);

        // Render continuous optical sensory fields (16x16 gaussian blobs)
        for (size_t y = 0; y < 16; ++y) {
            for (size_t x = 0; x < 16; ++x) {
                size_t idx = y * 16 + x;
                float d_agent = (x - agent_pos_x)*(x - agent_pos_x) + (y - agent_pos_y)*(y - agent_pos_y);
                float d_tgt = (x - target_pos_x)*(x - target_pos_x) + (y - target_pos_y)*(y - target_pos_y);
                sensory_curr[idx] = std::exp(-d_agent * 0.5f);
                sensory_target[idx] = std::exp(-d_tgt * 0.5f);
            }
        }

        // 1. Analytical adjoint action decision
        core.decide_continuous_action(sensory_curr, sensory_target, action);

        // Apply physical movement
        agent_pos_x += action[0] * 0.15f;
        agent_pos_y += action[1] * 0.15f;
        agent_pos_x = std::clamp(agent_pos_x, 1.0f, 14.0f);
        agent_pos_y = std::clamp(agent_pos_y, 1.0f, 14.0f);

        // Render next observed sensory state
        for (size_t y = 0; y < 16; ++y) {
            for (size_t x = 0; x < 16; ++x) {
                size_t idx = y * 16 + x;
                float d_agent = (x - agent_pos_x)*(x - agent_pos_x) + (y - agent_pos_y)*(y - agent_pos_y);
                sensory_next[idx] = std::exp(-d_agent * 0.5f);
            }
        }

        // 2. Closed-loop step through Laminar Core v7.3
        auto telem = core.step_environment(sensory_curr, sensory_target, sensory_next, action);
        cumulative_fe += telem.free_energy;

        if (telem.event_sprouted) {
            total_sprouts++;
            std::cout << "[⚡ SPORE SPROUT] Tick " << std::setw(5) << t
                      << " | FE: " << std::fixed << std::setprecision(5) << telem.free_energy
                      << " | NA: " << std::fixed << std::setprecision(3) << telem.noradrenaline_arousal
                      << " | Active Super-Primitives: " << telem.active_primitives_count << "\n";
        }

        if (t % 5000 == 0 || t == total_cycles) {
            float avg_fe = cumulative_fe / 5000.0f;
            std::cout << "Cycle " << std::setw(5) << t << "/" << total_cycles
                      << " | Active Primitives: " << core.get_primitives_count()
                      << " | Window FE: " << std::fixed << std::setprecision(5) << avg_fe
                      << " | NA: " << std::fixed << std::setprecision(3) << core.get_noradrenaline()
                      << " | Epigenetic Lock: " << std::fixed << std::setprecision(3) << core.get_average_epigenetic_lock()
                      << "\n";
            cumulative_fe = 0.0f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(total_cycles) / (elapsed_sec + 1e-6);

    std::cout << "\n=== FINAL TELEMETRY (EXP-487 LAMINAR CORE v7.3) ===\n";
    std::cout << "Initial Primitives:       " << initial_count << "\n";
    std::cout << "Final Evolved Primitives: " << core.get_primitives_count() << "\n";
    std::cout << "Total Endogenous Sprouts: " << total_sprouts << "\n";
    std::cout << "Throughput:               " << std::fixed << std::setprecision(1) << throughput << " Cycles/Sec\n";
    std::cout << "Elapsed Time:             " << std::fixed << std::setprecision(3) << elapsed_sec << " s\n";

    return 0;
}
