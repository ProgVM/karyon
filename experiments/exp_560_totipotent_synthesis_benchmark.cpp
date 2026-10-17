// ============================================================================
// EXP-560: Sovereign Totipotent Autopoietic Synthesis Benchmark
// ============================================================================

#include "karyon_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

int main() {
    std::cout << "============================================================\n";
    std::cout << "=== EXP-560: SOVEREIGN TOTIPOTENT SYNTHESIS BENCHMARK ===\n";
    std::cout << "============================================================\n";

    std::string path = "data/alpaca_data.json";
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return 1;
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Corpus: " << bytes.size() << " bytes loaded.\n";

    // Totipotent organism: controls ALL its own internal laws!
    karyon::SovereignTotipotentOrganism core;

    const size_t train_cycles = 250000;
    const float lr = 0.03f;

    size_t hits = 0;
    size_t last_20k_hits = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_cycles; ++step) {
        size_t idx = step % (bytes.size() - 1);
        uint8_t in_b = bytes[idx];
        uint8_t target_b = bytes[idx + 1];

        // 1. Organism perceives input using its self-synthesized internal clock & laws
        size_t action = core.perceive_and_decide(in_b, 0.01f, 0.0f);

        // 2. Environment signals honest feedback
        if (action == target_b) {
            hits++;
            if (step >= train_cycles - 20000) last_20k_hits++;
            core.assimilate_experience(1.0f, 1.0f, static_cast<int>(target_b), lr);
        } else {
            core.assimilate_experience(-1.0f, 1.0f, static_cast<int>(target_b), lr);
        }

        if (step % 50000 == 0 && step > 0) {
            auto telem = core.get_telemetry();
            std::cout << "  Step " << step 
                      << " | Hit Rate: " << (static_cast<double>(hits) / step) * 100.0 << "%"
                      << " | Modes: " << telem.active_modes
                      << " | Self-dt: " << telem.dt_eff
                      << " | Self-Refractory: " << telem.refractory_strength
                      << " | Self-Rollout Depth: " << telem.rollout_depth
                      << " | Curvature g_ij: " << telem.metric_curvature
                      << " | Free Energy: " << telem.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    auto final_telem = core.get_telemetry();

    std::cout << "\n============================================================\n";
    std::cout << "=== TEACHING SESSION COMPLETED (TOTIPOTENT AUTOPOIESIS) ===\n";
    std::cout << "Elapsed Time:          " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Throughput:            " << (train_cycles / elapsed_s) << " interactions/sec\n";
    std::cout << "Final Active Modes:    " << final_telem.active_modes << " (Phase Dim: " << final_telem.phase_dim << ")\n";
    std::cout << "Self-Synthesized Clock: " << final_telem.dt_eff << " s\n";
    std::cout << "Self-Synthesized Refr:  " << final_telem.refractory_strength << " (Recovery: " << final_telem.refractory_recovery << ")\n";
    std::cout << "Self-Synthesized Curv:  " << final_telem.metric_curvature << "\n";
    std::cout << "Self-Synthesized Sandb: " << final_telem.rollout_depth << " steps\n";
    std::cout << "Self-Synthesized Horiz: " << final_telem.avg_horizon << "\n";
    std::cout << "Final 20k Accuracy:    " << (static_cast<double>(last_20k_hits) / 20000.0) * 100.0 << "%\n";
    std::cout << "Final Free Energy:     " << final_telem.free_energy << "\n";

    // GENERATION EXPERIMENT WITH SELF-SYNTHESIZED LAWS
    std::cout << "\n============================================================\n";
    std::cout << "=== AUTONOMOUS TEXT GENERATION BENCHMARK (EXP-560) ===\n";
    std::cout << "============================================================\n";

    std::vector<std::string> prompts = {
        "The ",
        "Instruction: Give a definition of machine learning.\nOutput: Machine learning is ",
        "Python is a "
    };

    for (const auto& prompt : prompts) {
        std::cout << "\n--- PROMPT: \"" << prompt << "\" ---\n";
        std::cout << prompt;

        uint8_t last_b = prompt.front();
        for (char ch : prompt) {
            uint8_t b = static_cast<uint8_t>(ch);
            core.perceive_and_decide(b, 0.01f, 0.0f);
            last_b = b;
        }

        // Generate 150 bytes
        for (size_t g = 0; g < 150; ++g) {
            size_t next_b = core.perceive_and_decide(last_b, 0.01f, 0.20f);
            char c = static_cast<char>(next_b);
            std::cout << c << std::flush;
            last_b = static_cast<uint8_t>(next_b);
        }
        std::cout << "\n";
    }

    std::cout << "\n============================================================\n";
    return 0;
}
