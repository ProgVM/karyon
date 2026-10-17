// ============================================================================
// EXP-558: Sovereign Dynamic Memory Sprouting & Horizon Adaptation Benchmark
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
    std::cout << "=== EXP-558: AUTONOMOUS MEMORY CAPACITY & HORIZON BENCHMARK ===\n";
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

    // Organism with autonomous dynamic memory
    karyon::SovereignDynamicExperientialContinuum core;

    const float dt = 0.01f;
    const size_t train_cycles = 250000;
    const float lr = 0.03f;

    size_t hits = 0;
    size_t last_20k_hits = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_cycles; ++step) {
        size_t idx = step % (bytes.size() - 1);
        uint8_t in_b = bytes[idx];
        uint8_t target_b = bytes[idx + 1];

        // 1. Organism perceives input and generates action from its self-sized phase context
        size_t action = core.perceive_and_decide(in_b, dt, 0.0f);

        // 2. Environment signals feedback
        if (action == target_b) {
            hits++;
            if (step >= train_cycles - 20000) last_20k_hits++;
            core.assimilate_experience(1.0f, 1.0f, static_cast<int>(target_b), lr);
        } else {
            core.assimilate_experience(-1.0f, 1.0f, static_cast<int>(target_b), lr);
        }

        if (step % 50000 == 0 && step > 0) {
            std::cout << "  Step " << step 
                      << " | Hit Rate: " << (static_cast<double>(hits) / step) * 100.0 << "%"
                      << " | Active Modes: " << core.get_active_modes()
                      << " | Phase Dim: " << core.get_phase_dim()
                      << " | Avg Horizon: " << core.get_avg_memory_horizon()
                      << " | Free Energy: " << core.get_free_energy() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n============================================================\n";
    std::cout << "=== TEACHING SESSION COMPLETED ===\n";
    std::cout << "Elapsed Time:        " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Throughput:          " << (train_cycles / elapsed_s) << " interactions/sec\n";
    std::cout << "Final Active Modes:  " << core.get_active_modes() << " (Initial was 64)\n";
    std::cout << "Final Phase Dim:     " << core.get_phase_dim() << "\n";
    std::cout << "Final Avg Horizon:   " << core.get_avg_memory_horizon() << "\n";
    std::cout << "Final 20k Accuracy:  " << (static_cast<double>(last_20k_hits) / 20000.0) * 100.0 << "%\n";
    std::cout << "Final Free Energy:   " << core.get_free_energy() << "\n";

    // GENERATION EXPERIMENT
    std::cout << "\n============================================================\n";
    std::cout << "=== AUTONOMOUS TEXT GENERATION BENCHMARK ===\n";
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
            core.perceive_and_decide(b, dt, 0.0f);
            last_b = b;
        }

        // Generate 150 bytes with Temperature T=0.25
        for (size_t g = 0; g < 150; ++g) {
            size_t next_b = core.perceive_and_decide(last_b, dt, 0.25f);
            char c = static_cast<char>(next_b);
            std::cout << c << std::flush;
            last_b = static_cast<uint8_t>(next_b);
        }
        std::cout << "\n";
    }

    std::cout << "\n============================================================\n";
    return 0;
}
