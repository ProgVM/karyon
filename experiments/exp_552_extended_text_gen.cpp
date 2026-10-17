// ============================================================================
// EXP-552: Extended Autopoietic Evolution & Text Generation Experiment
// 
// Objectives:
// 1. Give the core massive temporal relaxation (1,000,000+ training cycles over Alpaca corpus).
// 2. Allow the wavefield to expand its mode spectrum to maximum capacity (N_modes -> 256).
// 3. Shape continuous Hopfield attractor basins and Fourier projections online.
// 4. Test closed-loop continuous generation: prompt the core with "Instruction: "
//    and observe autonomous byte generation.
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
    std::cout << "=== EXP-552: EXTENDED AUTOPOIETIC TEXT GENERATION EXPERIMENT ===\n";
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

    std::cout << "Loaded " << bytes.size() << " bytes from Alpaca dataset.\n";

    karyon::SovereignContinuumCore core;
    const float dt = 0.01f;
    const size_t total_train_steps = 1000000; // 1 MILLION relaxation steps!

    std::cout << "Training wavefield core over " << total_train_steps << " steps to allow full autopoietic growth...\n";

    double cum_loss = 0.0;
    double last_100k_loss = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < total_train_steps; ++step) {
        size_t idx = step % (bytes.size() - 1);
        uint8_t curr = bytes[idx];
        uint8_t target = bytes[idx + 1];

        float x = (static_cast<float>(curr) - 128.0f) / 128.0f;
        float prev_x = (idx > 0) ? (static_cast<float>(bytes[idx - 1]) - 128.0f) / 128.0f : 0.0f;
        float x_dot = (x - prev_x) / dt;

        // Step continuum
        core.perceive_and_act(x, x_dot, dt);

        // Train online byte predictor
        float lr = 0.01f;
        uint8_t pred = core.predict_and_train_byte(target, lr);

        // Approximate loss
        float err = std::abs(static_cast<float>(pred) - static_cast<float>(target)) / 256.0f;
        float nats = 0.5f * std::log(2.0f * M_PI * 16.0f) + (err * 256.0f * err * 256.0f) / (2.0f * 16.0f);
        nats = std::clamp(nats, 0.01f, 5.545f);

        cum_loss += nats;
        if (step >= total_train_steps - 100000) last_100k_loss += nats;

        if (step % 200000 == 0 && step > 0) {
            std::cout << "  Step " << step << " | Active Modes: " << core.get_active_modes()
                      << " | Avg Loss: " << cum_loss / step << " nats\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n============================================================\n";
    std::cout << "=== TRAINING COMPLETE ===\n";
    std::cout << "Elapsed Time:               " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Training Throughput:        " << (total_train_steps / elapsed_s) << " steps/sec\n";
    std::cout << "Initial Active Modes:       16\n";
    std::cout << "Final Active Evolved Modes: " << core.get_active_modes() << " / 256\n";
    std::cout << "Overall Mean Loss:          " << cum_loss / total_train_steps << " nats/byte\n";
    std::cout << "Final 100k Mean Loss:       " << last_100k_loss / 100000.0 << " nats/byte\n";
    std::cout << "Final Perplexity (PPL):     " << std::exp(last_100k_loss / 100000.0) << "\n";

    // GENERATION EXPERIMENT
    std::cout << "\n============================================================\n";
    std::cout << "=== AUTONOMOUS TEXT GENERATION BENCHMARK ===\n";
    std::cout << "============================================================\n";

    std::string prompt = "Instruction: Give a definition of machine learning.";
    std::cout << "Prompt text: \"" << prompt << "\"\n";
    std::cout << "\nGenerating 200 autonomous bytes from continuous wavefield:\n---\n";
    std::cout << prompt;

    // Feed prompt first to condition wavefield
    uint8_t last_byte = prompt.back();
    for (char ch : prompt) {
        uint8_t b = static_cast<uint8_t>(ch);
        float x = (static_cast<float>(b) - 128.0f) / 128.0f;
        core.perceive_and_act(x, 0.0f, dt);
        last_byte = b;
    }

    // Autoregressive Generation
    std::string generated_output = prompt;
    for (size_t gen = 0; gen < 200; ++gen) {
        float x = (static_cast<float>(last_byte) - 128.0f) / 128.0f;
        core.perceive_and_act(x, 0.0f, dt);

        // Predict next byte without updating weights (lr = 0.0)
        uint8_t next_b = core.predict_and_train_byte(0, 0.0f);
        
        // Prevent control char artifacts
        if (next_b < 32 && next_b != 10 && next_b != 13) {
            next_b = 32; // Map to space
        }

        char gen_char = static_cast<char>(next_b);
        std::cout << gen_char << std::flush;
        generated_output += gen_char;
        last_byte = next_b;
    }

    std::cout << "\n---\n\nGeneration Complete.\n";
    return 0;
}
