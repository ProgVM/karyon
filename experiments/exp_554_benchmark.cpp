// ============================================================================
// EXP-554: Benchmark True Epistemic Surprise & Real Text Generation
// ============================================================================

#include "experiments/exp_554_orthogonal_epistemic_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

int main() {
    std::cout << "============================================================\n";
    std::cout << "=== EXP-554: ORTHOGONAL EPISTEMIC SURPRISE BENCHMARK ===\n";
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

    std::cout << "Corpus: " << bytes.size() << " bytes.\n";

    karyon::OrthogonalEpistemicCore core;
    const float dt = 0.01f;
    const size_t train_steps = 300000;
    const float lr = 0.05f;

    std::cout << "Training Orthogonal Epistemic Core for " << train_steps << " steps...\n";

    double cum_surprise = 0.0;
    double last_20k_surprise = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        size_t idx = step % (bytes.size() - 1);
        uint8_t in_b = bytes[idx];
        uint8_t target_b = bytes[idx + 1];

        float surprise = core.process_byte_transition(in_b, target_b, dt, lr);
        cum_surprise += surprise;
        if (step >= train_steps - 20000) last_20k_surprise += surprise;

        if (step % 50000 == 0 && step > 0) {
            double current_avg = cum_surprise / step;
            std::cout << "  Step " << step << " | Surprise: " << current_avg
                      << " nats/byte | PPL: " << std::exp(current_avg) << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    double final_surprise = last_20k_surprise / 20000.0;
    std::cout << "\n============================================================\n";
    std::cout << "=== TRAINING COMPLETE ===\n";
    std::cout << "Elapsed Time:         " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Throughput:           " << (train_steps / elapsed_s) << " bytes/sec\n";
    std::cout << "Initial Random PPL:   256.0 (Surprise = 5.545 nats)\n";
    std::cout << "Final Mean Surprise:  " << final_surprise << " nats/byte\n";
    std::cout << "Final Perplexity:     " << std::exp(final_surprise) << "\n";

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
        
        // Prime the wavefield with prompt bytes
        uint8_t last_b = prompt.front();
        for (char ch : prompt) {
            uint8_t b = static_cast<uint8_t>(ch);
            core.process_byte_transition(last_b, b, dt, 0.0f); // Pure perception, no learning
            last_b = b;
        }

        std::cout << prompt;
        for (size_t g = 0; g < 150; ++g) {
            uint8_t next_b = core.sample_next_byte(last_b, dt, 0.5f); // T=0.5
            char c = static_cast<char>(next_b);
            std::cout << c << std::flush;
            last_b = next_b;
        }
        std::cout << "\n";
    }

    std::cout << "\n============================================================\n";
    return 0;
}
