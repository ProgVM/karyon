// ============================================================================
// EXP-556: Pure Teacher-Student Experiential Dialogue
// Standard: KEP v28.0 (Principles 1, 2, 5, 27, 28, 29)
//
// Role of this benchmark:
// 1. The Trainer is purely an external teacher / environment.
// 2. The Trainer NEVER modifies weights, computes gradients, or touches Karyon internals!
// 3. The Trainer only presents input and provides evaluative feedback:
//    - "CORRECT (+1.0)" when Karyon's self-directed prediction hits the target.
//    - "INCORRECT (-1.0)" when it makes an error, accompanied by a demonstration of truth.
// 4. Karyon autonomously assimilates this experiential feedback into its continuous
//    Hamiltonian wavefield and tests autonomous text generation.
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
    std::cout << "=== EXP-556: PURE EXPERIENTIAL TEACHER-STUDENT BENCHMARK ===\n";
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

    // Autonomous Experiential Organism
    karyon::SovereignExperientialCore karyon;

    const float dt = 0.01f;
    const size_t train_cycles = 200000;

    std::array<float, karyon::CHANNEL_IO_DIM> sensory_in{};

    size_t hits = 0;
    size_t last_10k_hits = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_cycles; ++step) {
        size_t idx = step % (bytes.size() - 1);
        uint8_t current_byte = bytes[idx];
        uint8_t target_next_byte = bytes[idx + 1];

        // 1. Teacher presents current byte as sensory input
        sensory_in.fill(0.0f);
        sensory_in[current_byte] = 1.0f;

        // 2. Karyon perceives, propagates internal field, and independently decides
        size_t decision = karyon.perceive_and_decide(sensory_in.data(), sensory_in.size(), dt);

        // 3. Teacher evaluates Karyon's decision (HONEST ENVIRONMENTAL FEEDBACK)
        // Trainer DOES NOT touch Karyon's brain. Trainer only speaks feedback:
        if (decision == target_next_byte) {
            // "Good job! Correct!"
            hits++;
            if (step >= train_cycles - 10000) last_10k_hits++;
            karyon.assimilate_experience(1.0f, 1.0f, static_cast<int>(target_next_byte));
        } else {
            // "Wrong! That was an error. The right answer was target_next_byte."
            karyon.assimilate_experience(-1.0f, 1.0f, static_cast<int>(target_next_byte));
        }

        if (step % 50000 == 0 && step > 0) {
            std::cout << "  Step " << step 
                      << " | Cumulative Hit Rate: " << (static_cast<double>(hits) / step) * 100.0 << "%"
                      << " | Positive Exp: " << karyon.get_positive_experience()
                      << " | Negative Exp: " << karyon.get_negative_experience()
                      << " | Internal Free Energy: " << karyon.get_free_energy() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n============================================================\n";
    std::cout << "=== TEACHING SESSION COMPLETED ===\n";
    std::cout << "Elapsed Time:               " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Teaching Throughput:        " << (train_cycles / elapsed_s) << " interactions/sec\n";
    std::cout << "Total Positive Feedback:    " << karyon.get_positive_experience() << "\n";
    std::cout << "Total Negative Feedback:    " << karyon.get_negative_experience() << "\n";
    std::cout << "Final 10k Accuracy:         " << (static_cast<double>(last_10k_hits) / 10000.0) * 100.0 << "%\n";
    std::cout << "Final Internal Free Energy: " << karyon.get_free_energy() << "\n";

    // AUTONOMOUS GENERATION
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

        // Condition Karyon with prompt bytes
        uint8_t last_b = prompt.front();
        for (char ch : prompt) {
            uint8_t b = static_cast<uint8_t>(ch);
            sensory_in.fill(0.0f);
            sensory_in[b] = 1.0f;
            karyon.perceive_and_decide(sensory_in.data(), sensory_in.size(), dt);
            last_b = b;
        }

        // Generate 150 bytes autonomously
        for (size_t g = 0; g < 150; ++g) {
            sensory_in.fill(0.0f);
            sensory_in[last_b] = 1.0f;
            size_t next_decision = karyon.perceive_and_decide(sensory_in.data(), sensory_in.size(), dt);

            char c = static_cast<char>(next_decision);
            std::cout << c << std::flush;
            last_b = static_cast<uint8_t>(next_decision);
        }
        std::cout << "\n";
    }

    std::cout << "\n============================================================\n";
    return 0;
}
