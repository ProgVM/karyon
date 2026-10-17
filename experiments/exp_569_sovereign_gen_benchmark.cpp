// ============================================================================
// EXP-569: Sovereign Pure Physics Autonomous Text Generation Benchmark
// Standard: KEP v30.0 (Principles 28, 29, 30)
// ============================================================================

#include "karyon_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

class InteractiveDynamicEnvironment {
public:
    InteractiveDynamicEnvironment(const std::vector<uint8_t>& corpus) 
        : corpus_(corpus), cursor_(0) {}

    uint8_t observe() const noexcept {
        return corpus_[cursor_ % corpus_.size()];
    }

    struct StepResult {
        float valence;
        float magnitude;
        uint8_t expected_target;
    };

    StepResult step(uint8_t action) noexcept {
        uint8_t expected = corpus_[(cursor_ + 1) % corpus_.size()];
        float valence = (action == expected) ? 1.0f : -1.0f;
        cursor_ = (cursor_ + 1) % corpus_.size();
        return StepResult{valence, 1.0f, expected};
    }

private:
    const std::vector<uint8_t>& corpus_;
    size_t cursor_;
};

int main() {
    std::cout << "============================================================\n";
    std::cout << "=== EXP-569: SOVEREIGN PURE PHYSICS TEXT GENERATION ===\n";
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

    karyon::SovereignFullPlasticityOrganism core;
    InteractiveDynamicEnvironment env(bytes);

    const size_t train_steps = 200000;
    const float lr = 0.05f;

    std::cout << "\n>>> Phase 1: Pure Physical Training for " << train_steps << " steps...\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t obs = env.observe();
        size_t motor_action = core.perceive_and_decide(obs, 0.01f);
        auto env_res = env.step(static_cast<uint8_t>(motor_action));
        core.assimilate_experience(obs, env_res.valence, env_res.magnitude, static_cast<int>(env_res.expected_target), lr);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double train_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << ">>> Phase 1 Finished in " << train_time << " s (" << (train_steps / train_time) << " steps/s).\n";

    auto telem = core.get_telemetry();
    std::cout << "  Active Modes: " << telem.active_modes << " (Phase Dim: " << telem.phase_dim << ")\n";
    std::cout << "  Endogenous Beta (Inverse Temp): " << telem.endogenous_beta << "\n";
    std::cout << "  Hippocampal Gain: " << telem.hippocampal_gain << "\n";
    std::cout << "  Stored Traces: " << telem.stored_phase_traces << "\n";
    std::cout << "  Internal Free Energy: " << telem.free_energy << "\n";

    // --- Phase 2: Autonomous Text Generation ---
    std::cout << "\n============================================================\n";
    std::cout << "=== PHASE 2: SOVEREIGN AUTONOMOUS TEXT GENERATION ===\n";
    std::cout << "============================================================\n";

    std::vector<std::string> prompts = {
        "Instruction: What is the capital of France?\nOutput:",
        "Instruction: Write a short sentence about artificial intelligence.\nOutput:",
        "The quick brown fox jumps over the lazy dog."
    };

    for (size_t p_idx = 0; p_idx < prompts.size(); ++p_idx) {
        const std::string& prompt = prompts[p_idx];
        std::cout << "\n--- Prompt [" << p_idx + 1 << "]: \"" << prompt << "\" ---\n";
        std::cout << "Generated Output: ";

        uint8_t last_b = 0;
        for (char ch : prompt) {
            uint8_t b = static_cast<uint8_t>(ch);
            core.perceive_and_decide(b, 0.01f);
            last_b = b;
            std::cout << ch;
        }

        // Generate 200 bytes continuously using 100% endogenous physics
        for (size_t g = 0; g < 200; ++g) {
            size_t next_b = core.perceive_and_decide(last_b, 0.01f);
            char c = static_cast<char>(next_b);
            std::cout << c << std::flush;
            last_b = static_cast<uint8_t>(next_b);
        }
        std::cout << "\n------------------------------------------------------------\n";
    }

    std::cout << "\n============================================================\n";
    return 0;
}
