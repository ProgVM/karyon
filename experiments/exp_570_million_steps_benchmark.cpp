// ============================================================================
// EXP-570: 1,000,000 Step Deep Physics Training & Generation Benchmark
// Standard: KEP v30.0 (Principles 28, 29, 30)
// ============================================================================

#include "karyon_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

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
    std::cout << "=== EXP-570: 1,000,000 STEP DEEP SOVEREIGN TEXT TRAINING ===\n";
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

    const size_t total_steps = 1000000;
    const size_t checkpoint_interval = 100000;
    const float lr = 0.05f;

    size_t hits = 0;
    size_t window_hits = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t_last = t0;

    std::cout << "\n>>> Starting 1M Step Continuous Physical Learning...\n";
    std::cout << "Step\t\tHits%\tWindowHits%\tFreeEnergy\tModes\tBeta\tSteps/s\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (size_t step = 1; step <= total_steps; ++step) {
        uint8_t obs = env.observe();
        size_t motor_action = core.perceive_and_decide(obs, 0.01f);
        auto env_res = env.step(static_cast<uint8_t>(motor_action));
        core.assimilate_experience(obs, env_res.valence, env_res.magnitude, static_cast<int>(env_res.expected_target), lr);

        if (motor_action == env_res.expected_target) {
            hits++;
            window_hits++;
        }

        if (step % checkpoint_interval == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_window = std::chrono::duration<double>(now - t_last).count();
            double steps_sec = checkpoint_interval / elapsed_window;
            t_last = now;

            float overall_pct = (static_cast<float>(hits) / static_cast<float>(step)) * 100.0f;
            float window_pct = (static_cast<float>(window_hits) / static_cast<float>(checkpoint_interval)) * 100.0f;
            window_hits = 0;

            auto telem = core.get_telemetry();
            std::cout << std::setw(8) << step << "\t"
                      << std::fixed << std::setprecision(2) << overall_pct << "%\t"
                      << std::fixed << std::setprecision(2) << window_pct << "%\t\t"
                      << std::fixed << std::setprecision(4) << telem.free_energy << "\t\t"
                      << telem.active_modes << "\t"
                      << std::fixed << std::setprecision(2) << telem.endogenous_beta << "\t"
                      << std::fixed << std::setprecision(0) << steps_sec << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << ">>> 1M Steps Finished in " << total_time << " s (" << (total_steps / total_time) << " steps/s).\n";

    // --- Phase 2: Generation Inspection ---
    std::cout << "\n============================================================\n";
    std::cout << "=== PHASE 2: AUTONOMOUS GENERATION AFTER 1M STEPS ===\n";
    std::cout << "============================================================\n";

    std::vector<std::string> prompts = {
        "The capital of France is ",
        "Artificial intelligence is ",
        "Once upon a time "
    };

    for (size_t p_idx = 0; p_idx < prompts.size(); ++p_idx) {
        const std::string& prompt = prompts[p_idx];
        std::cout << "\n--- Prompt [" << p_idx + 1 << "]: \"" << prompt << "\" ---\nGenerated: ";

        uint8_t last_b = 0;
        for (char ch : prompt) {
            uint8_t b = static_cast<uint8_t>(ch);
            core.perceive_and_decide(b, 0.01f);
            last_b = b;
            std::cout << ch;
        }

        for (size_t g = 0; g < 150; ++g) {
            size_t next_b = core.perceive_and_decide(last_b, 0.01f);
            char c = static_cast<char>(next_b);
            // Print printable ASCII or escape non-printable
            if (c >= 32 && c <= 126) {
                std::cout << c;
            } else if (c == '\n') {
                std::cout << '\n';
            } else {
                std::cout << '.';
            }
            last_b = static_cast<uint8_t>(next_b);
        }
        std::cout << "\n------------------------------------------------------------\n";
    }

    return 0;
}
