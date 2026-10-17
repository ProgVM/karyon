// ============================================================================
// EXP-563: Long-Horizon Endogenous Training & Memory Scaling Benchmark
// Standard: KEP v30.0 (Principles 28, 29, 30)
// ============================================================================

#include "karyon_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

class InteractiveDynamicEnvironment {
public:
    InteractiveDynamicEnvironment(const std::vector<uint8_t>& corpus) 
        : corpus_(corpus), cursor_(0), state_entropy_(1.0f) {}

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
        float valence = 0.0f;
        float magnitude = 1.0f;

        if (action == expected) {
            valence = 1.0f;
            state_entropy_ = std::max(0.1f, state_entropy_ * 0.999f);
            cursor_ = (cursor_ + 1) % corpus_.size();
        } else {
            valence = -1.0f;
            state_entropy_ = std::min(2.0f, state_entropy_ * 1.002f);
            cursor_ = (cursor_ + 1) % corpus_.size();
        }

        return StepResult{valence, magnitude, expected};
    }

private:
    const std::vector<uint8_t>& corpus_;
    size_t cursor_;
    float state_entropy_;
};

int main() {
    std::cout << "============================================================\n";
    std::cout << "=== EXP-563: ENDOGENOUS MEMORY SCALING LONG-HORIZON RUN ===\n";
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

    std::cout << "Interactive Corpus: " << bytes.size() << " bytes loaded.\n";

    karyon::SovereignFullPlasticityOrganism core;
    InteractiveDynamicEnvironment env(bytes);

    // Extended 500,000 Step Interactive Relaxation Run
    const size_t interaction_steps = 500000;
    const float lr = 0.05f;

    size_t successes = 0;
    size_t last_20k_successes = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < interaction_steps; ++step) {
        uint8_t obs = env.observe();
        size_t motor_action = core.perceive_and_decide(obs, 0.01f, 0.0f);

        auto env_res = env.step(static_cast<uint8_t>(motor_action));

        if (env_res.valence > 0.0f) {
            successes++;
            if (step >= interaction_steps - 20000) last_20k_successes++;
        }

        core.assimilate_experience(obs, env_res.valence, env_res.magnitude, static_cast<int>(env_res.expected_target), lr);

        if (step % 50000 == 0 && step > 0) {
            auto telem = core.get_telemetry();
            std::cout << "  Step " << step 
                      << " | Hit Rate: " << (static_cast<double>(successes) / step) * 100.0 << "%"
                      << " | Modes: " << telem.active_modes
                      << " | Phase Dim: " << telem.phase_dim
                      << " | Hippocampal Gain: " << telem.hippocampal_gain
                      << " | Phase Traces: " << telem.stored_phase_traces
                      << " | Free Energy: " << telem.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    auto final_telem = core.get_telemetry();

    std::cout << "\n============================================================\n";
    std::cout << "=== INTERACTIVE RUN COMPLETED (EXP-563) ===\n";
    std::cout << "Elapsed Time:            " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Throughput:              " << (interaction_steps / elapsed_s) << " interactions/sec\n";
    std::cout << "Final Active Modes:      " << final_telem.active_modes << " (Phase Dim: " << final_telem.phase_dim << ")\n";
    std::cout << "Hippocampal Memory Gain: " << final_telem.hippocampal_gain << " (Endogenous)\n";
    std::cout << "Phase Memory Traces:     " << final_telem.stored_phase_traces << " items stored\n";
    std::cout << "Latest Resonance:        " << final_telem.phase_retrieval_resonance << "\n";
    std::cout << "Self-Synthesized Clock:   " << final_telem.dt_eff << " s\n";
    std::cout << "Self-Synthesized Refr:    " << final_telem.refractory_strength << "\n";
    std::cout << "Final 20k Accuracy:      " << (static_cast<double>(last_20k_successes) / 20000.0) * 100.0 << "%\n";
    std::cout << "Final Free Energy:       " << final_telem.free_energy << "\n";

    std::cout << "\n============================================================\n";
    return 0;
}
