// ============================================================================
// EXP-561: Unified Tri-Vector Sovereign Autopoiesis Benchmark
// Vector 1 (Synaptogenesis J_ij) + Vector 2 (Phase Hippocampus) + Vector 3 (Real-time Interactive Environment)
// ============================================================================

#include "karyon_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

// Interactive Sensorimotor World: State changes based on Karyon's actions
class InteractiveDynamicEnvironment {
public:
    InteractiveDynamicEnvironment(const std::vector<uint8_t>& corpus) 
        : corpus_(corpus), cursor_(0), state_entropy_(1.0f) {}

    // Environment returns current sensory observation
    uint8_t observe() const noexcept {
        return corpus_[cursor_ % corpus_.size()];
    }

    // Step environment with Karyon's motor action
    // Returns: {sensory_feedback_valence, sensory_magnitude, correct_target}
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
            // On success, advance stream naturally
            cursor_ = (cursor_ + 1) % corpus_.size();
        } else {
            valence = -1.0f;
            state_entropy_ = std::min(2.0f, state_entropy_ * 1.002f);
            // On failure, environment provides resistive perturbation
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
    std::cout << "=== EXP-561: TRI-VECTOR SOVEREIGN AUTOPOIESIS BENCHMARK ===\n";
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

    // Substrate v36.0 with all 3 vectors active
    karyon::SovereignTriVectorOrganism core;
    InteractiveDynamicEnvironment env(bytes);

    const size_t interaction_steps = 250000;
    const float lr = 0.03f;

    size_t successes = 0;
    size_t last_20k_successes = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < interaction_steps; ++step) {
        // Sensorimotor Vector 3: Observe current physical state of environment
        uint8_t obs = env.observe();

        // Organism perceives & decides using self-synthesized laws + Synaptogenesis + Hippocampus
        size_t motor_action = core.perceive_and_decide(obs, 0.01f, 0.0f);

        // Environment reacts physically to Karyon's motor action
        auto env_res = env.step(static_cast<uint8_t>(motor_action));

        if (env_res.valence > 0.0f) {
            successes++;
            if (step >= interaction_steps - 20000) last_20k_successes++;
        }

        // Organism assimilates experience (updates J_ij chords, phase memory traces, and laws)
        core.assimilate_experience(env_res.valence, env_res.magnitude, static_cast<int>(env_res.expected_target), lr);

        if (step % 50000 == 0 && step > 0) {
            auto telem = core.get_telemetry();
            std::cout << "  Step " << step 
                      << " | Hit Rate: " << (static_cast<double>(successes) / step) * 100.0 << "%"
                      << " | Modes: " << telem.active_modes
                      << " | Phase Dim: " << telem.phase_dim
                      << " | Synaptic J_ij Density: " << telem.coupling_density
                      << " | Phase Traces: " << telem.stored_phase_traces
                      << " | Retrieval Resonance: " << telem.phase_retrieval_resonance
                      << " | Free Energy: " << telem.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    auto final_telem = core.get_telemetry();

    std::cout << "\n============================================================\n";
    std::cout << "=== INTERACTIVE TRI-VECTOR RUN COMPLETED ===\n";
    std::cout << "Elapsed Time:          " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Throughput:            " << (interaction_steps / elapsed_s) << " interactions/sec\n";
    std::cout << "Final Active Modes:    " << final_telem.active_modes << " (Phase Dim: " << final_telem.phase_dim << ")\n";
    std::cout << "Active Synaptic J_ij:  Mean weight = " << final_telem.mean_coupling_weight << "\n";
    std::cout << "Phase Memory Traces:   " << final_telem.stored_phase_traces << " items stored\n";
    std::cout << "Latest Resonance:      " << final_telem.phase_retrieval_resonance << "\n";
    std::cout << "Self-Synthesized Clock: " << final_telem.dt_eff << " s\n";
    std::cout << "Self-Synthesized Refr:  " << final_telem.refractory_strength << "\n";
    std::cout << "Final 20k Accuracy:    " << (static_cast<double>(last_20k_successes) / 20000.0) * 100.0 << "%\n";
    std::cout << "Final Free Energy:     " << final_telem.free_energy << "\n";

    // GENERATION EXPERIMENT WITH FULL TRI-VECTOR SUBSTRATE
    std::cout << "\n============================================================\n";
    std::cout << "=== CONTINUOUS INTERACTIVE GENERATION (EXP-561) ===\n";
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
