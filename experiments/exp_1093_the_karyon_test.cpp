// EXP-1093: The Karyon Test — Self-Referential Epistemic Closure & Unbounded Bit Substrate
// Objective:
// 1. Evaluate whether a purely native bit substrate with zero human ceilings can solve 
//    a recursive, self-referential dynamic environment (The Karyon Test).
// 2. Environment dynamics:
//    - Stage A: Deterministic recursive sequences (rule-discovery).
//    - Stage B: Causal Self-Referential Feedback (environment state = f(history, agent_action)).
//    - Stage C: Controlled Phase Transition (sudden law inversion requiring unlearning & restructuring).
// 3. Serialize the evolved substrate into .kcore v41 (Manifest + Kernel + Bit Substrate).
// 4. Dissect and analyze the sprouted bit-couplings topology.

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <random>
#include "../src/karyon_native_substrate.hpp"

// Synthetic Karyon Test Environment: Self-Referential Dynamical Bitstream
struct KaryonTestEnvironment {
    uint64_t env_state;
    uint64_t step_count;
    std::mt19937_64 rng;

    KaryonTestEnvironment(uint64_t seed = 42) : env_state(0x1337BEEFCAFEULL), step_count(0), rng(seed) {}

    uint64_t NextWord(uint64_t agent_predicted_word) {
        step_count++;

        // Self-referential feedback: agent's prediction causally influences environment state
        uint64_t agent_feedback = agent_predicted_word ^ (agent_predicted_word >> 17);
        env_state = (env_state ^ agent_feedback) + 0x9E3779B97F4A7C15ULL;
        env_state = (env_state << 5) | (env_state >> (64 - 5));

        if (step_count < 20000) {
            // Phase 1: Pure Deterministic Non-Linear Rule
            return env_state ^ (env_state >> 11) ^ (env_state << 7);
        } else if (step_count < 40000) {
            // Phase 2: Causal Self-Closure (Target depends directly on recursive feedback)
            return (env_state ^ agent_feedback) * 0xBF58476D1CE4E5B9ULL;
        } else {
            // Phase 3: Dynamic Phase Transition (Law Inversion)
            return ~(env_state ^ (env_state >> 13) ^ agent_feedback);
        }
    }
};

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1093: THE KARYON TEST (UNBOUNDED SUBSTRATE GENESIS)\n";
    std::cout << " Paradigm: Self-Referential Causal Closure & Anti-Static Genesis\n";
    std::cout << " Substrate: Native 64-Bit Substrate (V=2) & Zero Imposed Ceilings\n";
    std::cout << "======================================================================\n\n";

    KaryonNativeSubstrate karyon;
    KaryonTestEnvironment env(1093);

    const size_t TOTAL_WORDS = 60000;
    const uint64_t TOTAL_BITS = TOTAL_WORDS * 64;

    std::cout << " [1] Initializing The Karyon Test Environment:\n";
    std::cout << "     - Total Stream: " << TOTAL_WORDS << " words (" << TOTAL_BITS << " bits)\n";
    std::cout << "     - Phase 1: Non-Linear Rule Discovery (0 - 20k words)\n";
    std::cout << "     - Phase 2: Causal Self-Referential Closure (20k - 40k words)\n";
    std::cout << "     - Phase 3: Sudden Dynamic Phase Transition (40k - 60k words)\n";
    std::cout << " [2] Substrate Memory: 0 Initial Couplings (Unbounded Hardware Growth)\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    uint64_t last_prediction = 0ULL;
    // Pass 1: Active Evolution & Sprouting
    std::cout << " ---> Pass 1 (Autopoietic Learning & Adaptation)...\n";
    karyon.ResetStepCounters();
    env = KaryonTestEnvironment(1093);

    for (size_t i = 0; i < TOTAL_WORDS; ++i) {
        uint64_t target = env.NextWord(last_prediction);
        last_prediction = karyon.Step(target, true);

        if ((i + 1) % 20000 == 0) {
            size_t phase = (i + 1) / 20000;
            std::cout << "      [Phase " << phase << " Checkpoint] Word " << (i + 1) 
                      << " | Current Couplings: " << karyon.bit_couplings.size() 
                      << " | Bit Errors so far: " << karyon.bit_errors << "\n";
        }
    }

    double fe_pass1 = karyon.ComputeFreeEnergy();
    std::cout << "      Pass 1 Final: Bit Errors = " << karyon.bit_errors << " / " << TOTAL_BITS
              << " | Free Energy F = " << std::fixed << std::setprecision(8) << fe_pass1 << " nats"
              << " | Sprouted Couplings = " << karyon.peak_couplings << "\n\n";

    // Pass 2: Attractor Snapping & Equilibrium
    std::cout << " ---> Pass 2 (Attractor Snapping & Relaxation)...\n";
    karyon.ResetStepCounters();
    env = KaryonTestEnvironment(1093);
    last_prediction = 0ULL;

    for (size_t i = 0; i < TOTAL_WORDS; ++i) {
        uint64_t target = env.NextWord(last_prediction);
        last_prediction = karyon.Step(target, true);
    }
    double fe_pass2 = karyon.ComputeFreeEnergy();
    std::cout << "      Pass 2 Final: Bit Errors = " << karyon.bit_errors << " / " << TOTAL_BITS
              << " | Free Energy F = " << std::fixed << std::setprecision(8) << fe_pass2 << " nats"
              << " | Couplings = " << karyon.bit_couplings.size() << "\n\n";

    // Pass 3: The True Karyon Test (Zero-Sprouting / Read-only Evaluation)
    std::cout << " ---> Pass 3 (The Sovereign Evaluation: Read-Only Autopoiesis)...\n";
    karyon.ResetStepCounters();
    env = KaryonTestEnvironment(1093);
    last_prediction = 0ULL;

    for (size_t i = 0; i < TOTAL_WORDS; ++i) {
        uint64_t target = env.NextWord(last_prediction);
        last_prediction = karyon.Step(target, false); // Read-only
    }
    double fe_pass3 = karyon.ComputeFreeEnergy();
    std::cout << "      Pass 3 Final: Bit Errors = " << karyon.bit_errors << " / " << TOTAL_BITS
              << " | Free Energy F = " << std::fixed << std::setprecision(8) << fe_pass3 << " nats\n\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 4. Checkpoint Substrate to .kcore v41
    std::string checkpoint_path = "build/karyon_exp1093_test.kcore";
    std::cout << " [3] Checkpointing Substrate to Container: " << checkpoint_path << "...\n";
    if (karyon.Save(checkpoint_path)) {
        std::ifstream f(checkpoint_path, std::ios::binary | std::ios::ate);
        size_t file_size = f.tellg();
        std::cout << "     Container successfully created! Size on disk: " 
                  << std::fixed << std::setprecision(2) << (double)file_size / (1024.0 * 1024.0) << " MB (" 
                  << file_size << " bytes)\n";
    } else {
        std::cerr << "     Failed to save .kcore container!\n";
    }

    // 5. Substrate Endoscopic Dissection: Inspect Sprouted Couplings
    std::cout << "\n [4] Endoscopic Substrate Dissection (Sampling First 10 Sprouted Couplings):\n";
    size_t count = 0;
    for (const auto &pair : karyon.bit_couplings) {
        std::cout << "     Coupling [" << std::setw(2) << count << "] Signature: 0x" 
                  << std::hex << std::setw(16) << std::setfill('0') << pair.first
                  << " -> Target Word: 0x" << std::setw(16) << std::setfill('0') << pair.second 
                  << std::dec << "\n";
        if (++count >= 10) break;
    }

    std::cout << "\n======================================================================\n";
    std::cout << " KARYON TEST RESULTS SUMMARY:\n";
    std::cout << " - Total Bits Tested     : " << TOTAL_BITS << " bits\n";
    std::cout << " - Evaluation Bit Errors : " << karyon.bit_errors << " bits\n";
    std::cout << " - Final Free Energy F   : " << std::fixed << std::setprecision(8) << fe_pass3 << " nats\n";
    std::cout << " - Sprouted Substrate    : " << karyon.peak_couplings << " couplings\n";
    std::cout << " - Total Time Elapsed    : " << std::fixed << std::setprecision(2) << duration_ms << " ms\n";
    std::cout << "======================================================================\n";

    return 0;
}
