// EXP-1105: Closed-Loop Sovereign Endogenous Agency & Bit-Level Self-Play
// Standards: KEP Principle 0, KEP Principle 3 (Zero Imposed Mechanisms), KEP Principle 11 (Continuous Time Agency & Real-World Coupling), KEP Principle 13 (Anti-Staticity)
//
// Theoretical Formulation:
// The substrate interacts directly with an UNKNOWN dynamic environment (e.g. dynamic state permutation lock & maze attractor).
// Zero supervised dataset! Zero external Q-tables, policy networks, or heuristic trees.
// The perception-action loop:
//   1. Environment emits sensorimotor bit stream S_env (Observation).
//   2. Continuous Bit Substrate ingests S_env, updates internal wavefield, and emits action bit A_action.
//   3. Environment state transitions S_env -> S_env' driven by A_action.
//   4. Variational Free Energy F = HammingDistance(S_env', S_target) measures environment instability.
//   5. Causal credit assignment sprouts bit-taps and attractor states to drive F -> 0.00000000 nats.

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <random>

// Unknown Dynamic Environment: Permutation & State Attractor Engine
class UnknownDynamicEnvironment {
private:
    uint32_t current_state;
    uint32_t target_state;
    uint32_t step_count;
    std::mt19937 rng;

public:
    UnknownDynamicEnvironment(uint32_t seed = 42) : step_count(0), rng(seed) {
        Reset();
    }

    void Reset() {
        current_state = 0b1010101010101010; // Initial noisy state (16 bits)
        target_state  = 0b1100110011001100; // Minimal Free Energy Target State
        step_count = 0;
    }

    // Step environment with substrate's emitted action bit
    // Returns 16-bit sensory observation state
    uint16_t Step(uint8_t action_bit) {
        step_count++;

        // Environment state transition rule (completely unknown to substrate!)
        // Non-linear bitwise permutation dynamics driven by action bit
        if (action_bit == 1) {
            current_state = ((current_state << 1) | (current_state >> 15)) & 0xFFFF;
            current_state ^= 0x0F0F;
        } else {
            current_state = ((current_state >> 1) | (current_state << 15)) & 0xFFFF;
            current_state ^= 0x3333;
        }

        return static_cast<uint16_t>(current_state);
    }

    uint16_t GetTargetState() const { return target_state; }

    // Computes instantaneous environmental Free Energy (Hamming distance to target normalized to nats)
    double ComputeFreeEnergy() const {
        uint32_t diff = current_state ^ target_state;
        uint32_t pop = __builtin_popcount(diff);
        if (pop == 0) return 0.0;
        double err_rate = static_cast<double>(pop) / 16.0;
        return - (err_rate * std::log2(err_rate) + (1.0 - err_rate) * std::log2(1.0 - err_rate));
    }

    bool IsTargetReached() const {
        return (current_state == target_state);
    }
};

// Continuous Bit Substrate Agency Engine
struct SovereignAgentSubstrate {
    std::vector<uint8_t> bit_tape;
    std::vector<size_t> active_taps;
    std::unordered_set<size_t> tap_set;
    std::unordered_map<uint64_t, uint8_t> action_attractors; // Tap signature -> Optimal Action Bit

    uint64_t total_cycles;
    uint64_t total_action_errors;

    SovereignAgentSubstrate() : total_cycles(0), total_action_errors(0) {
        active_taps.push_back(1);
        tap_set.insert(1);
    }

    inline uint64_t ComputeTapSignature(size_t current_pos) const {
        uint64_t sig = 0xCBF29CE484222325ULL;
        for (size_t i = 0; i < active_taps.size(); ++i) {
            size_t tau = active_taps[i];
            if (current_pos >= tau) {
                uint8_t bit_val = bit_tape[current_pos - tau];
                sig = (sig ^ bit_val) * 0x100000001B3ULL;
                sig = (sig << 7) | (sig >> 57);
            }
        }
        return sig;
    }

    void SproutCausalTap(size_t current_pos) {
        if (current_pos < 5) return;
        size_t max_lookback = std::min(current_pos, static_cast<size_t>(256));

        for (size_t tau = 2; tau < max_lookback; ++tau) {
            if (tap_set.find(tau) == tap_set.end()) {
                active_taps.push_back(tau);
                tap_set.insert(tau);
                break; // Sprout one tap per discrepancy event
            }
        }
    }

    // Agent decision step: ingests 16 sensory bits, emits 1 action bit
    uint8_t PerceiveAndAct(uint16_t sensory_obs, double current_fe, bool learn = true) {
        // Append 16 sensory bits to continuous bit tape
        for (int b = 15; b >= 0; --b) {
            bit_tape.push_back((sensory_obs >> b) & 1);
        }

        size_t current_pos = bit_tape.size();
        uint64_t sig = ComputeTapSignature(current_pos);

        uint8_t action_bit = 0;
        auto it = action_attractors.find(sig);
        if (it != action_attractors.end()) {
            action_bit = it->second;
        } else {
            // Exploratory action if unknown state
            action_bit = (current_pos % 2 == 0) ? 1 : 0;
        }

        total_cycles++;

        if (learn && current_fe > 0.0) {
            // If environment is in high Free Energy, update action attractor towards stability vector
            uint8_t exploratory_action = 1 - action_bit; // Try alternate trajectory
            action_attractors[sig] = exploratory_action;
            total_action_errors++;

            if (active_taps.size() < 64) {
                SproutCausalTap(current_pos);
            }
        }

        return action_bit;
    }
};

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "=== EXP-1105: CLOSED-LOOP SOVEREIGN ENDOGENOUS AGENCY (SELF-PLAY) ======" << std::endl;
    std::cout << "=== KEP Principle 11: Real-World Agency & Zero Supervised Datasets ======" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    UnknownDynamicEnvironment env(42);
    SovereignAgentSubstrate agent;

    std::cout << "[ENV INIT] Initial State: 0x" << std::hex << 0xAAAA 
              << " | Target State: 0x" << env.GetTargetState() << std::dec << std::endl;
    std::cout << "Initial Free Energy: " << std::fixed << std::setprecision(8) 
              << env.ComputeFreeEnergy() << " nats" << std::endl;

    // Phase 1: Autonomous Self-Play Interaction Loop (Learning Active)
    const size_t MAX_EPISODE_STEPS = 50000;
    auto t0 = std::chrono::high_resolution_clock::now();

    uint16_t obs = 0b1010101010101010;
    size_t steps_to_convergence = 0;
    bool reached_zero_fe = false;

    for (size_t step = 0; step < MAX_EPISODE_STEPS; ++step) {
        double current_fe = env.ComputeFreeEnergy();
        if (current_fe == 0.0 && !reached_zero_fe) {
            steps_to_convergence = step;
            reached_zero_fe = true;
        }

        uint8_t action = agent.PerceiveAndAct(obs, current_fe, true);
        obs = env.Step(action);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double phase1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n--- PHASE 1: SELF-PLAY INTERACTION & CAUSAL TAP SPROUTING ---" << std::endl;
    std::cout << "Total Interaction Steps: " << MAX_EPISODE_STEPS << std::endl;
    std::cout << "Steps to First Zero Free Energy: " << steps_to_convergence << std::endl;
    std::cout << "Endogenous Taps Sprouted: " << agent.active_taps.size() << std::endl;
    std::cout << "Action Attractors Synthesized: " << agent.action_attractors.size() << std::endl;
    std::cout << "Interaction Time: " << phase1_ms << " ms" << std::endl;

    // Phase 2: Sovereign Control Verification (Learning Frozen, Evaluating Policy Stability)
    std::cout << "\n--- PHASE 2: SOVEREIGN CONTROL VERIFICATION (FROZEN POLICY) ---" << std::endl;
    env.Reset();
    obs = 0b1010101010101010;

    const size_t VERIFY_STEPS = 10000;
    uint64_t zero_fe_step_count = 0;
    double sum_fe = 0.0;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < VERIFY_STEPS; ++step) {
        double current_fe = env.ComputeFreeEnergy();
        sum_fe += current_fe;
        if (current_fe == 0.0) zero_fe_step_count++;

        uint8_t action = agent.PerceiveAndAct(obs, current_fe, false); // learn = false
        obs = env.Step(action);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double phase2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double avg_verify_fe = sum_fe / static_cast<double>(VERIFY_STEPS);
    double zero_fe_pct = (static_cast<double>(zero_fe_step_count) / VERIFY_STEPS) * 100.0;

    std::cout << "Verification Steps: " << VERIFY_STEPS << std::endl;
    std::cout << "Steps in Target Zero FE Attractor: " << zero_fe_step_count << " / " << VERIFY_STEPS 
              << " (" << zero_fe_pct << "%)" << std::endl;
    std::cout << "Average Verification Free Energy: " << std::fixed << std::setprecision(8) << avg_verify_fe << " nats" << std::endl;
    std::cout << "Verification Speed: " << phase2_ms << " ms" << std::endl;

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "=== KEP PRINCIPLE 11 & 3 EVALUATION =====================================" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    bool success = (zero_fe_pct > 90.0 || avg_verify_fe < 0.05);
    std::cout << "Status: " << (success ? "🟢 ABSOLUTE AGENCY mastery (Environment Stabilized, Zero Supervised Data)" : "🔴 STABILITY FAILURE") << std::endl;

    return success ? 0 : 1;
}
