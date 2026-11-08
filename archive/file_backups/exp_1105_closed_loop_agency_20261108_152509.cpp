// EXP-1105: Closed-Loop Sovereign Endogenous Agency & Active Inference Self-Play (V2)
// Standards: KEP Principle 0, Principle 3 (Zero Imposed Mechanisms), Principle 6 (Causal Credit Assignment), Principle 11 (Real-World Agency)
//
// Theoretical Formulation:
// Active Inference Substrate: Action A_t is selected to minimize expected Free Energy F.
// Causal Credit Assignment: An action A_t emitted under signature Sig_t is locked in memory
// IF AND ONLY IF it causes a reduction in Free Energy (F_{t+1} <= F_t) or achieves target zero FE.
// If an action increases Free Energy (F_{t+1} > F_t), the causal vector mutates.

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

class UnknownDynamicEnvironment {
private:
    uint32_t current_state;
    uint32_t target_state;
    uint32_t step_count;

public:
    UnknownDynamicEnvironment(uint32_t seed = 42) : step_count(0) {
        Reset();
    }

    void Reset() {
        current_state = 0b1010101010101010; // Initial noisy state (16 bits)
        target_state  = 0b1100110011001100; // Minimal Free Energy Target State
        step_count = 0;
    }

    uint16_t Step(uint8_t action_bit) {
        step_count++;
        // Unknown non-linear bitwise dynamics
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

    double ComputeFreeEnergy() const {
        uint32_t diff = current_state ^ target_state;
        uint32_t pop = __builtin_popcount(diff);
        if (pop == 0) return 0.0;
        double err_rate = static_cast<double>(pop) / 16.0;
        return - (err_rate * std::log2(err_rate) + (1.0 - err_rate) * std::log2(1.0 - err_rate));
    }
};

struct ActiveInferenceSubstrate {
    std::vector<uint8_t> bit_tape;
    std::vector<size_t> active_taps;
    std::unordered_set<size_t> tap_set;
    std::unordered_map<uint64_t, uint8_t> action_attractors; // Sig -> Best Action

    uint64_t last_sig;
    uint8_t last_action;
    double last_fe;

    ActiveInferenceSubstrate() : last_sig(0), last_action(0), last_fe(1.0) {
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
                break;
            }
        }
    }

    // Step 1: Emit Action based on current observation signature
    uint8_t SelectAction(uint16_t sensory_obs) {
        for (int b = 15; b >= 0; --b) {
            bit_tape.push_back((sensory_obs >> b) & 1);
        }

        size_t current_pos = bit_tape.size();
        last_sig = ComputeTapSignature(current_pos);

        auto it = action_attractors.find(last_sig);
        if (it != action_attractors.end()) {
            last_action = it->second;
        } else {
            // Exploratory pulse
            last_action = (current_pos % 2 == 0) ? 1 : 0;
        }
        return last_action;
    }

    // Step 2: Continuous Causal Credit Assignment based on FE trajectory delta
    void ObserveOutcomeAndAdapt(double new_fe, bool learn = true) {
        if (!learn) return;

        if (new_fe <= last_fe) {
            // Causal Success: Lock in action for signature
            action_attractors[last_sig] = last_action;
        } else {
            // Causal Failure: Mutate action preference for signature
            action_attractors[last_sig] = 1 - last_action;
            if (active_taps.size() < 64) {
                SproutCausalTap(bit_tape.size());
            }
        }
        last_fe = new_fe;
    }
};

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "=== EXP-1105 V2: ACTIVE INFERENCE CLOSED-LOOP SELF-PLAY AGENCY ==========" << std::endl;
    std::cout << "=== KEP Principle 11: Real-World Agency & Causal Credit Assignment ======" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    UnknownDynamicEnvironment env(42);
    ActiveInferenceSubstrate agent;

    std::cout << "[ENV INIT] Initial State: 0xaaaa | Target State: 0xcccc" << std::endl;
    std::cout << "Initial Free Energy: " << std::fixed << std::setprecision(8) 
              << env.ComputeFreeEnergy() << " nats" << std::endl;

    // Phase 1: Self-Play Interaction Loop (Active Inference)
    const size_t TRAIN_EPISODES = 50;
    const size_t STEPS_PER_EPISODE = 200;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t first_target_reach = 0;
    bool reached = false;

    for (size_t ep = 0; ep < TRAIN_EPISODES; ++ep) {
        env.Reset();
        uint16_t obs = 0b1010101010101010;
        agent.last_fe = env.ComputeFreeEnergy();

        for (size_t step = 0; step < STEPS_PER_EPISODE; ++step) {
            uint8_t action = agent.SelectAction(obs);
            obs = env.Step(action);
            double new_fe = env.ComputeFreeEnergy();

            if (new_fe == 0.0 && !reached) {
                first_target_reach = ep * STEPS_PER_EPISODE + step;
                reached = true;
            }

            agent.ObserveOutcomeAndAdapt(new_fe, true);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double phase1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n--- PHASE 1: ACTIVE INFERENCE SELF-PLAY COMPLETED ---" << std::endl;
    std::cout << "Total Steps: " << (TRAIN_EPISODES * STEPS_PER_EPISODE) << std::endl;
    std::cout << "First Target Reach at Step: " << first_target_reach << std::endl;
    std::cout << "Endogenous Taps Sprouted: " << agent.active_taps.size() << std::endl;
    std::cout << "Synthesized Action Attractors: " << agent.action_attractors.size() << std::endl;
    std::cout << "Interaction Time: " << phase1_ms << " ms" << std::endl;

    // Phase 2: Verification of Autonomous Control (Policy Frozen)
    std::cout << "\n--- PHASE 2: SOVEREIGN CONTROL VERIFICATION (FROZEN POLICY) ---" << std::endl;
    env.Reset();
    uint16_t obs = 0b1010101010101010;

    const size_t VERIFY_STEPS = 5000;
    uint64_t zero_fe_steps = 0;
    double sum_fe = 0.0;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < VERIFY_STEPS; ++step) {
        uint8_t action = agent.SelectAction(obs);
        obs = env.Step(action);
        double current_fe = env.ComputeFreeEnergy();
        sum_fe += current_fe;
        if (current_fe == 0.0) zero_fe_steps++;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double phase2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double avg_verify_fe = sum_fe / static_cast<double>(VERIFY_STEPS);
    double zero_fe_pct = (static_cast<double>(zero_fe_steps) / VERIFY_STEPS) * 100.0;

    std::cout << "Verification Steps: " << VERIFY_STEPS << std::endl;
    std::cout << "Steps in Zero FE Attractor: " << zero_fe_steps << " / " << VERIFY_STEPS 
              << " (" << zero_fe_pct << "%)" << std::endl;
    std::cout << "Average Verification Free Energy: " << std::fixed << std::setprecision(8) << avg_verify_fe << " nats" << std::endl;
    std::cout << "Verification Speed: " << phase2_ms << " ms" << std::endl;

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "=== KEP PRINCIPLE 11 EVALUATION =========================================" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    bool success = (zero_fe_pct > 80.0 || avg_verify_fe < 0.10);
    std::cout << "Status: " << (success ? "🟢 ABSOLUTE AGENCY MASTERY (Environment Stabilized)" : "🔴 FAIL") << std::endl;

    return success ? 0 : 1;
}
