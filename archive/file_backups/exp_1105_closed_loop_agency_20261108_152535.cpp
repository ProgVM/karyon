// EXP-1105 V3: Closed-Loop Sovereign Agency with Soft Attractor Basin Snapping & Closed Sensorimotor Tape
// Standards: KEP Principle 0, Principle 3 (Zero Imposed Mechanisms), Principle 7 (Soft Attractor Basin Snapping), Principle 11 (Real-World Agency)
//
// Theoretical Formulation:
// 1. Closed Sensorimotor Tape: The continuous bit tape ingests BOTH sensory bits S_env AND efferent action bits A_action.
// 2. Causal Credit Assignment: Actions reducing Free Energy (F -> 0) solidify action attractors.
// 3. Soft Attractor Basin Snapping (KEP Principle 7):
//    When encountering an unseen signature in frozen verification, the substrate snaps to the nearest
//    stability basin via minimum Hamming distance across sprouted attractor states.

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
    uint16_t current_state;
    uint16_t target_state;
    uint32_t step_count;

public:
    UnknownDynamicEnvironment(uint32_t seed = 42) : step_count(0) {
        Reset();
    }

    void Reset() {
        current_state = 0b1010101010101010; // Initial noisy state (0xAAAA)
        target_state  = 0b1100110011001100; // Minimal Free Energy Target State (0xCCCC)
        step_count = 0;
    }

    uint16_t Step(uint8_t action_bit) {
        step_count++;
        // Unknown dynamic permutation rules
        if (action_bit == 1) {
            current_state = ((current_state << 1) | (current_state >> 15)) & 0xFFFF;
            current_state ^= 0x0F0F;
        } else {
            current_state = ((current_state >> 1) | (current_state << 15)) & 0xFFFF;
            current_state ^= 0x3333;
        }
        return current_state;
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

struct SovereignActiveInferenceAgency {
    std::vector<uint8_t> bit_tape;
    std::vector<size_t> active_taps;
    std::unordered_set<size_t> tap_set;

    // Map: Tap Signature -> (Best Action Bit, Associated Free Energy Value)
    struct AttractorNode {
        uint8_t action_bit;
        double min_fe;
    };
    std::unordered_map<uint64_t, AttractorNode> action_attractors;

    uint64_t last_sig;
    uint8_t last_action;
    double last_fe;

    SovereignActiveInferenceAgency() : last_sig(0), last_action(0), last_fe(1.0) {
        active_taps.push_back(1);
        tap_set.insert(1);
        active_taps.push_back(16); // Look back 1 full sensory word
        tap_set.insert(16);
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

    // Principle 7: Soft Attractor Basin Snapping for unseen signatures
    uint8_t SnapToNearestAttractorAction(uint64_t query_sig) const {
        if (action_attractors.empty()) return 0;
        
        auto it = action_attractors.find(query_sig);
        if (it != action_attractors.end()) {
            return it->second.action_bit;
        }

        // Search for nearest signature in Hamming space
        uint8_t best_action = 0;
        int min_hamming = 64;
        double min_fe = 100.0;

        for (const auto& kv : action_attractors) {
            int dist = __builtin_popcountll(query_sig ^ kv.first);
            if (dist < min_hamming || (dist == min_hamming && kv.second.min_fe < min_fe)) {
                min_hamming = dist;
                min_fe = kv.second.min_fe;
                best_action = kv.second.action_bit;
            }
        }
        return best_action;
    }

    // Step 1: Perceive observation & select action bit
    uint8_t SelectAction(uint16_t sensory_obs, bool learn = true) {
        // Continuous sensorimotor tape: append sensory bits
        for (int b = 15; b >= 0; --b) {
            bit_tape.push_back((sensory_obs >> b) & 1);
        }

        size_t current_pos = bit_tape.size();
        last_sig = ComputeTapSignature(current_pos);

        if (learn) {
            auto it = action_attractors.find(last_sig);
            if (it != action_attractors.end()) {
                last_action = it->second.action_bit;
            } else {
                // Exploratory pulse
                last_action = (current_pos % 2 == 0) ? 1 : 0;
            }
        } else {
            // Principle 7 Soft Attractor Snapping during verification
            last_action = SnapToNearestAttractorAction(last_sig);
        }

        // Continuous sensorimotor tape: append emitted action bit
        bit_tape.push_back(last_action);

        return last_action;
    }

    // Step 2: Causal Credit Assignment
    void AdaptCausalAttractors(double new_fe, bool learn = true) {
        if (!learn) return;

        auto it = action_attractors.find(last_sig);
        if (it == action_attractors.end() || new_fe <= it->second.min_fe) {
            // Solidify action in attractor memory
            action_attractors[last_sig] = { last_action, new_fe };
        } else {
            // Mutate action if Free Energy degraded
            action_attractors[last_sig] = { static_cast<uint8_t>(1 - last_action), new_fe };
            if (active_taps.size() < 64) {
                SproutCausalTap(bit_tape.size());
            }
        }
        last_fe = new_fe;
    }
};

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "=== EXP-1105 V3: CLOSED-LOOP SOVEREIGN AGENCY (SOFT ATTRACTOR SNAPPING) =" << std::endl;
    std::cout << "=== KEP Principle 7 & 11: Real-World Agency & Zero Supervised Data ======" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    UnknownDynamicEnvironment env(42);
    SovereignActiveInferenceAgency agent;

    std::cout << "[ENV INIT] Initial State: 0xaaaa | Target State: 0xcccc" << std::endl;
    std::cout << "Initial Free Energy: " << std::fixed << std::setprecision(8) 
              << env.ComputeFreeEnergy() << " nats" << std::endl;

    // Phase 1: Self-Play Interaction & Attractor Synthesis
    const size_t TRAIN_EPISODES = 100;
    const size_t STEPS_PER_EPISODE = 200;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t zero_fe_count_train = 0;
    for (size_t ep = 0; ep < TRAIN_EPISODES; ++ep) {
        env.Reset();
        uint16_t obs = 0b1010101010101010;
        agent.last_fe = env.ComputeFreeEnergy();

        for (size_t step = 0; step < STEPS_PER_EPISODE; ++step) {
            uint8_t action = agent.SelectAction(obs, true);
            obs = env.Step(action);
            double new_fe = env.ComputeFreeEnergy();
            if (new_fe == 0.0) zero_fe_count_train++;
            agent.AdaptCausalAttractors(new_fe, true);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double phase1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n--- PHASE 1: SELF-PLAY ATTRACTOR SYNTHESIS ---" << std::endl;
    std::cout << "Total Steps: " << (TRAIN_EPISODES * STEPS_PER_EPISODE) << std::endl;
    std::cout << "Training Zero FE Trajectories Reached: " << zero_fe_count_train << std::endl;
    std::cout << "Endogenous Taps Sprouted: " << agent.active_taps.size() << std::endl;
    std::cout << "Synthesized Action Attractor Basins: " << agent.action_attractors.size() << std::endl;
    std::cout << "Interaction Time: " << phase1_ms << " ms" << std::endl;

    // Phase 2: Autonomous Verification (Policy Frozen, Soft Attractor Snapping Active)
    std::cout << "\n--- PHASE 2: SOVEREIGN CONTROL VERIFICATION (POLICY FROZEN) ---" << std::endl;
    env.Reset();
    uint16_t obs = 0b1010101010101010;

    const size_t VERIFY_STEPS = 5000;
    uint64_t zero_fe_steps = 0;
    double sum_fe = 0.0;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < VERIFY_STEPS; ++step) {
        uint8_t action = agent.SelectAction(obs, false); // Frozen learning
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
    std::cout << "=== KEP PRINCIPLE 7 & 11 EVALUATION =====================================" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    bool success = (zero_fe_pct > 80.0 || avg_verify_fe < 0.10);
    std::cout << "Status: " << (success ? "🟢 ABSOLUTE AGENCY MASTERY (Environment Stabilized, Zero Supervised Data)" : "🔴 FAIL") << std::endl;

    return success ? 0 : 1;
}
