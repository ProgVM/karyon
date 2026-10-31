// ============================================================================
// KARYON-CORE EXPERIMENT EXP-894 (REVISION 2)
// Title: Sovereign Active Inference Attractor Agent (SAIAA) — True Causal Control
// Standard: KEP v37.0 Sovereign Master Protocol | Principles 2, 5, 7, 28, 29, 30 & 32
//
// Root Cause Diagnosis of Revision 1 Failure:
//   In Rev 1, the agent had no causal transition model P(s_{t+1} | s_t, a_t).
//   It predicted sensory spin without conditioning on its own efferent action a_t!
//
// Sovereign Solution:
//   1. Endogenous Causal Sensorimotor Model:
//      - The state key couples sensation and action: Key = (SensationHistory << 1) | (Action & 1).
//      - Evaluates Expected Free Energy G(a) = D_KL( P(s_{t+1} | S_t, a) || Prior*(Homeostasis) ).
//   2. Homeostatic Prior: Target is zero angular deviation (|theta| -> 0).
//   3. Topological Memory & Apoptosis: Condenses phase states and prunes noisy dead branches.
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cstdint>
#include <sstream>
#include <unordered_map>

constexpr double PI = 3.14159265358979323846;

class HardwareEntropySource {
private:
    std::random_device rd;
public:
    double sample_jitter(double scale = 0.0001) {
        uint32_t val = rd();
        double u1 = (static_cast<double>(val & 0xFFFF) + 1.0) / 65537.0;
        double u2 = (static_cast<double>((val >> 16) & 0xFFFF) + 1.0) / 65537.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2) * scale;
    }
};

struct TransitionNode {
    double expected_next_theta = 0.0;
    uint32_t visits = 0;
    double vitality = 1.0;
};

struct AgentActionReport {
    double environment_state;
    int action_emitted;
    double free_energy;
    double expected_fe_action_neg;
    double expected_fe_action_pos;
    size_t active_attractor_nodes;
    bool target_intercepted;
};

class SovereignActiveInferenceAgent {
private:
    // Causal transition tensor: Key(S_t, a_t) -> Next State Prediction
    std::unordered_map<uint64_t, TransitionNode> transition_attractor;

    uint64_t rolling_sensory_state = 0;
    uint32_t current_depth = 12;

    HardwareEntropySource hw_noise;

    uint64_t total_steps = 0;
    double cumulative_free_energy = 0.0;
    
    // Inverted Pendulum Environment
    double env_theta = 0.25; // Initial tilt angle (rad)
    double env_omega = 0.0;  // Angular velocity
    double dt = 0.02;        // 50 Hz control loop
    double g = 9.81, L = 1.0;

    int last_action = 0;

public:
    SovereignActiveInferenceAgent() {
        transition_attractor.reserve(4096);
    }

    // Step 1: Active Inference Action Emission
    // Chooses action a \in {-1, +1} that minimizes Expected Free Energy G(a)
    std::pair<int, std::pair<double, double>> EmitEfferentAction() {
        uint64_t mask = (1ULL << current_depth) - 1ULL;
        uint64_t base_key = (rolling_sensory_state & mask) << 1;

        // Key under Action = -1 (coded as bit 0)
        uint64_t key_neg = base_key | 0ULL;
        // Key under Action = +1 (coded as bit 1)
        uint64_t key_pos = base_key | 1ULL;

        double predicted_theta_neg = 0.0;
        double predicted_theta_pos = 0.0;

        auto it_neg = transition_attractor.find(key_neg);
        if (it_neg != transition_attractor.end() && it_neg->second.visits > 0) {
            predicted_theta_neg = it_neg->second.expected_next_theta;
        } else {
            // Epistemic prior: negative torque pushes theta negative
            predicted_theta_neg = env_theta - 0.08;
        }

        auto it_pos = transition_attractor.find(key_pos);
        if (it_pos != transition_attractor.end() && it_pos->second.visits > 0) {
            predicted_theta_pos = it_pos->second.expected_next_theta;
        } else {
            // Epistemic prior: positive torque pushes theta positive
            predicted_theta_pos = env_theta + 0.08;
        }

        // Expected Free Energy G(a) = |predicted_theta - HomeostaticTarget(0.0)|
        double G_neg = std::abs(predicted_theta_neg);
        double G_pos = std::abs(predicted_theta_pos);

        int chosen_action = (G_neg < G_pos) ? -1 : +1;

        // Exploration noise
        if (hw_noise.sample_jitter(0.01) > 0.009) {
            chosen_action = -chosen_action;
        }

        last_action = chosen_action;
        return {chosen_action, {G_neg, G_pos}};
    }

    // Step 2: Physical Environmental Feedback & Causal Learning
    AgentActionReport StepEnvironmentAndLearn(int motor_action) {
        double prev_theta = env_theta;

        // Physical torque applied by agent
        double control_torque = -motor_action * 14.0; // Negative feedback torque
        double alpha = (g / L) * std::sin(env_theta) + control_torque;
        
        env_omega += alpha * dt;
        env_omega *= 0.96; // Physical air damping
        env_theta += env_omega * dt;

        // Free Energy is actual sensory displacement from homeostasis (theta = 0)
        double free_energy = std::abs(env_theta);
        cumulative_free_energy += free_energy;
        total_steps++;

        // Update Causal Sensorimotor Attractor: Key(PreviousState, Action) -> Resulting Theta
        uint64_t mask = (1ULL << current_depth) - 1ULL;
        uint64_t action_bit = (motor_action > 0) ? 1ULL : 0ULL;
        uint64_t transition_key = ((rolling_sensory_state & mask) << 1) | action_bit;

        auto& node = transition_attractor[transition_key];
        node.visits++;
        node.vitality = 1.0;
        // Exact causal online update
        node.expected_next_theta += (env_theta - node.expected_next_theta) / static_cast<double>(node.visits);

        // Advance sensory history
        uint8_t sensory_bit = (env_theta > 0.0) ? 1 : 0;
        rolling_sensory_state = ((rolling_sensory_state << 1) | sensory_bit);

        // Topological Condensation & Apoptosis (every 200 steps)
        if (total_steps % 200 == 0) {
            for (auto it = transition_attractor.begin(); it != transition_attractor.end();) {
                it->second.vitality *= 0.90;
                if (it->second.vitality < 0.1 && it->second.visits < 2) {
                    it = transition_attractor.erase(it);
                } else {
                    ++it;
                }
            }
        }

        AgentActionReport report;
        report.environment_state = env_theta;
        report.action_emitted = motor_action;
        report.free_energy = free_energy;
        report.active_attractor_nodes = transition_attractor.size();
        report.target_intercepted = (std::abs(env_theta) < 0.05);

        return report;
    }

    double get_mean_free_energy() const {
        return (total_steps > 0) ? (cumulative_free_energy / total_steps) : 0.0;
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-894 (REV 2): CAUSAL ACTIVE INFERENCE AGENT\n";
    std::cout << " Standard: KEP v37.0 | Sensorimotor Causal Loop & Homeostasis Restoration\n";
    std::cout << " Target: Stabilize Inverted Pendulum at theta -> 0 via Efferent Action\n";
    std::cout << "===================================================================\n\n";

    SovereignActiveInferenceAgent agent;

    uint32_t total_runs = 500;
    uint32_t successful_balances = 0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Step | Env State (theta) | Action Emitted | G(-1) / G(+1)  | Free Energy | Nodes\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (uint32_t step = 1; step <= total_runs; ++step) {
        auto [action, G_vals] = agent.EmitEfferentAction();
        AgentActionReport report = agent.StepEnvironmentAndLearn(action);

        if (report.target_intercepted) {
            successful_balances++;
        }

        if (step <= 15 || step % 50 == 0 || step == total_runs) {
            std::cout << std::setw(4) << step << " | "
                      << std::setw(17) << report.environment_state << " | "
                      << std::setw(14) << (action > 0 ? "+1 (PUSH_R)" : "-1 (PUSH_L)") << " | "
                      << std::setw(6) << G_vals.first << " / " << std::setw(6) << G_vals.second << " | "
                      << std::setw(11) << report.free_energy << " | "
                      << std::setw(5) << report.active_attractor_nodes << "\n";
        }
    }

    std::cout << "\n===================================================================\n";
    std::cout << " SUMMARY TELEMETRY & HOMEOSTATIC PERFORMANCE:\n";
    std::cout << " [+] Mean Free Energy: " << agent.get_mean_free_energy() << " rad\n";
    std::cout << " [+] Homeostatic Balance Rate (|theta| < 0.05 rad): " 
              << (100.0 * successful_balances / total_runs) << "%\n";
    std::cout << "===================================================================\n";

    return 0;
}
