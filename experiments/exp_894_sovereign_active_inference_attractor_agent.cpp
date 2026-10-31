// ============================================================================
// KARYON-CORE EXPERIMENT EXP-894 (REVISION 3)
// Title: Sovereign Active Inference Attractor Agent (SAIAA) — True Physics Alignment
// Standard: KEP v37.0 Sovereign Master Protocol | Principles 2, 5, 7, 28, 29, 30 & 32
//
// Biophysical Diagnosis & Calibration:
//   1. Physical Force Vector Alignment:
//      - When theta > 0 (tilted right), restorative force MUST be negative (push left).
//      - When theta < 0 (tilted left), restorative force MUST be positive (push right).
//   2. Continuous Phase-Space Memory:
//      - Key couples continuous phase position + action vector -> next position attractor.
//   3. Active Inference Expected Free Energy G(a):
//      - Agent evaluates G(a) = |E_{P}[theta_{t+1} | s_t, a_t]|
//      - Spontaneously discovers exact balance equilibrium theta -> 0.0 rad.
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
};

struct AgentActionReport {
    double environment_state;
    int action_emitted;
    double free_energy;
    size_t active_attractor_nodes;
    bool target_intercepted;
};

class SovereignActiveInferenceAgent {
private:
    // Transition Memory: Key(DiscretizedPhase, Action) -> Expected Next Theta
    std::unordered_map<uint64_t, TransitionNode> transition_attractor;

    HardwareEntropySource hw_noise;

    uint64_t total_steps = 0;
    double cumulative_free_energy = 0.0;
    
    // Inverted Pendulum Environment
    double env_theta = 0.25; // Initial tilt angle (rad)
    double env_omega = 0.0;  // Angular velocity
    double dt = 0.02;        // 50 Hz control loop
    double g = 9.81, L = 1.0;

public:
    SovereignActiveInferenceAgent() {
        transition_attractor.reserve(4096);
    }

    // Convert continuous phase state (theta, omega) to uint64 state key
    uint64_t DiscretizePhaseState(double theta, double omega) const {
        int th_bin = std::clamp(static_cast<int>((theta + 2.0) * 20.0), 0, 79);
        int om_bin = std::clamp(static_cast<int>((omega + 5.0) * 10.0), 0, 99);
        return (static_cast<uint64_t>(th_bin) << 16) | static_cast<uint64_t>(om_bin);
    }

    // Step 1: Active Inference Efferent Action Selection
    // Evaluates Expected Free Energy G(a) for motor actions a \in {-1, +1}
    std::pair<int, std::pair<double, double>> EmitEfferentAction() {
        uint64_t state_key = DiscretizePhaseState(env_theta, env_omega);

        uint64_t key_neg = (state_key << 1) | 0ULL; // Action = -1
        uint64_t key_pos = (state_key << 1) | 1ULL; // Action = +1

        double predicted_theta_neg = 0.0;
        double predicted_theta_pos = 0.0;

        auto it_neg = transition_attractor.find(key_neg);
        if (it_neg != transition_attractor.end() && it_neg->second.visits > 0) {
            predicted_theta_neg = it_neg->second.expected_next_theta;
        } else {
            // Internal Prior: Motor action -1 pushes theta towards negative
            predicted_theta_neg = env_theta - 0.04;
        }

        auto it_pos = transition_attractor.find(key_pos);
        if (it_pos != transition_attractor.end() && it_pos->second.visits > 0) {
            predicted_theta_pos = it_pos->second.expected_next_theta;
        } else {
            // Internal Prior: Motor action +1 pushes theta towards positive
            predicted_theta_pos = env_theta + 0.04;
        }

        // Homeostatic Target is theta = 0.0 rad
        double G_neg = std::abs(predicted_theta_neg);
        double G_pos = std::abs(predicted_theta_pos);

        // Select action minimizing Expected Free Energy G(a)
        int chosen_action = (G_neg < G_pos) ? -1 : +1;

        // Jitter for exploration
        if (hw_noise.sample_jitter(0.01) > 0.0095) {
            chosen_action = -chosen_action;
        }

        return {chosen_action, {G_neg, G_pos}};
    }

    // Step 2: Physical Environmental Feedback & Causal Learning Loop
    AgentActionReport StepEnvironmentAndLearn(int motor_action) {
        uint64_t prev_state_key = DiscretizePhaseState(env_theta, env_omega);

        // CORRECT PHYSICAL FORCE ALIGNMENT:
        // Motor action +1 applies positive torque, -1 applies negative torque.
        double motor_torque = motor_action * 12.0;
        double alpha = (g / L) * std::sin(env_theta) + motor_torque;
        
        env_omega += alpha * dt;
        env_omega *= 0.95; // Mechanical friction damping
        env_theta += env_omega * dt;

        // Keep theta bound within [-PI, +PI]
        while (env_theta > PI) env_theta -= 2.0 * PI;
        while (env_theta < -PI) env_theta += 2.0 * PI;

        // Free Energy is sensory deviation from homeostatic target (theta = 0)
        double free_energy = std::abs(env_theta);
        cumulative_free_energy += free_energy;
        total_steps++;

        // Update Causal Sensorimotor Memory Node
        uint64_t action_bit = (motor_action > 0) ? 1ULL : 0ULL;
        uint64_t transition_key = (prev_state_key << 1) | action_bit;

        auto& node = transition_attractor[transition_key];
        node.visits++;
        node.expected_next_theta += (env_theta - node.expected_next_theta) / static_cast<double>(node.visits);

        AgentActionReport report;
        report.environment_state = env_theta;
        report.action_emitted = motor_action;
        report.free_energy = free_energy;
        report.active_attractor_nodes = transition_attractor.size();
        report.target_intercepted = (std::abs(env_theta) < 0.08);

        return report;
    }

    double get_mean_free_energy() const {
        return (total_steps > 0) ? (cumulative_free_energy / total_steps) : 0.0;
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-894 (REV 3): SOVEREIGN ACTIVE INFERENCE AGENT\n";
    std::cout << " Standard: KEP v37.0 | Physical Sensorimotor Balance Verification\n";
    std::cout << " Target: Steady-State Inverted Pendulum Balance (|theta| < 0.08 rad)\n";
    std::cout << "===================================================================\n\n";

    SovereignActiveInferenceAgent agent;

    uint32_t total_runs = 1000;
    uint32_t successful_balances = 0;
    uint32_t last_100_balances = 0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Step | Env State (theta) | Action Emitted | G(-1) / G(+1)  | Free Energy | Nodes\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (uint32_t step = 1; step <= total_runs; ++step) {
        auto [action, G_vals] = agent.EmitEfferentAction();
        AgentActionReport report = agent.StepEnvironmentAndLearn(action);

        if (report.target_intercepted) {
            successful_balances++;
            if (step > 900) last_100_balances++;
        }

        if (step <= 15 || step % 100 == 0 || step == total_runs) {
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
    std::cout << " [+] Overall Balance Rate (|theta| < 0.08 rad): " 
              << (100.0 * successful_balances / total_runs) << "%\n";
    std::cout << " [+] Final Steady-State Balance Rate (Last 100 Steps): " 
              << (100.0 * last_100_balances / 100) << "%\n";
    std::cout << "===================================================================\n";

    return 0;
}
