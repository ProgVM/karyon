// ============================================================================
// EXP-548: Causal Active Inference Morphogenesis (Current & Expected Free Energy)
// Grounding: KEP Principle 5 (Causal Credit Assignment - Death of Blind Dice-Throwing),
//            KEP Principle 28 (Authentic Behavioral Competence),
//            KEP Principle 29 (Law of Sovereign Self-Control & Sufficient Time).
//
// BAZILEVS DIRECTIVE:
// "Karyon must perceive both CURRENT Free Energy F_t and EXPECTED Free Energy G_{t+1},
// and derive causal lessons from this delta to guide the autopoietic evolution of its
// internal coordinates and metric tensor g_ij."
//
// MATHEMATICAL FORMULATION:
// 1. Current Variational Free Energy:
//    F_t = D_KL(Q || P) + Accuracy_Error = 0.5 * (x_env^2 + 0.2 * v_env^2)
// 2. Expected Free Energy (EFE) Look-ahead Rollout:
//    G_{t+1} = E_Q [ ln Q(s_{t+1} | pi) - ln P(s_{t+1}) ]
//    Evaluated via internal forward simulation of the environment-organism manifold.
// 3. Epistemic Causal Vector (Lesson Extraction):
//    Delta_FE = G_{t+1} - F_t
//    - If Delta_FE < 0: Action successfully reduces future surprise -> consolidate g_ij.
//    - If Delta_FE >= 0: Failure/Stagnation -> trigger directed morphogenetic mutation
//      along the endogenous causal gradient: dg_ij/dt = - eta * (dG/dg_ij).
// 4. Autonomous Morphogenesis:
//    New modes sprout if and only if persistent expected surprise cannot be minimized
//    by the existing sub-dimensional manifold.
//
// ENVIRONMENT:
// Highly non-linear coupled Duffing oscillator with continuous chaotic drive and
// stochastic perturbation shocks.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <x86intrin.h>
#include <omp.h>

struct EnvironmentState {
    float x;
    float v;
};

class ActiveInferenceContinuumOrganism {
public:
    ActiveInferenceContinuumOrganism(size_t initial_dim = 4, size_t max_dim = 32)
        : current_dim_(initial_dim), max_dim_(max_dim), total_cycles_(0) {
        
        q_.assign(max_dim_, 0.0f);
        p_.assign(max_dim_, 0.0f);
        
        // Internal Metric Tensor g_ij (Coupling between internal modes)
        g_metric_.assign(max_dim_ * max_dim_, 0.0f);
        for (size_t i = 0; i < current_dim_; ++i) {
            g_metric_[i * max_dim_ + i] = 2.0f + 0.2f * static_cast<float>(i);
        }

        // Efference motor weights w_motor_i
        w_motor_.assign(max_dim_, 0.0f);
        for (size_t i = 0; i < current_dim_; ++i) {
            w_motor_[i] = (i % 2 == 0 ? 1.0f : -1.0f) / std::sqrt(static_cast<float>(i + 1));
        }

        stiffness_.assign(max_dim_, 2.5f);
        damping_.assign(max_dim_, 1.2f);

        // Causal Lesson Accumulator (Success / Error Traces)
        causal_credit_trace_.assign(max_dim_, 0.0f);
        persistent_expected_surprise_ = 0.0f;
    }

    // Step the organism with Current and Expected Free Energy evaluation
    float step_active_inference(const EnvironmentState& env, float dt = 0.005f) {
        total_cycles_++;

        // --------------------------------------------------------------------
        // 1. EVALUATE CURRENT FREE ENERGY (F_t)
        // --------------------------------------------------------------------
        float F_t = 0.5f * (env.x * env.x + 0.25f * env.v * env.v);

        // --------------------------------------------------------------------
        // 2. INTERNAL MENTAL SIMULATION: EXPECTED FREE ENERGY G_{t+1} (EFE)
        // Simulate immediate forward rollout under current internal configuration
        // --------------------------------------------------------------------
        float pred_action = compute_action();
        float pred_env_v = env.v + (env.x - env.x * env.x * env.x - 0.25f * env.v - 2.0f * pred_action) * dt;
        float pred_env_x = env.x + pred_env_v * dt;
        float G_expected = 0.5f * (pred_env_x * pred_env_x + 0.25f * pred_env_v * pred_env_v);

        // Causal Epistemic Gradient: Delta = Future Surprise - Current Surprise
        float delta_FE = G_expected - F_t;
        persistent_expected_surprise_ = 0.995f * persistent_expected_surprise_ + 0.005f * G_expected;

        // --------------------------------------------------------------------
        // 3. CAUSAL LESSON EXTRACTION & SOVEREIGN REINFORCEMENT (Principle 5)
        // The organism adjusts its metric g_ij and motor weights w_motor
        // strictly based on whether the action minimized or exacerbated surprise!
        // --------------------------------------------------------------------
        float learning_rate = 0.08f;
        for (size_t i = 0; i < current_dim_; ++i) {
            // Directional alignment: did mode q_i contribute to reducing G?
            float mode_gradient = pred_env_x * q_[i];
            
            // If delta_FE > 0 (Surprise increasing), apply strong corrective counter-force
            float correction = (delta_FE > 0.0f) ? 1.5f : 0.5f;
            w_motor_[i] -= learning_rate * correction * mode_gradient * dt;
            w_motor_[i] = std::clamp(w_motor_[i], -5.0f, 5.0f);

            // Causal credit assignment: record which modes are effective
            causal_credit_trace_[i] = 0.99f * causal_credit_trace_[i] - 0.01f * (delta_FE * q_[i] * q_[i]);

            // Metric adaptation: stiffen successful modes, relax unsuccessful
            if (causal_credit_trace_[i] > 0.0f) {
                // Consolidate successful structural pathways
                g_metric_[i * max_dim_ + i] += 0.01f * dt;
            } else {
                // Mutate and search alternative phase frequencies
                g_metric_[i * max_dim_ + i] -= 0.02f * dt;
            }
            g_metric_[i * max_dim_ + i] = std::clamp(g_metric_[i * max_dim_ + i], 0.2f, 40.0f);
        }

        // --------------------------------------------------------------------
        // 4. CONTINUOUS SUBSTRATE FIELD DYNAMICS (Hamiltonian Leapfrog)
        // --------------------------------------------------------------------
        float sensory_plucking = (env.x + 0.3f * env.v) * 4.0f;

        for (size_t i = 0; i < current_dim_; ++i) {
            float internal_force = 0.0f;
            for (size_t j = 0; j < current_dim_; ++j) {
                internal_force += g_metric_[i * max_dim_ + j] * q_[j];
            }
            float ext_drive = sensory_plucking / std::sqrt(static_cast<float>(i + 1));

            // dp/dt = - F_internal - damping * p + F_sensory
            float dp = (-internal_force - damping_[i] * p_[i] + ext_drive);
            p_[i] += dp * dt;
            p_[i] = std::clamp(p_[i], -25.0f, 25.0f);

            // dq/dt = p
            q_[i] += p_[i] * dt;
            q_[i] = std::clamp(q_[i], -15.0f, 15.0f);
        }

        // --------------------------------------------------------------------
        // 5. CAUSAL MORPHOGENESIS (Autopoietic Growth Driven by EFE Deficit)
        // If Expected Free Energy remains high despite maximum internal relaxation,
        // sprout a new physical coordinate mode along the unmodelled residual!
        // --------------------------------------------------------------------
        if (total_cycles_ % 1500 == 0 && current_dim_ < max_dim_) {
            if (persistent_expected_surprise_ > 0.10f) {
                size_t new_dim = current_dim_;
                q_[new_dim] = 0.001f * q_[new_dim - 1]; // Net2Net smooth sprouting
                p_[new_dim] = 0.0f;
                stiffness_[new_dim] = stiffness_[new_dim - 1] * 1.3f;
                damping_[new_dim] = damping_[new_dim - 1] * 1.05f;
                g_metric_[new_dim * max_dim_ + new_dim] = stiffness_[new_dim];
                
                // Causal coupling to parent
                g_metric_[new_dim * max_dim_ + (new_dim - 1)] = 0.1f;
                g_metric_[(new_dim - 1) * max_dim_ + new_dim] = 0.1f;
                
                // Initialize new motor weight to compensate residual
                w_motor_[new_dim] = -0.5f * (pred_env_x > 0 ? 1.0f : -1.0f);

                current_dim_++;
            }
        }

        // Final actual action emitted
        return compute_action();
    }

    float compute_action() const {
        float act = 0.0f;
        for (size_t i = 0; i < current_dim_; ++i) {
            act += w_motor_[i] * q_[i];
        }
        return std::clamp(act, -12.0f, 12.0f);
    }

    size_t get_active_dim() const { return current_dim_; }
    float get_persistent_expected_surprise() const { return persistent_expected_surprise_; }

private:
    size_t current_dim_;
    size_t max_dim_;
    size_t total_cycles_;

    std::vector<float> q_;
    std::vector<float> p_;
    std::vector<float> g_metric_;
    std::vector<float> w_motor_;
    std::vector<float> stiffness_;
    std::vector<float> damping_;
    std::vector<float> causal_credit_trace_;
    float persistent_expected_surprise_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-548: CAUSAL ACTIVE INFERENCE MORPHOGENESIS (F_t & G_{t+1}) =====\n";
    std::cout << "=== Closed-Loop Causal Learning Over 100,000 Continuous Epochs =========\n";
    std::cout << "========================================================================\n\n";

    ActiveInferenceContinuumOrganism karyon(4, 32);

    std::cout << "Initial Active Modes: " << karyon.get_active_dim() << "\n";
    std::cout << "Target: Tame non-linear chaotic Duffing drift using Current & Expected FE.\n\n";

    // Non-linear environment
    EnvironmentState env{0.75f, 0.0f};
    const float dt = 0.005f;

    std::mt19937 rng(5480);
    std::normal_distribution<float> shock_dist(0.0f, 0.35f);

    const size_t total_steps = 100000;
    size_t controlled_steps = 0;
    float cumulative_FE = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < total_steps; ++step) {
        float action = karyon.step_active_inference(env, dt);

        // Stochastic environmental shocks every 2500 steps
        float external_shock = 0.0f;
        if (step % 2500 == 0 && step > 0) {
            external_shock = shock_dist(rng);
        }

        // Non-linear Duffing dynamics:
        // x'' = x - x^3 - 0.25 * v + 0.3 * cos(1.2 * t) - 2.0 * action + shock
        float t = step * dt;
        float f_duffing = env.x - env.x * env.x * env.x - 0.25f * env.v + 0.3f * std::cos(1.2f * t);
        
        env.v += (f_duffing - 2.0f * action + external_shock) * dt;
        env.x += env.v * dt;

        float FE = 0.5f * (env.x * env.x + 0.25f * env.v * env.v);
        cumulative_FE += FE;

        if (std::abs(env.x) < 0.25f) {
            controlled_steps++;
        }

        if (step % 20000 == 0 && step > 0) {
            std::cout << "  [Step " << std::setw(6) << step << "] "
                      << "Modes: " << std::setw(2) << karyon.get_active_dim()
                      << " | Pos: " << std::fixed << std::setprecision(4) << env.x
                      << " | Vel: " << std::setw(7) << env.v
                      << " | Action: " << std::setw(7) << action
                      << " | EFE: " << karyon.get_persistent_expected_surprise() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double steps_per_sec = total_steps / elapsed;
    float retention_pct = (static_cast<float>(controlled_steps) / total_steps) * 100.0f;
    float avg_FE = cumulative_FE / total_steps;

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-548 FINAL SCIENTIFIC TELEMETRY REPORT ===\n";
    std::cout << "Total Steps Run:                  " << total_steps << "\n";
    std::cout << "Wall-Clock Duration:              " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "Silicon Throughput:               " << std::fixed << std::setprecision(1) << steps_per_sec << " steps/sec\n";
    std::cout << "Initial Active Modes:             4\n";
    std::cout << "Final Evolved Modes:              " << karyon.get_active_dim() << "\n";
    std::cout << "Chaotic Basin Retention (|x|<0.25): " << std::fixed << std::setprecision(2) << retention_pct << "%\n";
    std::cout << "Average Variational Free Energy:  " << std::scientific << std::setprecision(4) << avg_FE << "\n";
    std::cout << "Final Environment State:          x = " << env.x << ", v = " << env.v << "\n";

    if (retention_pct > 80.0f) {
        std::cout << "VERDICT: POSITIVE (Causal Active Inference succeeded: Karyon learned from Current & Expected FE and mastered chaos!)\n";
    } else {
        std::cout << "VERDICT: REJECTED\n";
    }
    std::cout << "========================================================================\n";

    return 0;
}
