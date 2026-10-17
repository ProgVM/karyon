// ============================================================================
// EXP-548.1: Sovereign Causal Active Inference (Correct Force Sign & Feedback)
// Grounding: KEP Principle 5, Principle 28, Principle 29
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

struct EnvironmentState {
    float x;
    float v;
};

class SovereignActiveInferenceOrganism {
public:
    SovereignActiveInferenceOrganism(size_t initial_dim = 4, size_t max_dim = 32)
        : current_dim_(initial_dim), max_dim_(max_dim), total_cycles_(0) {
        
        q_.assign(max_dim_, 0.0f);
        p_.assign(max_dim_, 0.0f);
        
        g_metric_.assign(max_dim_ * max_dim_, 0.0f);
        for (size_t i = 0; i < current_dim_; ++i) {
            g_metric_[i * max_dim_ + i] = 4.0f + 0.5f * static_cast<float>(i);
        }

        // Efference motor weights w_motor_i: opposing displacement
        w_motor_.assign(max_dim_, 0.0f);
        for (size_t i = 0; i < current_dim_; ++i) {
            w_motor_[i] = 1.0f / std::sqrt(static_cast<float>(i + 1));
        }

        damping_.assign(max_dim_, 2.0f);
        persistent_expected_surprise_ = 0.0f;
    }

    float step_active_inference(const EnvironmentState& env, float dt = 0.005f) {
        total_cycles_++;

        // 1. Current Variational Free Energy
        float F_t = 0.5f * (env.x * env.x + 0.2f * env.v * env.v);

        // 2. Expected Free Energy Lookahead Rollout
        float current_act = compute_action();
        // Environmental physics: corrective action must counter displacement (+x requires negative force)
        float pred_env_v = env.v + (env.x - env.x * env.x * env.x - 0.25f * env.v - current_act) * dt;
        float pred_env_x = env.x + pred_env_v * dt;
        float G_expected = 0.5f * (pred_env_x * pred_env_x + 0.2f * pred_env_v * pred_env_v);

        float delta_FE = G_expected - F_t;
        persistent_expected_surprise_ = 0.99f * persistent_expected_surprise_ + 0.01f * G_expected;

        // 3. Continuous Substrate Field Dynamics
        // Direct sensory immersion: displacement plucks internal modes
        float sensory_drive = (env.x + 0.4f * env.v) * 10.0f;

        for (size_t i = 0; i < current_dim_; ++i) {
            float inner_force = 0.0f;
            for (size_t j = 0; j < current_dim_; ++j) {
                inner_force += g_metric_[i * max_dim_ + j] * q_[j];
            }
            float ext_force = sensory_drive / std::sqrt(static_cast<float>(i + 1));

            p_[i] += (-inner_force - damping_[i] * p_[i] + ext_force) * dt;
            p_[i] = std::clamp(p_[i], -30.0f, 30.0f);

            q_[i] += p_[i] * dt;
            q_[i] = std::clamp(q_[i], -15.0f, 15.0f);
        }

        // 4. Causal Epistemic Gradient Descent on Motor Weights:
        // dG/dw_i = (dG/dx_pred) * (dx_pred/dv_pred) * (dv_pred/d_act) * (d_act/dw_i)
        // dG/dx_pred = pred_env_x
        // dv_pred/d_act = -dt
        // dx_pred/dv_pred = dt
        // Therefore: dG/dw_i = - pred_env_x * q_i * dt^2
        // To minimize G: w_i -= lr * dG/dw_i = w_i + lr * pred_env_x * q_i
        float lr = 0.15f;
        for (size_t i = 0; i < current_dim_; ++i) {
            float grad_G_w = pred_env_x * q_[i];
            w_motor_[i] += lr * grad_G_w * dt;
            w_motor_[i] = std::clamp(w_motor_[i], -10.0f, 10.0f);

            // Self-tuning metric tensor: stiffen if surprise is high to resist drift
            if (delta_FE > 0.0f) {
                g_metric_[i * max_dim_ + i] += 0.05f * dt;
            } else {
                g_metric_[i * max_dim_ + i] = std::max(1.0f, g_metric_[i * max_dim_ + i] - 0.01f * dt);
            }
        }

        // 5. Morphogenesis: sprout new mode if persistent surprise > 0.05
        if (total_cycles_ % 2000 == 0 && current_dim_ < max_dim_) {
            if (persistent_expected_surprise_ > 0.05f) {
                size_t new_dim = current_dim_;
                q_[new_dim] = 0.001f * q_[new_dim - 1];
                p_[new_dim] = 0.0f;
                g_metric_[new_dim * max_dim_ + new_dim] = 6.0f + 0.5f * static_cast<float>(new_dim);
                w_motor_[new_dim] = (pred_env_x > 0.0f ? 1.0f : -1.0f) / std::sqrt(static_cast<float>(new_dim + 1));
                current_dim_++;
            }
        }

        return compute_action();
    }

    float compute_action() const {
        float act = 0.0f;
        for (size_t i = 0; i < current_dim_; ++i) {
            act += w_motor_[i] * q_[i];
        }
        return std::clamp(act, -30.0f, 30.0f);
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
    std::vector<float> damping_;
    float persistent_expected_surprise_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-548.1: SOVEREIGN CAUSAL ACTIVE INFERENCE (CORRECTED SIGN) ======\n";
    std::cout << "========================================================================\n\n";

    SovereignActiveInferenceOrganism karyon(4, 32);

    EnvironmentState env{0.65f, 0.0f};
    const float dt = 0.005f;

    std::mt19937 rng(5481);
    std::normal_distribution<float> shock_dist(0.0f, 0.35f);

    const size_t total_steps = 100000;
    size_t controlled_steps = 0;
    float cumulative_FE = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < total_steps; ++step) {
        float action = karyon.step_active_inference(env, dt);

        float external_shock = 0.0f;
        if (step % 2500 == 0 && step > 0) {
            external_shock = shock_dist(rng);
        }

        float t = step * dt;
        float f_duffing = env.x - env.x * env.x * env.x - 0.25f * env.v + 0.3f * std::cos(1.2f * t);
        
        // Corrective physical coupling: action counters drift
        env.v += (f_duffing - action + external_shock) * dt;
        env.x += env.v * dt;

        float FE = 0.5f * (env.x * env.x + 0.2f * env.v * env.v);
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
    std::cout << "=== EXP-548.1 FINAL SCIENTIFIC TELEMETRY REPORT ===\n";
    std::cout << "Total Steps Run:                  " << total_steps << "\n";
    std::cout << "Wall-Clock Duration:              " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "Silicon Throughput:               " << std::fixed << std::setprecision(1) << steps_per_sec << " steps/sec\n";
    std::cout << "Initial Active Modes:             4\n";
    std::cout << "Final Evolved Modes:              " << karyon.get_active_dim() << "\n";
    std::cout << "Chaotic Basin Retention (|x|<0.25): " << std::fixed << std::setprecision(2) << retention_pct << "%\n";
    std::cout << "Average Variational Free Energy:  " << std::scientific << std::setprecision(4) << avg_FE << "\n";
    std::cout << "Final Environment State:          x = " << env.x << ", v = " << env.v << "\n";

    if (retention_pct > 80.0f) {
        std::cout << "VERDICT: POSITIVE (Causal Active Inference mastered non-linear chaos!)\n";
    } else {
        std::cout << "VERDICT: REJECTED\n";
    }
    std::cout << "========================================================================\n";

    return 0;
}
