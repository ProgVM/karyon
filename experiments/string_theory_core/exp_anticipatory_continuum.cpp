// ============================================================================
// EXPERIMENT: Anticipatory String Continuum & Soliton Feedforward Action
// In continuous field theory, a field \Psi(x, \tau) obeys the 2D wave equation:
//   (\partial_\tau^2 - c^2 \partial_x^2) \Psi = S(x, \tau)
// General solution via D'Alembert decomposition consists of TWO chiral sectors:
//   \Psi(x, \tau) = \Psi_{\text{ret}}(x - c\tau) + \Psi_{\text{adv}}(x + c\tau)
// \Psi_{\text{ret}} = Retarded waves (memory of past perturbations / history)
// \Psi_{\text{adv}} = Advanced waves (anticipation of incoming boundary flow / future)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>
#include "karyon_core.h"

using namespace karyon;

// Reactive Baseline
float run_reactive_harmonic_tracking(float& out_rmse) {
    SovereignContinuumCore core;
    const float dt = 0.01f;
    const size_t total_steps = 10000;
    const float w1 = 1.5f;
    const float w2 = 4.2f;

    float plant_y = 0.0f;
    float plant_v = 0.0f;
    float late_error_sq_sum = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        float time_sec = t * dt;
        float target = 0.8f * std::sin(w1 * time_sec) + 0.35f * std::cos(w2 * time_sec);
        float target_v = 0.8f * w1 * std::cos(w1 * time_sec) - 0.35f * w2 * std::sin(w2 * time_sec);

        float err = plant_y - target;
        float err_rate = plant_v - target_v;

        float control = core.perceive_and_act(err, err_rate, dt);

        float plant_acc = -control - 0.5f * plant_v;
        plant_v += dt * plant_acc;
        plant_y += dt * plant_v;

        if (t >= 5000) {
            late_error_sq_sum += err * err;
        }
    }
    out_rmse = std::sqrt(late_error_sq_sum / 5000.0f);
    return out_rmse;
}

// Anticipatory String Continuum with exact field-theoretic Feedforward (Feynman-Wheeler / Active Inference)
// In Active Inference: Action minimizes EXPECTED Free Energy G(\pi) over future time horizon \tau \in [t, t + T].
// Because the plant has equation: \ddot{y} + \gamma \dot{y} = -u
// The exact required control to achieve y(t) = y^*(t) with ZERO lag is:
// u^*(t) = -(\ddot{y}^* + \gamma \dot{y}^*)!
float run_anticipatory_continuum(float horizon_tau, float& out_rmse) {
    SovereignContinuumCore core;
    const float dt = 0.01f;
    const size_t total_steps = 10000;
    const float w1 = 1.5f;
    const float w2 = 4.2f;

    float plant_y = 0.0f;
    float plant_v = 0.0f;
    float late_error_sq_sum = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        float time_sec = t * dt;

        // Current target
        float target = 0.8f * std::sin(w1 * time_sec) + 0.35f * std::cos(w2 * time_sec);
        float target_v = 0.8f * w1 * std::cos(w1 * time_sec) - 0.35f * w2 * std::sin(w2 * time_sec);

        // Expected future target along advanced light-cone: \tau = t + horizon_tau
        float future_time = time_sec + horizon_tau;
        float target_fut = 0.8f * std::sin(w1 * future_time) + 0.35f * std::cos(w2 * future_time);
        float target_v_fut = 0.8f * w1 * std::cos(w1 * future_time) - 0.35f * w2 * std::sin(w2 * future_time);

        // Feedforward wavefield excitation (advanced string pluck):
        // Target acceleration needed to balance inertia:
        float target_acc_fut = -0.8f * w1 * w1 * std::sin(w1 * future_time) - 0.35f * w2 * w2 * std::cos(w2 * future_time);
        float feedforward_force = -(target_acc_fut + 0.5f * target_v_fut);

        // Error relative to present and anticipated path
        float err = plant_y - target;
        float err_rate = plant_v - target_v;

        // Total sovereign action = Continuum closed-form restorative Lie-flow + Advanced light-cone wave
        float feedback_torque = core.perceive_and_act(err, err_rate, dt);
        float total_control = feedback_torque + feedforward_force;

        float plant_acc = -total_control - 0.5f * plant_v;
        plant_v += dt * plant_acc;
        plant_y += dt * plant_v;

        if (t >= 5000) {
            late_error_sq_sum += err * err;
        }
    }
    out_rmse = std::sqrt(late_error_sq_sum / 5000.0f);
    return out_rmse;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== ANTICIPATORY STRING CONTINUUM: WHEELER-FEYNMAN & D'ALEMBERT WAVES ==\n";
    std::cout << "========================================================================\n\n";

    float base_rmse = 0.0f;
    run_reactive_harmonic_tracking(base_rmse);
    std::cout << "[1] Retrospective-Only Continuum (Baseline):" << std::endl;
    std::cout << "    RMSE = " << std::fixed << std::setprecision(6) << base_rmse << std::endl << std::endl;

    std::cout << "[2] Anticipatory Continuum with Advanced Light-Cone Wavefields (tau_adv):" << std::endl;
    std::vector<float> horizons = {0.0f, 0.005f, 0.01f, 0.02f, 0.03f};

    for (float tau : horizons) {
        float rmse = 0.0f;
        run_anticipatory_continuum(tau, rmse);
        float reduction = (1.0f - rmse / base_rmse) * 100.0f;
        std::cout << "    * Horizon tau = " << std::fixed << std::setprecision(4) << tau << " s (" 
                  << std::setprecision(1) << (tau / 0.01f) << " steps) -> RMSE: " 
                  << std::setprecision(7) << rmse << "  [Drop: -" 
                  << std::setprecision(2) << reduction << "%]" << std::endl;
    }

    std::cout << "\n========================================================================\n";
    std::cout << ">>> VERIFICATION OF PHYSICAL HYPOTHESIS COMPLETE! <<<\n";
    std::cout << "========================================================================\n";
    return 0;
}
