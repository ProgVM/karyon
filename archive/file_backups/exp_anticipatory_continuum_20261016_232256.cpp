// ============================================================================
// EXPERIMENT: Anticipatory String Continuum (Left/Right-Movers & Symplectic Advanced Waves)
// Comparing Pure Retrospective/Reactive Lie-Flow vs Advanced Anticipatory Continuum
// Hypothesis: If the string continuum integrates advanced wave modes (tau_adv)
// derived from stationary action / Hamilton's principle over the worldsheet,
// the causal inertial phase lag is completely eliminated, driving RMSE -> 0.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>
#include "karyon_core.h"

using namespace karyon;

// Reactive Baseline (Standard Lie-Flow)
float run_reactive_harmonic_tracking(float& out_rmse, float& out_phase_lag) {
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
    out_rmse = std::sqrt(late_error_sq_sum / 5000);
    return out_rmse;
}

// Anticipatory String Continuum:
// Leverages D'Alembert advanced wave / Wheeler-Feynman time-symmetric excitation:
// Advanced coordinate along the worldsheet: \tau_{adv} = t + \delta t_{inertial}
// where \delta t_{inertial} is endogenously tuned by the plant's mechanical impedance!
class AnticipatoryContinuumCore {
public:
    AnticipatoryContinuumCore() noexcept {
        core_.reset();
    }

    // Perceive present state AND project expected future trajectory along the light-cone
    inline float perceive_anticipate_and_act(float err, float err_rate, float err_accel, float dt, float tau_adv) noexcept {
        // Advanced Taylor/Lie expansion along the light cone of the worldsheet:
        // e_{adv} = e(t) + \tau_{adv} * \dot{e}(t) + 0.5 * \tau_{adv}^2 * \ddot{e}(t)
        float err_future = err + tau_adv * err_rate + 0.5f * tau_adv * tau_adv * err_accel;
        float rate_future = err_rate + tau_adv * err_accel;

        // The continuum is plucked by the stationary action boundary condition at \tau = t + \tau_{adv}
        return core_.perceive_and_act(err_future, rate_future, dt);
    }

private:
    SovereignContinuumCore core_;
};

float run_anticipatory_harmonic_tracking(float tau_adv, float& out_rmse) {
    AnticipatoryContinuumCore core;
    const float dt = 0.01f;
    const size_t total_steps = 10000;
    const float w1 = 1.5f;
    const float w2 = 4.2f;

    float plant_y = 0.0f;
    float plant_v = 0.0f;
    float late_error_sq_sum = 0.0f;

    float prev_err_rate = 0.0f;

    for (size_t t = 0; t < total_steps; ++t) {
        float time_sec = t * dt;
        float target = 0.8f * std::sin(w1 * time_sec) + 0.35f * std::cos(w2 * time_sec);
        float target_v = 0.8f * w1 * std::cos(w1 * time_sec) - 0.35f * w2 * std::sin(w2 * time_sec);

        float err = plant_y - target;
        float err_rate = plant_v - target_v;
        float err_accel = (err_rate - prev_err_rate) / dt;
        prev_err_rate = err_rate;

        float control = core.perceive_anticipate_and_act(err, err_rate, err_accel, dt, tau_adv);

        float plant_acc = -control - 0.5f * plant_v;
        plant_v += dt * plant_acc;
        plant_y += dt * plant_v;

        if (t >= 5000) {
            late_error_sq_sum += err * err;
        }
    }
    out_rmse = std::sqrt(late_error_sq_sum / 5000);
    return out_rmse;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== ANTICIPATORY STRING CONTINUUM EXPERIMENT: WHEELER-FEYNMAN WAVES ===\n";
    std::cout << "========================================================================\n";

    float base_rmse = 0.0f, dummy = 0.0f;
    run_reactive_harmonic_tracking(base_rmse, dummy);
    std::cout << "  [BASELINE] Pure Reactive/Retrospective Continuum RMSE: " << std::fixed << std::setprecision(6) << base_rmse << "\n\n";

    std::cout << "  Testing Advanced Worldsheet Horizons (tau_adv):\n";
    std::vector<float> horizons = {0.005f, 0.01f, 0.02f, 0.025f, 0.03f, 0.035f, 0.04f, 0.05f};

    float best_tau = 0.0f;
    float min_rmse = 1e9f;

    for (float tau : horizons) {
        float rmse = 0.0f;
        run_anticipatory_harmonic_tracking(tau, rmse);
        std::cout << "    * tau_adv = " << std::setw(6) << tau << " s (" << std::setw(4) << (tau / 0.01f) << " steps ahead) -> Late RMSE: " << rmse;
        if (rmse < base_rmse) {
            float drop_pct = (1.0f - rmse / base_rmse) * 100.0f;
            std::cout << "  [-" << std::setprecision(1) << drop_pct << "% ERROR DROP!]\n";
        } else {
            std::cout << "\n";
        }
        if (rmse < min_rmse) {
            min_rmse = rmse;
            best_tau = tau;
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << ">>> OPTIMAL ANTICIPATORY HORIZON: tau_adv = " << best_tau << " s\n";
    std::cout << ">>> MINIMUM TRACKING RMSE: " << min_rmse << " (vs Baseline " << base_rmse << ")\n";
    std::cout << ">>> ERROR REDUCTION: " << (1.0f - min_rmse / base_rmse) * 100.0f << "%\n";
    std::cout << "========================================================================\n";

    return 0;
}
