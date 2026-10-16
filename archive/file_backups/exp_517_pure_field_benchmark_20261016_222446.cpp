// ============================================================================
// EXP-517: Benchmark of Pure Sovereign Metric Field Tensor (100.00% IDEAL TARGET)
// Validating Principle 29 without discrete Left/Right sub-classes or manual PD terms!
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/sovereign_metric_field_organism.hpp"

using namespace karyon;

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-517: PURE SOVEREIGN METRIC FIELD TENSOR BENCHMARK =============\n";
    std::cout << "=== Zero Sub-Classes, Zero Hardcoded PD Terms, Pure 1D Continuum ======\n";
    std::cout << "========================================================================\n\n";

    SovereignFieldConfig cfg;
    cfg.num_spatial_nodes = 32;
    cfg.eta_metric = 0.08f;

    SovereignMetricFieldOrganism organism(cfg);

    float theta = 0.50f; // Initial massive deflection (28.6 degrees!)
    float omega = 0.0f;
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;
    const size_t total_steps = 30000;

    size_t upright_steps_total = 0;
    size_t upright_last_10k = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < total_steps; ++t) {
        // Evaluate upright: |theta| < 0.20 rad (~11 degrees)
        bool is_upright = (std::abs(theta) < 0.20f);
        if (is_upright) {
            upright_steps_total++;
            if (t >= 20000) upright_last_10k++;
        }

        // Active control from pure spatial field tensor
        float torque = organism.perceive_and_act(theta, omega, dt);

        // Physical pendulum dynamics: alpha = (g/L)*sin(theta) - torque - damping
        float alpha = (g / L) * std::sin(theta) - torque - 0.1f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        // Telemetry checkpoints
        if ((t + 1) % 5000 == 0 || t == total_steps - 1) {
            std::cout << "  Step " << (t + 1) << "/" << total_steps
                      << " | Theta: " << theta << " rad"
                      << " | Omega: " << omega
                      << " | Cumulative Upright: " << (upright_steps_total * 100 / (t + 1)) << "%"
                      << " | Avg Metric Tension T(x): " << organism.get_avg_tension()
                      << " | Avg Metric Damping Gamma(x): " << organism.get_avg_damping()
                      << " | Field Restoring Torque: " << organism.get_torque()
                      << " | Free Energy: " << organism.get_free_energy() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    float overall_pct = static_cast<float>(upright_steps_total * 100) / static_cast<float>(total_steps);
    float last_10k_pct = static_cast<float>(upright_last_10k * 100) / 10000.0f;

    std::cout << "\n========================================================================\n";
    std::cout << ">>> PURE CONTINUUM BENCHMARK COMPLETE!\n";
    std::cout << ">>> Overall Upright Ratio: " << overall_pct << "% (" << upright_steps_total << "/" << total_steps << ")\n";
    std::cout << ">>> Steady-State Upright Ratio (Last 10k Steps): " << last_10k_pct << "%\n";
    std::cout << ">>> Final Equilibrium Angle Theta: " << theta << " rad\n";
    std::cout << ">>> Free Energy: " << organism.get_free_energy() << "\n";
    std::cout << ">>> Throughput: " << (total_steps / elapsed) << " steps/s\n";
    std::cout << "========================================================================\n";

    return 0;
}
