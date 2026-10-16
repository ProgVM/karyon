// ============================================================================
// EXP-521: Benchmark of 2D Continuous Metric Membrane \Psi(x, y, t)
// Stabilizing an Inverted Spherical Pendulum (2-DOF: \theta_x, \theta_y)
// ============================================================================

#include <iostream>
#include <chrono>
#include <cmath>
#include "experiments/string_theory_core/sovereign_2d_field_continuum.hpp"

using namespace karyon;

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-521: 2D CONTINUOUS METRIC MEMBRANE BENCHMARK (64 MODES) ========\n";
    std::cout << "=== 2-DOF Inverted Spherical Pendulum Stabilization via 2D Field ======\n";
    std::cout << "========================================================================\n\n";

    Sovereign2DContinuum organism;

    Vector2D theta = { 0.40f, -0.35f }; // Diagonal 2D deflection (~30 deg total angle)
    Vector2D omega = { 0.0f, 0.0f };
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;
    const size_t total_steps = 50000;

    size_t upright_steps_total = 0;
    size_t upright_last_25k = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < total_steps; ++t) {
        float angle_magnitude = std::sqrt(theta.x * theta.x + theta.y * theta.y);
        bool is_upright = (angle_magnitude < 0.20f);
        if (is_upright) {
            upright_steps_total++;
            if (t >= 25000) upright_last_25k++;
        }

        Vector2D torque = organism.perceive_and_act_2d(theta, omega, dt);

        // Spherical pendulum 2D dynamics
        float alpha_x = (g / L) * std::sin(theta.x) - torque.x - 0.1f * omega.x;
        float alpha_y = (g / L) * std::sin(theta.y) - torque.y - 0.1f * omega.y;

        omega.x += dt * alpha_x;
        omega.y += dt * alpha_y;

        theta.x += dt * omega.x;
        theta.y += dt * omega.y;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    float final_mag = std::sqrt(theta.x * theta.x + theta.y * theta.y);
    float overall_pct = static_cast<float>(upright_steps_total * 100) / static_cast<float>(total_steps);
    float last_25k_pct = static_cast<float>(upright_last_25k * 100) / 25000.0f;
    double throughput = static_cast<double>(total_steps) / elapsed;

    std::cout << ">>> BENCHMARK RESULTS (50,000 STEPS - 2D CONTINUUM):\n";
    std::cout << "  * Total Elapsed Time : " << (elapsed * 1000.0) << " ms\n";
    std::cout << "  * Throughput         : " << throughput << " steps/sec\n";
    std::cout << "  * Latency per Step   : " << (elapsed * 1e9 / total_steps) << " ns/step\n";
    std::cout << "  * Overall Upright %  : " << overall_pct << "%\n";
    std::cout << "  * Steady-State %     : " << last_25k_pct << "% (Target: 100%)\n";
    std::cout << "  * Final Theta X      : " << theta.x << " rad\n";
    std::cout << "  * Final Theta Y      : " << theta.y << " rad\n";
    std::cout << "  * Final Angle Norm   : " << final_mag << " rad\n";
    std::cout << "  * Final Free Energy  : " << organism.get_free_energy() << "\n";
    std::cout << "========================================================================\n";

    return 0;
}
