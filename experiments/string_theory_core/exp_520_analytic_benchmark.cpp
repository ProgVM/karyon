// ============================================================================
// EXP-520: Benchmark of Analytic Matrix Exponential Lie-Flow Sovereign Continuum
// ============================================================================

#include <iostream>
#include <chrono>
#include <cmath>
#include "experiments/string_theory_core/analytic_field_continuum.hpp"

using namespace karyon;

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-520: ANALYTIC LIE-FLOW MATRIX EXPONENTIAL CONTINUUM BENCHMARK ==\n";
    std::cout << "=== Exact Closed-Form Flow, Zero Substepping, Unconditional Stability =\n";
    std::cout << "========================================================================\n\n";

    AnalyticSovereignContinuum organism;

    float theta = 0.45f;
    float omega = 0.0f;
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;
    const size_t total_steps = 100000;

    size_t upright_steps_total = 0;
    size_t upright_last_50k = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < total_steps; ++t) {
        bool is_upright = (std::abs(theta) < 0.20f);
        if (is_upright) {
            upright_steps_total++;
            if (t >= 50000) upright_last_50k++;
        }

        float torque = organism.perceive_and_act_analytic(theta, omega, dt);

        float alpha = (g / L) * std::sin(theta) - torque - 0.1f * omega;
        omega += dt * alpha;
        theta += dt * omega;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    float overall_pct = static_cast<float>(upright_steps_total * 100) / static_cast<float>(total_steps);
    float last_50k_pct = static_cast<float>(upright_last_50k * 100) / 50000.0f;
    double throughput = static_cast<double>(total_steps) / elapsed;

    std::cout << ">>> BENCHMARK RESULTS (100,000 STEPS - EXACT ANALYTIC FLOW):\n";
    std::cout << "  * Total Elapsed Time : " << (elapsed * 1000.0) << " ms\n";
    std::cout << "  * Throughput         : " << throughput << " steps/sec\n";
    std::cout << "  * Latency per Step   : " << (elapsed * 1e9 / total_steps) << " ns/step\n";
    std::cout << "  * Overall Upright %  : " << overall_pct << "%\n";
    std::cout << "  * Steady-State %     : " << last_50k_pct << "% (100% target)\n";
    std::cout << "  * Final Theta        : " << theta << " rad\n";
    std::cout << "  * Final Free Energy  : " << organism.get_free_energy() << "\n";
    std::cout << "  * Final Stiffness K  : " << organism.get_avg_stiffness() << "\n";
    std::cout << "  * Final Damping G    : " << organism.get_avg_damping() << "\n";
    std::cout << "========================================================================\n";

    return 0;
}
