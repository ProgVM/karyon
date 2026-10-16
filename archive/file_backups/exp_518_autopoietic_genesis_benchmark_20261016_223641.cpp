// ============================================================================
// EXP-518: Autopoietic Genesis Benchmark
// Validating that "atoms", "molecules", and "cells" spontaneously condense
// as non-linear topological solitons, breathers, and domain walls in the continuous
// string field WITHOUT a single hardcoded struct!
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "experiments/string_theory_core/emergent_soliton_genesis_organism.hpp"

using namespace karyon;

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-518: AUTOPOIETIC SOLITON CONDENSATION & PARTICLE GENESIS ======\n";
    std::cout << "=== Pure Continuous Field: Zero Hardcoded Structs, Pure Wavefields ====\n";
    std::cout << "========================================================================\n\n";

    AutopoieticGenesisConfig cfg;
    cfg.spatial_resolution = 64;
    cfg.eta_autopoiesis = 0.08f;
    cfg.symplectic_substeps = 8;

    AutopoieticSolitonContinuum organism(cfg);

    float theta = 0.40f; // Initial angle perturbation (~23 degrees)
    float omega = 0.0f;
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;
    const size_t total_steps = 30000;

    size_t upright_steps_total = 0;
    size_t upright_last_10k = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < total_steps; ++t) {
        bool is_upright = (std::abs(theta) < 0.20f);
        if (is_upright) {
            upright_steps_total++;
            if (t >= 20000) upright_last_10k++;
        }

        float torque = organism.perceive_act_and_condense(theta, omega, dt);

        float alpha = (g / L) * std::sin(theta) - torque - 0.1f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        if ((t + 1) % 5000 == 0 || t == total_steps - 1) {
            auto telem = organism.analyze_emergent_structures();
            std::cout << "  Step " << (t + 1) << "/" << total_steps
                      << " | Theta: " << theta << " rad"
                      << " | Upright: " << (upright_steps_total * 100 / (t + 1)) << "%"
                      << " | Emergent Solitons ('Atoms'): " << telem.emergent_solitons
                      << " | Bound Breathers ('Molecules'): " << telem.emergent_bound_states
                      << " | Domain Walls ('Membranes'): " << telem.emergent_domain_walls
                      << " | Synthesized K: " << telem.self_synthesized_stiffness
                      << " | Free Energy: " << telem.free_energy << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    auto final_telem = organism.analyze_emergent_structures();
    float overall_pct = static_cast<float>(upright_steps_total * 100) / static_cast<float>(total_steps);
    float last_10k_pct = static_cast<float>(upright_last_10k * 100) / 10000.0f;

    std::cout << "\n========================================================================\n";
    std::cout << ">>> AUTOPOIETIC CONDENSATION BENCHMARK COMPLETE!\n";
    std::cout << ">>> Overall Upright Ratio: " << overall_pct << "% (" << upright_steps_total << "/" << total_steps << ")\n";
    std::cout << ">>> Steady-State Upright Ratio (Last 10k Steps): " << last_10k_pct << "%\n";
    std::cout << ">>> Final Equilibrium Angle Theta: " << theta << " rad\n";
    std::cout << ">>> Final Emergent Solitons (Condensed Particles): " << final_telem.emergent_solitons << "\n";
    std::cout << ">>> Final Emergent Bound Pairs (Molecules): " << final_telem.emergent_bound_states << "\n";
    std::cout << ">>> Final Domain Walls (Cell Membranes): " << final_telem.emergent_domain_walls << "\n";
    std::cout << ">>> Synthesized Continuum Stiffness: " << final_telem.self_synthesized_stiffness << "\n";
    std::cout << ">>> Final Free Energy: " << final_telem.free_energy << "\n";
    std::cout << ">>> Throughput: " << (total_steps / elapsed) << " steps/s\n";
    std::cout << "========================================================================\n";

    return 0;
}
