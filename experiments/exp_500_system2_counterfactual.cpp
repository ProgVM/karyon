// ============================================================================
// EXP-500: System 2 Counterfactual Latent Rollout & Fundamental Invariants
// Karyon-CoRE v28.0 Sovereign Benchmark (NO MAGIC NUMBERS)
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include "karyon_native_core.h"

#define main benchmark_main_impl
#include "experiments/exp_493_sovereign_multitask_genesis.cpp"
#undef main

using namespace karyon;

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-500: SYSTEM 2 COUNTERFACTUAL ROLLOUT & INVARIANTS (NO MAGIC) ===\n";
    std::cout << "=== Pure Shannon Entropy Horizon, Kramers Phase Fluctuation, Oja Scale ===\n";
    std::cout << "====================================================================\n\n";

    CoreConfig cfg;
    cfg.sensory_dim = 256;
    cfg.initial_latent_dim = 16;
    cfg.max_latent_dim = 64;
    cfg.action_dim = 4;

    KaryonSovereignCore core(cfg);

    // Pass 1: Multi-task sequential evaluation
    std::cout << "--- PASS 1: INITIAL MULTI-TASK EVALUATION ---\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    auto repA1 = run_regime_pursuit(core, 10000);
    std::cout << repA1.task_name << " | FE: " << repA1.final_fe
              << " | " << repA1.competence_label << "\n";

    auto repB1 = run_regime_pendulum(core, 15000);
    std::cout << repB1.task_name << " | FE: " << repB1.final_fe
              << " | " << repB1.competence_label << "\n";

    auto repC1 = run_regime_bistable(core, 15000);
    std::cout << repC1.task_name << " | FE: " << repC1.final_fe
              << " | " << repC1.competence_label << "\n";

    auto repD1 = run_regime_lorenz(core, 15000);
    std::cout << repD1.task_name << " | FE: " << repD1.final_fe
              << " | " << repD1.competence_label << "\n";

    // Pass 2: Re-evaluating skills without memory resets
    std::cout << "\n--- PASS 2: CUMULATIVE SKILL RE-EVALUATION ---\n";
    auto repA2 = run_regime_pursuit(core, 10000);
    std::cout << repA2.task_name << " (Pass 2) | FE: " << repA2.final_fe
              << " | " << repA2.competence_label << "\n";

    auto repB2 = run_regime_pendulum(core, 15000);
    std::cout << repB2.task_name << " (Pass 2) | FE: " << repB2.final_fe
              << " | " << repB2.competence_label << "\n";

    auto repC2 = run_regime_bistable(core, 15000);
    std::cout << repC2.task_name << " (Pass 2) | FE: " << repC2.final_fe
              << " | " << repC2.competence_label << "\n";

    auto repD2 = run_regime_lorenz(core, 15000);
    std::cout << repD2.task_name << " (Pass 2) | FE: " << repD2.final_fe
              << " | " << repD2.competence_label << "\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n====================================================================\n";
    std::cout << "=== EXP-500 SUMMARY TELEMETRY ===\n";
    std::cout << "Pass 1 Orbit FE: " << repA1.final_fe << " | Pass 2 Orbit FE: " << repA2.final_fe << "\n";
    std::cout << "Pass 1 Pendulum FE: " << repB1.final_fe << " | Pass 2 Pendulum FE: " << repB2.final_fe << "\n";
    std::cout << "Pass 1 Double-Well FE: " << repC1.final_fe << " | Pass 2 Double-Well FE: " << repC2.final_fe << "\n";
    std::cout << "Pass 1 Lorenz FE: " << repD1.final_fe << " | Pass 2 Lorenz FE: " << repD2.final_fe << "\n";
    std::cout << "Total Elapsed Time: " << elapsed_s << " s\n";
    std::cout << "====================================================================\n";

    return 0;
}
