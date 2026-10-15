// ============================================================================
// EXP-496: Deep Counterfactual Latent Rollout & Multi-Step Expected Free Energy
// Karyon-CoRE v28.0 Sovereign Benchmark
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include "karyon_native_core.h"

#define main real_benchmark_main
#include "experiments/exp_493_sovereign_multitask_genesis.cpp"
#undef main

using namespace karyon;

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-496: DEEP COUNTERFACTUAL LATENT ROLLOUT & EXPECTED FE   ===\n";
    std::cout << "=== Multi-Step System 2 Mental Sandbox Engine                   ===\n";
    std::cout << "====================================================================\n\n";

    CoreConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.initial_latent_dim = 32;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 8;
    cfg.plasticity_rate = 0.02f;
    cfg.initial_hormones = 2;
    cfg.max_hormones = 6;
    cfg.min_horizon_k = 2;
    cfg.max_horizon_k = 6;

    KaryonSovereignCore core(cfg);

    std::cout << "[EXP-496] Executing 4-Regime Multi-Task Benchmark with Deep System 2 Rollout...\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<TaskReport> reports;
    reports.push_back(run_regime_pursuit(core));
    reports.push_back(run_regime_pendulum(core));
    reports.push_back(run_regime_bistable(core));
    reports.push_back(run_regime_lorenz(core));

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_elapsed = std::chrono::duration<float>(t_end - t_start).count();

    std::cout << "\n====================================================================\n";
    std::cout << "=== MULTI-TASK EXP-496 SYSTEM 2 ROLLOUT SUMMARY                   ===\n";
    std::cout << "====================================================================\n";
    for (const auto& rep : reports) {
        std::cout << " - " << rep.task_name << ":\n";
        std::cout << "     Competence: " << rep.competence_label << "\n";
        std::cout << "     FE: " << rep.initial_fe << " -> " << rep.final_fe << "\n";
        std::cout << "     Primitives: " << rep.primitives_count << " | Hormones: " << rep.hormones_count << "\n";
        std::cout << "     Throughput: " << rep.throughput_sps << " steps/sec\n";
    }
    std::cout << "Total Elapsed Time: " << total_elapsed << "s\n";
    std::cout << "====================================================================\n";

    return 0;
}
