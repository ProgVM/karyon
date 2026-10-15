// ============================================================================
// EXP-498: Fast Contextual Resonant Recall via Dynamic Episodic Memory
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
    std::cout << "=== EXP-498: FAST CONTEXTUAL RESONANT RECALL VIA EPISODIC MEMORY ===\n";
    std::cout << "=== Non-Parametric Phase Jump & Instant Attractor Resurgence     ===\n";
    std::cout << "====================================================================\n\n";

    CoreConfig cfg;
    cfg.sensory_dim = 256;
    cfg.action_dim = 4;
    cfg.initial_latent_dim = 16;
    cfg.max_latent_dim = 64;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 8;
    cfg.plasticity_rate = 0.02f;
    cfg.initial_hormones = 2;
    cfg.max_hormones = 6;
    cfg.sprouting_fe_threshold = 0.010f;
    cfg.sprouting_cooldown_ticks = 100;

    KaryonSovereignCore core(cfg);

    std::cout << "[EXP-498] Phase 1: Initial Benchmark Run (Populating Episodic Memory Bank)...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<TaskReport> reports_pass1;
    reports_pass1.push_back(run_regime_pursuit(core));
    reports_pass1.push_back(run_regime_pendulum(core));
    reports_pass1.push_back(run_regime_bistable(core));
    reports_pass1.push_back(run_regime_lorenz(core));

    std::cout << "\n[EXP-498] Phase 2: Immediate Re-Evaluation (Testing Fast Resonant Recall & Zero-Lag Transfer)...\n";

    std::vector<TaskReport> reports_pass2;
    reports_pass2.push_back(run_regime_pursuit(core));
    reports_pass2.push_back(run_regime_pendulum(core));
    reports_pass2.push_back(run_regime_bistable(core));
    reports_pass2.push_back(run_regime_lorenz(core));

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_elapsed = std::chrono::duration<float>(t_end - t_start).count();

    std::cout << "\n====================================================================\n";
    std::cout << "=== MULTI-TASK EXP-498 FAST RESONANT RECALL SUMMARY               ===\n";
    std::cout << "====================================================================\n";
    std::cout << "PASS 1 (Populating Memory):\n";
    for (const auto& rep : reports_pass1) {
        std::cout << " - " << rep.task_name << " | Competence: " << rep.competence_label 
                  << " | FE: " << rep.initial_fe << " -> " << rep.final_fe << "\n";
    }

    std::cout << "\nPASS 2 (Resonant Recall Active):\n";
    for (const auto& rep : reports_pass2) {
        std::cout << " - " << rep.task_name << " | Competence: " << rep.competence_label 
                  << " | FE: " << rep.initial_fe << " -> " << rep.final_fe << "\n";
    }

    std::cout << "\nTotal Benchmark Time: " << total_elapsed << "s\n";
    std::cout << "====================================================================\n";

    return 0;
}
