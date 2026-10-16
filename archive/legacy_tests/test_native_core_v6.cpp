// tests/test_native_core_v6.cpp
// Verification test for Karyon-CoRE Native Core v6.0 Sovereign Substrate

#include "karyon_native_core.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=== Running Karyon-CoRE v6.0 Core Verification Test ===" << std::endl;

    karyon::CoreConfig cfg;
    cfg.state_dim = 8;
    cfg.num_primitives = 2;
    cfg.num_actions = 4;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 6;

    karyon::KaryonSovereignCore core(cfg);

    std::vector<float> state(8, 0.0f);
    std::vector<float> goal = {0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f};

    size_t test_ticks = 5000;
    size_t goals = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 1; t <= test_ticks; ++t) {
        // 1. Decide action with mental sandbox rollout
        size_t act = core.decide_action(state, goal);

        // 2. Physical environment simulation
        std::vector<float> next_state = state;
        float force = 0.20f;
        if (act == 0) for (size_t i = 0; i < 4; ++i) next_state[i] += force;
        else if (act == 1) for (size_t i = 0; i < 4; ++i) next_state[i] -= force;
        else if (act == 2) for (size_t i = 4; i < 8; ++i) next_state[i] += force;
        else if (act == 3) for (size_t i = 4; i < 8; ++i) next_state[i] -= force;

        for (auto& s : next_state) s = std::clamp(s * 0.96f, -1.0f, 1.0f);

        // 3. Step Core
        auto telem = core.step_environment(state, goal, next_state, act);
        state = next_state;

        if (telem.goal_reached) {
            goals++;
            // Re-randomize state
            for (auto& s : state) s = ((rand() % 100) / 100.0f) * 0.8f - 0.4f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double tps = test_ticks / (elapsed_sec + 1e-6);

    std::cout << "Test Passed: Completed " << test_ticks << " ticks.\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << tps << " ticks/sec\n";
    std::cout << "Goals Achieved: " << goals << "\n";
    std::cout << "Noradrenaline: " << std::fixed << std::setprecision(3) << core.get_noradrenaline() << "\n";
    std::cout << "Current Horizon K: " << core.get_current_horizon() << "\n";

    assert(goals > 0);
    assert(tps > 50000.0);
    std::cout << "All Core Assertions PASSED successfully!" << std::endl;

    return 0;
}
