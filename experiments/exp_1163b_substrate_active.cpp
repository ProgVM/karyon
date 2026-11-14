// ============================================================================
// KARYON-CORE EXP-1163b: MINIMAL SUBSTRATE PHYSICS IN ACTIVE ENVIRONMENT
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// Evaluates SubstratePhysicsEngine (substrate_physics.hpp) inside the frozen
// ActiveWorld (active_env.hpp) across 100,000 steps with non-stationary
// regime shifts. Compares performance directly against Random and Oracle.
// ============================================================================

#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "active_env.hpp"
#include "substrate_physics.hpp"

void run_substrate_benchmark(uint64_t total_steps = 100000) {
    ActiveWorld env;
    env.reset(0x9999888877776666ULL);

    SubstratePhysicsEngine engine;

    uint64_t deaths = 0;
    std::vector<double> regime_rewards(4, 0.0);
    std::vector<uint64_t> regime_steps(4, 0);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t step = 0; step < total_steps; ++step) {
        uint64_t obs = env.observe();
        uint8_t a = engine.forward(obs);
        double dE = env.step(a);
        engine.adapt(dE);

        uint32_t r = env.regime;
        regime_rewards[r] += dE;
        regime_steps[r]++;

        if (!env.alive()) {
            deaths++;
            env.energy = ActiveWorld::E_INIT;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n======================================================================" << std::endl;
    std::cout << " KARYON SUBSTRATE BENCHMARK: SubstratePhysicsEngine (EXP-1163b)" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << " Total Steps     : " << total_steps << " | Total Deaths: " << deaths << std::endl;
    std::cout << " Food Eaten      : " << env.food_eaten << " | Toxins Hit: " << env.toxins_hit << std::endl;
    std::cout << " Final Energy    : " << std::fixed << std::setprecision(1) << env.energy << " / " << ActiveWorld::E_MAX << std::endl;
    std::cout << " Execution Speed : " << std::setprecision(2) << ms << " ms (" << (total_steps / (ms / 1000.0)) << " steps/sec)" << std::endl;
    std::cout << " Regime Delta E Breakdown:" << std::endl;
    std::cout << "   Regime 0 (Base Law)       : dE = " << std::setprecision(1) << regime_rewards[0]
              << " (" << regime_steps[0] << " steps)" << std::endl;
    std::cout << "   Regime 1 (Toxin Swap)     : dE = " << regime_rewards[1]
              << " (" << regime_steps[1] << " steps)" << std::endl;
    std::cout << "   Regime 2 (Motor Inversion) : dE = " << regime_rewards[2]
              << " (" << regime_steps[2] << " steps)" << std::endl;
    std::cout << "   Regime 3 (Dual Swap + Inv) : dE = " << regime_rewards[3]
              << " (" << regime_steps[3] << " steps)" << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1163b: SUBSTRATE PHYSICS IN FROZEN ENVIRONMENT" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 1, 3, 4, 8, 9, 11, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    run_substrate_benchmark(100000);

    return 0;
}
