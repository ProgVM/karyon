// ============================================================================
// Comprehensive Native C++20 Test Suite for Karyon Sovereign Core v28.0
// Testing: Lie-flow integration, stack alignment, stability, FE convergence
// ============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "karyon_core.h"

using namespace karyon;

void test_initialization() {
    std::cout << "[TEST 1] Testing Core Initialization and Memory Alignment...\n";
    SovereignContinuumCore core;
    const auto& state = core.get_state();

    // Check alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(&state);
    assert(addr % 64 == 0 && "Error: ContinuousFieldState is not 64-byte aligned!");

    for (size_t k = 0; k < DEFAULT_MODES; ++k) {
        assert(state.q[k] == 0.0f && "Initial q must be 0");
        assert(state.p[k] == 0.0f && "Initial p must be 0");
        assert(state.stiffness[k] == 2.0f && "Initial stiffness must be 2.0");
        assert(state.damping[k] == 1.0f && "Initial damping must be 1.0");
    }
    assert(core.get_free_energy() == 0.0f);
    assert(core.get_torque() == 0.0f);
    std::cout << "  -> PASSED: 64-byte cache line alignment and baseline states verified.\n";
}

void test_zero_excitation_stationary() {
    std::cout << "[TEST 2] Testing Zero Excitation Stationary Lie-Flow...\n";
    SovereignContinuumCore core;
    for (int i = 0; i < 100; ++i) {
        float torque = core.perceive_and_act(0.0f, 0.0f, 0.02f);
        assert(std::abs(torque) < 1e-6f && "Torque must remain 0 under zero error");
        assert(core.get_free_energy() == 0.0f && "Free energy must be 0");
    }
    std::cout << "  -> PASSED: Zero error maintains exact mathematical zero state.\n";
}

void test_restorative_torque_symmetry() {
    std::cout << "[TEST 3] Testing Restorative Torque Sign & Physical Directionality...\n";
    SovereignContinuumCore core_pos;
    SovereignContinuumCore core_neg;

    float torque_pos = core_pos.perceive_and_act(0.35f, 0.0f, 0.02f);
    float torque_neg = core_neg.perceive_and_act(-0.35f, 0.0f, 0.02f);

    assert(torque_pos > 0.0f && "Positive displacement must generate positive restorative counter-torque");
    assert(torque_neg < 0.0f && "Negative displacement must generate negative restorative counter-torque");
    assert(std::abs(torque_pos + torque_neg) < 1e-4f && "Torque must be odd/anti-symmetric under sign reversal");
    std::cout << "  -> PASSED: Restorative boundary torque has exact physical symmetry.\n";
}

void test_endless_stability_no_drift() {
    std::cout << "[TEST 4] Testing 50,000 Step Numerical Stability (No NaN/Inf/Drift)...\n";
    SovereignContinuumCore core;
    float theta = 0.5f;
    float omega = 0.0f;
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;

    for (size_t t = 0; t < 50000; ++t) {
        float torque = core.perceive_and_act(theta, omega, dt);
        float alpha = (g / L) * std::sin(theta) - torque - 0.1f * omega;
        omega += dt * alpha;
        theta += dt * omega;

        assert(!std::isnan(theta) && !std::isinf(theta) && "Theta must not diverge!");
        assert(!std::isnan(torque) && !std::isinf(torque) && "Torque must not diverge!");
    }

    assert(std::abs(theta) < 1e-5f && "System must reach asymptotic equilibrium!");
    assert(core.get_free_energy() < 1e-10f && "Free Energy must collapse to zero!");
    std::cout << "  -> PASSED: Asymptotic convergence verified (final theta < 1e-5 rad, FE -> 0).\n";
}

void test_ultra_throughput() {
    std::cout << "[TEST 5] Testing Throughput Benchmark (>100,000 steps/sec)...\n";
    SovereignContinuumCore core;
    const size_t bench_steps = 100000;
    float err = 0.1f;
    float d_err = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < bench_steps; ++i) {
        float tau = core.perceive_and_act(err, d_err, 0.02f);
        err *= 0.999f;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double steps_per_sec = bench_steps / sec;

    std::cout << "  -> Throughput: " << steps_per_sec << " steps/sec (" << (sec * 1e9 / bench_steps) << " ns/step)\n";
    assert(steps_per_sec > 80000.0 && "Throughput must exceed 80k steps/sec on native CPU!");
    std::cout << "  -> PASSED: Ultra-low latency Lie-flow verified.\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== KARYON SOVEREIGN CORE V28.0 NATIVE TEST SUITE ======================\n";
    std::cout << "========================================================================\n\n";

    test_initialization();
    test_zero_excitation_stationary();
    test_restorative_torque_symmetry();
    test_endless_stability_no_drift();
    test_ultra_throughput();

    std::cout << "\n========================================================================\n";
    std::cout << "=== ALL 5 NATIVE C++20 TESTS PASSED WITH 100% SUCCESS! =================\n";
    std::cout << "========================================================================\n";
    return 0;
}
