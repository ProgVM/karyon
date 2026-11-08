// tests/test_v30_unbounded_core.cpp
// Rigorous Sanity Test for Updated Karyon-CoRE Wavefield Engine (EXP-1104 / EXP-1103 Core)

#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include "include/karyon_core_wavefield.hpp"

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "=== KARYON-CORE V30.0 UNBOUNDED BIT-STREAM CORE INTEGRATION TEST ========" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    karyon::ContinuousWavefieldEngine engine;

    const size_t TOTAL_BITS = 40000;
    std::vector<uint8_t> stream(TOTAL_BITS);

    for (size_t t = 0; t < TOTAL_BITS; ++t) {
        uint8_t b1 = (t % 7 == 0) ? 1 : 0;
        uint8_t b2 = ((t / 3) % 11 == 0) ? 1 : 0;
        uint8_t b3 = (t >= 17) ? stream[t - 17] : 0;
        stream[t] = b1 ^ b2 ^ b3;
    }

    std::cout << "[TEST 1] Ingesting continuous raw bit stream (" << TOTAL_BITS << " bits)..." << std::endl;
    for (size_t t = 0; t < TOTAL_BITS; ++t) {
        engine.StepBit(stream[t], true, false);
    }

    std::cout << "Bits Processed: " << engine.GetTotalBitsEvaluated() << std::endl;
    std::cout << "Initial Bit Errors: " << engine.GetTotalBitErrors() << std::endl;
    std::cout << "Initial Free Energy: " << std::fixed << std::setprecision(8) << engine.GetFreeEnergyNats() << " nats" << std::endl;
    std::cout << "Endogenous Taps Sprouted: " << engine.GetSproutedTapCount() << std::endl;
    std::cout << "Sprouted States in Memory: " << engine.GetSproutedNodeCount() << std::endl;

    assert(engine.GetSproutedTapCount() > 1 && "Endogenous taps must sprout beyond initial seed!");
    assert(engine.GetSproutedNodeCount() > 0 && "Substrate memory must sprout attractor nodes!");

    std::cout << "\n[TEST 2] Verification on continuation stream (Learning Frozen, F -> 0)..." << std::endl;
    engine.ResetEvaluationStats();

    const size_t EVAL_BITS = 20000;
    std::vector<uint8_t> eval_stream(EVAL_BITS);
    for (size_t t = 0; t < EVAL_BITS; ++t) {
        size_t global_t = TOTAL_BITS + t;
        uint8_t b1 = (global_t % 7 == 0) ? 1 : 0;
        uint8_t b2 = ((global_t / 3) % 11 == 0) ? 1 : 0;
        uint8_t b3 = (t >= 17) ? eval_stream[t - 17] : stream[TOTAL_BITS - 17 + t];
        eval_stream[t] = b1 ^ b2 ^ b3;
    }

    for (size_t t = 0; t < EVAL_BITS; ++t) {
        engine.StepBit(eval_stream[t], false, false);
    }

    double final_fe = engine.GetFreeEnergyNats();
    uint64_t eval_errors = engine.GetTotalBitErrors();

    std::cout << "Evaluation Bits: " << engine.GetTotalBitsEvaluated() << std::endl;
    std::cout << "Evaluation Bit Errors: " << eval_errors << " / " << engine.GetTotalBitsEvaluated() << std::endl;
    std::cout << "Final Free Energy: " << std::fixed << std::setprecision(8) << final_fe << " nats" << std::endl;

    assert(eval_errors == 0 && "Verification errors must be 0 for deterministic sequence!");
    assert(final_fe == 0.0 && "Final Free Energy must converge to absolute zero (0.00000000 nats)!");

    std::cout << "\n=========================================================================" << std::endl;
    std::cout << ">>> VERDICT: 🟢 100% CORE INTEGRATION SANITY PASSED <<<" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}
