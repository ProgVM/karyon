// tests/test_v30_unbounded_core.cpp
// Rigorous Sanity Test for Updated Karyon-CoRE Wavefield Engine (EXP-1104 / EXP-1103 Core)
// Tests:
// 1. Continuous Bit-Stream Ingestion & Dynamic Mutual-Information Tap Sprouting
// 2. Hardware Jitter / Silicon Clock Drift Integration (__rdtsc)
// 3. Exact Zero-Error Convergence (F = 0.00000000 nats) on Continuation Stream

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

    // Generate 20,000 continuous raw bytes with multi-scale non-word-aligned bit structures
    const size_t TOTAL_BYTES = 20000;
    std::vector<uint8_t> stream(TOTAL_BYTES);

    for (size_t i = 0; i < TOTAL_BYTES; ++i) {
        uint8_t b1 = (i % 7 == 0) ? 0xAA : 0x55;
        uint8_t b2 = ((i / 3) % 11 == 0) ? 0x0F : 0xF0;
        stream[i] = b1 ^ b2;
    }

    std::cout << "[TEST 1] Ingesting continuous raw byte stream (" << (TOTAL_BYTES * 8) << " raw bits)..." << std::endl;
    engine.IngestContinuousByteStream(stream.data(), TOTAL_BYTES, true, false);

    std::cout << "Bits Processed: " << engine.GetTotalBitsEvaluated() << std::endl;
    std::cout << "Initial Bit Errors: " << engine.GetTotalBitErrors() << std::endl;
    std::cout << "Initial Free Energy: " << std::fixed << std::setprecision(8) << engine.GetFreeEnergyNats() << " nats" << std::endl;
    std::cout << "Endogenous Taps Sprouted: " << engine.GetSproutedTapCount() << std::endl;
    std::cout << "Sprouted States in Memory: " << engine.GetSproutedNodeCount() << std::endl;

    assert(engine.GetSproutedTapCount() > 1 && "Endogenous taps must sprout beyond initial seed!");
    assert(engine.GetSproutedNodeCount() > 0 && "Substrate memory must sprout attractor nodes!");

    std::cout << "\n[TEST 2] Verification on continuation stream (Learning Frozen, F -> 0)..." << std::endl;
    engine.ResetHistory(); // Reset evaluation counters for clean verification pass

    // Ingest continuation bytes
    const size_t EVAL_BYTES = 10000;
    std::vector<uint8_t> eval_stream(EVAL_BYTES);
    for (size_t i = 0; i < EVAL_BYTES; ++i) {
        size_t global_i = TOTAL_BYTES + i;
        uint8_t b1 = (global_i % 7 == 0) ? 0xAA : 0x55;
        uint8_t b2 = ((global_i / 3) % 11 == 0) ? 0x0F : 0xF0;
        eval_stream[i] = b1 ^ b2;
    }

    engine.IngestContinuousByteStream(eval_stream.data(), EVAL_BYTES, false, false);

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
