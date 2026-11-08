// EXP-1107: Dual Deterministic + Non-Deterministic Sovereign Bit-Coupling Substrate
// Standards: KEP Principle 0 (Zero Imposed Mechanisms), KEP Principle 4 (V=2), KEP Principle 13 (Anti-Staticity)

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include "include/karyon_core_wavefield.hpp"

std::vector<uint8_t> GenerateMultimodalBitStream() {
    std::vector<uint8_t> bytes;

    // Modality 1: Audio PCM Waveform
    for (int t = 0; t < 500; ++t) {
        int16_t pcm_sample = static_cast<int16_t>(16384.0 * std::sin(2.0 * M_PI * t / 20.0));
        bytes.push_back(static_cast<uint8_t>(pcm_sample & 0xFF));
        bytes.push_back(static_cast<uint8_t>((pcm_sample >> 8) & 0xFF));
    }

    // Modality 2: UTF-8 Text Formant
    std::string text = "Karyon-CoRE Principle 13 Dual Substrate: Determinism + Non-Determinism!";
    for (char c : text) {
        bytes.push_back(static_cast<uint8_t>(c));
    }

    // Modality 3: x86 Bytecode
    uint8_t bytecode[] = { 0x90, 0x90, 0x48, 0x31, 0xC0, 0x48, 0xFF, 0xC0, 0x0F, 0x05, 0xC3 };
    for (int i = 0; i < 5; ++i) {
        for (uint8_t b : bytecode) {
            bytes.push_back(b);
        }
    }

    return bytes;
}

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "=== EXP-1107: DUAL DETERMINISTIC + NON-DETERMINISTIC BIT SUBSTRATE ======" << std::endl;
    std::cout << "=== KEP Principle 0 & 13: Native Bits, Dual Phase Attractors ============" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    std::vector<uint8_t> stream = GenerateMultimodalBitStream();
    size_t total_bits = stream.size() * 8;

    karyon::ContinuousWavefieldEngine engine;

    // -------------------------------------------------------------------------
    // TEST 1: DETERMINISTIC CONVERGENCE (Zero Noise -> F must be 0.00000000 nats)
    // -------------------------------------------------------------------------
    std::cout << "\n--- TEST 1: DETERMINISTIC CONVERGENCE TEST ---" << std::endl;
    engine.IngestContinuousByteStream(stream.data(), stream.size(), true, false, 0.0);
    engine.IngestContinuousByteStream(stream.data(), stream.size(), true, false, 0.0);

    engine.ResetEvaluationStats();
    engine.IngestContinuousByteStream(stream.data(), stream.size(), false, false, 0.0);

    double det_fe = engine.GetFreeEnergyNats();
    uint64_t det_errors = engine.GetTotalBitErrors();

    std::cout << "Deterministic Bits: " << engine.GetTotalBitsEvaluated() << std::endl;
    std::cout << "Deterministic Bit Errors: " << det_errors << std::endl;
    std::cout << "Deterministic Free Energy: " << std::fixed << std::setprecision(8) << det_fe << " nats" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 2: NON-DETERMINISTIC NOISE RESILIENCE (5% Jitter Noise -> Soft Attractor Snapping)
    // -------------------------------------------------------------------------
    std::cout << "\n--- TEST 2: NON-DETERMINISTIC CHAOS / NOISE RESILIENCE (5% Jitter) ---" << std::endl;
    engine.ResetEvaluationStats();
    engine.IngestContinuousByteStream(stream.data(), stream.size(), false, true, 0.05);

    double noisy_fe = engine.GetFreeEnergyNats();
    uint64_t noisy_errors = engine.GetTotalBitErrors();
    double noisy_err_rate = static_cast<double>(noisy_errors) / engine.GetTotalBitsEvaluated() * 100.0;

    std::cout << "Noisy Bits Evaluated: " << engine.GetTotalBitsEvaluated() << std::endl;
    std::cout << "Noisy Bit Errors: " << noisy_errors << " (" << std::fixed << std::setprecision(2) << noisy_err_rate << "%)" << std::endl;
    std::cout << "Noisy Free Energy: " << std::fixed << std::setprecision(8) << noisy_fe << " nats" << std::endl;

    // -------------------------------------------------------------------------
    // KEP EVALUATION
    // -------------------------------------------------------------------------
    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "=== KEP DUAL-PHASE PRINCIPLE 13 EVALUATION ==============================" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    bool det_passed = (det_fe == 0.0 && det_errors == 0);
    bool noisy_passed = (noisy_fe < 0.40 && noisy_err_rate < 10.0); // Soft snapping prevents crash

    std::cout << "Deterministic Phase (F = 0.00000000): " << (det_passed ? "🟢 PASSED" : "🔴 FAILED") << std::endl;
    std::cout << "Non-Deterministic Soft Basin Snapping: " << (noisy_passed ? "🟢 PASSED" : "🔴 FAILED") << std::endl;

    bool overall = det_passed && noisy_passed;
    std::cout << "Overall Substrate Status: " << (overall ? "🟢 DUAL-PHASE SOVEREIGN SUCCESS" : "🔴 FAIL") << std::endl;

    return overall ? 0 : 1;
}
