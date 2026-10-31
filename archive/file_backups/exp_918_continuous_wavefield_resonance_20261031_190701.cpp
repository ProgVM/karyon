// ============================================================================
// KARYON-CORE EXPERIMENT EXP-918
// Title: Continuous Wavefield Spin-Resonance Substrate Benchmark (Bazilevs' Dream)
// Standard: KEP v40.0 Sovereign Master | Principle 0: Bazilevs' Dream
//
// OBJECTIVE:
// Audit the pure continuous wavefield spin-resonance engine (EXP-918):
// - ABSOLUTELY ZERO HASH MAPS, DISCRETE TABLES, OR TOP-DOWN DICTIONARIES
// - Continuous physical wave spin prediction \hat{s}_t = tanh(W_{tap}^T \Psi(t))
// - Local Online Free Energy Plasticity Update
// - Continuous Hilbert Space Phase Trace Resonance under noise
//
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cassert>

#include "../include/karyon_core_wavefield.hpp"

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-918: CONTINUOUS WAVEFIELD SPIN-RESONANCE BENCHMARK\n";
    std::cout << " Standard: KEP v40.0 | Pure Continuous Wavefield Substrate (Bazilevs' Dream)\n";
    std::cout << " Features: Zero Hash Maps | Zero Discrete Tables | Pure Wave Dynamics\n";
    std::cout << "===================================================================\n\n";

    std::string ref_text = 
        "Karyon Sovereign Engine v40.0 operates on pure continuous wavefield spin-resonance. "
        "Zero imposed mechanisms, zero hash tables, zero backprop, absolute zero error!";

    std::cout << "[STEP 1] Training Continuous Substrate Engine on reference text (" << (ref_text.size() * 8) << " bits)...\n";
    karyon::ContinuousWavefieldEngine engine(0.08); // Online learning rate 0.08

    auto start_time = std::chrono::high_resolution_clock::now();
    engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(ref_text.data()), ref_text.size(), true);
    auto end_time = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "  -> Training Completed in: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "  -> Phase Traces Synthesized: " << engine.GetPhaseTraceCount() << "\n\n";

    // ------------------------------------------------------------------------
    // TEST A: Deterministic Sequence Mastery (Zero Error Verification)
    // ------------------------------------------------------------------------
    std::string clean_seed_str = "Karyon Sovereign Engine v40.0 ";
    std::vector<uint8_t> clean_seed_bytes(clean_seed_str.begin(), clean_seed_str.end());

    std::string expected_target = ref_text.substr(clean_seed_str.size());
    size_t target_gen_bytes = expected_target.size();

    std::cout << "[TEST A] Autoregressive Generation from Clean Seed:\n";
    std::cout << "  -> Seed: \"" << clean_seed_str << "\"\n";

    engine.ResetHistory();
    engine.IngestRawBytes(clean_seed_bytes.data(), clean_seed_bytes.size(), false);

    auto gen_bytes_clean = engine.GenerateAutoregressiveBytes(target_gen_bytes, 0.20, true);
    std::string gen_clean_str(gen_bytes_clean.begin(), gen_bytes_clean.end());

    size_t match_clean = 0;
    for (size_t i = 0; i < target_gen_bytes; ++i) {
        if (gen_bytes_clean[i] == expected_target[i]) match_clean++;
    }
    double acc_clean = 100.0 * static_cast<double>(match_clean) / target_gen_bytes;

    std::cout << "  -> Generated Continuation:\n\"" << gen_clean_str << "\"\n";
    std::cout << "  -> Continuation Accuracy (Clean): " << std::fixed << std::setprecision(2) << acc_clean << "%\n\n";

    // ------------------------------------------------------------------------
    // TEST B: Transmission Bit Noise Recovery (Phase Resonance under Noise)
    // ------------------------------------------------------------------------
    std::vector<uint8_t> noisy_seed_bytes = clean_seed_bytes;
    noisy_seed_bytes[noisy_seed_bytes.size() - 2] ^= 0x05; // Flip 2 bits in second-to-last byte

    std::cout << "[TEST B] Autoregressive Generation from Noisy Seed (Injected Bit Flips):\n";
    std::cout << "  -> Noisy Seed: \"" << std::string(noisy_seed_bytes.begin(), noisy_seed_bytes.end()) << "\"\n";

    engine.ResetHistory();
    engine.IngestRawBytes(noisy_seed_bytes.data(), noisy_seed_bytes.size(), false);

    auto gen_bytes_noisy = engine.GenerateAutoregressiveBytes(target_gen_bytes, 0.25, true);
    std::string gen_noisy_str(gen_bytes_noisy.begin(), gen_bytes_noisy.end());

    size_t match_noisy = 0;
    for (size_t i = 0; i < target_gen_bytes; ++i) {
        if (gen_bytes_noisy[i] == expected_target[i]) match_noisy++;
    }
    double acc_noisy = 100.0 * static_cast<double>(match_noisy) / target_gen_bytes;

    std::cout << "  -> Generated Continuation (EXP-918 Phase Resonance ON):\n\"" << gen_noisy_str << "\"\n";
    std::cout << "  -> Continuation Accuracy (Noisy): " << std::fixed << std::setprecision(2) << acc_noisy << "%\n\n";

    // ------------------------------------------------------------------------
    // TEST C: High-Speed .kcore v40 State Persistence (.kcore)
    // ------------------------------------------------------------------------
    std::cout << "[TEST C] Testing High-Speed Binary .kcore v40 Serialization...\n";
    std::string kcore_file = "exp_918_wavefield.kcore";
    
    bool save_ok = engine.SaveState(kcore_file);
    std::cout << "  -> Save State (.kcore v40): " << (save_ok ? "SUCCESS" : "FAILED") << "\n";

    karyon::ContinuousWavefieldEngine loaded_engine(0.08);
    bool load_ok = loaded_engine.LoadState(kcore_file);
    std::cout << "  -> Load State (.kcore v40): " << (load_ok ? "SUCCESS" : "FAILED") << "\n";

    loaded_engine.ResetHistory();
    loaded_engine.IngestRawBytes(clean_seed_bytes.data(), clean_seed_bytes.size(), false);

    auto loaded_gen_bytes = loaded_engine.GenerateAutoregressiveBytes(target_gen_bytes, 0.20, true);
    bool parity_match = (gen_bytes_clean == loaded_gen_bytes);

    std::cout << "  -> State Parity Match Post-Deserialization: " << (parity_match ? "100.00% MATCH 🏆" : "DISCREPANCY DETECTED") << "\n\n";

    // ------------------------------------------------------------------------
    // SUMMARY REPORT
    // ------------------------------------------------------------------------
    bool overall_pass = (acc_clean == 100.0) && (acc_noisy == 100.0) && parity_match;

    std::cout << "===================================================================\n";
    std::cout << " EXP-918 BAZILEVS' DREAM CONTINUOUS BENCHMARK SUMMARY REPORT:\n";
    std::cout << "  - Substrate Model: Pure Continuous Wavefield Spin-Resonance (EXP-918)\n";
    std::cout << "  - Imposed Hash Maps / Discrete Tables: ZERO (0)\n";
    std::cout << "  - Clean Seed Accuracy: " << acc_clean << "% (Deterministic Mastery)\n";
    std::cout << "  - Noisy Seed Accuracy: " << acc_noisy << "% (Phase Space Snapping)\n";
    std::cout << "  - .kcore v40 Parity Match: " << (parity_match ? "100.00%" : "Failed") << "\n";
    std::cout << "  -> OVERALL VERDICT: " << (overall_pass ? "🟢 EXP-918 BAZILEVS' DREAM VERIFIED SUCCESS 🏆" : "🔴 REGRESSION DETECTED") << "\n";
    std::cout << "===================================================================\n";

    return overall_pass ? 0 : 1;
}
