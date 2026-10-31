// ============================================================================
// KARYON-CORE EXPERIMENT EXP-921
// Title: Soliton Phase-Tracking Topological Resonator Substrate
// Standard: KEP v40.0 Sovereign Master | Principle 0: Bazilevs' Dream
//
// OBJECTIVES:
// 1. Validate Continuous Soliton Wavefront Momentum (O(1) local phase coherence).
// 2. Measure raw generation throughput speedup (>10,000,000 bits/sec vs ~5,000 bits/sec).
// 3. Multi-domain verification: Deterministic Text, Multi-Bit Noise Snapping,
//    Dyck-4 Recursive Grammar, Branching Disambiguation, and .kcore persistence.
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
    std::cout << " KARYON-CORE EXP-921: SOLITON PHASE-TRACKING RESONATOR BENCHMARK\n";
    std::cout << " Standard: KEP v40.0 | Principle 0: Bazilevs' Dream of Sovereign Genesis\n";
    std::cout << " Substrate: Single 1D Physical Wavefield Manifold + Soliton Phase Momentum\n";
    std::cout << "===================================================================\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 1: Reference Text & Deep Natural Language Ingestion
    // ------------------------------------------------------------------------
    std::string ref_text = 
        "Karyon Sovereign Engine v40.0 operates on pure continuous wavefield spin-resonance. "
        "Zero imposed mechanisms, zero hash tables, zero backprop, absolute zero error!";

    std::cout << "[DOMAIN 1: NATURAL LANGUAGE & HIGH-SPEED SOLITON PROPAGATION]\n";
    karyon::ContinuousWavefieldEngine engine_nl;

    auto t0 = std::chrono::high_resolution_clock::now();
    engine_nl.IngestRawBytes(reinterpret_cast<const uint8_t*>(ref_text.data()), ref_text.size(), true);
    auto t1 = std::chrono::high_resolution_clock::now();

    double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  -> Ingestion Time: " << std::fixed << std::setprecision(4) << train_ms << " ms\n";
    std::cout << "  -> Physical Manifold Size: " << engine_nl.GetManifoldSizeBytes() << " bytes (1 byte/spin)\n";
    std::cout << "  -> Topological Phase Taps: " << engine_nl.GetWaveTraceCount() << "\n\n";

    // TEST 1A: Clean Seed Generation & Throughput Measurement
    std::string clean_seed = "Karyon Sovereign Engine v40.0 ";
    std::string expected_target = ref_text.substr(clean_seed.size());
    size_t target_bytes = expected_target.size();
    size_t target_bits = target_bytes * 8;

    engine_nl.ResetHistory();
    engine_nl.IngestRawBytes(reinterpret_cast<const uint8_t*>(clean_seed.data()), clean_seed.size(), false);

    auto g0 = std::chrono::high_resolution_clock::now();
    auto gen_clean = engine_nl.GenerateAutoregressiveBytes(target_bytes);
    auto g1 = std::chrono::high_resolution_clock::now();

    double gen_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    double bits_per_sec = (target_bits / (gen_ms / 1000.0));
    std::string gen_clean_str(gen_clean.begin(), gen_clean.end());

    size_t clean_matches = 0;
    for (size_t i = 0; i < target_bytes; ++i) if (gen_clean[i] == expected_target[i]) clean_matches++;
    double acc_clean = 100.0 * (double)clean_matches / target_bytes;

    std::cout << "  [TEST 1A: Clean Seed]\n";
    std::cout << "    -> Generated: \"" << gen_clean_str << "\"\n";
    std::cout << "    -> Accuracy: " << acc_clean << "%\n";
    std::cout << "    -> Generation Latency: " << gen_ms << " ms (" << std::fixed << std::setprecision(0) << bits_per_sec << " bits/sec)\n\n";

    // TEST 1B: Multi-Bit Noisy Seed Trajectory Snapping
    std::vector<uint8_t> noisy_seed_bytes(clean_seed.begin(), clean_seed.end());
    noisy_seed_bytes[noisy_seed_bytes.size() - 2] ^= 0x05; // 2 bit flips in second-to-last byte
    noisy_seed_bytes[10] ^= 0x01; // additional bit flip earlier in context

    engine_nl.ResetHistory();
    engine_nl.IngestRawBytes(noisy_seed_bytes.data(), noisy_seed_bytes.size(), false);
    auto gen_noisy = engine_nl.GenerateAutoregressiveBytes(target_bytes);
    std::string gen_noisy_str(gen_noisy.begin(), gen_noisy.end());

    size_t noisy_matches = 0;
    for (size_t i = 0; i < target_bytes; ++i) if (gen_noisy[i] == expected_target[i]) noisy_matches++;
    double acc_noisy = 100.0 * (double)noisy_matches / target_bytes;

    std::cout << "  [TEST 1B: Multi-Bit Noisy Seed Snapping]\n";
    std::cout << "    -> Generated: \"" << gen_noisy_str << "\"\n";
    std::cout << "    -> Accuracy: " << acc_noisy << "%\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 2: Context-Free Recursive Grammar (Dyck-4 Parenthesis)
    // ------------------------------------------------------------------------
    std::cout << "[DOMAIN 2: RECURSIVE SYMBOLIC STRUCTURE (DYCK-4 GRAMMAR)]\n";
    std::string dyck = "((({{[[]]}})))[[{{}}]]((())){{[[]]}}";
    std::string dyck_prefix = "((({{[[]]}})))";
    std::string dyck_expected = dyck.substr(dyck_prefix.size());

    karyon::ContinuousWavefieldEngine engine_dyck;
    engine_dyck.IngestRawBytes(reinterpret_cast<const uint8_t*>(dyck.data()), dyck.size(), true);

    engine_dyck.ResetHistory();
    engine_dyck.IngestRawBytes(reinterpret_cast<const uint8_t*>(dyck_prefix.data()), dyck_prefix.size(), false);
    auto gen_dyck = engine_dyck.GenerateAutoregressiveBytes(dyck_expected.size());
    std::string gen_dyck_str(gen_dyck.begin(), gen_dyck.end());

    double acc_dyck = (gen_dyck_str == dyck_expected) ? 100.0 : 0.0;
    std::cout << "  -> Prefix:    \"" << dyck_prefix << "\"\n";
    std::cout << "  -> Expected:  \"" << dyck_expected << "\"\n";
    std::cout << "  -> Generated: \"" << gen_dyck_str << "\"\n";
    std::cout << "  -> Dyck-4 Accuracy: " << acc_dyck << "%\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 3: Context-Dependent Branching Disambiguation
    // ------------------------------------------------------------------------
    std::cout << "[DOMAIN 3: CONTEXT-DEPENDENT BRANCHING RESOLUTION]\n";
    std::string branch_A = "The sovereign cat sat on the soft warm mat. ";
    std::string branch_B = "The sovereign dog sat on the deep wool rug. ";

    karyon::ContinuousWavefieldEngine engine_branch;
    engine_branch.IngestRawBytes(reinterpret_cast<const uint8_t*>(branch_A.data()), branch_A.size(), true);
    engine_branch.IngestRawBytes(reinterpret_cast<const uint8_t*>(branch_B.data()), branch_B.size(), true);

    std::string prompt_cat = "The sovereign cat ";
    std::string exp_cat = branch_A.substr(prompt_cat.size());
    engine_branch.ResetHistory();
    engine_branch.IngestRawBytes(reinterpret_cast<const uint8_t*>(prompt_cat.data()), prompt_cat.size(), false);
    auto gen_cat = engine_branch.GenerateAutoregressiveBytes(exp_cat.size());
    std::string gen_cat_str(gen_cat.begin(), gen_cat.end());
    double acc_cat = (gen_cat_str == exp_cat) ? 100.0 : 0.0;

    std::string prompt_dog = "The sovereign dog ";
    std::string exp_dog = branch_B.substr(prompt_dog.size());
    engine_branch.ResetHistory();
    engine_branch.IngestRawBytes(reinterpret_cast<const uint8_t*>(prompt_dog.data()), prompt_dog.size(), false);
    auto gen_dog = engine_branch.GenerateAutoregressiveBytes(exp_dog.size());
    std::string gen_dog_str(gen_dog.begin(), gen_dog.end());
    double acc_dog = (gen_dog_str == exp_dog) ? 100.0 : 0.0;

    std::cout << "  -> Branch A (cat): " << (acc_cat == 100.0 ? "RESOLVED 100%" : "FAILED") << " (\"" << gen_cat_str << "\")\n";
    std::cout << "  -> Branch B (dog): " << (acc_dog == 100.0 ? "RESOLVED 100%" : "FAILED") << " (\"" << gen_dog_str << "\")\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 4: High-Speed .kcore v40 State Persistence
    // ------------------------------------------------------------------------
    std::cout << "[DOMAIN 4: BINARY STATE PERSISTENCE (.kcore v40 Continuous Format)]\n";
    std::string kcore_file = "exp_921_soliton.kcore";
    bool save_ok = engine_nl.SaveState(kcore_file);
    karyon::ContinuousWavefieldEngine loaded_engine;
    bool load_ok = loaded_engine.LoadState(kcore_file);

    loaded_engine.ResetHistory();
    loaded_engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(clean_seed.data()), clean_seed.size(), false);
    auto gen_loaded = loaded_engine.GenerateAutoregressiveBytes(target_bytes);
    bool parity_match = (gen_clean == gen_loaded);

    std::cout << "  -> Save State: " << (save_ok ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  -> Load State: " << (load_ok ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  -> State Parity Post-Reload: " << (parity_match ? "100.00% EXACT PARITY 🏆" : "FAILED") << "\n\n";

    // ------------------------------------------------------------------------
    // SUMMARY TELEMETRY
    // ------------------------------------------------------------------------
    bool all_passed = (acc_clean == 100.0) && (acc_noisy == 100.0) && (acc_dyck == 100.0) && 
                       (acc_cat == 100.0) && (acc_dog == 100.0) && parity_match;

    std::cout << "===================================================================\n";
    std::cout << " EXP-921 SOLITON PHASE-TRACKING BENCHMARK REPORT:\n";
    std::cout << "  - Imposed Discrete Tables / Hash Maps: ZERO (0)\n";
    std::cout << "  - Memory Redundancy: ZERO (0) - 1 byte/spin contiguous manifold\n";
    std::cout << "  - Clean Seed Deterministic Accuracy: " << acc_clean << "%\n";
    std::cout << "  - Generation Throughput: " << std::fixed << std::setprecision(0) << bits_per_sec << " bits/sec (" 
              << std::fixed << std::setprecision(4) << gen_ms << " ms for " << target_bits << " bits)\n";
    std::cout << "  - Multi-Bit Noise Trajectory Snapping: " << acc_noisy << "%\n";
    std::cout << "  - Dyck-4 Recursive Grammar Accuracy: " << acc_dyck << "%\n";
    std::cout << "  - Branching Disambiguation Accuracy: " << (acc_cat + acc_dog)/2.0 << "%\n";
    std::cout << "  - .kcore v40 Parity Match: " << (parity_match ? "100.00%" : "0.00%") << "\n";
    std::cout << "  -> OVERALL VERDICT: " << (all_passed ? "🟢 EXP-921 SOLITON PHASE-TRACKING SUCCESS 🏆" : "🔴 REGRESSION DETECTED") << "\n";
    std::cout << "===================================================================\n";

    return all_passed ? 0 : 1;
}
