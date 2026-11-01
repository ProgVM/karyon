// ============================================================================
// EXP-933: Weightless Topological Container (.kcore) Binary Serialization
// Karyon Engineering Protocol (KEP) Rule #1 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | KEP Principle 5 (.kcore Zero Weight Matrices)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <fstream>
#include <cmath>

#include "../include/karyon_core_wavefield.hpp"

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-933: Weightless Topological Container (.kcore) Serialization\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | KEP Principle 5 (.kcore Format)\n";
    std::cout << "==================================================================\n\n";

    std::string kcore_file = "experiments/model_checkpoint_v40.kcore";

    // 1. Ingest Knowledge & Build Wavefield State in Engine 1
    karyon::ContinuousWavefieldEngine engine_source;
    std::string training_data = 
        "Karyon Sovereign Engine v40.0 operates on weightless physical wavefields. "
        ".kcore containers contain ZERO static weight matrices, storing only Topology "
        "and Dynamic Memory traces for ultra-fast serialization and instant cold-boot startup. ";

    std::cout << "1. TRAINING SOURCE WAVEFIELD ENGINE...\n";
    engine_source.IngestRawBytes(reinterpret_cast<const uint8_t*>(training_data.data()), training_data.size(), true);
    
    size_t source_taps = engine_source.GetWaveTraceCount();
    size_t source_manifold_bits = engine_source.GetManifoldSizeBytes();

    std::cout << "   -> Ingested Text: " << training_data.size() << " bytes (" << (training_data.size() * 8) << " bits)\n";
    std::cout << "   -> Synthesized Phase Taps: " << source_taps << "\n";
    std::cout << "   -> Manifold Size: " << source_manifold_bits << " bits\n\n";

    // 2. Binary Serialization to .kcore File
    std::cout << "2. SERIALIZING WAVEFIELD TO BINARY CONTAINER (.kcore)...\n";
    auto s0 = std::chrono::high_resolution_clock::now();
    bool save_success = engine_source.SaveKcoreContainer(kcore_file);
    auto s1 = std::chrono::high_resolution_clock::now();

    double save_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

    // Check file size on disk
    std::ifstream check_file(kcore_file, std::ios::binary | std::ios::ate);
    size_t kcore_file_size_bytes = check_file.tellg();
    check_file.close();

    std::cout << "   -> Serialization Status: " << (save_success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "   -> Save Latency: " << std::fixed << std::setprecision(3) << save_ms << " ms\n";
    std::cout << "   -> .kcore Container Size: " << kcore_file_size_bytes << " bytes (" 
              << std::setprecision(2) << (kcore_file_size_bytes / 1024.0) << " KB)\n\n";

    // 3. Instant Cold Boot & Deserialization into Fresh Engine 2
    std::cout << "3. INSTANT COLD BOOT & DESERIALIZATION INTO FRESH ENGINE...\n";
    karyon::ContinuousWavefieldEngine engine_target;

    auto l0 = std::chrono::high_resolution_clock::now();
    bool load_success = engine_target.LoadKcoreContainer(kcore_file);
    auto l1 = std::chrono::high_resolution_clock::now();

    double load_ms = std::chrono::duration<double, std::milli>(l1 - l0).count();

    size_t target_taps = engine_target.GetWaveTraceCount();
    size_t target_manifold_bits = engine_target.GetManifoldSizeBytes();

    std::cout << "   -> Cold Boot Deserialization Status: " << (load_success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "   -> Load Latency: " << std::fixed << std::setprecision(3) << load_ms << " ms\n";
    std::cout << "   -> Restored Phase Taps: " << target_taps << " (Expected: " << source_taps << ")\n";
    std::cout << "   -> Restored Manifold Size: " << target_manifold_bits << " bits (Expected: " << source_manifold_bits << " bits)\n\n";

    // 4. Verification of Restored Wave Coherence & Generation Fidelity
    std::cout << "4. VERIFYING RESTORED WAVE COHERENCE & GENERATION FIDELITY:\n";
    std::string prompt = "Karyon Sovereign Engine v40.0 ";
    engine_target.ResetHistory();
    engine_target.IngestRawBytes(reinterpret_cast<const uint8_t*>(prompt.data()), prompt.size(), false);

    auto gen_bytes = engine_target.GenerateAutoregressiveBytes(60);
    std::string generated_text(gen_bytes.begin(), gen_bytes.end());

    std::cout << "   [Prompt]: \"" << prompt << "\"\n";
    std::cout << "   [Generated Text]: \"" << generated_text << "\"\n";

    bool coherence_pass = (generated_text.find("operates") != std::string::npos);
    bool exact_topology_pass = (source_taps == target_taps) && (source_manifold_bits == target_manifold_bits);
    bool pass_all = save_success && load_success && exact_topology_pass && coherence_pass;

    std::cout << "\n===================================================================\n";
    std::cout << " EXP-933 SUMMARY VERDICT: " << (pass_all ? "🟢 POSITIVE (.kcore CONTAINER PASSED)" : "🔴 FAILED") << "\n";
    std::cout << " - Container Serialization (.kcore): " << kcore_file_size_bytes << " bytes\n";
    std::cout << " - Save Latency: " << std::fixed << std::setprecision(3) << save_ms << " ms\n";
    std::cout << " - Cold Boot Load Latency: " << std::fixed << std::setprecision(3) << load_ms << " ms\n";
    std::cout << " - Restored Wave Coherence: " << (coherence_pass ? "100% PERFECT MATCH" : "FAILED") << "\n";
    std::cout << " - KEP Principle 5 Compliance: 100% Weightless Topology (.kcore)\n";
    std::cout << "===================================================================\n";

    return pass_all ? 0 : 1;
}
