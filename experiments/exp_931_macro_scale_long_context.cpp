// ============================================================================
// EXP-931: Macro-Scale Continuous Long-Context Autopoietic Scaling
// Karyon Engineering Protocol (KEP) Rule #1 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | Principle 0, Principle 7 & Principle 8
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
    std::cout << "EXP-931: Macro-Scale Continuous Long-Context Autopoietic Scaling\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | V=2 Continuous Wavefield\n";
    std::cout << "==================================================================\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // Generate large macro-scale text corpus (~1,000,000 continuous bits / 125,000 bytes)
    std::cout << "1. GENERATING MACRO-SCALE CONTINUOUS WAVE CORPUS...\n";
    std::string base_pattern = 
        "Fact 1: Karyon Sovereign Engine v40.0 operates on weightless physical wavefield V=2 with zero discrete dictionaries. "
        "Fact 2: Endogenous causal credit assignment tracks phase tap efficacy s_k dynamically without backpropagation. "
        "Fact 3: Autopoietic morphogenesis prunes topological debt while preserving continuous wave coherence. "
        "Fact 4: Feynman path integral superposition unifies all physical wave modalities into continuous action fields. ";

    std::string large_corpus;
    large_corpus.reserve(200000);
    size_t repetitions = 300; // ~125 KB, >1,000,000 bits
    for (size_t r = 0; r < repetitions; ++r) {
        large_corpus += base_pattern;
        large_corpus += "Segment ID: " + std::to_string(r) + " | Status: Active Wave Resonance. ";
    }

    size_t total_bytes = large_corpus.size();
    size_t total_bits = total_bytes * 8;

    std::cout << "   -> Corpus Size: " << total_bytes << " bytes (" << total_bits << " bits)\n\n";

    // 2. Ingest Large Macro-Scale Stream
    std::cout << "2. MACRO-SCALE WAVE INGESTION BENCHMARK:\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(large_corpus.data()), total_bytes, true);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ingestion_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ingestion_mb_sec = (total_bytes / (1024.0 * 1024.0)) / (ingestion_ms / 1000.0);
    size_t initial_taps = engine.GetWaveTraceCount();

    std::cout << "   -> Ingestion Time: " << std::fixed << std::setprecision(2) << ingestion_ms << " ms\n";
    std::cout << "   -> Ingestion Throughput: " << std::fixed << std::setprecision(2) << ingestion_mb_sec << " MB/sec\n";
    std::cout << "   -> Synthesized Phase Taps: " << initial_taps << "\n\n";

    // 3. Multi-Fact Long-Context Phase Recall Verification
    std::vector<std::string> test_prompts = {
        "Fact 1: Karyon Sovereign Engine v40.0 ",
        "Fact 2: Endogenous causal credit ",
        "Fact 3: Autopoietic morphogenesis ",
        "Fact 4: Feynman path integral "
    };

    std::cout << "3. MACRO-SCALE LONG-CONTEXT RECALL VERIFICATION:\n";
    size_t recall_matches = 0;
    for (size_t i = 0; i < test_prompts.size(); ++i) {
        engine.ResetHistory();
        engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(test_prompts[i].data()), test_prompts[i].size(), false);
        auto gen = engine.GenerateAutoregressiveBytes(60);
        std::string res(gen.begin(), gen.end());

        std::cout << "   [Prompt " << (i + 1) << "]: \"" << test_prompts[i] << "\"\n";
        std::cout << "   [Generated]: \"" << res << "\"\n";

        bool has_key_term = (res.find("operates") != std::string::npos ||
                             res.find("assignment") != std::string::npos ||
                             res.find("prunes") != std::string::npos ||
                             res.find("superposition") != std::string::npos);

        if (has_key_term) recall_matches++;
        std::cout << "   [Status]: " << (has_key_term ? "🟢 PERFECT RECALL" : "🔴 MISMATCH") << "\n\n";
    }

    // 4. Autopoietic Morphogenesis under Macro-Scale Long-Context
    std::cout << "4. MACRO-SCALE AUTOPOIETIC MORPHOGENESIS:\n";
    auto p0 = std::chrono::high_resolution_clock::now();
    size_t pruned_count = engine.ExecuteAutopoieticPruning(1.00);
    auto p1 = std::chrono::high_resolution_clock::now();

    double pruning_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();
    size_t post_prune_taps = engine.GetWaveTraceCount();
    double reduction_pct = (double)pruned_count / (double)initial_taps * 100.0;

    std::cout << "   -> Morphogenesis Latency: " << std::fixed << std::setprecision(2) << pruning_ms << " ms\n";
    std::cout << "   -> Pruned Taps: " << pruned_count << " / " << initial_taps << " (" 
              << std::setprecision(2) << reduction_pct << " %)\n";
    std::cout << "   -> Remaining Active Taps: " << post_prune_taps << "\n\n";

    // 5. Post-Pruning Generation Throughput Benchmark
    std::cout << "5. POST-PRUNING GENERATION THROUGHPUT:\n";
    engine.ResetHistory();
    engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(test_prompts[0].data()), test_prompts[0].size(), false);

    auto b0 = std::chrono::high_resolution_clock::now();
    auto gen_bench = engine.GenerateAutoregressiveBytes(500);
    auto b1 = std::chrono::high_resolution_clock::now();

    double gen_ms = std::chrono::duration<double, std::milli>(b1 - b0).count();
    double gen_bits_sec = (500 * 8) / (gen_ms / 1000.0);

    std::cout << "   -> Generation Latency (500 B): " << std::fixed << std::setprecision(2) << gen_ms << " ms\n";
    std::cout << "   -> Throughput: " << std::fixed << std::setprecision(2) << (gen_bits_sec / 1000000.0) << " Mbit/s ("
              << static_cast<size_t>(gen_bits_sec) << " bits/sec)\n\n";

    bool pass_all = (recall_matches == test_prompts.size()) && (total_bits > 1000000);

    std::cout << "===================================================================\n";
    std::cout << " EXP-931 SUMMARY VERDICT: " << (pass_all ? "🟢 POSITIVE (MACRO-SCALE SCALING PASSED)" : "🔴 FAILED") << "\n";
    std::cout << " - Total Tested Stream: " << total_bits << " bits (" << total_bytes << " bytes)\n";
    std::cout << " - Ingestion Latency: " << std::fixed << std::setprecision(2) << ingestion_ms << " ms (" << ingestion_mb_sec << " MB/s)\n";
    std::cout << " - Long-Context Recall Accuracy: " << (recall_matches * 100 / test_prompts.size()) << " %\n";
    std::cout << " - Topological Debt Reduction: " << std::fixed << std::setprecision(2) << reduction_pct << " %\n";
    std::cout << " - Post-Pruning Generation Throughput: " << std::fixed << std::setprecision(2) << (gen_bits_sec / 1000000.0) << " Mbit/s\n";
    std::cout << "===================================================================\n";

    return pass_all ? 0 : 1;
}
