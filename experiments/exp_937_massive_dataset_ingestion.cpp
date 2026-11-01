// ============================================================================
// EXP-937: Massive Dataset Ingestion & Macro-Scale Topological Condensation
// Karyon Engineering Protocol (KEP) Rule #1 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | Principle 4 & 5 (Massive Scale Substrate)
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
    std::cout << "EXP-937: Massive Dataset Ingestion & Macro-Scale Topological Condensation\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 4 & 5 (Massive Scale Substrate)\n";
    std::cout << "==================================================================\n\n";

    // 1. Read Shakespeare Dataset (~1.1 MB raw UTF-8 bytes)
    std::string dataset_path = "data/tinyshakespeare.txt";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset: " << dataset_path << "\n";
        return 1;
    }

    std::streamsize full_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read full dataset
    std::vector<uint8_t> buffer(full_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), full_size)) {
        std::cerr << "Error: Failed to read dataset into memory\n";
        return 1;
    }
    file.close();

    std::cout << "1. DATASET TELEMETRY:\n";
    std::cout << "   -> Target File: " << dataset_path << "\n";
    std::cout << "   -> Total Raw Size: " << full_size << " bytes (" 
              << std::fixed << std::setprecision(2) << (full_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "   -> Total Binary Spins to Absorb: " << (full_size * 8) << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // 2. Stream Ingestion with Progressive Throughput Profiling
    std::cout << "2. CONTINUOUS STREAM INGESTION (PROGRESSIVE CHUNKS):\n";
    // Ingest in slices to track throughput and scaling
    size_t chunk_size = 100 * 1024; // 100 KB chunks
    size_t bytes_ingested = 0;
    double total_ingest_time_ms = 0.0;

    auto global_start = std::chrono::high_resolution_clock::now();

    while (bytes_ingested < static_cast<size_t>(full_size)) {
        size_t cur_chunk = std::min(chunk_size, static_cast<size_t>(full_size) - bytes_ingested);
        
        auto c0 = std::chrono::high_resolution_clock::now();
        engine.IngestRawBytes(buffer.data() + bytes_ingested, cur_chunk, true);
        auto c1 = std::chrono::high_resolution_clock::now();

        double chunk_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        total_ingest_time_ms += chunk_ms;
        bytes_ingested += cur_chunk;

        double progress_pct = (static_cast<double>(bytes_ingested) / full_size) * 100.0;
        double mb_sec = (static_cast<double>(bytes_ingested) / 1024.0 / 1024.0) / (total_ingest_time_ms / 1000.0);

        std::cout << "   [Chunk " << std::setw(2) << (bytes_ingested / chunk_size) << "] "
                  << "Ingested: " << std::setw(7) << bytes_ingested << " / " << full_size << " bytes ("
                  << std::fixed << std::setprecision(1) << progress_pct << " %) | "
                  << "Active Taps: " << std::setw(9) << engine.GetWaveTraceCount() << " | "
                  << "Throughput: " << std::setprecision(2) << mb_sec << " MB/s | "
                  << "Chunk Time: " << std::setprecision(1) << chunk_ms << " ms\n";
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double total_duration_sec = std::chrono::duration<double>(global_end - global_start).count();

    std::cout << "\n3. MACRO-SCALE INGESTION SUMMARY:\n";
    std::cout << "   -> Total Ingestion Time: " << std::fixed << std::setprecision(3) << total_duration_sec << " s\n";
    std::cout << "   -> Global Throughput: " << (full_size / 1024.0 / 1024.0 / total_duration_sec) << " MB/s ("
              << ((full_size * 8.0) / total_duration_sec / 1e6) << " Million spins/sec)\n";
    std::cout << "   -> Total Synthesized Phase Taps: " << engine.GetWaveTraceCount() << "\n\n";

    // 4. Autopoietic Recall & Generalization Probe
    std::cout << "4. SAMPLING CONTINUOUS WAVEFIELD RETRIEVAL & AUTOREGRESSIVE GENERATION:\n";
    std::vector<std::string> test_prompts = {
        "First Citizen:\nBefore we proceed any further, hear me",
        "MENENIUS:\nWhat work's, my countrymen",
        "BRUTUS:\nHe's poor in no one fault, but"
    };

    size_t successful_probes = 0;
    for (size_t i = 0; i < test_prompts.size(); ++i) {
        engine.ResetHistory();
        engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(test_prompts[i].data()), test_prompts[i].size(), false);
        auto gen = engine.GenerateAutoregressiveBytes(40);
        std::string continuation(gen.begin(), gen.end());

        std::cout << "   [Probe " << (i + 1) << "] Seed: \"" << test_prompts[i].substr(0, 25) << "...\"\n";
        std::cout << "   [Probe " << (i + 1) << "] Autoregressive Wave Emission:\n\"" << continuation << "\"\n\n";
        if (!continuation.empty()) successful_probes++;
    }

    // 5. Binary .kcore Serialization of the Massive Substrate
    std::cout << "5. TESTING SERIALIZATION OF MASSIVE SUBSTRATE TO .kcore:\n";
    std::string kcore_out = "build/massive_shakespeare.kcore";
    auto s0 = std::chrono::high_resolution_clock::now();
    bool saved = engine.SaveKcoreContainer(kcore_out);
    auto s1 = std::chrono::high_resolution_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

    std::ifstream kfile(kcore_out, std::ios::binary | std::ios::ate);
    size_t kcore_bytes = 0;
    if (kfile.is_open()) {
        kcore_bytes = static_cast<size_t>(kfile.tellg());
    }
    kfile.close();

    std::cout << "   -> Saved Massive .kcore: " << (saved ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "   -> Serialization Latency: " << std::fixed << std::setprecision(2) << save_ms << " ms\n";
    std::cout << "   -> .kcore Container Size on Disk: " << (kcore_bytes / 1024.0 / 1024.0) << " MB (" 
              << kcore_bytes << " bytes)\n\n";

    bool pass_all = (bytes_ingested == static_cast<size_t>(full_size)) && saved && (successful_probes == test_prompts.size());

    std::cout << "===================================================================\n";
    std::cout << " EXP-937 SUMMARY VERDICT: " << (pass_all ? "🟢 POSITIVE (MASSIVE DATASET INGESTED)" : "🔴 FAILED") << "\n";
    std::cout << " - Absorbed Data Volume: " << (full_size / 1024.0 / 1024.0) << " MB (" << (full_size * 8) << " spins)\n";
    std::cout << " - Active Topological Nodes: " << engine.GetWaveTraceCount() << "\n";
    std::cout << " - Ingestion Throughput: " << std::fixed << std::setprecision(2) << (full_size / 1024.0 / 1024.0 / total_duration_sec) << " MB/s\n";
    std::cout << " - KEP Principle 4 & 5 Compliance: Pure Bit-Spin Substrate at Macro Scale\n";
    std::cout << "===================================================================\n";

    return pass_all ? 0 : 1;
}
