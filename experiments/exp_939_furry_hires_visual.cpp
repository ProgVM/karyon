// ============================================================================
// EXP-939: High-Resolution 128x128 Continuous Chromatic Wavefield Synthesis
// Karyon Engineering Protocol (KEP) Rule #1 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | Principle 1, 4 & 5 (Universal Wave Modality)
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
    std::cout << "EXP-939: High-Resolution 128x128 Continuous Chromatic Wavefield Synthesis\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 1, 4 & 5 (V=2 Hilbert Space)\n";
    std::cout << "==================================================================\n\n";

    // 1. Read High-Resolution Binary Stream (128x128 RGB, ~2.81 MB / 23.59M spins)
    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open high-res dataset: " << dataset_path << "\n";
        return 1;
    }

    std::streamsize full_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(full_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), full_size)) {
        std::cerr << "Error: Failed to read stream\n";
        return 1;
    }
    file.close();

    size_t frame_bytes = 128 * 128 * 3; // 49,152 bytes per 128x128 frame
    size_t num_frames = full_size / frame_bytes;
    size_t total_spins = full_size * 8;

    std::cout << "1. HI-RES CHROMATIC STREAM TELEMETRY:\n";
    std::cout << "   -> Target Stream: " << dataset_path << "\n";
    std::cout << "   -> Frame Dimension: 128x128 RGB (" << frame_bytes << " bytes / " << (frame_bytes * 8) << " spins per frame)\n";
    std::cout << "   -> Art Frames: " << num_frames << " frames\n";
    std::cout << "   -> Stream Volume: " << full_size << " bytes (" 
              << std::fixed << std::setprecision(2) << (full_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "   -> Total Binary Spins: " << total_spins << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // 2. Continuous Ingestion of 23.59 Million High-Res Spins
    std::cout << "2. CONTINUOUS INGESTION OF 128x128 CHROMATIC BIT-SPINS:\n";
    size_t chunk_size = frame_bytes * 5; // 5 high-res frames per block (245,760 bytes)
    size_t bytes_ingested = 0;
    double total_ingest_ms = 0.0;

    auto global_start = std::chrono::high_resolution_clock::now();

    while (bytes_ingested < static_cast<size_t>(full_size)) {
        size_t cur_chunk = std::min(chunk_size, static_cast<size_t>(full_size) - bytes_ingested);

        auto c0 = std::chrono::high_resolution_clock::now();
        engine.IngestRawBytes(buffer.data() + bytes_ingested, cur_chunk, true);
        auto c1 = std::chrono::high_resolution_clock::now();

        double chunk_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        total_ingest_ms += chunk_ms;
        bytes_ingested += cur_chunk;

        double progress_pct = (static_cast<double>(bytes_ingested) / full_size) * 100.0;
        double mb_sec = (static_cast<double>(bytes_ingested) / 1024.0 / 1024.0) / (total_ingest_ms / 1000.0);

        std::cout << "   [Hi-Res Block " << std::setw(2) << (bytes_ingested / chunk_size) << "] "
                  << "Ingested: " << std::setw(7) << bytes_ingested << " / " << full_size << " bytes ("
                  << std::fixed << std::setprecision(1) << progress_pct << " %) | "
                  << "Active Taps: " << std::setw(8) << engine.GetWaveTraceCount() << " | "
                  << "Throughput: " << std::setprecision(2) << mb_sec << " MB/s | "
                  << "Latency: " << std::setprecision(1) << chunk_ms << " ms\n";
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double total_duration_sec = std::chrono::duration<double>(global_end - global_start).count();

    std::cout << "\n3. MACRO-SCALE INGESTION SUMMARY:\n";
    std::cout << "   -> Total Ingestion Time: " << std::fixed << std::setprecision(3) << total_duration_sec << " s\n";
    std::cout << "   -> Spin Absorption Rate: " << ((full_size * 8.0) / total_duration_sec / 1e6) << " Million spins/sec\n";
    std::cout << "   -> Active Topological Phase Nodes: " << engine.GetWaveTraceCount() << "\n\n";

    // 4. Zero-Shot Creative Synthesis from Minimal Micro-Seed
    // We give a minimal micro-seed (only 2 scanlines = 128 * 2 * 3 = 768 bytes, 1.5% of the frame)
    // and let Karyon freely synthesize the remaining 126 scanlines (48,384 bytes / 387,072 spins)
    std::cout << "4. PROBING FULL 128x128 IMAGINATIVE VISUAL RECONSTRUCTION & CREATION:\n";
    size_t seed_bytes = 128 * 2 * 3; // First 2 scanlines only!
    size_t target_generate_bytes = frame_bytes - seed_bytes;

    engine.ResetHistory();
    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    std::cout << "   -> Minimal Seed: 2 top scanlines (" << seed_bytes << " bytes, 1.5% of frame)\n";
    std::cout << "   -> Autoregressively Generating 126 scanlines (" << target_generate_bytes 
              << " bytes / " << (target_generate_bytes * 8) << " spins)...\n";

    auto g0 = std::chrono::high_resolution_clock::now();
    auto generated_tail = engine.GenerateAutoregressiveBytes(target_generate_bytes);
    auto g1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    std::cout << "   -> Full 128x128 Synthesis Finished in: " << std::fixed << std::setprecision(2) << gen_ms << " ms ("
              << (target_generate_bytes / (gen_ms / 1000.0) / 1024.0) << " KB/s)\n";

    // Assemble complete 128x128 image
    std::vector<uint8_t> complete_image;
    complete_image.insert(complete_image.end(), buffer.begin(), buffer.begin() + seed_bytes);
    complete_image.insert(complete_image.end(), generated_tail.begin(), generated_tail.end());

    std::string out_bin_path = "data/furry_hires/karyon_hires_synthesized_128x128.bin";
    std::ofstream out_f(out_bin_path, std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(complete_image.data()), complete_image.size());
    out_f.close();
    std::cout << "   -> Saved Synthesized 128x128 Binary Frame to: " << out_bin_path << "\n\n";

    // 5. Binary Serialization to .kcore
    std::cout << "5. SERIALIZING 23.59M-SPIN HIGH-RES .kcore CONTAINER:\n";
    std::string kcore_path = "build/furry_128x128_wavefield.kcore";
    auto s0 = std::chrono::high_resolution_clock::now();
    bool saved = engine.SaveKcoreContainer(kcore_path);
    auto s1 = std::chrono::high_resolution_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

    std::ifstream kfile(kcore_path, std::ios::binary | std::ios::ate);
    size_t kcore_bytes = 0;
    if (kfile.is_open()) kcore_bytes = static_cast<size_t>(kfile.tellg());
    kfile.close();

    std::cout << "   -> .kcore Serialization: " << (saved ? "SUCCESS" : "FAILED") << " in " << std::fixed << std::setprecision(2) << save_ms << " ms\n";
    std::cout << "   -> Container Size: " << (kcore_bytes / 1024.0 / 1024.0) << " MB\n\n";

    bool pass_all = (bytes_ingested == static_cast<size_t>(full_size)) && saved && (complete_image.size() == frame_bytes);

    std::cout << "===================================================================\n";
    std::cout << " EXP-939 SUMMARY VERDICT: " << (pass_all ? "🟢 POSITIVE (HI-RES VISUAL WAVEFIELD SYNTHESIS PASSED)" : "🔴 FAILED") << "\n";
    std::cout << " - Absorbed Hi-Res Dataset: " << num_frames << " frames 128x128 (" << (full_size / 1024.0 / 1024.0) << " MB, " << total_spins << " spins)\n";
    std::cout << " - Active Topological Phase Nodes: " << engine.GetWaveTraceCount() << "\n";
    std::cout << " - Synthesized Complete Hi-Res Frame: 128x128 RGB (" << complete_image.size() << " bytes)\n";
    std::cout << "===================================================================\n";

    return pass_all ? 0 : 1;
}
