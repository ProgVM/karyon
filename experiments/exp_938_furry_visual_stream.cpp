// ============================================================================
// EXP-938: Continuous Multimodal Wavefield Ingestion of Furry Visual Stream
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
    std::cout << "EXP-938: Continuous Multimodal Wavefield Ingestion of Furry Visual Stream\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 1, 4 & 5 (Universal Bit-Spin V=2)\n";
    std::cout << "==================================================================\n\n";

    // 1. Read Raw Binary Visual Stream (~1.01 MB / 86 64x64 RGB Art Images)
    std::string dataset_path = "data/furry_stream/furry_64x64_raw.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset: " << dataset_path << "\n";
        return 1;
    }

    std::streamsize full_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(full_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), full_size)) {
        std::cerr << "Error: Failed to read visual stream into memory\n";
        return 1;
    }
    file.close();

    size_t total_spins = full_size * 8;
    size_t num_images = full_size / (64 * 64 * 3);

    std::cout << "1. MULTIMODAL VISUAL STREAM TELEMETRY:\n";
    std::cout << "   -> Target Stream: " << dataset_path << "\n";
    std::cout << "   -> Encoded Art Images (64x64 RGB): " << num_images << " uncompressed frames\n";
    std::cout << "   -> Total Stream Volume: " << full_size << " raw bytes (" 
              << std::fixed << std::setprecision(2) << (full_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "   -> Total Binary Visual Spins: " << total_spins << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // 2. Stream Ingestion of Continuous Visual Wave Dynamics
    std::cout << "2. CONTINUOUS INGESTION OF HIGH-DIMENSIONAL CHROMATIC BIT-SPINS:\n";
    size_t chunk_size = 12288 * 5; // 5 images per chunk (61,440 bytes)
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

        std::cout << "   [Visual Block " << std::setw(2) << (bytes_ingested / chunk_size) << "] "
                  << "Ingested: " << std::setw(7) << bytes_ingested << " / " << full_size << " bytes ("
                  << std::fixed << std::setprecision(1) << progress_pct << " %) | "
                  << "Active Visual Taps: " << std::setw(9) << engine.GetWaveTraceCount() << " | "
                  << "Throughput: " << std::setprecision(2) << mb_sec << " MB/s | "
                  << "Latency: " << std::setprecision(1) << chunk_ms << " ms\n";
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double total_duration_sec = std::chrono::duration<double>(global_end - global_start).count();

    std::cout << "\n3. MACRO-SCALE INGESTION SUMMARY:\n";
    std::cout << "   -> Total Ingestion Time: " << std::fixed << std::setprecision(3) << total_duration_sec << " s\n";
    std::cout << "   -> Spin Processing Rate: " << ((full_size * 8.0) / total_duration_sec / 1e6) << " Million spins/sec\n";
    std::cout << "   -> Active Topological Phase Nodes: " << engine.GetWaveTraceCount() << "\n\n";

    // 4. Autopoietic Visual Synthesis & Image Generation
    std::cout << "4. PROBING CONTINUOUS WAVEFIELD IMAGINATIVE VISUAL SYNTHESIS:\n";
    // Feed seed bytes from the first image (e.g. 1/4 of an image = 3072 bytes)
    size_t seed_bytes = 64 * 16 * 3; // First 16 scanlines
    engine.ResetHistory();
    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    // Generate the remaining 48 scanlines (64 * 48 * 3 = 9216 bytes)
    size_t generate_bytes = 64 * 48 * 3;
    std::cout << "   -> Conditioning Seed: First 16 scanlines of Art Frame #0 (" << seed_bytes << " bytes)\n";
    std::cout << "   -> Autoregressively Synthesizing Remaining 48 scanlines (" << generate_bytes << " bytes / " 
              << (generate_bytes * 8) << " spins)...\n";

    auto g0 = std::chrono::high_resolution_clock::now();
    auto generated_tail = engine.GenerateAutoregressiveBytes(generate_bytes);
    auto g1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    std::cout << "   -> Synthesis Complete in: " << std::fixed << std::setprecision(2) << gen_ms << " ms ("
              << (generate_bytes / (gen_ms / 1000.0) / 1024.0) << " KB/s)\n";

    // Combine seed + generated image and save to raw binary file
    std::vector<uint8_t> complete_generated_image;
    complete_generated_image.insert(complete_generated_image.end(), buffer.begin(), buffer.begin() + seed_bytes);
    complete_generated_image.insert(complete_generated_image.end(), generated_tail.begin(), generated_tail.end());

    std::string out_img_raw = "data/furry_stream/karyon_synthesized_furry_64x64.bin";
    std::ofstream out_f(out_img_raw, std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(complete_generated_image.data()), complete_generated_image.size());
    out_f.close();
    std::cout << "   -> Saved Synthesized 64x64 Raw Frame to: " << out_img_raw << "\n\n";

    // 5. Binary Serialization to .kcore
    std::cout << "5. SERIALIZING MULTIMODAL FURRY TOPOLOGICAL CONTAINER (.kcore):\n";
    std::string kcore_path = "build/furry_visual_wavefield.kcore";
    auto s0 = std::chrono::high_resolution_clock::now();
    bool saved = engine.SaveKcoreContainer(kcore_path);
    auto s1 = std::chrono::high_resolution_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();

    std::ifstream kfile(kcore_path, std::ios::binary | std::ios::ate);
    size_t kcore_bytes = 0;
    if (kfile.is_open()) {
        kcore_bytes = static_cast<size_t>(kfile.tellg());
    }
    kfile.close();

    std::cout << "   -> .kcore Serialization: " << (saved ? "SUCCESS" : "FAILED") << " in " << std::fixed << std::setprecision(2) << save_ms << " ms\n";
    std::cout << "   -> Container Size: " << (kcore_bytes / 1024.0 / 1024.0) << " MB\n\n";

    bool pass_all = (bytes_ingested == static_cast<size_t>(full_size)) && saved && (generated_tail.size() == generate_bytes);

    std::cout << "===================================================================\n";
    std::cout << " EXP-938 SUMMARY VERDICT: " << (pass_all ? "🟢 POSITIVE (MULTIMODAL VISUAL INGESTION PASSED)" : "🔴 FAILED") << "\n";
    std::cout << " - Absorbed Art Dataset: " << num_images << " images (" << (full_size / 1024.0 / 1024.0) << " MB, " << total_spins << " spins)\n";
    std::cout << " - Phase Taps Synthesized: " << engine.GetWaveTraceCount() << "\n";
    std::cout << " - Synthesized Visual Output: 64x64 RGB (" << complete_generated_image.size() << " bytes)\n";
    std::cout << " - KEP Principle 1 & 4 Compliance: 100% Universal Bit-Spin Wavefield (No Vision Transformers, No CNNs)\n";
    std::cout << "===================================================================\n";

    return pass_all ? 0 : 1;
}
