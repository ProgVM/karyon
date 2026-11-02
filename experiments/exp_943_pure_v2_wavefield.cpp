// ============================================================================
// EXP-943: Pure Substrate V=2 Wavefield Synthesis (Zero Image-Specific Formulations)
// Karyon Engineering Protocol (KEP) Rule #1 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | Principle 1, 4, 5 & 10 (Universal Bit-Spin Wavefield)
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

#include "karyon_core_wavefield.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-943: Pure Substrate V=2 Continuous Wavefield Synthesis\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principles 1, 4 & 5 (Pure V=2 Substrate)\n";
    std::cout << "==================================================================\n\n";

    // 1. Ingest Raw Universal Dataset Stream (128x128 RGB, 2.81 MB / 23.59M spins)
    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset stream: " << dataset_path << "\n";
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

    size_t frame_bytes = 128 * 128 * 3; // 49,152 bytes per frame
    size_t num_frames = full_size / frame_bytes;
    size_t total_spins = full_size * 8;

    std::cout << "1. PURE V=2 SUBSTRATE STREAM TELEMETRY:\n";
    std::cout << "   -> Target Stream: " << dataset_path << "\n";
    std::cout << "   -> Frame Dimension: 128x128 RGB (" << frame_bytes << " bytes / " << (frame_bytes * 8) << " spins per frame)\n";
    std::cout << "   -> Stream Volume: " << full_size << " bytes (" 
              << std::fixed << std::setprecision(2) << (full_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "   -> Total Binary Bit-Spins: " << total_spins << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // 2. Continuous Ingestion of Raw Bit-Spins into Topological Memory
    std::cout << "2. CONTINUOUS INGESTION OF RAW SPINS INTO PHYSICAL WAVEFIELD:\n";
    auto t_ingest_start = std::chrono::high_resolution_clock::now();

    // Ingest complete raw stream using universal IngestRawBytes
    engine.IngestRawBytes(buffer.data(), full_size, true);

    auto t_ingest_end = std::chrono::high_resolution_clock::now();
    double ingest_duration_sec = std::chrono::duration<double>(t_ingest_end - t_ingest_start).count();

    std::cout << "   -> Ingestion Completed in: " << std::fixed << std::setprecision(3) << ingest_duration_sec << " s\n";
    std::cout << "   -> Spin Absorption Throughput: " << std::setprecision(2) << ((total_spins / ingest_duration_sec) / 1e6) << " Million spins/sec\n";
    std::cout << "   -> Total Active Topological Taps: " << engine.GetWaveTraceCount() << "\n\n";

    // 3. Save .kcore Weightless Topological Container
    std::string kcore_path = "data/exp_943_pure_wavefield.kcore";
    std::cout << "3. WEIGHTLESS TOPOLOGICAL CONTAINER SERIALIZATION:\n";
    bool save_ok = engine.SaveKcoreContainer(kcore_path);
    std::cout << "   -> Container Path: " << kcore_path << "\n";
    std::cout << "   -> Serialization Status: " << (save_ok ? "SUCCESS" : "FAILED") << "\n\n";

    // 4. Pure Wavefield Autoregressive Synthesis directly by Karyon (Zero Ad-Hoc Image Formulations)
    // Giving a minimal seed of first 2 scanlines (768 bytes), and letting Karyon autoregressively
    // predict the remaining 48,384 bytes (387,072 spins) directly through PredictNextSpin().
    std::cout << "4. PURE KARYON SUBSTRATE AUTOREGRESSIVE WAVE SYNTHESIS:\n";
    engine.ResetHistory();

    size_t seed_bytes = 128 * 2 * 3; // First 2 scanlines (768 bytes)
    size_t target_generate_bytes = frame_bytes - seed_bytes;

    std::cout << "   -> Micro-Seed Context: " << seed_bytes << " bytes (" << (seed_bytes * 8) << " spins)\n";
    std::cout << "   -> Autoregressive Generation Target: " << target_generate_bytes << " bytes (" << (target_generate_bytes * 8) << " spins)\n";

    // Seed engine with history without updating topological memory
    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    // Perform pure autoregressive byte synthesis via Feynman Wavefield Superposition
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> generated_tail = engine.GenerateAutoregressiveBytes(target_generate_bytes);

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_duration_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();

    double spins_per_sec = (target_generate_bytes * 8.0) / gen_duration_sec;
    double fps = 1.0 / (gen_duration_sec * (frame_bytes / static_cast<double>(target_generate_bytes)));

    std::cout << "   -> Generation Time: " << std::fixed << std::setprecision(3) << gen_duration_sec << " s\n";
    std::cout << "   -> Generation Throughput: " << std::setprecision(2) << (spins_per_sec / 1e3) << " K-spins/sec\n";
    std::cout << "   -> Effective FPS: " << std::setprecision(2) << fps << " FPS\n\n";

    // Reconstruct full frame for visual verification
    std::vector<uint8_t> synthesized_frame(frame_bytes);
    std::copy(buffer.begin(), buffer.begin() + seed_bytes, synthesized_frame.begin());
    std::copy(generated_tail.begin(), generated_tail.end(), synthesized_frame.begin() + seed_bytes);

    // 5. Visual Output Serialization
    std::string out_png = "data/karyon_pure_v2_furry_128.png";
    int write_res = stbi_write_png(out_png.c_str(), 128, 128, 3, synthesized_frame.data(), 128 * 3);

    std::cout << "5. VISUAL WAVEFIELD SERIALIZATION:\n";
    std::cout << "   -> Output PNG: " << out_png << "\n";
    std::cout << "   -> Write Result: " << (write_res ? "SUCCESS" : "FAILED") << "\n\n";

    // 6. Quantitative Bit Fidelity Check
    size_t exact_bit_matches = 0;
    size_t total_eval_bytes = target_generate_bytes;
    for (size_t i = 0; i < total_eval_bytes; ++i) {
        uint8_t ref_byte = buffer[seed_bytes + i];
        uint8_t gen_byte = generated_tail[i];
        for (int b = 7; b >= 0; --b) {
            if (((ref_byte >> b) & 1) == ((gen_byte >> b) & 1)) {
                exact_bit_matches++;
            }
        }
    }

    double bit_fidelity_pct = (static_cast<double>(exact_bit_matches) / (total_eval_bytes * 8)) * 100.0;
    std::cout << "6. QUANTITATIVE WAVEFIELD FIDELITY:\n";
    std::cout << "   -> Evaluated Bit-Spins: " << (total_eval_bytes * 8) << " spins\n";
    std::cout << "   -> Exact Matching Spins: " << exact_bit_matches << " spins\n";
    std::cout << "   -> Bit-Spin Fidelity: " << std::fixed << std::setprecision(2) << bit_fidelity_pct << " %\n\n";

    std::cout << "==================================================================\n";
    std::cout << "EXP-943 COMPLETED SUCCESSFULLY\n";
    std::cout << "==================================================================\n";

    return 0;
}
