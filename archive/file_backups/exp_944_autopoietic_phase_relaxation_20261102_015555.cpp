// ============================================================================
// EXP-944: Autopoietic Multi-Pass Phase Relaxation & Soliton Energy Minimization
// Karyon Engineering Protocol (KEP) Rule #1 & Principle 9 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | Principle 0, 3, 6 & 9 (Phase Relaxation & Soliton Stability)
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
    std::cout << "EXP-944: Autopoietic Multi-Pass Phase Relaxation & Soliton Energy Minimization\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principles 0, 3, 6 & 9 (Phase Relaxation)\n";
    std::cout << "==================================================================\n\n";

    // 1. Load Target Stream (128x128 RGB, 2.81 MB / 23.59M spins)
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
    size_t total_spins = full_size * 8;

    std::cout << "1. MULTI-PASS PHASE RELAXATION SETUP:\n";
    std::cout << "   -> Stream Volume: " << full_size << " bytes (" 
              << std::fixed << std::setprecision(2) << (full_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "   -> Target Stream Spins: " << total_spins << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // 2. Multi-Pass Autopoietic Phase Relaxation (3 Passes over the physical manifold)
    std::cout << "2. EXECUTING MULTI-PASS PHASE RELAXATION (3 REPETITION PASSES):\n";
    auto t_ingest_start = std::chrono::high_resolution_clock::now();

    const int NUM_PASSES = 3;
    for (int p = 1; p <= NUM_PASSES; ++p) {
        auto t_pass_start = std::chrono::high_resolution_clock::now();
        engine.IngestRawBytes(buffer.data(), full_size, true);
        auto t_pass_end = std::chrono::high_resolution_clock::now();
        double pass_dur = std::chrono::duration<double>(t_pass_end - t_pass_start).count();
        std::cout << "   -> Pass " << p << "/" << NUM_PASSES << " Complete: " 
                  << std::fixed << std::setprecision(3) << pass_dur << " s | Active Taps: " << engine.GetWaveTraceCount() << "\n";
    }

    auto t_ingest_end = std::chrono::high_resolution_clock::now();
    double total_ingest_dur = std::chrono::duration<double>(t_ingest_end - t_ingest_start).count();
    std::cout << "   -> Total Multi-Pass Ingestion Time: " << std::fixed << std::setprecision(3) << total_ingest_dur << " s\n\n";

    // 3. Autopoietic Topological Morphogenesis & Pruning of Sub-Critical Taps
    std::cout << "3. AUTOPOIETIC TOPOLOGICAL MORPHOGENESIS (CAUSAL PRUNING):\n";
    size_t pruned_count = engine.ExecuteAutopoieticPruning(1.10); // Prune taps with <1.10 efficacy
    std::cout << "   -> Sub-Critical Taps Pruned: " << pruned_count << "\n";
    std::cout << "   -> Surviving Epigenetically Locked Taps: " << engine.GetWaveTraceCount() << "\n\n";

    // 4. Save .kcore Weightless Topological Container
    std::string kcore_path = "data/exp_944_relaxed_wavefield.kcore";
    std::cout << "4. WEIGHTLESS TOPOLOGICAL CONTAINER SERIALIZATION:\n";
    bool save_ok = engine.SaveKcoreContainer(kcore_path);
    std::cout << "   -> Container Path: " << kcore_path << "\n";
    std::cout << "   -> Serialization Status: " << (save_ok ? "SUCCESS" : "FAILED") << "\n\n";

    // 5. Autoregressive Synthesis from Minimal 2-Line Seed
    std::cout << "5. AUTOREGRESSIVE SYNTHESIS FROM RELAXED ATTRACTOR BASIN:\n";
    engine.ResetHistory();

    size_t seed_bytes = 128 * 2 * 3; // First 2 scanlines (768 bytes)
    size_t target_generate_bytes = frame_bytes - seed_bytes;

    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    auto t_gen_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> generated_tail = engine.GenerateAutoregressiveBytes(target_generate_bytes);
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_duration_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();

    double spins_per_sec = (target_generate_bytes * 8.0) / gen_duration_sec;
    double fps = 1.0 / (gen_duration_sec * (frame_bytes / static_cast<double>(target_generate_bytes)));

    std::cout << "   -> Generation Time: " << std::fixed << std::setprecision(3) << gen_duration_sec << " s\n";
    std::cout << "   -> Generation Throughput: " << std::setprecision(2) << (spins_per_sec / 1e3) << " K-spins/sec\n";
    std::cout << "   -> Effective FPS: " << std::setprecision(2) << fps << " FPS\n\n";

    // 6. Visual Output Serialization
    std::vector<uint8_t> synthesized_frame(frame_bytes);
    std::copy(buffer.begin(), buffer.begin() + seed_bytes, synthesized_frame.begin());
    std::copy(generated_tail.begin(), generated_tail.end(), synthesized_frame.begin() + seed_bytes);

    std::string out_png = "data/karyon_relaxed_v2_furry_128.png";
    int write_res = stbi_write_png(out_png.c_str(), 128, 128, 3, synthesized_frame.data(), 128 * 3);

    std::cout << "6. VISUAL WAVEFIELD SERIALIZATION:\n";
    std::cout << "   -> Output PNG: " << out_png << "\n";
    std::cout << "   -> Write Result: " << (write_res ? "SUCCESS" : "FAILED") << "\n\n";

    // 7. Quantitative Bit Fidelity Slices Comparison
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
    std::cout << "7. QUANTITATIVE FIDELITY TELEMETRY:\n";
    std::cout << "   -> Evaluated Bit-Spins: " << (total_eval_bytes * 8) << " spins\n";
    std::cout << "   -> Exact Matching Spins: " << exact_bit_matches << " spins\n";
    std::cout << "   -> Bit-Spin Fidelity (EXP-943 Baseline: 58.16%): " << std::fixed << std::setprecision(2) << bit_fidelity_pct << " %\n\n";

    std::cout << "==================================================================\n";
    std::cout << "EXP-944 COMPLETED SUCCESSFULLY\n";
    std::cout << "==================================================================\n";

    return 0;
}
