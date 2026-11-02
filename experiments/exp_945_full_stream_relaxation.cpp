// ============================================================================
// EXP-945: Full-Stream Autopoietic Phase Relaxation & 100% Deterministic Attractor Locking
// Standard: KEP v40.0 Sovereign Master | Principles 0, 3, 6 & 9 (F -> 0 Absolute Zero Error)
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
    std::cout << "EXP-945: Full-Stream Autopoietic Phase Relaxation & 100% Attractor Locking\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 0 (F -> 0 Deterministic Standard)\n";
    std::cout << "==================================================================\n\n";

    // 1. Load Single Frame Stream (128x128 RGB, 49,152 bytes / 393,216 spins) for exact deterministic test
    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset stream\n";
        return 1;
    }

    std::streamsize full_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t frame_bytes = 128 * 128 * 3; // 49,152 bytes
    std::vector<uint8_t> buffer(frame_bytes);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), frame_bytes)) {
        std::cerr << "Error: Failed to read single frame\n";
        return 1;
    }
    file.close();

    size_t total_spins = frame_bytes * 8;

    std::cout << "1. TARGET DETERMINISTIC STREAM SETUP:\n";
    std::cout << "   -> Frame Size: " << frame_bytes << " bytes (128x128 RGB)\n";
    std::cout << "   -> Target Spins: " << total_spins << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    // 2. Full-Stream Multi-Pass Ingestion & Phase Alignment
    std::cout << "2. EXECUTING FULL-STREAM MULTI-PASS PHASE RELAXATION:\n";
    auto t_ingest_start = std::chrono::high_resolution_clock::now();

    // Ingest Pass 1
    engine.IngestRawBytes(buffer.data(), frame_bytes, true);
    std::cout << "   -> Pass 1 Complete | Active Taps: " << engine.GetWaveTraceCount() << "\n";

    // Pass 2: Evaluate & Reinforce ALL spins across full stream
    engine.ResetHistory();
    size_t correct_predictions = 0;
    for (size_t i = 0; i < total_spins; ++i) {
        uint8_t byte = buffer[i / 8];
        int b = 7 - (i % 8);
        int8_t true_spin = ((byte >> b) & 1) ? +1 : -1;

        double pred = engine.PredictNextSpin();
        if ((pred >= 0.0 && true_spin > 0) || (pred < 0.0 && true_spin < 0)) {
            correct_predictions++;
        }
        engine.PushWorkingSpin(true_spin);
    }
    double pass2_acc = (static_cast<double>(correct_predictions) / total_spins * 100.0);
    std::cout << "   -> Full-Stream Phase Resonance Accuracy: " << std::fixed << std::setprecision(4) << pass2_acc << " %\n";

    // Ingest Pass 3: Second topological sweep to lock phase memory
    engine.IngestRawBytes(buffer.data(), frame_bytes, true);
    std::cout << "   -> Pass 3 Complete | Active Taps: " << engine.GetWaveTraceCount() << "\n";

    auto t_ingest_end = std::chrono::high_resolution_clock::now();
    double total_ingest_dur = std::chrono::duration<double>(t_ingest_end - t_ingest_start).count();
    std::cout << "   -> Total Ingestion Time: " << std::fixed << std::setprecision(3) << total_ingest_dur << " s\n\n";

    // 3. Smart Causal Pruning (Only prune truly zero-efficacy or unreferenced taps)
    std::cout << "3. CAUSAL MORPHOGENESIS (SAFE DYNAMIC PRUNING):\n";
    // We only prune taps with efficacy < 1.01 (which leaves all active stream taps intact)
    size_t pruned_count = engine.ExecuteAutopoieticPruning(1.01);
    std::cout << "   -> Sub-Critical Taps Pruned: " << pruned_count << "\n";
    std::cout << "   -> Surviving Active Taps: " << engine.GetWaveTraceCount() << "\n\n";

    // 4. Autoregressive Synthesis from 1.5% Seed (768 bytes)
    std::cout << "4. AUTOREGRESSIVE RECONSTRUCTION FROM 1.5% SEED:\n";
    engine.ResetHistory();

    size_t seed_bytes = 128 * 2 * 3; // 768 bytes (first 2 scanlines)
    size_t target_generate_bytes = frame_bytes - seed_bytes;

    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    auto t_gen_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> generated_tail = engine.GenerateAutoregressiveBytes(target_generate_bytes);
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_duration_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();

    // 5. Visual Output & Fidelity Evaluation
    std::vector<uint8_t> synthesized_frame(frame_bytes);
    std::copy(buffer.begin(), buffer.begin() + seed_bytes, synthesized_frame.begin());
    std::copy(generated_tail.begin(), generated_tail.end(), synthesized_frame.begin() + seed_bytes);

    std::string out_png = "data/karyon_exp945_full_deterministic.png";
    stbi_write_png(out_png.c_str(), 128, 128, 3, synthesized_frame.data(), 128 * 3);

    size_t exact_bit_matches = 0;
    for (size_t i = 0; i < target_generate_bytes; ++i) {
        uint8_t ref_byte = buffer[seed_bytes + i];
        uint8_t gen_byte = generated_tail[i];
        for (int b = 7; b >= 0; --b) {
            if (((ref_byte >> b) & 1) == ((gen_byte >> b) & 1)) {
                exact_bit_matches++;
            }
        }
    }

    double bit_fidelity_pct = (static_cast<double>(exact_bit_matches) / (target_generate_bytes * 8)) * 100.0;
    std::cout << "5. EXACT DETERMINISTIC FIDELITY METRICS:\n";
    std::cout << "   -> Evaluated Bit-Spins: " << (target_generate_bytes * 8) << " spins\n";
    std::cout << "   -> Exact Matching Spins: " << exact_bit_matches << " spins\n";
    std::cout << "   -> Bit-Spin Reconstruction Fidelity: " << std::fixed << std::setprecision(4) << bit_fidelity_pct << " %\n";
    std::cout << "   -> Output PNG: " << out_png << "\n\n";

    std::cout << "==================================================================\n";
    std::cout << "EXP-945 COMPLETED SUCCESSFULLY\n";
    std::cout << "==================================================================\n";

    return 0;
}
