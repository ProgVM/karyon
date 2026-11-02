// ============================================================================
// EXP-947: Deep Soliton Relaxation & 100% Convergence Target (Principle 0)
// Standard: KEP v40.0 Sovereign Master | Principle 0 (F = 0.0000000 nats)
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
    std::cout << "EXP-947: Deep Soliton Phase Relaxation & 100% Convergence Target\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 0 (Zero Error Standard)\n";
    std::cout << "==================================================================\n\n";

    // Load Single Target Frame Stream (128x128 RGB, 49,152 bytes)
    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file.is_open()) return 1;

    size_t frame_bytes = 128 * 128 * 3;
    std::vector<uint8_t> buffer(frame_bytes);
    file.read(reinterpret_cast<char*>(buffer.data()), frame_bytes);
    file.close();

    size_t total_spins = frame_bytes * 8;

    karyon::ContinuousWavefieldEngine engine;

    // 1. Ingest Pass 1 & Pass 2 (Autopoietic Phase Alignment & Reinforcement)
    engine.IngestRawBytes(buffer.data(), frame_bytes, true);
    
    // Evaluate & Reinforce phase momentum across full stream
    engine.ResetHistory();
    for (size_t i = 0; i < total_spins; ++i) {
        uint8_t byte = buffer[i / 8];
        int b = 7 - (i % 8);
        int8_t true_spin = ((byte >> b) & 1) ? +1 : -1;
        engine.PredictNextSpin();
        engine.PushWorkingSpin(true_spin);
    }

    // Ingest Pass 3
    engine.IngestRawBytes(buffer.data(), frame_bytes, true);

    std::cout << "1. DEEP PHASE RELAXATION COMPLETE:\n";
    std::cout << "   -> Active Topological Taps: " << engine.GetWaveTraceCount() << "\n\n";

    // 2. Autoregressive Generation from 1.5% Seed (768 bytes)
    engine.ResetHistory();
    size_t seed_bytes = 768;
    size_t target_generate_bytes = frame_bytes - seed_bytes;

    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    auto t_gen_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> generated_tail = engine.GenerateAutoregressiveBytes(target_generate_bytes);
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_duration = std::chrono::duration<double>(t_gen_end - t_gen_start).count();

    // 3. Evaluate Exact Bit Fidelity
    size_t exact_bit_matches = 0;
    size_t total_eval_spins = target_generate_bytes * 8;
    for (size_t i = 0; i < target_generate_bytes; ++i) {
        uint8_t ref_byte = buffer[seed_bytes + i];
        uint8_t gen_byte = generated_tail[i];
        for (int b = 7; b >= 0; --b) {
            if (((ref_byte >> b) & 1) == ((gen_byte >> b) & 1)) {
                exact_bit_matches++;
            }
        }
    }

    double bit_fidelity_pct = (static_cast<double>(exact_bit_matches) / total_eval_spins) * 100.0;
    double free_energy = -std::log(bit_fidelity_pct / 100.0);

    std::vector<uint8_t> synthesized_frame(frame_bytes);
    std::copy(buffer.begin(), buffer.begin() + seed_bytes, synthesized_frame.begin());
    std::copy(generated_tail.begin(), generated_tail.end(), synthesized_frame.begin() + seed_bytes);

    std::string out_png = "data/karyon_exp947_100pct.png";
    stbi_write_png(out_png.c_str(), 128, 128, 3, synthesized_frame.data(), 128 * 3);

    std::cout << "2. TELEMETRY RESULTS:\n";
    std::cout << "   -> Evaluated Bit-Spins: " << total_eval_spins << " spins\n";
    std::cout << "   -> Exact Matching Spins: " << exact_bit_matches << " spins\n";
    std::cout << "   -> Bit-Spin Reconstruction Fidelity: " << std::fixed << std::setprecision(4) << bit_fidelity_pct << " %\n";
    std::cout << "   -> Variational Free Energy F: " << std::setprecision(7) << free_energy << " nats\n";
    std::cout << "   -> Generation Time: " << std::setprecision(3) << gen_duration << " s\n";
    std::cout << "   -> Output PNG: " << out_png << "\n\n";

    return 0;
}
