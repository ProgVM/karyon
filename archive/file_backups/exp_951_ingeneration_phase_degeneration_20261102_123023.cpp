#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include "karyon_core_wavefield.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset at " << dataset_path << "\n";
        return 1;
    }

    size_t frame_bytes = 128 * 128 * 3;
    size_t num_frames_to_ingest = 10;
    std::vector<uint8_t> multi_frame_buffer(frame_bytes * num_frames_to_ingest);
    file.read(reinterpret_cast<char*>(multi_frame_buffer.data()), multi_frame_buffer.size());
    file.close();

    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-951: IN-GENERATION CONTINUOUS WAVEFIELD PHASE DEGENERATION\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 4 & 7\n";
    std::cout << " Substrate: Dynamic Phase Jitter s_t -> s_t + N(0, sigma^2) WITHIN Autoregressive Loop\n";
    std::cout << "===================================================================\n\n";

    karyon::ContinuousWavefieldEngine engine;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.IngestRawBytes(multi_frame_buffer.data(), multi_frame_buffer.size(), true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ingest_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[+] Ingested " << num_frames_to_ingest << " frames (" << (multi_frame_buffer.size() * 8) 
              << " spins) in " << ingest_sec << " sec. Taps: " << engine.GetWaveTraceCount() << "\n";

    std::vector<double> sigma_levels = {0.30, 0.60, 0.90, 1.20, 1.50};
    std::mt19937 rng(42);

    for (double sigma : sigma_levels) {
        engine.ResetHistory();
        size_t seed_bytes = 768; // 256 RGB pixels seed
        
        std::vector<uint8_t> clean_seed(multi_frame_buffer.begin(), multi_frame_buffer.begin() + seed_bytes);
        engine.IngestRawBytes(clean_seed.data(), seed_bytes, false);

        size_t gen_bytes = frame_bytes - seed_bytes;
        std::vector<uint8_t> full(frame_bytes);
        std::copy(clean_seed.begin(), clean_seed.end(), full.begin());

        std::normal_distribution<double> continuous_noise(0.0, sigma);

        size_t byte_idx = seed_bytes;
        size_t flipped_bits = 0;
        size_t total_gen_bits = gen_bytes * 8;

        for (size_t i = 0; i < gen_bytes; ++i) {
            uint8_t current_byte = 0;
            for (int bit = 7; bit >= 0; --bit) {
                // 1. Predict raw wavefield spin
                double raw_spin = engine.PredictNextSpin();
                
                // 2. Inject Continuous Phase Noise directly into dynamic wave prediction
                double eta = continuous_noise(rng);
                double perturbed_spin = raw_spin + eta;

                // 3. Attractor Basin Thresholding & Phase Snapping
                int8_t perceived_spin = (perturbed_spin >= 0.0) ? +1 : -1;
                uint8_t bit_val = (perceived_spin == 1) ? 1 : 0;
                current_byte |= (bit_val << bit);

                if ((raw_spin >= 0.0 ? 1 : -1) != perceived_spin) {
                    flipped_bits++;
                }

                // 4. Push perceived spin into working history to mutate future phase momentum
                engine.PushWorkingSpin(perceived_spin);
            }
            full[byte_idx++] = current_byte;
        }

        double perturbation_rate = (double)flipped_bits / total_gen_bits * 100.0;

        std::string filename = "data/karyon_exp951_ingeneration_noise_sigma_" + std::to_string(static_cast<int>(sigma * 100)) + ".png";
        stbi_write_png(filename.c_str(), 128, 128, 3, full.data(), 128 * 3);
        std::cout << "[+] Saved IN-GENERATION wavefield excitation (sigma=" << std::fixed << std::setprecision(2) << sigma << "): " << filename 
                  << " | Dynamic Phase Bifurcations: " << flipped_bits << "/" << total_gen_bits 
                  << " (" << std::fixed << std::setprecision(2) << perturbation_rate << "%)\n";
    }

    return 0;
}
