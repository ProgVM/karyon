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
    std::cout << " KARYON-CORE EXP-950: CONTINUOUS PHASE-WAVEFIELD NOISE EXCITATION (CP-WNE)\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 4 & 7\n";
    std::cout << " Substrate: Continuous Spin Wavefield Perturbation s_t -> s_t + N(0, sigma^2)\n";
    std::cout << "===================================================================\n\n";

    karyon::ContinuousWavefieldEngine engine;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.IngestRawBytes(multi_frame_buffer.data(), multi_frame_buffer.size(), true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ingest_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[+] Ingested " << num_frames_to_ingest << " frames (" << (multi_frame_buffer.size() * 8) 
              << " spins) in " << ingest_sec << " sec. Taps: " << engine.GetWaveTraceCount() << "\n";

    std::vector<double> sigma_levels = {0.02, 0.05, 0.12, 0.25};
    std::mt19937 rng(42);

    for (double sigma : sigma_levels) {
        engine.ResetHistory();
        size_t seed_bytes = 768; // 256 RGB pixels micro seed
        std::normal_distribution<double> continuous_noise(0.0, sigma);

        std::vector<uint8_t> perturbed_seed(seed_bytes);

        for (size_t i = 0; i < seed_bytes; ++i) {
            uint8_t orig_byte = multi_frame_buffer[i];
            uint8_t continuous_perturbed_byte = 0;

            for (int bit = 7; bit >= 0; --bit) {
                // Continuous spin wave amplitude s_t \in {-1.0, +1.0}
                double spin = ((orig_byte >> bit) & 1) ? +1.0 : -1.0;
                
                // Continuous Phase Noise Injection (Gaussian Wave Shift)
                double eta = continuous_noise(rng);
                double continuous_spin = spin + eta;

                // Continuous physical phase alignment (Attractor Boundary Thresholding)
                uint8_t bit_val = (continuous_spin >= 0.0) ? 1 : 0;
                continuous_perturbed_byte |= (bit_val << bit);
            }
            perturbed_seed[i] = continuous_perturbed_byte;
        }

        engine.IngestRawBytes(perturbed_seed.data(), seed_bytes, false);
        size_t gen_bytes = frame_bytes - seed_bytes;
        auto gen = engine.GenerateAutoregressiveBytes(gen_bytes);

        std::vector<uint8_t> full(frame_bytes);
        std::copy(perturbed_seed.begin(), perturbed_seed.end(), full.begin());
        std::copy(gen.begin(), gen.end(), full.begin() + seed_bytes);

        std::string filename = "data/karyon_exp950_continuous_noise_sigma_" + std::to_string(static_cast<int>(sigma * 100)) + ".png";
        stbi_write_png(filename.c_str(), 128, 128, 3, full.data(), 128 * 3);
        std::cout << "[+] Saved CONTINUOUS wavefield excitation (sigma=" << sigma << "): " << filename << "\n";
    }

    return 0;
}
