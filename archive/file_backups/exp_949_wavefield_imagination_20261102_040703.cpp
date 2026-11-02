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

    size_t frame_bytes = 128 * 128 * 3; // 49,152 bytes
    size_t num_frames_to_ingest = 10; // Ingest 10 diverse frames
    std::vector<uint8_t> multi_frame_buffer(frame_bytes * num_frames_to_ingest);
    file.read(reinterpret_cast<char*>(multi_frame_buffer.data()), multi_frame_buffer.size());
    file.close();

    std::cout << "==================================================================\n";
    std::cout << "EXP-949: Sovereign Wavefield Imagination & Superposition Genesis\n";
    std::cout << "Ingested " << num_frames_to_ingest << " diverse physical manifold frames ("
              << multi_frame_buffer.size() << " bytes, " << multi_frame_buffer.size() * 8 << " spins)\n";
    std::cout << "==================================================================\n\n";

    // 1. Build Multi-Frame Wavefield Engine
    karyon::ContinuousWavefieldEngine engine;
    engine.IngestRawBytes(multi_frame_buffer.data(), multi_frame_buffer.size(), true);

    std::cout << "Wavefield topology built with " << engine.GetWaveTraceCount() << " phase taps.\n\n";

    // Test 1: Ground Truth Deterministic Reconstruction of Frame 0
    {
        engine.ResetHistory();
        size_t seed_bytes = 768; // 1.56% micro prefix of Frame 0
        engine.IngestRawBytes(multi_frame_buffer.data(), seed_bytes, false);

        size_t gen_bytes = frame_bytes - seed_bytes;
        auto gen = engine.GenerateAutoregressiveBytes(gen_bytes);

        std::vector<uint8_t> full(frame_bytes);
        std::copy(multi_frame_buffer.begin(), multi_frame_buffer.begin() + seed_bytes, full.begin());
        std::copy(gen.begin(), gen.end(), full.begin() + seed_bytes);

        stbi_write_png("data/karyon_exp949_reconstruction_frame0.png", 128, 128, 3, full.data(), 128 * 3);
        std::cout << "Saved deterministic reconstruction: data/karyon_exp949_reconstruction_frame0.png\n";
    }

    // Test 2: Phase Superposition (Hybrid Seed - 50% Frame 0 + 50% Frame 1)
    {
        engine.ResetHistory();
        size_t seed_bytes = 1536; // 3% seed context
        std::vector<uint8_t> hybrid_seed(seed_bytes);
        for (size_t i = 0; i < seed_bytes; ++i) {
            // Quantum/Wave linear blend of frame 0 & frame 1
            hybrid_seed[i] = static_cast<uint8_t>((static_cast<uint16_t>(multi_frame_buffer[i]) +
                                                    static_cast<uint16_t>(multi_frame_buffer[frame_bytes + i])) / 2);
        }

        engine.IngestRawBytes(hybrid_seed.data(), seed_bytes, false);
        size_t gen_bytes = frame_bytes - seed_bytes;
        auto gen = engine.GenerateAutoregressiveBytes(gen_bytes);

        std::vector<uint8_t> full(frame_bytes);
        std::copy(hybrid_seed.begin(), hybrid_seed.begin() + seed_bytes, full.begin());
        std::copy(gen.begin(), gen.end(), full.begin() + seed_bytes);

        stbi_write_png("data/karyon_exp949_imagination_hybrid_superposition.png", 128, 128, 3, full.data(), 128 * 3);
        std::cout << "Saved phase superposition synthesis: data/karyon_exp949_imagination_hybrid_superposition.png\n";
    }

    // Test 3: Phase Noise / Stochastic Excitation (\sigma_{hw} = 0.05, 0.15, 0.30)
    std::vector<double> noise_levels = {0.02, 0.05, 0.12, 0.25};
    std::mt19937 rng(42);

    for (double sigma : noise_levels) {
        engine.ResetHistory();
        size_t seed_bytes = 768; // Micro seed
        std::vector<uint8_t> noisy_seed(seed_bytes);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < seed_bytes; ++i) {
            uint8_t b = multi_frame_buffer[i];
            if (dist(rng) < sigma) {
                b ^= (1 << (dist(rng) * 8)); // Stochastic phase bit excitation
            }
            noisy_seed[i] = b;
        }

        engine.IngestRawBytes(noisy_seed.data(), seed_bytes, false);
        size_t gen_bytes = frame_bytes - seed_bytes;
        auto gen = engine.GenerateAutoregressiveBytes(gen_bytes);

        std::vector<uint8_t> full(frame_bytes);
        std::copy(noisy_seed.begin(), noisy_seed.begin() + seed_bytes, full.begin());
        std::copy(gen.begin(), gen.end(), full.begin() + seed_bytes);

        std::string filename = "data/karyon_exp949_stochastic_noise_sigma_" + std::to_string(static_cast<int>(sigma * 100)) + ".png";
        stbi_write_png(filename.c_str(), 128, 128, 3, full.data(), 128 * 3);
        std::cout << "Saved stochastic excitation (sigma=" << sigma << "): " << filename << "\n";
    }

    std::cout << "\nEXP-949 Superposition & Excitation Complete.\n";
    return 0;
}
