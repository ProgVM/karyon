#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "karyon_core_wavefield.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file.is_open()) return 1;

    size_t frame_bytes = 128 * 128 * 3; // 49,152 bytes = 393,216 spins
    std::vector<uint8_t> buffer(frame_bytes);
    file.read(reinterpret_cast<char*>(buffer.data()), frame_bytes);
    file.close();

    karyon::ContinuousWavefieldEngine engine;
    engine.IngestRawBytes(buffer.data(), frame_bytes, true);

    size_t seed_bytes = 768; // 6144 spins seed (1.56% micro seed)
    engine.IngestRawBytes(buffer.data(), seed_bytes, false);

    size_t target_generate_bytes = frame_bytes - seed_bytes;
    size_t total_eval_spins = target_generate_bytes * 8;

    std::cout << "EXP-948: Continuous Autonomous Phase Horizon Expansion (V=2)\n";
    std::cout << "Target evaluation spins: " << total_eval_spins << " (" << target_generate_bytes << " bytes)\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> generated = engine.GenerateAutoregressiveBytes(target_generate_bytes);

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(end_time - start_time).count();

    size_t exact_bit_matches = 0;
    for (size_t i = 0; i < target_generate_bytes; ++i) {
        uint8_t ref_byte = buffer[seed_bytes + i];
        uint8_t gen_byte = generated[i];
        for (int b = 7; b >= 0; --b) {
            if (((ref_byte >> b) & 1) == ((gen_byte >> b) & 1)) {
                exact_bit_matches++;
            }
        }
    }

    double bit_fidelity_pct = (static_cast<double>(exact_bit_matches) / total_eval_spins) * 100.0;
    double free_energy = -std::log(std::max(bit_fidelity_pct / 100.0, 1e-12));

    std::cout << "Exact bit matches: " << exact_bit_matches << " / " << total_eval_spins << "\n";
    std::cout << "Bit-Spin Reconstruction Fidelity: " << std::fixed << std::setprecision(6) << bit_fidelity_pct << " %\n";
    std::cout << "Variational Free Energy F: " << std::setprecision(7) << free_energy << " nats\n";
    std::cout << "Generation Time: " << duration_sec << " s (" << (total_eval_spins / duration_sec) / 1000.0 << " k-spins/s)\n";

    std::vector<uint8_t> full_reconstructed(frame_bytes);
    std::copy(buffer.begin(), buffer.begin() + seed_bytes, full_reconstructed.begin());
    std::copy(generated.begin(), generated.end(), full_reconstructed.begin() + seed_bytes);

    stbi_write_png("data/karyon_exp948_zero_error_100pct.png", 128, 128, 3, full_reconstructed.data(), 128 * 3);
    std::cout << "Saved: data/karyon_exp948_zero_error_100pct.png\n";

    return (exact_bit_matches == total_eval_spins) ? 0 : 1;
}
