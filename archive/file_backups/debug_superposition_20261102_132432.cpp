#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "stb_image_write.h"

int main() {
    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) return 1;

    size_t frame_bytes = 128 * 128 * 3;
    std::vector<uint8_t> f0(frame_bytes), f5(frame_bytes), f10(frame_bytes);
    file.read(reinterpret_cast<char*>(f0.data()), frame_bytes);
    file.seekg(5 * frame_bytes, std::ios::beg);
    file.read(reinterpret_cast<char*>(f5.data()), frame_bytes);
    file.seekg(10 * frame_bytes, std::ios::beg);
    file.read(reinterpret_cast<char*>(f10.data()), frame_bytes);
    file.close();

    // 1. Save frame 0 raw
    stbi_write_png("debug_frame0.png", 128, 128, 3, f0.data(), 128 * 3);

    // 2. Average in RGB continuous space vs bit-spin space
    std::vector<uint8_t> rgb_avg(frame_bytes);
    std::vector<uint8_t> bit_majority(frame_bytes);

    for (size_t i = 0; i < frame_bytes; ++i) {
        // RGB Continuous linear interpolation / superposition
        rgb_avg[i] = static_cast<uint8_t>((static_cast<int>(f0[i]) + static_cast<int>(f5[i]) + static_cast<int>(f10[i])) / 3);

        // Bitwise majority
        uint8_t b_out = 0;
        for (int bit = 7; bit >= 0; --bit) {
            int s0 = ((f0[i] >> bit) & 1) ? 1 : -1;
            int s5 = ((f5[i] >> bit) & 1) ? 1 : -1;
            int s10 = ((f10[i] >> bit) & 1) ? 1 : -1;
            int sum_s = s0 + s5 + s10;
            if (sum_s > 0) b_out |= (1 << bit);
        }
        bit_majority[i] = b_out;
    }

    stbi_write_png("debug_rgb_avg.png", 128, 128, 3, rgb_avg.data(), 128 * 3);
    stbi_write_png("debug_bit_majority.png", 128, 128, 3, bit_majority.data(), 128 * 3);

    std::cout << "Debug frames generated successfully.\n";
    return 0;
}
