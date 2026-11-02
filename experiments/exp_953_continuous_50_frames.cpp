#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "karyon_core_wavefield.hpp"

int main() {
    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) return 1;

    file.seekg(0, std::ios::end);
    size_t file_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> dataset(file_bytes);
    file.read(reinterpret_cast<char*>(dataset.data()), file_bytes);
    file.close();

    ContinuousWavefieldEngine engine;
    engine.IngestDataset(dataset.data(), dataset.size(), 128 * 128 * 3);

    // Continuous Field Superposition across ALL 50 vehicle frames
    std::vector<size_t> all_frames(engine.GetFrameCount());
    for (size_t f = 0; f < engine.GetFrameCount(); ++f) all_frames[f] = f;

    engine.InitializeSuperpositionWavefronts(all_frames, 0);

    size_t frame_bytes = 128 * 128 * 3;
    std::vector<uint8_t> synth_frame(frame_bytes);

    for (size_t byte_idx = 0; byte_idx < frame_bytes; ++byte_idx) {
        uint8_t b_out = 0;
        for (int bit = 7; bit >= 0; --bit) {
            double spin_val = engine.StepSuperpositionWavefield(1.0); // T=1.0 continuous wave superposition
            if (spin_val >= 0.0) {
                b_out |= (1 << bit);
            }
        }
        synth_frame[byte_idx] = b_out;
    }

    stbi_write_png("exp_953_continuous_50_frames.png", 128, 128, 3, synth_frame.data(), 128 * 3);
    std::cout << "EXP-953 Continuous 50-frame wavefield synthesis complete.\n";
    return 0;
}
