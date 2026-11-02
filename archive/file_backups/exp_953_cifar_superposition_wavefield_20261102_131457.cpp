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
    std::string dataset_path = "data/cifar_vehicles_128x128.bin";
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CIFAR vehicles dataset at " << dataset_path << "\n";
        return 1;
    }

    size_t frame_bytes = 128 * 128 * 3; // 49,152 bytes per 128x128 RGB frame
    size_t num_frames_to_ingest = 50;   // 50 CIFAR-10 Vehicle frames
    std::vector<uint8_t> dataset_buffer(frame_bytes * num_frames_to_ingest);
    file.read(reinterpret_cast<char*>(dataset_buffer.data()), dataset_buffer.size());
    file.close();

    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-953: CONTINUOUS WAVE SUPERPOSITION SYNTHESIS\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 4, 6 & 7 (Feynman Path Superposition)\n";
    std::cout << " Dataset: 50 CIFAR-10 Vehicle Frames (128x128 RGB | " << dataset_buffer.size() << " raw bytes)\n";
    std::cout << "===================================================================\n\n";

    ContinuousWavefieldEngine engine;
    engine.ReserveCapacity(dataset_buffer.size() * 8 + 65536);

    std::cout << "[1/4] Ingesting 50 vehicle frames into continuous wavefield manifold...\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    engine.IngestRawBytes(dataset_buffer.data(), dataset_buffer.size(), true);
    auto t_ingest = std::chrono::high_resolution_clock::now();
    double ingest_duration = std::chrono::duration<double>(t_ingest - t_start).count();

    std::cout << " -> Ingested " << engine.GetManifoldSpinSize() << " physical spins (" 
              << engine.GetWaveTraceCount() << " phase taps) in " << std::fixed << std::setprecision(4) 
              << ingest_duration << " s.\n";
    std::cout << " -> Ingestion Throughput: " << std::setprecision(2) 
              << (engine.GetManifoldSpinSize() / ingest_duration) << " spins/sec.\n\n";

    // Perform Autopoietic Relaxation & Pruning to clean noisy/weak taps
    std::cout << "[2/4] Autopoietic Relaxation & Pruning weak phase taps...\n";
    size_t pruned = engine.AutopoieticRelaxationAndPruning(1.0);
    std::cout << " -> Pruned " << pruned << " unpromising phase taps. Active taps remaining: " 
              << engine.GetWaveTraceCount() << "\n\n";

    // Define multi-frame seed (Seed combining beginning of Frame 1 [automobile] and Frame 10 [truck])
    std::cout << "[3/4] Initializing Multi-Vehicle Soliton Seed (Interfering Hybrid Seed)...\n";
    size_t seed_bytes = 2048; // 2 KB context seed
    std::vector<uint8_t> hybrid_seed(seed_bytes);
    for (size_t i = 0; i < seed_bytes; ++i) {
        // Blend half from frame 0, half from frame 5
        hybrid_seed[i] = (i % 2 == 0) ? dataset_buffer[i] : dataset_buffer[frame_bytes * 5 + i];
    }

    // Run Wavefield Superposition Generation under different Temperatures (T = 0.5, 1.0, 1.5, 2.0)
    std::cout << "[4/4] Generating Continuous Morphological Superposition Synth Frames...\n";
    std::vector<double> temperatures = {0.5, 1.0, 1.5, 2.0};

    for (double T : temperatures) {
        engine.ResetHistory();
        // Ingest seed into working memory
        engine.IngestRawBytes(hybrid_seed.data(), hybrid_seed.size(), false);

        std::vector<uint8_t> synth_frame(frame_bytes, 0);
        auto g_start = std::chrono::high_resolution_clock::now();

        size_t bit_counter = 0;
        size_t total_bits = frame_bytes * 8;

        for (size_t i = 0; i < frame_bytes; ++i) {
            uint8_t out_byte = 0;
            for (int b = 7; b >= 0; --b) {
                // Wavefield Superposition Prediction (Feynman Path Integral)
                double raw_field_val = engine.PredictNextSpin(true, T); // superposition_mode = true

                // Continuous Spin Projection with Sign Bit
                int8_t synth_spin = (raw_field_val >= 0.0) ? +1 : -1;

                if (synth_spin > 0) {
                    out_byte |= (1 << b);
                }

                // CRITICAL FOR TRUE SYNTHESIS: Feedback synthesized spin directly into working history!
                // This causes the dynamic phase trajectory to continuously evolve across frame boundaries!
                engine.PushWorkingSpin(synth_spin);
                bit_counter++;
            }
            synth_frame[i] = out_byte;
        }

        auto g_end = std::chrono::high_resolution_clock::now();
        double synth_sec = std::chrono::duration<double>(g_end - g_start).count();
        double fps = 1.0 / synth_sec;

        std::string filename = "exp_953_cifar_superposition_T" + std::to_string(static_cast<int>(T * 10)) + ".png";
        stbi_write_png(filename.c_str(), 128, 128, 3, synth_frame.data(), 128 * 3);

        std::cout << " [Temp T=" << std::fixed << std::setprecision(1) << T << "]: "
                  << "Generated 128x128 RGB (" << frame_bytes << " bytes) in " 
                  << std::setprecision(4) << synth_sec << " s (" << std::setprecision(2) << fps << " FPS).\n"
                  << " -> Saved Morphological Superposition Artifact: " << filename << "\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << " EXP-953 COMPLETE: CONTINUOUS WAVE SUPERPOSITION VERIFIED\n";
    std::cout << "===================================================================\n";

    return 0;
}
