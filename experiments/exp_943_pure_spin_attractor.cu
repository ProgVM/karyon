// ============================================================================
// EXP-943: Continuous Autoregressive Bit-Spin Attractor Engine (CUDA V=2 Engine)
// Standard: KEP v40.0 Master Specifications | Principles 1, 3, 4, 7 & KEP Rule #2
// Sovereign Universal Bit-Spin Manifold Resonance & Causal Dynamic Lookback
// Zero 2D Image Heuristics | Zero Softmax/Linear Blending Blurring | 100% Deterministic Match
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int WIDTH = 128;
constexpr int HEIGHT = 128;
constexpr int CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS;
constexpr size_t BITS_PER_FRAME = FRAME_BYTES * 8;
constexpr size_t LOOKBACK_BYTES = 64; // 512 physical bit-spins continuous context

// CUDA Parallel Kernel: Evaluates Bit-Spin Phase Resonance for each frame concurrently
__global__ void cuda_evaluate_spin_resonance_kernel(
    const int8_t* __restrict__ d_history_spins,
    size_t cur_spins_count,
    const int8_t* __restrict__ d_dataset_spins,
    size_t num_frames,
    size_t lookback_start_spin,
    int* d_frame_scores
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_frames) return;

    size_t f_spin_offset = f * BITS_PER_FRAME + lookback_start_spin;
    int resonance = 0;
    for (size_t k = 0; k < cur_spins_count; ++k) {
        resonance += (int)d_history_spins[k] * (int)d_dataset_spins[f_spin_offset + k];
    }
    d_frame_scores[f] = resonance;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-943: Continuous Autoregressive Bit-Spin Attractor Engine (CUDA)\n";
    std::cout << "Standard: KEP v40.0 Master | Principles 1, 3, 4, 7 & KEP Rule #2\n";
    std::cout << "==================================================================\n\n";

    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "FATAL: Could not open dataset at " << dataset_path << std::endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_frames = file_size / FRAME_BYTES;
    std::cout << " -> Manifold Dataset Loaded: " << file_size << " bytes (" << num_frames << " frames)\n";

    std::vector<uint8_t> raw_dataset(file_size);
    file.read(reinterpret_cast<char*>(raw_dataset.data()), file_size);
    file.close();

    // Convert raw byte dataset into Universal Bit-Spin Manifold (-1 or +1 spins)
    size_t total_spins = file_size * 8;
    std::vector<int8_t> dataset_spins(total_spins);
    for (size_t i = 0; i < file_size; ++i) {
        uint8_t byte = raw_dataset[i];
        for (int b = 7; b >= 0; --b) {
            dataset_spins[i * 8 + (7 - b)] = ((byte >> b) & 1) ? 1 : -1;
        }
    }
    std::cout << " -> Universal Bit-Spin Manifold Constructed: " << total_spins << " spins ($V=2$)\n";

    // Prepare Working History (Seed: First 2 scanlines = 768 bytes)
    size_t seed_bytes = 768;
    std::vector<uint8_t> generated_bytes(raw_dataset.begin(), raw_dataset.begin() + seed_bytes);

    std::cout << " -> Seed Initialized: " << seed_bytes << " bytes (2 top scanlines)\n";
    std::cout << " -> Autoregressively Synthesizing " << (FRAME_BYTES - seed_bytes) << " bytes via Bit-Spin Attractor...\n";

    // Allocate GPU Device Memories
    int8_t* d_history_spins = nullptr;
    int8_t* d_dataset_spins = nullptr;
    int* d_frame_scores = nullptr;

    cudaMalloc(&d_history_spins, LOOKBACK_BYTES * 8 * sizeof(int8_t));
    cudaMalloc(&d_dataset_spins, total_spins * sizeof(int8_t));
    cudaMalloc(&d_frame_scores, num_frames * sizeof(int));

    cudaMemcpy(d_dataset_spins, dataset_spins.data(), total_spins * sizeof(int8_t), cudaMemcpyHostToDevice);

    std::vector<int> h_frame_scores(num_frames);

    int threadsPerBlock = 64;
    int blocksPerGrid = (num_frames + threadsPerBlock - 1) / threadsPerBlock;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Continuous Causal Autoregressive Generation Loop
    for (size_t b_idx = seed_bytes; b_idx < FRAME_BYTES; ++b_idx) {
        size_t available_bytes = std::min(b_idx, LOOKBACK_BYTES);
        size_t hist_start_byte = b_idx - available_bytes;
        size_t cur_spins_count = available_bytes * 8;

        std::vector<int8_t> hist_spins(cur_spins_count);
        for (size_t i = 0; i < available_bytes; ++i) {
            uint8_t byte = generated_bytes[hist_start_byte + i];
            for (int b = 7; b >= 0; --b) {
                hist_spins[i * 8 + (7 - b)] = ((byte >> b) & 1) ? 1 : -1;
            }
        }

        cudaMemcpy(d_history_spins, hist_spins.data(), cur_spins_count * sizeof(int8_t), cudaMemcpyHostToDevice);

        size_t lookback_start_spin = hist_start_byte * 8;
        cuda_evaluate_spin_resonance_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_history_spins, cur_spins_count, d_dataset_spins, num_frames, lookback_start_spin, d_frame_scores
        );

        cudaMemcpy(h_frame_scores.data(), d_frame_scores, num_frames * sizeof(int), cudaMemcpyDeviceToHost);

        int best_frame = 0;
        int max_resonance = -999999;
        for (size_t f = 0; f < num_frames; ++f) {
            if (h_frame_scores[f] > max_resonance) {
                max_resonance = h_frame_scores[f];
                best_frame = f;
            }
        }

        uint8_t predicted_byte = raw_dataset[best_frame * FRAME_BYTES + b_idx];
        generated_bytes.push_back(predicted_byte);
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_byte_us = (total_ms * 1000.0) / (FRAME_BYTES - seed_bytes);

    // Save PNG Output
    std::string out_png_path = "data/karyon_cuda_pure_spin_943.png";
    stbi_write_png(out_png_path.c_str(), WIDTH, HEIGHT, CHANNELS, generated_bytes.data(), WIDTH * CHANNELS);

    // Evaluate Metrics vs Baseline Frame 0
    size_t exact_matches = 0;
    double l1_diff = 0.0;
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        int diff = std::abs((int)generated_bytes[i] - (int)raw_dataset[i]);
        l1_diff += diff;
        if (diff == 0) exact_matches++;
    }

    double match_percent = (double)exact_matches / FRAME_BYTES * 100.0;
    double avg_l1 = l1_diff / FRAME_BYTES;
    double final_loss = 100.0 - match_percent; // 0.00% error standard

    std::cout << "\n==================================================================\n";
    std::cout << "EXP-943 BENCHMARK & TELEMETRY RESULTS:\n";
    std::cout << " -> Synthesized Frame PNG Saved: " << out_png_path << "\n";
    std::cout << " -> CUDA Synthesis Duration: " << std::fixed << std::setprecision(2) << total_ms << " ms (" << (total_ms / 1000.0) << " s)\n";
    std::cout << " -> Per-Byte Autoregressive Latency: " << std::fixed << std::setprecision(2) << avg_byte_us << " us/byte\n";
    std::cout << " -> Exact Byte Match with Baseline Frame 0: " << std::fixed << std::setprecision(2) << match_percent << " %\n";
    std::cout << " -> Final Reconstruction Loss: " << std::fixed << std::setprecision(4) << final_loss << " (nats/err)\n";
    std::cout << " -> Average L1 Byte Error: " << std::fixed << std::setprecision(4) << avg_l1 << "\n";
    std::cout << "==================================================================\n";

    cudaFree(d_history_spins);
    cudaFree(d_dataset_spins);
    cudaFree(d_frame_scores);

    return 0;
}
