// ============================================================================
// EXP-943: Pure Continuous Bit-Spin Attractor Snapping (V=2 Universal Substrate)
// Karyon Engineering Protocol (KEP) Master Specification | Standard v40.0
// Axiom Principle 1, 3, 4 & 7: Universal Bit-Spin Stream, Hard Attractor Resonance,
// Zero Image-Specific 2D Heuristics, Zero Softmax/Linear Blending Blurring.
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

// CUDA Kernel: Pure Bit-Spin Universal Manifold Resonance & Attractor Snapping
// Evaluates exact physical phase alignment over universal lookback history L_context (1024 spins)
// Output is selected strictly via minimum Free Energy / Maximum Phase Resonance (Hard Snapping)
__global__ void cuda_pure_spin_attractor_kernel(
    const int8_t* __restrict__ d_history_spins, // Current sequence working history spins (-1 or +1)
    size_t history_len,
    const int8_t* __restrict__ d_dataset_spins, // Master bit-spin manifold of stored continuous dynamics
    size_t num_dataset_frames,
    uint8_t* d_output_frame_bytes
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= FRAME_BYTES) return;

    // Each byte consists of 8 consecutive physical bit-spins in the universal manifold
    size_t byte_spin_offset = pixel_idx * 8;

    // We compare a window of lookback spins (e.g. up to 64 bytes = 512 spins)
    constexpr size_t LOOKBACK_SPINS = 512;
    size_t current_lookback = (history_len < LOOKBACK_SPINS) ? history_len : LOOKBACK_SPINS;

    float max_resonance_score = -1e9f;
    size_t best_frame_idx = 0;

    // Search across all stored frames in the manifold for the deepest attractor basin (phase lock)
    for (size_t f = 0; f < num_dataset_frames; ++f) {
        size_t frame_spin_start = f * BITS_PER_FRAME;

        // Calculate phase resonance score over lookback window
        float resonance = 0.0f;
        if (current_lookback > 0) {
            size_t history_start = history_len - current_lookback;
            size_t dataset_lookback_start = frame_spin_start + (byte_spin_offset > current_lookback ? byte_spin_offset - current_lookback : 0);

            for (size_t k = 0; k < current_lookback; ++k) {
                int8_t s_hist = d_history_spins[history_start + k];
                int8_t s_data = d_dataset_spins[dataset_lookback_start + k];
                // Spin product: +1 for parallel alignment, -1 for anti-parallel
                resonance += (float)(s_hist * s_data);
            }
        }

        if (resonance > max_resonance_score) {
            max_resonance_score = resonance;
            best_frame_idx = f;
        }
    }

    // Hard Attractor Snapping (Zero Blurring): Select exact candidate byte from winning physical basin
    size_t target_byte_offset = best_frame_idx * FRAME_BYTES + pixel_idx;
    // Reconstruction of 8 spins back into pure byte without interpolation
    uint8_t snapped_byte = 0;
    for (int b = 7; b >= 0; --b) {
        int8_t spin = d_dataset_spins[target_byte_offset * 8 + (7 - b)];
        if (spin > 0) {
            snapped_byte |= (1 << b);
        }
    }

    d_output_frame_bytes[pixel_idx] = snapped_byte;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-943: Universal Bit-Spin Attractor Snapping (V=2 Sovereign Substrate)\n";
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

    // Convert raw byte manifold into Universal Bit-Spin Manifold (-1 or +1 spins)
    size_t total_spins = file_size * 8;
    std::vector<int8_t> dataset_spins(total_spins);
    for (size_t i = 0; i < file_size; ++i) {
        uint8_t byte = raw_dataset[i];
        for (int b = 7; b >= 0; --b) {
            dataset_spins[i * 8 + (7 - b)] = ((byte >> b) & 1) ? 1 : -1;
        }
    }
    std::cout << " -> Universal Bit-Spin Manifold Constructed: " << total_spins << " spins ($V=2$)\n";

    // Prepare Working History (Seed: First 2 scanlines = 768 bytes = 6144 spins)
    size_t seed_bytes = 768;
    size_t seed_spins = seed_bytes * 8;
    std::vector<int8_t> history_spins(dataset_spins.begin(), dataset_spins.begin() + seed_spins);

    // Allocate Device GPU Buffers
    int8_t* d_history_spins = nullptr;
    int8_t* d_dataset_spins = nullptr;
    uint8_t* d_output_frame_bytes = nullptr;

    cudaMalloc(&d_history_spins, seed_spins * sizeof(int8_t));
    cudaMalloc(&d_dataset_spins, total_spins * sizeof(int8_t));
    cudaMalloc(&d_output_frame_bytes, FRAME_BYTES * sizeof(uint8_t));

    cudaMemcpy(d_history_spins, history_spins.data(), seed_spins * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataset_spins, dataset_spins.data(), total_spins * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Launch CUDA Kernel for Universal Attractor Snapping
    int threadsPerBlock = 256;
    int blocksPerGrid = (FRAME_BYTES + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << " -> Launching CUDA Bit-Spin Attractor Kernel (" << blocksPerGrid << " blocks x " << threadsPerBlock << " threads)...\n";

    // Warmup
    cuda_pure_spin_attractor_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_history_spins, seed_spins, d_dataset_spins, num_frames, d_output_frame_bytes
    );
    cudaDeviceSynchronize();

    // High Precision Timed Benchmark
    constexpr int BENCHMARK_RUNS = 50;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < BENCHMARK_RUNS; ++r) {
        cuda_pure_spin_attractor_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_history_spins, seed_spins, d_dataset_spins, num_frames, d_output_frame_bytes
        );
    }
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_frame_latency_ms = total_ms / BENCHMARK_RUNS;
    double fps = 1000.0 / avg_frame_latency_ms;

    // Retrieve Output Frame
    std::vector<uint8_t> output_frame(FRAME_BYTES);
    cudaMemcpy(output_frame.data(), d_output_frame_bytes, FRAME_BYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    std::string out_png_path = "data/karyon_cuda_pure_spin_943.png";
    stbi_write_png(out_png_path.c_str(), WIDTH, HEIGHT, CHANNELS, output_frame.data(), WIDTH * CHANNELS);

    // Quantitative Evaluation against baseline frame 0
    double l1_diff = 0.0;
    size_t exact_matches = 0;
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        int diff = std::abs((int)output_frame[i] - (int)raw_dataset[i]);
        l1_diff += diff;
        if (diff == 0) exact_matches++;
    }
    double avg_l1 = l1_diff / FRAME_BYTES;
    double match_percent = (double)exact_matches / FRAME_BYTES * 100.0;

    std::cout << "\n==================================================================\n";
    std::cout << "EXP-943 BENCHMARK & TELEMETRY RESULTS:\n";
    std::cout << " -> Synthesized Frame PNG Saved: " << out_png_path << "\n";
    std::cout << " -> CUDA Average Frame Latency: " << std::fixed << std::setprecision(4) << avg_frame_latency_ms << " ms (" << (avg_frame_latency_ms * 1000.0) << " us)\n";
    std::cout << " -> CUDA Generation Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS\n";
    std::cout << " -> Exact Byte Match with Baseline Frame 0: " << std::fixed << std::setprecision(2) << match_percent << " %\n";
    std::cout << " -> Average L1 Byte Error: " << std::fixed << std::setprecision(4) << avg_l1 << "\n";
    std::cout << "==================================================================\n";

    // Free CUDA Resources
    cudaFree(d_history_spins);
    cudaFree(d_dataset_spins);
    cudaFree(d_output_frame_bytes);

    return 0;
}
