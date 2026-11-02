#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// KARYON-CORE EXP-956 CUDA: ULTRA-FAST BIT-PACKED CUDA SPIN WAVEFIELD KERNEL
// KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5
// Substrate: Pure Continuous Bit-Spin Manifold (V=2), Zero Weight Matrices, O(K) Wavefront Taps

constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152 bytes
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216 bits/spins
constexpr size_t WORDS_PER_FRAME = FRAME_BITS / 64;       // 6,144 uint64_t words

// CUDA Kernel: Bit-Packed 64-Spin Parallel Superposition Kernel
// Each CUDA Thread processes 64 physical spins (one uint64_t word) simultaneously
__global__ void cuda_bitpacked_wavefield_kernel(
    const uint64_t* __restrict__ d_packed_manifold,
    const size_t* __restrict__ d_ensemble_indices,
    int K,
    float beta,
    uint64_t* __restrict__ d_output_words
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= WORDS_PER_FRAME) return;

    uint64_t out_word = 0;

    // Process 64 bits/spins in parallel inside thread
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        int ones = 0;
        for (int k = 0; k < K; ++k) {
            size_t f_idx = d_ensemble_indices[k];
            uint64_t word = d_packed_manifold[f_idx * WORDS_PER_FRAME + w];
            ones += (int)((word >> bit) & 1ULL);
        }

        float mean_s = (2.0f * ones - (float)K) / (float)K;
        if (tanhf(beta * mean_s) >= 0.0f) {
            out_word |= (1ULL << bit);
        }
    }

    d_output_words[w] = out_word;
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-956 CUDA: ULTRA-FAST BIT-PACKED WAVEFIELD KERNEL\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5\n";
    std::cout << " Substrate: Pure Continuous Bit-Spin Manifold (V=2), Zero Matrices\n";
    std::cout << " Target GPU: NVIDIA Tesla T4 (sm_75)\n";
    std::cout << "===================================================================\n\n";

    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open data/cifar_vehicles_128x128.bin\n";
        return 1;
    }

    file.seekg(0, std::ios::end);
    size_t file_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> dataset(file_bytes);
    file.read(reinterpret_cast<char*>(dataset.data()), file_bytes);
    file.close();

    size_t num_frames = file_bytes / FRAME_BYTES;
    std::vector<uint64_t> packed_manifold(num_frames * WORDS_PER_FRAME);

    // Bit-pack raw bytes into uint64_t words
    for (size_t f = 0; f < num_frames; ++f) {
        const uint8_t* frame_ptr = dataset.data() + f * FRAME_BYTES;
        uint64_t* word_ptr = packed_manifold.data() + f * WORDS_PER_FRAME;

        for (size_t w = 0; w < WORDS_PER_FRAME; ++w) {
            uint64_t word = 0;
            size_t byte_base = w * 8;
            for (int b = 0; b < 8; ++b) {
                uint64_t val = frame_ptr[byte_base + b];
                word |= (val << (b * 8));
            }
            word_ptr[w] = word;
        }
    }

    // Allocate GPU Device Memory
    uint64_t *d_packed_manifold = nullptr, *d_output_words = nullptr;
    size_t *d_ensemble_indices = nullptr;

    cudaMalloc(&d_packed_manifold, packed_manifold.size() * sizeof(uint64_t));
    cudaMalloc(&d_output_words, WORDS_PER_FRAME * sizeof(uint64_t));
    cudaMemcpy(d_packed_manifold, packed_manifold.data(), packed_manifold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<size_t> active_ensemble = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
    int K = active_ensemble.size();

    cudaMalloc(&d_ensemble_indices, K * sizeof(size_t));
    cudaMemcpy(d_ensemble_indices, active_ensemble.data(), K * sizeof(size_t), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (WORDS_PER_FRAME + block_size - 1) / block_size;

    // Warmup CUDA kernel execution
    cuda_bitpacked_wavefield_kernel<<<grid_size, block_size>>>(
        d_packed_manifold, d_ensemble_indices, K, 3.0f, d_output_words
    );
    cudaDeviceSynchronize();

    // High-Precision Benchmark (10,000 runs)
    int runs = 10000;
    auto cuda_t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        cuda_bitpacked_wavefield_kernel<<<grid_size, block_size>>>(
            d_packed_manifold, d_ensemble_indices, K, 3.0f, d_output_words
        );
    }
    cudaDeviceSynchronize();

    auto cuda_t1 = std::chrono::high_resolution_clock::now();
    double cuda_sec = std::chrono::duration<double>(cuda_t1 - cuda_t0).count();

    double fps = runs / cuda_sec;
    double latency_ms = (cuda_sec / runs) * 1000.0;
    double mspins_sec = (runs * K * FRAME_BITS) / (cuda_sec * 1e6);

    std::cout << "[+] CUDA Bit-Packed Kernel Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS\n";
    std::cout << "[+] CUDA Frame Latency: " << std::setprecision(4) << latency_ms << " ms (" << latency_ms * 1000.0 << " us)\n";
    std::cout << "[+] CUDA Wavefield Spin Throughput: " << std::setprecision(2) << mspins_sec << " M-Spins/sec (" << mspins_sec / 1000.0 << " G-Spins/s)\n\n";

    // Copy result back to CPU and write image
    std::vector<uint64_t> host_out_words(WORDS_PER_FRAME);
    cudaMemcpy(host_out_words.data(), d_output_words, WORDS_PER_FRAME * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> output_bytes(FRAME_BYTES);
    for (size_t w = 0; w < WORDS_PER_FRAME; ++w) {
        uint64_t word = host_out_words[w];
        size_t byte_base = w * 8;
        for (int b = 0; b < 8; ++b) {
            output_bytes[byte_base + b] = static_cast<uint8_t>((word >> (b * 8)) & 0xFF);
        }
    }

    stbi_write_png("exp_956_cuda_bitpacked_wavefield.png", WIDTH, HEIGHT, CHANNELS, output_bytes.data(), WIDTH * CHANNELS);
    std::cout << "[+] Output saved to exp_956_cuda_bitpacked_wavefield.png\n";

    cudaFree(d_packed_manifold);
    cudaFree(d_output_words);
    cudaFree(d_ensemble_indices);

    return 0;
}
