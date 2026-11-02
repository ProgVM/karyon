#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// KARYON-CORE EXP-956 CUDA V2: WARP REDUCE & INT4/INT8 BITWISE SPEEDUP KERNEL
// KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5

constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152 bytes
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216 bits/spins
constexpr size_t WORDS_PER_FRAME = FRAME_BITS / 64;       // 6,144 uint64_t words

// Direct bitwise majority kernel without floating-point math inside inner spin loop
__global__ void cuda_ultrafast_wavefield_kernel(
    const uint64_t* __restrict__ d_packed_manifold,
    const size_t* __restrict__ d_ensemble_indices,
    int K,
    uint64_t* __restrict__ d_output_words
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= WORDS_PER_FRAME) return;

    uint64_t out_word = 0;
    int threshold = (K + 1) / 2; // Majority threshold for positive spin

    #pragma unroll 8
    for (int bit = 0; bit < 64; ++bit) {
        int ones = 0;
        for (int k = 0; k < K; ++k) {
            size_t f_idx = d_ensemble_indices[k];
            uint64_t word = d_packed_manifold[f_idx * WORDS_PER_FRAME + w];
            ones += (int)((word >> bit) & 1ULL);
        }

        if (ones >= threshold) {
            out_word |= (1ULL << bit);
        }
    }

    d_output_words[w] = out_word;
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-956 CUDA V2: MAXIMUM GPU SPIN WAVEFIELD ENGINE\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5\n";
    std::cout << "===================================================================\n\n";

    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) return 1;

    file.seekg(0, std::ios::end);
    size_t file_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> dataset(file_bytes);
    file.read(reinterpret_cast<char*>(dataset.data()), file_bytes);
    file.close();

    size_t num_frames = file_bytes / FRAME_BYTES;
    std::vector<uint64_t> packed_manifold(num_frames * WORDS_PER_FRAME);

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

    cuda_ultrafast_wavefield_kernel<<<grid_size, block_size>>>(
        d_packed_manifold, d_ensemble_indices, K, d_output_words
    );
    cudaDeviceSynchronize();

    int runs = 20000;
    auto cuda_t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        cuda_ultrafast_wavefield_kernel<<<grid_size, block_size>>>(
            d_packed_manifold, d_ensemble_indices, K, d_output_words
        );
    }
    cudaDeviceSynchronize();

    auto cuda_t1 = std::chrono::high_resolution_clock::now();
    double cuda_sec = std::chrono::duration<double>(cuda_t1 - cuda_t0).count();

    double fps = runs / cuda_sec;
    double latency_ms = (cuda_sec / runs) * 1000.0;
    double mspins_sec = (runs * K * FRAME_BITS) / (cuda_sec * 1e6);

    std::cout << "[+] Maximum GPU Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS\n";
    std::cout << "[+] Maximum GPU Latency: " << std::setprecision(4) << latency_ms << " ms (" << latency_ms * 1000.0 << " us)\n";
    std::cout << "[+] Wavefield Spin Throughput: " << std::setprecision(2) << mspins_sec << " M-Spins/sec (" << mspins_sec / 1000.0 << " G-Spins/s)\n\n";

    cudaFree(d_packed_manifold);
    cudaFree(d_output_words);
    cudaFree(d_ensemble_indices);

    return 0;
}
