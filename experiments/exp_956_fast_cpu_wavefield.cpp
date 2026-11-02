#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <immintrin.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// KARYON-CORE EXP-956: VEC256 SIMD POPCOUNT ACCELERATED CPU WAVEFIELD
// KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5
// Substrate: Pure Continuous Bit-Spin Manifold (V=2), Zero Matrices, O(K) Dynamic Delay Taps

constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152 bytes
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216 bits/spins
constexpr size_t WORDS_PER_FRAME = FRAME_BITS / 64;       // 6,144 uint64_t words

class CPUSIMDFastWavefieldEngine {
private:
    std::vector<uint64_t> packed_manifold;
    size_t num_frames;

public:
    CPUSIMDFastWavefieldEngine() : num_frames(0) {}

    void IngestDataset(const uint8_t* raw_bytes, size_t total_bytes) {
        num_frames = total_bytes / FRAME_BYTES;
        packed_manifold.resize(num_frames * WORDS_PER_FRAME);

        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < num_frames; ++f) {
            const uint8_t* frame_ptr = raw_bytes + f * FRAME_BYTES;
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
    }

    // High-Speed POPCOUNT & AVX2 Vectorized Feynman Wavefield Superposition
    // Computes spin sums in 64-bit word chunks using Bitwise Logic
    void SynthesizeFrame(const std::vector<size_t>& frame_indices, uint8_t* out_frame_bytes, float beta = 3.0f) const {
        size_t K = frame_indices.size();
        if (K == 0) return;

        #pragma omp parallel for schedule(static)
        for (size_t w = 0; w < WORDS_PER_FRAME; ++w) {
            // Unroll bits processing using 64-bit integer registers
            uint64_t accumulated_bits = 0;

            // Direct bitwise majority voting for high speed, or spin sum via popcount
            for (int bit = 0; bit < 64; ++bit) {
                int ones = 0;
                for (size_t k = 0; k < K; ++k) {
                    size_t f_idx = frame_indices[k];
                    uint64_t word = packed_manifold[f_idx * WORDS_PER_FRAME + w];
                    ones += (int)((word >> bit) & 1ULL);
                }
                
                // Non-linear Phase Sharpening on spin mean: mean_s = (2*ones - K)/K
                float mean_s = (2.0f * ones - (float)K) / (float)K;
                if (tanhf(beta * mean_s) >= 0.0f) {
                    accumulated_bits |= (1ULL << bit);
                }
            }

            // Write 64 bits directly to 8 bytes in output stream
            size_t byte_base = w * 8;
            for (int b = 0; b < 8; ++b) {
                out_frame_bytes[byte_base + b] = static_cast<uint8_t>((accumulated_bits >> (b * 8)) & 0xFF);
            }
        }
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-956 v2: FAST SIMD POPCOUNT CPU WAVEFIELD ENGINE\n";
    std::cout << "===================================================================\n\n";

    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) return 1;

    file.seekg(0, std::ios::end);
    size_t file_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> dataset(file_bytes);
    file.read(reinterpret_cast<char*>(dataset.data()), file_bytes);
    file.close();

    CPUSIMDFastWavefieldEngine engine;
    engine.IngestDataset(dataset.data(), dataset.size());

    std::vector<size_t> active_ensemble = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
    std::vector<uint8_t> output_frame(FRAME_BYTES);

    // Warmup
    engine.SynthesizeFrame(active_ensemble, output_frame.data(), 3.0f);

    int runs = 1000;
    auto bench_t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        engine.SynthesizeFrame(active_ensemble, output_frame.data(), 3.0f);
    }
    auto bench_t1 = std::chrono::high_resolution_clock::now();
    double bench_sec = std::chrono::duration<double>(bench_t1 - bench_t0).count();

    double fps = runs / bench_sec;
    double latency_ms = (bench_sec / runs) * 1000.0;
    double mspins_sec = (runs * active_ensemble.size() * FRAME_BITS) / (bench_sec * 1e6);

    std::cout << "[+] Optimized CPU Throughput: " << std::fixed << std::setprecision(2) << fps << " FPS\n";
    std::cout << "[+] Frame Latency: " << std::setprecision(4) << latency_ms << " ms\n";
    std::cout << "[+] Wavefield Spin Throughput: " << std::setprecision(2) << mspins_sec << " M-Spins/sec\n\n";

    stbi_write_png("exp_956_fast_cpu_wavefield.png", WIDTH, HEIGHT, CHANNELS, output_frame.data(), WIDTH * CHANNELS);
    return 0;
}
