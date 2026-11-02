#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ============================================================================
// KARYON-CORE EXP-958: SOVEREIGN GENESIS OF SPACE (TOPOLOGICAL PHASE DISCOVERY)
// KEP v40.0 Sovereign Master | Principles 1, 3, 4, 5, 8
// Substrate: Pure 1D Unrolled Bit-Spin Manifold (V=2), Zero Imposed 2D Coordinates!
// Target: Autonomous Discovery of Spatial Stride & Dynamic Topological Laplacian
// ============================================================================

constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152 bytes
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216 bits/spins
constexpr size_t WORDS_PER_FRAME = FRAME_BITS / 64;       // 6,144 uint64_t words

// CUDA Kernel 1: Calculate Continuous Phase Means across Ensemble
__global__ void cuda_compute_phase_means_kernel(
    const uint64_t* __restrict__ d_packed_manifold,
    const size_t* __restrict__ d_ensemble_indices,
    int K,
    float* __restrict__ d_continuous_spins
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= WORDS_PER_FRAME) return;

    #pragma unroll 8
    for (int bit = 0; bit < 64; ++bit) {
        int ones = 0;
        for (int k = 0; k < K; ++k) {
            size_t f_idx = d_ensemble_indices[k];
            uint64_t word = d_packed_manifold[f_idx * WORDS_PER_FRAME + w];
            ones += (int)((word >> bit) & 1ULL);
        }
        
        float mean_s = (2.0f * ones - (float)K) / (float)K;
        int spin_idx = w * 64 + bit;
        d_continuous_spins[spin_idx] = mean_s;
    }
}

// CUDA Kernel 2: Autonomous Topological Phase Diffusion Kernel
// Uses dynamic topological neighbor offsets (discovered purely from phase correlation!)
// NO 2D (x, y) coordinates passed to kernel — operates purely on 1D unrolled index!
__global__ void cuda_emergent_topological_diffusion_kernel(
    const float* __restrict__ d_in_spins,
    float* __restrict__ d_out_spins,
    const int* __restrict__ d_topological_taps, // Discovered topological offset taps!
    int num_taps,
    float D_diff,
    float beta,
    float gamma,
    float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FRAME_BITS) return;

    float s_center = d_in_spins[i];
    float laplacian_topo = 0.0f;

    // Accumulate topological Laplacian from dynamically discovered phase neighbors
    for (int t = 0; t < num_taps; ++t) {
        int offset = d_topological_taps[t];
        int nbr_idx = i + offset;

        // Physical boundary handling on unrolled manifold
        if (nbr_idx < 0) nbr_idx = 0;
        if (nbr_idx >= (int)FRAME_BITS) nbr_idx = FRAME_BITS - 1;

        laplacian_topo += (d_in_spins[nbr_idx] - s_center);
    }

    // Reaction-Diffusion PDE over emergent topological manifold
    float ds_dt = D_diff * laplacian_topo + tanhf(beta * s_center) - gamma * s_center;
    float s_new = s_center + dt * ds_dt;

    s_new = fmaxf(-1.0f, fminf(1.0f, s_new));
    d_out_spins[i] = s_new;
}

// CUDA Kernel 3: Spins to Output Bytes
__global__ void cuda_spins_to_bytes_1d_kernel(
    const float* __restrict__ d_spins,
    uint8_t* __restrict__ d_out_bytes
) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= FRAME_BYTES) return;

    int spin_base = byte_idx * 8;
    uint8_t b_val = 0;

    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        if (d_spins[spin_base + bit] >= 0.0f) {
            b_val |= (1 << bit);
        }
    }

    d_out_bytes[byte_idx] = b_val;
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-958: SOVEREIGN GENESIS OF SPACE (TOPOLOGICAL MAP)\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 3, 4, 5, 8\n";
    std::cout << " Substrate: 1D Unrolled Continuous Bit-Spin Manifold (V=2)\n";
    std::cout << " Goal: Zero Hardcoded 2D Coordinates — Autonomous Spatial Genesis\n";
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

    // Step 1: Autonomous Spatial Topology Discovery
    // The system analyzes autocorrelation of unrolled 1D bit stream across lag distances
    std::cout << "[*] Phase 1: Autonomous Topology Scanner analyzing 1D unrolled stream...\n";
    
    size_t max_lag = 4000;
    std::vector<double> lag_autocorrelation(max_lag, 0.0);

    // Compute lag covariance on CPU over dataset
    for (size_t lag = 1; lag < max_lag; ++lag) {
        double acc = 0.0;
        size_t count = 0;
        for (size_t f = 0; f < std::min<size_t>(num_frames, 10); ++f) {
            const uint8_t* f_ptr = dataset.data() + f * FRAME_BYTES;
            for (size_t i = 0; i < FRAME_BYTES - lag; i += 7) {
                int s1 = (f_ptr[i] > 127) ? 1 : -1;
                int s2 = (f_ptr[i + lag] > 127) ? 1 : -1;
                acc += (s1 * s2);
                count++;
            }
        }
        lag_autocorrelation[lag] = acc / count;
    }

    // Find top structural lag peaks (autonomously discovering stride lengths)
    std::vector<std::pair<double, int>> lag_peaks;
    for (int lag = 1; lag < (int)max_lag; ++lag) {
        if (lag > 2 && lag < (int)max_lag - 1) {
            if (lag_autocorrelation[lag] > lag_autocorrelation[lag-1] &&
                lag_autocorrelation[lag] > lag_autocorrelation[lag+1]) {
                lag_peaks.push_back({lag_autocorrelation[lag], lag});
            }
        }
    }
    std::sort(lag_peaks.rbegin(), lag_peaks.rend());

    std::cout << "[+] Top Autonomously Discovered Topological Lags (Byte Offsets):\n";
    for (int k = 0; k < std::min<int>(5, lag_peaks.size()); ++k) {
        std::cout << "    Rank " << (k+1) << ": Offset = " << lag_peaks[k].second 
                  << " bytes (" << lag_peaks[k].second * 8 << " bits), Correlation = " 
                  << std::fixed << std::setprecision(4) << lag_peaks[k].first << "\n";
    }

    // Convert discovered byte offsets to bit offsets for dynamic CUDA taps
    // The top peak discovered will naturally be 384 bytes (which is 128 pixels * 3 channels!)
    std::vector<int> discovered_taps_bits;
    // Local 1D neighbors (adjacent bytes)
    discovered_taps_bits.push_back(-8);
    discovered_taps_bits.push_back(+8);
    // Discovered long-range 2D stride neighbors (autonomously discovered!)
    for (int k = 0; k < 2 && k < (int)lag_peaks.size(); ++k) {
        int byte_off = lag_peaks[k].second;
        discovered_taps_bits.push_back(-byte_off * 8);
        discovered_taps_bits.push_back(+byte_off * 8);
    }

    std::cout << "[+] Dynamic Topological Taps Allocated: " << discovered_taps_bits.size() << " dynamic connections.\n\n";

    // Allocate Device GPU Memory
    uint64_t* d_packed_manifold = nullptr;
    size_t* d_ensemble_indices = nullptr;
    float* d_continuous_spins_A = nullptr;
    float* d_continuous_spins_B = nullptr;
    int* d_topological_taps = nullptr;
    uint8_t* d_out_bytes = nullptr;

    cudaMalloc(&d_packed_manifold, packed_manifold.size() * sizeof(uint64_t));
    cudaMemcpy(d_packed_manifold, packed_manifold.data(), packed_manifold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<size_t> active_ensemble = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
    int K = active_ensemble.size();

    cudaMalloc(&d_ensemble_indices, K * sizeof(size_t));
    cudaMemcpy(d_ensemble_indices, active_ensemble.data(), K * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_topological_taps, discovered_taps_bits.size() * sizeof(int));
    cudaMemcpy(d_topological_taps, discovered_taps_bits.data(), discovered_taps_bits.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_continuous_spins_A, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_continuous_spins_B, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_out_bytes, FRAME_BYTES * sizeof(uint8_t));

    int block_1d = 256;
    int grid_words = (WORDS_PER_FRAME + block_1d - 1) / block_1d;
    int grid_bits  = (FRAME_BITS + block_1d - 1) / block_1d;
    int grid_bytes = (FRAME_BYTES + block_1d - 1) / block_1d;

    // Step 2: Compute Baseline (No Topological Diffusion)
    cuda_compute_phase_means_kernel<<<grid_words, block_1d>>>(
        d_packed_manifold, d_ensemble_indices, K, d_continuous_spins_A
    );
    cuda_spins_to_bytes_1d_kernel<<<grid_bytes, block_1d>>>(d_continuous_spins_A, d_out_bytes);
    std::vector<uint8_t> baseline_bytes(FRAME_BYTES);
    cudaMemcpy(baseline_bytes.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);

    // Step 3: Run Emergent Topological Diffusion Loop over Unrolled Stream
    float D_diff = 0.15f;
    float beta = 3.0f;
    float gamma = 0.8f;
    float dt = 0.05f;
    int steps = 10;

    float* src_spins = d_continuous_spins_A;
    float* dst_spins = d_continuous_spins_B;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) {
        cuda_emergent_topological_diffusion_kernel<<<grid_bits, block_1d>>>(
            src_spins, dst_spins, d_topological_taps, (int)discovered_taps_bits.size(),
            D_diff, beta, gamma, dt
        );
        std::swap(src_spins, dst_spins);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cuda_spins_to_bytes_1d_kernel<<<grid_bytes, block_1d>>>(src_spins, d_out_bytes);
    std::vector<uint8_t> emergent_bytes(FRAME_BYTES);
    cudaMemcpy(emergent_bytes.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);

    stbi_write_png("exp_958_emergent_topological_solitons.png", WIDTH, HEIGHT, CHANNELS, emergent_bytes.data(), WIDTH * CHANNELS);
    std::cout << "[+] Output saved to exp_958_emergent_topological_solitons.png\n";

    // Quantitative Telemetry
    double mean_base = 0.0, sq_base = 0.0;
    double mean_em   = 0.0, sq_em   = 0.0;
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        mean_base += baseline_bytes[i];
        sq_base += baseline_bytes[i] * baseline_bytes[i];
        mean_em += emergent_bytes[i];
        sq_em += emergent_bytes[i] * emergent_bytes[i];
    }
    mean_base /= FRAME_BYTES;
    mean_em   /= FRAME_BYTES;
    double std_base = std::sqrt(sq_base / FRAME_BYTES - mean_base * mean_base);
    double std_em   = std::sqrt(sq_em / FRAME_BYTES - mean_em * mean_em);

    std::cout << "\n=== EXP-958 EMPIRICAL TELEMETRY (SOVEREIGN GENESIS OF SPACE) ===\n";
    std::cout << "[+] Baseline Contrast (Std Dev)      : " << std::fixed << std::setprecision(2) << std_base << "\n";
    std::cout << "[+] Emergent Topological Contrast    : " << std::setprecision(2) << std_em << "\n";
    std::cout << "[+] Autonomous Topology Contrast Delta: " << std::showpos << (std_em - std_base) << std::noshowpos << "\n";
    std::cout << "[+] PDE Latency (Unrolled 1D Grid)    : " << std::setprecision(4) << latency_ms << " ms\n";

    cudaFree(d_packed_manifold);
    cudaFree(d_ensemble_indices);
    cudaFree(d_topological_taps);
    cudaFree(d_continuous_spins_A);
    cudaFree(d_continuous_spins_B);
    cudaFree(d_out_bytes);

    return 0;
}
