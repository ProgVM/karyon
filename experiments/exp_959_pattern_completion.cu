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
// KARYON-CORE EXP-959: ZERO-SHOT AUTOPOIETIC PATTERN COMPLETION
// KEP v40.0 Sovereign Master | Principles 1, 3, 4, 5, 7, 8
// Substrate: Pure 1D Unrolled Continuous Bit-Spin Manifold (V=2)
// Mechanism: Emergent Topological Soliton Relaxation (F -> 0)
// Target: Zero-Shot Recovery of Corrupted/Masked Regions Without Weight Matrices
// ============================================================================

constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152 bytes
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216 bits/spins

// CUDA Kernel: Autopoietic Pattern Completion Relaxation Kernel
// Unmasked positions stay anchored to observed sensory evidence.
// Corrupted/Masked positions evolve dynamically according to emergent topological field relaxation.
__global__ void cuda_autopoietic_pattern_completion_kernel(
    const float* __restrict__ d_in_spins,
    float* __restrict__ d_out_spins,
    const float* __restrict__ d_anchor_spins,
    const uint8_t* __restrict__ d_mask, // 1 = corrupted/masked (free to relax), 0 = unmasked (anchored)
    const int* __restrict__ d_topological_taps,
    int num_taps,
    float D_diff,
    float beta,
    float gamma,
    float dt,
    float anchor_strength
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FRAME_BITS) return;

    int byte_idx = i / 8;
    bool is_masked = (d_mask[byte_idx] == 1);

    float s_center = d_in_spins[i];
    float laplacian_topo = 0.0f;

    // Topological Laplacian from discovered spatial lag taps (-8, +8, -3072, +3072 bits)
    for (int t = 0; t < num_taps; ++t) {
        int offset = d_topological_taps[t];
        int nbr_idx = i + offset;

        // Boundary reflection handling
        if (nbr_idx < 0) nbr_idx = 0;
        if (nbr_idx >= (int)FRAME_BITS) nbr_idx = FRAME_BITS - 1;

        laplacian_topo += (d_in_spins[nbr_idx] - s_center);
    }

    // Reaction-Diffusion Soliton dynamics
    float ds_dt = D_diff * laplacian_topo + tanhf(beta * s_center) - gamma * s_center;

    if (!is_masked) {
        // Soft anchoring pull toward original observed evidence
        float s_anchor = d_anchor_spins[i];
        ds_dt += anchor_strength * (s_anchor - s_center);
    }

    float s_new = s_center + dt * ds_dt;
    s_new = fmaxf(-1.0f, fminf(1.0f, s_new));

    d_out_spins[i] = s_new;
}

// CUDA Kernel: Spin array to RGB bytes
__global__ void cuda_spins_to_bytes_kernel(
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
    std::cout << " KARYON-CORE EXP-959: ZERO-SHOT AUTOPOIETIC PATTERN COMPLETION\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 3, 4, 5, 7, 8\n";
    std::cout << " Substrate: 1D Continuous Bit-Spin Wavefield (V=2)\n";
    std::cout << " Goal: Zero-Shot Repair of 20% Corrupted Image Region via Solitons\n";
    std::cout << "===================================================================\n\n";

    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Error: Failed to open dataset binary!\n";
        return 1;
    }

    std::vector<uint8_t> frame_gt(FRAME_BYTES);
    file.read(reinterpret_cast<char*>(frame_gt.data()), FRAME_BYTES);
    file.close();

    // 1. Create Ground Truth Spin Array
    std::vector<float> spins_gt(FRAME_BITS);
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        uint8_t b = frame_gt[i];
        for (int bit = 0; bit < 8; ++bit) {
            spins_gt[i * 8 + bit] = ((b >> bit) & 1) ? 1.0f : -1.0f;
        }
    }

    // 2. Create Mask and Corrupted Frame
    // Mask: 20% central rectangle mask (Y: 40-88, X: 40-88 across all 3 channels)
    std::vector<uint8_t> mask(FRAME_BYTES, 0);
    std::vector<uint8_t> frame_corrupted = frame_gt;
    std::vector<float> spins_corrupted = spins_gt;

    size_t masked_bytes_count = 0;
    for (size_t y = 40; y < 88; ++y) {
        for (size_t x = 40; x < 88; ++x) {
            for (size_t c = 0; c < CHANNELS; ++c) {
                size_t byte_idx = (y * WIDTH + x) * CHANNELS + c;
                mask[byte_idx] = 1;
                masked_bytes_count++;

                // Corrupt: inject zero-spin / random phase noise in masked box
                frame_corrupted[byte_idx] = 128; // neutral gray
                for (int bit = 0; bit < 8; ++bit) {
                    spins_corrupted[byte_idx * 8 + bit] = 0.0f; // neutral zero spin
                }
            }
        }
    }

    double mask_percentage = (100.0 * masked_bytes_count) / FRAME_BYTES;
    std::cout << "[+] Applied " << std::fixed << std::setprecision(2) << mask_percentage 
              << "% Central Mask (" << masked_bytes_count << " bytes / " 
              << masked_bytes_count * 8 << " bits masked).\n";

    // Measure Initial Error inside Masked Region before relaxation
    double initial_mae = 0.0;
    size_t initial_bit_flips = 0;
    size_t total_masked_bits = masked_bytes_count * 8;

    for (size_t y = 40; y < 88; ++y) {
        for (size_t x = 40; x < 88; ++x) {
            for (size_t c = 0; c < CHANNELS; ++c) {
                size_t byte_idx = (y * WIDTH + x) * CHANNELS + c;
                initial_mae += std::abs((int)frame_gt[byte_idx] - (int)frame_corrupted[byte_idx]);
                for (int bit = 0; bit < 8; ++bit) {
                    float s_gt = spins_gt[byte_idx * 8 + bit];
                    float s_corr = spins_corrupted[byte_idx * 8 + bit];
                    if ((s_gt >= 0.0f) != (s_corr >= 0.0f)) {
                        initial_bit_flips++;
                    }
                }
            }
        }
    }
    initial_mae /= masked_bytes_count;
    double initial_bit_error_rate = (100.0 * initial_bit_flips) / total_masked_bits;

    std::cout << "[+] Initial Masked Region MAE           : " << std::fixed << std::setprecision(2) << initial_mae << " / 255\n";
    std::cout << "[+] Initial Masked Bit Error Rate       : " << std::fixed << std::setprecision(2) << initial_bit_error_rate << "%\n";

    // 3. Discovered Topological Taps (from EXP-958)
    // Offset -8, +8 bits (adjacent byte neighbor)
    // Offset -3072, +3072 bits (128 pixels * 3 channels * 8 bits/byte = 3072 bits vertical neighbor)
    std::vector<int> topological_taps = {-8, +8, -3072, +3072};

    // Allocate Device Memory
    float *d_in_spins = nullptr, *d_out_spins = nullptr, *d_anchor_spins = nullptr;
    uint8_t *d_mask = nullptr, *d_out_bytes = nullptr;
    int *d_topological_taps = nullptr;

    cudaMalloc(&d_in_spins, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_out_spins, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_anchor_spins, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_mask, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_out_bytes, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_topological_taps, topological_taps.size() * sizeof(int));

    cudaMemcpy(d_in_spins, spins_corrupted.data(), FRAME_BITS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anchor_spins, spins_corrupted.data(), FRAME_BITS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), FRAME_BYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_topological_taps, topological_taps.data(), topological_taps.size() * sizeof(int), cudaMemcpyHostToDevice);

    int block_1d = 256;
    int grid_bits = (FRAME_BITS + block_1d - 1) / block_1d;
    int grid_bytes = (FRAME_BYTES + block_1d - 1) / block_1d;

    // 4. Execute Multi-Step Autopoietic Soliton Pattern Completion
    float D_diff = 0.20f;
    float beta = 3.5f;
    float gamma = 0.7f;
    float dt = 0.04f;
    float anchor_strength = 2.0f;
    int total_relaxation_steps = 30;

    std::cout << "\n[*] Executing " << total_relaxation_steps << " Autopoietic Soliton Relaxation Steps...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < total_relaxation_steps; ++step) {
        cuda_autopoietic_pattern_completion_kernel<<<grid_bits, block_1d>>>(
            d_in_spins, d_out_spins, d_anchor_spins, d_mask, d_topological_taps,
            (int)topological_taps.size(), D_diff, beta, gamma, dt, anchor_strength
        );
        std::swap(d_in_spins, d_out_spins);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5. Transfer Reconstructed Image Back to Host
    cuda_spins_to_bytes_kernel<<<grid_bytes, block_1d>>>(d_in_spins, d_out_bytes);
    std::vector<uint8_t> frame_reconstructed(FRAME_BYTES);
    cudaMemcpy(frame_reconstructed.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);

    // Save PNG Images for Visual Auditing
    stbi_write_png("exp_959_ground_truth.png", WIDTH, HEIGHT, CHANNELS, frame_gt.data(), WIDTH * CHANNELS);
    stbi_write_png("exp_959_corrupted.png", WIDTH, HEIGHT, CHANNELS, frame_corrupted.data(), WIDTH * CHANNELS);
    stbi_write_png("exp_959_reconstructed.png", WIDTH, HEIGHT, CHANNELS, frame_reconstructed.data(), WIDTH * CHANNELS);

    // 6. Quantitative Evaluation inside Masked Region
    double post_mae = 0.0;
    size_t post_bit_flips = 0;
    for (size_t y = 40; y < 88; ++y) {
        for (size_t x = 40; x < 88; ++x) {
            for (size_t c = 0; c < CHANNELS; ++c) {
                size_t byte_idx = (y * WIDTH + x) * CHANNELS + c;
                post_mae += std::abs((int)frame_gt[byte_idx] - (int)frame_reconstructed[byte_idx]);

                uint8_t gt_b = frame_gt[byte_idx];
                uint8_t rec_b = frame_reconstructed[byte_idx];
                for (int bit = 0; bit < 8; ++bit) {
                    bool gt_bit = (gt_b >> bit) & 1;
                    bool rec_bit = (rec_b >> bit) & 1;
                    if (gt_bit != rec_bit) {
                        post_bit_flips++;
                    }
                }
            }
        }
    }
    post_mae /= masked_bytes_count;
    double post_bit_error_rate = (100.0 * post_bit_flips) / total_masked_bits;
    double mae_improvement_percent = (100.0 * (initial_mae - post_mae)) / initial_mae;
    double bit_recovery_percent = 100.0 - post_bit_error_rate;

    std::cout << "\n=== EXP-959 EMPIRICAL TELEMETRY (ZERO-SHOT PATTERN COMPLETION) ===\n";
    std::cout << "[+] Initial Corrupted MAE (Masked Box) : " << std::fixed << std::setprecision(2) << initial_mae << "\n";
    std::cout << "[+] Post-Relaxation MAE (Masked Box)  : " << std::setprecision(2) << post_mae << "\n";
    std::cout << "[+] MAE Error Reduction               : " << std::setprecision(2) << mae_improvement_percent << "%\n";
    std::cout << "[+] Post-Relaxation Bit Error Rate    : " << std::setprecision(2) << post_bit_error_rate << "%\n";
    std::cout << "[+] Zero-Shot Bit Spin Recovery Rate  : " << std::setprecision(2) << bit_recovery_percent << "%\n";
    std::cout << "[+] Total Relaxation Latency (30 steps): " << std::setprecision(4) << latency_ms << " ms\n";
    std::cout << "[+] Per-Step Latency                  : " << std::setprecision(4) << latency_ms / total_relaxation_steps << " ms\n";

    cudaFree(d_in_spins);
    cudaFree(d_out_spins);
    cudaFree(d_anchor_spins);
    cudaFree(d_mask);
    cudaFree(d_out_bytes);
    cudaFree(d_topological_taps);

    return 0;
}
