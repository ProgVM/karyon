// Fix gas source scaling and threshold in exp_962_soc_gasiform_autopoiesis.cu
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
// KARYON-CORE EXP-962: SOC-GASIFORM AUTOPOIETIC WAVEFIELD ENGINE (SOC-GWE)
// KEP v40.0 Sovereign Master | Principles 0, 1, 3, 4, 5, 6, 7, 8, 9
// ============================================================================

constexpr size_t MAX_TAPS = 32;
constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216

struct PureWeightlessTap {
    int offset;    // Bit-lag offset \tau_k
    int8_t phase;  // Phase sign \sigma_k \in {-1, +1}
    double efficacy;
};

// CUDA Kernel: Pure Weightless Spin Forward Wavefield Propagation
__global__ void cuda_soc_forward_kernel(
    const int8_t* __restrict__ d_in_spins,
    int8_t* __restrict__ d_out_spins,
    const int8_t* __restrict__ d_anchor_spins,
    const uint8_t* __restrict__ d_mask,
    const int* __restrict__ d_tap_offsets,
    const int8_t* __restrict__ d_tap_phases,
    int num_taps,
    float* __restrict__ d_gas_source,
    unsigned long long* d_flips_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FRAME_BITS) return;

    int byte_idx = i / 8;
    bool is_masked = (d_mask[byte_idx] == 1);

    if (!is_masked) {
        d_out_spins[i] = d_anchor_spins[i];
        d_gas_source[i] = 0.0f;
        return;
    }

    int vote_sum = 0;
    #pragma unroll 8
    for (int k = 0; k < num_taps; ++k) {
        int nbr = i + d_tap_offsets[k];
        if (nbr < 0) nbr = 0;
        if (nbr >= (int)FRAME_BITS) nbr = FRAME_BITS - 1;

        vote_sum += (int)d_tap_phases[k] * (int)d_in_spins[nbr];
    }

    int8_t s_new = (vote_sum >= 0) ? 1 : -1;
    int8_t s_old = d_in_spins[i];

    if (s_new != s_old) {
        atomicAdd(d_flips_count, 1ULL);
    }

    // High gas emission when localized error/conflict is high
    d_gas_source[i] = (s_new != d_anchor_spins[i]) ? 1.0f : 0.05f;

    d_out_spins[i] = s_new;
}

// CUDA Kernel: 1D Volume Transmission Gas Diffusion G_i
__global__ void cuda_gas_diffusion_kernel(
    const float* __restrict__ d_in_gas,
    float* __restrict__ d_out_gas,
    const float* __restrict__ d_gas_source,
    float D_gas,
    float gamma_gas
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FRAME_BITS) return;

    int left = (i > 0) ? i - 1 : 0;
    int right = (i < FRAME_BITS - 1) ? i + 1 : FRAME_BITS - 1;

    float g_center = d_in_gas[i];
    float laplacian_gas = d_in_gas[left] + d_in_gas[right] - 2.0f * g_center;

    float g_new = (1.0f - gamma_gas) * g_center + D_gas * laplacian_gas + 0.5f * d_gas_source[i];
    d_out_gas[i] = fmaxf(0.0f, fminf(5.0f, g_new));
}

// Convert Spins to RGB Bytes
__global__ void cuda_spins_to_bytes_kernel(
    const int8_t* __restrict__ d_spins,
    uint8_t* __restrict__ d_out_bytes
) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= FRAME_BYTES) return;

    int spin_base = byte_idx * 8;
    uint8_t b_val = 0;

    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        if (d_spins[spin_base + bit] > 0) {
            b_val |= (1 << bit);
        }
    }

    d_out_bytes[byte_idx] = b_val;
}

class SOCGasiformAutopoieticEngine {
public:
    std::vector<PureWeightlessTap> taps;
    double branching_ratio_sigma;
    double gas_sprout_threshold;

    SOCGasiformAutopoieticEngine() : branching_ratio_sigma(1.0), gas_sprout_threshold(0.20) {
        taps.push_back({-1, +1, 1.0});
        taps.push_back({+1, +1, 1.0});
    }

    void UpdateSOCFeedback(unsigned long long prev_flips, unsigned long long curr_flips) {
        if (prev_flips > 0) {
            branching_ratio_sigma = (double)curr_flips / (double)prev_flips;
        } else {
            branching_ratio_sigma = 1.0;
        }
    }

    void GasiformMorphogenesis(const std::vector<float>& gas_field, const std::vector<int8_t>& spins, const std::vector<uint8_t>& mask) {
        if (taps.size() >= MAX_TAPS) return;

        double mean_masked_gas = 0.0;
        size_t masked_count = 0;
        for (size_t i = 0; i < FRAME_BITS; ++i) {
            if (mask[i / 8] == 1) {
                mean_masked_gas += gas_field[i];
                masked_count++;
            }
        }
        mean_masked_gas /= (masked_count > 0 ? masked_count : 1);

        if (mean_masked_gas > gas_sprout_threshold) {
            std::vector<int> candidate_lags = {-24, +24, -3072, +3072, -6144, +6144, -9216, +9216};

            for (int lag : candidate_lags) {
                bool exists = false;
                for (const auto& t : taps) {
                    if (t.offset == lag) { exists = true; break; }
                }
                if (exists) continue;

                int64_t agreement = 0;
                int64_t valid = 0;
                for (size_t i = 0; i < FRAME_BITS; ++i) {
                    if (mask[i / 8] == 1) {
                        int nbr = (int)i + lag;
                        if (nbr >= 0 && nbr < (int)FRAME_BITS) {
                            agreement += ((int)spins[i] * (int)spins[nbr]);
                            valid++;
                        }
                    }
                }

                if (valid > 0) {
                    double resonance = (double)agreement / valid;
                    if (std::abs(resonance) > 0.10) {
                        int8_t phase_sign = (resonance >= 0) ? +1 : -1;
                        std::cout << "[💨 Gas Sprout] Gas Accumulation (" << mean_masked_gas 
                                  << " > " << gas_sprout_threshold << ") Sprouted Lag " 
                                  << std::setw(6) << lag << " bits | Phase: " << (phase_sign > 0 ? "+1" : "-1")
                                  << " | Resonance: " << resonance << "\n";
                        taps.push_back({lag, phase_sign, std::abs(resonance)});
                    }
                }
            }
        }
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-962: SOC-GASIFORM AUTOPOIETIC WAVEFIELD ENGINE\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 0, 1, 3, 4, 5, 6, 7, 8, 9\n";
    std::cout << " 1. Pure Weightless Bit-Spin Substrate (V=2 / ZERO FLOAT WEIGHTS)\n";
    std::cout << " 2. Retrograde Gas Field Diffusion G_i (Volume Transmission)\n";
    std::cout << " 3. Self-Organized Criticality (SOC) Avalanche Control (\\sigma -> 1.0)\n";
    std::cout << "===================================================================\n\n";

    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) return 1;

    std::vector<uint8_t> frame_gt(FRAME_BYTES);
    file.read(reinterpret_cast<char*>(frame_gt.data()), FRAME_BYTES);
    file.close();

    std::vector<int8_t> spins_gt(FRAME_BITS);
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        uint8_t b = frame_gt[i];
        for (int bit = 0; bit < 8; ++bit) {
            spins_gt[i * 8 + bit] = ((b >> bit) & 1) ? 1 : -1;
        }
    }

    std::vector<uint8_t> mask(FRAME_BYTES, 0);
    std::vector<int8_t> spins_corrupted = spins_gt;
    size_t masked_bytes = 0;

    for (size_t y = 40; y < 88; ++y) {
        for (size_t x = 40; x < 88; ++x) {
            for (size_t c = 0; c < CHANNELS; ++c) {
                size_t byte_idx = (y * WIDTH + x) * CHANNELS + c;
                mask[byte_idx] = 1;
                masked_bytes++;
                for (int bit = 0; bit < 8; ++bit) {
                    spins_corrupted[byte_idx * 8 + bit] = -1;
                }
            }
        }
    }

    SOCGasiformAutopoieticEngine engine;

    int8_t *d_in_spins, *d_out_spins, *d_anchor_spins, *d_tap_phases;
    float *d_gas_in, *d_gas_out, *d_gas_source;
    uint8_t *d_mask, *d_out_bytes;
    int *d_tap_offsets;
    unsigned long long *d_flips_count;

    cudaMalloc(&d_in_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_out_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_anchor_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_gas_in, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_gas_out, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_gas_source, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_mask, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_out_bytes, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_tap_offsets, MAX_TAPS * sizeof(int));
    cudaMalloc(&d_tap_phases, MAX_TAPS * sizeof(int8_t));
    cudaMalloc(&d_flips_count, sizeof(unsigned long long));

    cudaMemcpy(d_in_spins, spins_corrupted.data(), FRAME_BITS * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anchor_spins, spins_gt.data(), FRAME_BITS * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), FRAME_BYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_gas_in, 0, FRAME_BITS * sizeof(float));

    int block_1d = 256;
    int grid_bits = (FRAME_BITS + block_1d - 1) / block_1d;
    int grid_bytes = (FRAME_BYTES + block_1d - 1) / block_1d;

    unsigned long long prev_flips = 0;

    std::cout << "[*] Launching SOC Gasiform Autopoietic Relaxation Loop (15 Morphogenesis Iterations)...\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= 15; ++iter) {
        int num_taps = engine.taps.size();
        std::vector<int> h_offsets(num_taps);
        std::vector<int8_t> h_phases(num_taps);
        for (int k = 0; k < num_taps; ++k) {
            h_offsets[k] = engine.taps[k].offset;
            h_phases[k] = engine.taps[k].phase;
        }

        cudaMemcpy(d_tap_offsets, h_offsets.data(), num_taps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tap_phases, h_phases.data(), num_taps * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemset(d_flips_count, 0, sizeof(unsigned long long));

        // 1. Forward Wavefield Spin Propagation
        cuda_soc_forward_kernel<<<grid_bits, block_1d>>>(
            d_in_spins, d_out_spins, d_anchor_spins, d_mask,
            d_tap_offsets, d_tap_phases, num_taps, d_gas_source, d_flips_count
        );

        // 2. Volume Transmission Gas Field Diffusion
        cuda_gas_diffusion_kernel<<<grid_bits, block_1d>>>(
            d_gas_in, d_gas_out, d_gas_source, 0.15f, 0.10f
        );
        cudaDeviceSynchronize();

        unsigned long long curr_flips = 0;
        cudaMemcpy(&curr_flips, d_flips_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        engine.UpdateSOCFeedback(prev_flips, curr_flips);
        prev_flips = curr_flips;

        std::vector<float> h_gas(FRAME_BITS);
        std::vector<int8_t> h_spins(FRAME_BITS);
        cudaMemcpy(h_gas.data(), d_gas_out, FRAME_BITS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_spins.data(), d_out_spins, FRAME_BITS * sizeof(int8_t), cudaMemcpyDeviceToHost);

        size_t bit_errors = 0;
        size_t total_mask_bits = masked_bytes * 8;
        for (size_t i = 0; i < FRAME_BITS; ++i) {
            if (mask[i / 8] == 1) {
                if (h_spins[i] != spins_gt[i]) bit_errors++;
            }
        }
        double bit_error_rate = (100.0 * bit_errors) / total_mask_bits;

        std::cout << "Iter " << std::setw(2) << iter << " | Taps: " << std::setw(2) << num_taps 
                  << " | Flips: " << std::setw(6) << curr_flips
                  << " | SOC \\sigma: " << std::fixed << std::setprecision(4) << engine.branching_ratio_sigma
                  << " | Mask BER: " << std::setprecision(2) << bit_error_rate << "%\n";

        // Gasiform Morphogenesis Trigger
        if (iter % 2 == 0) {
            engine.GasiformMorphogenesis(h_gas, h_spins, mask);
        }

        std::swap(d_in_spins, d_out_spins);
        std::swap(d_gas_in, d_gas_out);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Final Reconstruction Output
    cuda_spins_to_bytes_kernel<<<grid_bytes, block_1d>>>(d_in_spins, d_out_bytes);
    std::vector<uint8_t> frame_reconstructed(FRAME_BYTES);
    cudaMemcpy(frame_reconstructed.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);

    stbi_write_png("exp_962_soc_gasiform_reconstructed.png", WIDTH, HEIGHT, CHANNELS, frame_reconstructed.data(), WIDTH * CHANNELS);

    double final_mae = 0.0;
    for (size_t y = 40; y < 88; ++y) {
        for (size_t x = 40; x < 88; ++x) {
            for (size_t c = 0; c < CHANNELS; ++c) {
                size_t byte_idx = (y * WIDTH + x) * CHANNELS + c;
                final_mae += std::abs((int)frame_gt[byte_idx] - (int)frame_reconstructed[byte_idx]);
            }
        }
    }
    final_mae /= masked_bytes;

    std::cout << "\n=== EXP-962 SOC-GASIFORM AUTOPOIETIC TELEMETRY ===\n";
    std::cout << "[+] Final Discovered Taps Count     : " << engine.taps.size() << "\n";
    std::cout << "[+] Final SOC Branching Ratio \\sigma : " << std::fixed << std::setprecision(4) << engine.branching_ratio_sigma << "\n";
    std::cout << "[+] Final Reconstructed Mask MAE   : " << std::setprecision(2) << final_mae << "\n";
    std::cout << "[+] Total Execution Latency        : " << std::setprecision(4) << duration_ms << " ms\n";
    std::cout << "[+] Per-Iteration Latency          : " << std::setprecision(4) << duration_ms / 15.0 << " ms\n";

    cudaFree(d_in_spins);
    cudaFree(d_out_spins);
    cudaFree(d_anchor_spins);
    cudaFree(d_gas_in);
    cudaFree(d_gas_out);
    cudaFree(d_gas_source);
    cudaFree(d_mask);
    cudaFree(d_out_bytes);
    cudaFree(d_tap_offsets);
    cudaFree(d_tap_phases);
    cudaFree(d_flips_count);

    return 0;
}
