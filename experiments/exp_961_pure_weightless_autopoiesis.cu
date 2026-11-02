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
// KARYON-CORE EXP-961: PURE WEIGHTLESS AUTOPOIETIC GENESIS ENGINE
// KEP v40.0 Sovereign Master | Principles 0, 1, 3, 4, 5, 6, 8, 9
// STRICT REJECTION OF FLOATING-POINT WEIGHTS & GRADIENT DESCENT!
// Substrate: Pure Weightless Topological Delay Taps O(K) with Binary Phase (+1/-1)
// Mechanism: Bitwise POPCOUNT Spin Resonance & Structural Morphogenesis
// ============================================================================

constexpr size_t MAX_TAPS = 32;
constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216

// Pure Weightless Topological Tap: ZERO float weights!
struct PureWeightlessTap {
    int offset;    // Bit-lag offset \tau_k
    int8_t phase;  // Phase sign \sigma_k \in {-1, +1}
    double efficacy; // Pure unweighted resonance score
};

// CUDA Kernel: Pure Weightless Wavefield Synthesis via Spin Resonance Voting (ZERO WEIGHTS)
__global__ void cuda_pure_weightless_forward_kernel(
    const int8_t* __restrict__ d_in_spins,
    int8_t* __restrict__ d_out_spins,
    const int8_t* __restrict__ d_anchor_spins,
    const uint8_t* __restrict__ d_mask,
    const int* __restrict__ d_tap_offsets,
    const int8_t* __restrict__ d_tap_phases,
    int num_taps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FRAME_BITS) return;

    int byte_idx = i / 8;
    bool is_masked = (d_mask[byte_idx] == 1);

    if (!is_masked) {
        // Unmasked position is anchored to observed sensory evidence
        d_out_spins[i] = d_anchor_spins[i];
        return;
    }

    // Pure Weightless Spin Resonance Voting (Sum of signed binary spins)
    int vote_sum = 0;
    #pragma unroll 8
    for (int k = 0; k < num_taps; ++k) {
        int nbr = i + d_tap_offsets[k];
        if (nbr < 0) nbr = 0;
        if (nbr >= (int)FRAME_BITS) nbr = FRAME_BITS - 1;

        vote_sum += (int)d_tap_phases[k] * (int)d_in_spins[nbr];
    }

    // Binary Majority Voting / Soliton Step
    d_out_spins[i] = (vote_sum >= 0) ? 1 : -1;
}

// Convert Bit-Spins (int8_t) to RGB Bytes
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

class PureWeightlessAutopoieticEngine {
public:
    std::vector<PureWeightlessTap> taps;

    PureWeightlessAutopoieticEngine() {
        // Start with ZERO weight matrices and ZERO hardcoded 2D spatial assumptions.
        // Seed with only 2 basic 1D adjacent bit-lags
        taps.push_back({-1, +1, 1.0});
        taps.push_back({+1, +1, 1.0});
    }

    // Autopoietic Morphogenesis: Pure Structural Discovery of Lag Taps from Spin Covariance
    void AutopoieticMorphogenesis(const std::vector<int8_t>& current_spins, const std::vector<uint8_t>& mask) {
        if (taps.size() >= MAX_TAPS) return;

        // Candidate 1D unrolled bit-lags
        std::vector<int> candidate_lags = {-24, +24, -3072, +3072, -6144, +6144, -9216, +9216};

        for (int lag : candidate_lags) {
            // Check if lag already exists in topology
            bool exists = false;
            for (const auto& t : taps) {
                if (t.offset == lag) { exists = true; break; }
            }
            if (exists) continue;

            int64_t agreement_sum = 0;
            int64_t valid_pairs = 0;

            for (size_t i = 0; i < FRAME_BITS; ++i) {
                if (mask[i / 8] == 0) continue; // Measure resonance against unmasked context boundary

                int nbr = (int)i + lag;
                if (nbr >= 0 && nbr < (int)FRAME_BITS) {
                    agreement_sum += ((int)current_spins[i] * (int)current_spins[nbr]);
                    valid_pairs++;
                }
            }

            if (valid_pairs > 0) {
                double resonance = (double)agreement_sum / valid_pairs;
                if (std::abs(resonance) > 0.15) {
                    int8_t phase_sign = (resonance >= 0) ? +1 : -1;
                    std::cout << "[⚡ Weightless Sprout] Sprouted Topology Lag " << std::setw(6) << lag 
                              << " bits | Phase: " << (phase_sign > 0 ? "+1" : "-1")
                              << " | Resonance: " << std::fixed << std::setprecision(4) << resonance << "\n";

                    taps.push_back({lag, phase_sign, std::abs(resonance)});
                }
            }
        }
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-961: PURE WEIGHTLESS AUTOPOIETIC GENESIS ENGINE\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 0, 1, 3, 4, 5, 6, 8, 9\n";
    std::cout << " ZERO FLOATING-POINT WEIGHTS! ZERO GRADIENT DESCENT!\n";
    std::cout << " Substrate: Pure Weightless Topological Taps & Bitwise Spin Resonance\n";
    std::cout << "===================================================================\n\n";

    // Load Dataset
    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) return 1;

    std::vector<uint8_t> frame_gt(FRAME_BYTES);
    file.read(reinterpret_cast<char*>(frame_gt.data()), FRAME_BYTES);
    file.close();

    // Target Ground Truth Spins (int8_t {-1, +1})
    std::vector<int8_t> spins_gt(FRAME_BITS);
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        uint8_t b = frame_gt[i];
        for (int bit = 0; bit < 8; ++bit) {
            spins_gt[i * 8 + bit] = ((b >> bit) & 1) ? 1 : -1;
        }
    }

    // 20% Central Corrupted Mask
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
                    spins_corrupted[byte_idx * 8 + bit] = -1; // Default spin in mask
                }
            }
        }
    }

    std::cout << "[+] Prepared Pure Bit-Spin Stream with " << (100.0 * masked_bytes) / FRAME_BYTES 
              << "% Corrupted Mask.\n";

    PureWeightlessAutopoieticEngine engine;

    // CUDA Allocations
    int8_t *d_in_spins, *d_out_spins, *d_anchor_spins, *d_tap_phases;
    uint8_t *d_mask, *d_out_bytes;
    int *d_tap_offsets;

    cudaMalloc(&d_in_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_out_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_anchor_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_mask, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_out_bytes, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_tap_offsets, MAX_TAPS * sizeof(int));
    cudaMalloc(&d_tap_phases, MAX_TAPS * sizeof(int8_t));

    cudaMemcpy(d_in_spins, spins_corrupted.data(), FRAME_BITS * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anchor_spins, spins_gt.data(), FRAME_BITS * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), FRAME_BYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int block_1d = 256;
    int grid_bits = (FRAME_BITS + block_1d - 1) / block_1d;
    int grid_bytes = (FRAME_BYTES + block_1d - 1) / block_1d;

    std::cout << "\n[*] Launching Pure Weightless Autopoietic Relaxation Loop (10 Morphogenesis Iterations)...\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= 10; ++iter) {
        int num_taps = engine.taps.size();
        std::vector<int> h_offsets(num_taps);
        std::vector<int8_t> h_phases(num_taps);
        for (int k = 0; k < num_taps; ++k) {
            h_offsets[k] = engine.taps[k].offset;
            h_phases[k] = engine.taps[k].phase;
        }

        cudaMemcpy(d_tap_offsets, h_offsets.data(), num_taps * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tap_phases, h_phases.data(), num_taps * sizeof(int8_t), cudaMemcpyHostToDevice);

        // Pure Weightless Spin Resonance Forward Kernel
        cuda_pure_weightless_forward_kernel<<<grid_bits, block_1d>>>(
            d_in_spins, d_out_spins, d_anchor_spins, d_mask,
            d_tap_offsets, d_tap_phases, num_taps
        );
        cudaDeviceSynchronize();

        // Copy current spins back to host for structural morphogenesis audit
        std::vector<int8_t> h_spins(FRAME_BITS);
        cudaMemcpy(h_spins.data(), d_out_spins, FRAME_BITS * sizeof(int8_t), cudaMemcpyDeviceToHost);

        // Measure Bit Error Rate inside Mask
        size_t bit_errors = 0;
        size_t total_mask_bits = masked_bytes * 8;
        for (size_t i = 0; i < FRAME_BITS; ++i) {
            if (mask[i / 8] == 1) {
                if (h_spins[i] != spins_gt[i]) bit_errors++;
            }
        }
        double bit_error_rate = (100.0 * bit_errors) / total_mask_bits;

        std::cout << "Iteration " << std::setw(2) << iter << " | Active Topology Taps: " << std::setw(2) << num_taps 
                  << " | Mask Bit Error Rate: " << std::fixed << std::setprecision(2) << bit_error_rate << "%\n";

        // Autopoietic Morphogenesis: Discovery of new topological taps purely from resonance
        if (iter % 2 == 0) {
            engine.AutopoieticMorphogenesis(h_spins, mask);
        }

        std::swap(d_in_spins, d_out_spins);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Final Reconstruction Output
    cuda_spins_to_bytes_kernel<<<grid_bytes, block_1d>>>(d_in_spins, d_out_bytes);
    std::vector<uint8_t> frame_reconstructed(FRAME_BYTES);
    cudaMemcpy(frame_reconstructed.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);

    stbi_write_png("exp_961_pure_weightless_reconstructed.png", WIDTH, HEIGHT, CHANNELS, frame_reconstructed.data(), WIDTH * CHANNELS);

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

    std::cout << "\n=== EXP-961 PURE WEIGHTLESS AUTOPOIETIC TELEMETRY ===\n";
    std::cout << "[+] Floating-Point Weight Count     : 0 (PURE WEIGHTLESS!)\n";
    std::cout << "[+] Final Topological Taps Count   : " << engine.taps.size() << "\n";
    std::cout << "[+] Final Reconstructed Mask MAE   : " << std::fixed << std::setprecision(2) << final_mae << "\n";
    std::cout << "[+] Total Execution Latency (10 iter): " << std::setprecision(4) << duration_ms << " ms\n";
    std::cout << "[+] Per-Iteration Latency          : " << std::setprecision(4) << duration_ms / 10.0 << " ms\n";

    std::cout << "\n[*] Synthesized Pure Weightless Topology (Zero Weights):\n";
    for (const auto& t : engine.taps) {
        std::cout << "    - Offset Lag: " << std::setw(6) << t.offset << " bits | Phase Sign: " 
                  << (t.phase > 0 ? "+1" : "-1") << " | Resonance Score: " << t.efficacy << "\n";
    }

    cudaFree(d_in_spins);
    cudaFree(d_out_spins);
    cudaFree(d_anchor_spins);
    cudaFree(d_mask);
    cudaFree(d_out_bytes);
    cudaFree(d_tap_offsets);
    cudaFree(d_tap_phases);

    return 0;
}
