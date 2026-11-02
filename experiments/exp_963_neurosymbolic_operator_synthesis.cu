#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ============================================================================
// KARYON-CORE EXP-963: SOVEREIGN NEUROSYMBOLIC OPERATOR GENESIS ENGINE (NC-VFS)
// KEP v40.0 Sovereign Master | Principles 0, 1, 3, 4, 5, 6, 7, 8, 9
// 1. Pure Weightless Bit-Spin Substrate (V=2 / int8_t {-1, +1})
// 2. Differentiable-Free Neurosymbolic Operator Genesis (Algebraic Primitives)
// 3. Dynamic Selection & Locking of Non-Linear Interaction Rules:
//    - Bilinear Spin Product (s_a * s_b)
//    - Bitwise Parity XOR (s_a ^ s_b)
//    - Triadic Soliton Majority Gate (Majority(s_a, s_b, s_c))
// ============================================================================

constexpr size_t WIDTH = 128;
constexpr size_t HEIGHT = 128;
constexpr size_t CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS; // 49,152
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;            // 393,216

enum SymbolOperatorType {
    OP_LINEAR_SUM = 0,
    OP_BILINEAR_PRODUCT = 1,
    OP_BITWISE_XOR = 2,
    OP_TRIADIC_SOLITON = 3
};

struct NeurosymbolicRule {
    SymbolOperatorType type;
    int lag_a;
    int lag_b;
    int lag_c;
    int8_t phase;
    double efficacy;
    std::string name;
};

// CUDA Kernel: Evaluate Neurosymbolic Interaction Rules on Bit-Spin Substrate
__global__ void cuda_neurosymbolic_forward_kernel(
    const int8_t* __restrict__ d_in_spins,
    int8_t* __restrict__ d_out_spins,
    const int8_t* __restrict__ d_anchor_spins,
    const uint8_t* __restrict__ d_mask,
    const int* __restrict__ d_op_types,
    const int* __restrict__ d_lags_a,
    const int* __restrict__ d_lags_b,
    const int* __restrict__ d_lags_c,
    const int8_t* __restrict__ d_op_phases,
    int num_rules
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FRAME_BITS) return;

    int byte_idx = i / 8;
    bool is_masked = (d_mask[byte_idx] == 1);

    if (!is_masked) {
        d_out_spins[i] = d_anchor_spins[i];
        return;
    }

    int total_field = 0;

    #pragma unroll 4
    for (int r = 0; r < num_rules; ++r) {
        int idx_a = i + d_lags_a[r];
        int idx_b = i + d_lags_b[r];
        int idx_c = i + d_lags_c[r];

        if (idx_a < 0) idx_a = 0; if (idx_a >= (int)FRAME_BITS) idx_a = FRAME_BITS - 1;
        if (idx_b < 0) idx_b = 0; if (idx_b >= (int)FRAME_BITS) idx_b = FRAME_BITS - 1;
        if (idx_c < 0) idx_c = 0; if (idx_c >= (int)FRAME_BITS) idx_c = FRAME_BITS - 1;

        int8_t s_a = d_in_spins[idx_a];
        int8_t s_b = d_in_spins[idx_b];
        int8_t s_c = d_in_spins[idx_c];

        int rule_val = 0;
        int op_type = d_op_types[r];

        if (op_type == OP_LINEAR_SUM) {
            rule_val = (int)s_a;
        } else if (op_type == OP_BILINEAR_PRODUCT) {
            rule_val = (int)s_a * (int)s_b;
        } else if (op_type == OP_BITWISE_XOR) {
            rule_val = (s_a != s_b) ? 1 : -1;
        } else if (op_type == OP_TRIADIC_SOLITON) {
            int maj = s_a + s_b + s_c;
            rule_val = (maj >= 0) ? 1 : -1;
        }

        total_field += (int)d_op_phases[r] * rule_val;
    }

    d_out_spins[i] = (total_field >= 0) ? 1 : -1;
}

// Convert Bit-Spins to RGB Bytes
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

class SovereignNeurosymbolicEngine {
public:
    std::vector<NeurosymbolicRule> active_rules;

    SovereignNeurosymbolicEngine() {
        // Seed with basic linear lag rules
        active_rules.push_back({OP_LINEAR_SUM, -24, 0, 0, +1, 1.0, "Linear_RGB_Left(-24)"});
        active_rules.push_back({OP_LINEAR_SUM, +24, 0, 0, +1, 1.0, "Linear_RGB_Right(+24)"});
        active_rules.push_back({OP_LINEAR_SUM, -3072, 0, 0, +1, 1.0, "Linear_Row_Up(-3072)"});
        active_rules.push_back({OP_LINEAR_SUM, +3072, 0, 0, +1, 1.0, "Linear_Row_Down(+3072)"});
    }

    // Neurosymbolic Genesis: Synthesize & Lock High-Order Interaction Rules
    void SynthesizeHighOrderRules(const std::vector<int8_t>& spins, const std::vector<uint8_t>& mask) {
        std::vector<NeurosymbolicRule> candidates = {
            // Bilinear Phase Products
            {OP_BILINEAR_PRODUCT, -24, -3072, 0, +1, 0.0, "Bilinear(RGB_Left * Row_Up)"},
            {OP_BILINEAR_PRODUCT, +24, +3072, 0, +1, 0.0, "Bilinear(RGB_Right * Row_Down)"},

            // Bitwise Parity XOR
            {OP_BITWISE_XOR, -24, +24, 0, +1, 0.0, "Parity_XOR(RGB_Left ^ RGB_Right)"},
            {OP_BITWISE_XOR, -3072, +3072, 0, +1, 0.0, "Parity_XOR(Row_Up ^ Row_Down)"},

            // Triadic Soliton Majority Gate
            {OP_TRIADIC_SOLITON, -24, +24, -3072, +1, 0.0, "TriadicSoliton(RGB_L, RGB_R, Row_Up)"},
            {OP_TRIADIC_SOLITON, -3072, +3072, +24, +1, 0.0, "TriadicSoliton(Row_U, Row_D, RGB_R)"}
        };

        for (auto& cand : candidates) {
            int64_t agreement = 0;
            int64_t valid = 0;

            for (size_t i = 0; i < FRAME_BITS; ++i) {
                if (mask[i / 8] == 0) continue;

                int idx_a = (int)i + cand.lag_a;
                int idx_b = (int)i + cand.lag_b;
                int idx_c = (int)i + cand.lag_c;

                if (idx_a >= 0 && idx_a < (int)FRAME_BITS &&
                    idx_b >= 0 && idx_b < (int)FRAME_BITS &&
                    idx_c >= 0 && idx_c < (int)FRAME_BITS) {

                    int8_t s_a = spins[idx_a];
                    int8_t s_b = spins[idx_b];
                    int8_t s_c = spins[idx_c];

                    int val = 0;
                    if (cand.type == OP_BILINEAR_PRODUCT) val = (int)s_a * (int)s_b;
                    else if (cand.type == OP_BITWISE_XOR) val = (s_a != s_b) ? 1 : -1;
                    else if (cand.type == OP_TRIADIC_SOLITON) val = (s_a + s_b + s_c >= 0) ? 1 : -1;

                    agreement += ((int)spins[i] * val);
                    valid++;
                }
            }

            if (valid > 0) {
                cand.efficacy = (double)agreement / valid;
                if (std::abs(cand.efficacy) > 0.15) {
                    cand.phase = (cand.efficacy >= 0) ? +1 : -1;
                    std::cout << "[⚛️ Symbol Genesis] Synthesized & Locked Rule: " << std::setw(38) << cand.name 
                              << " | Efficacy: " << std::fixed << std::setprecision(4) << cand.efficacy << "\n";
                    active_rules.push_back(cand);
                }
            }
        }
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-963: SOVEREIGN NEUROSYMBOLIC OPERATOR GENESIS ENGINE\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 0, 1, 3, 4, 5, 6, 7, 8, 9\n";
    std::cout << " 1. Pure Weightless Bit-Spin Substrate (V=2 / ZERO FLOAT WEIGHTS)\n";
    std::cout << " 2. Dynamic Synthesis of Non-Linear Symbolic Interaction Rules\n";
    std::cout << " 3. Zero Backprop — Epigenetic Lock of High-Efficacy Primitives\n";
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

    SovereignNeurosymbolicEngine engine;

    int8_t *d_in_spins, *d_out_spins, *d_anchor_spins, *d_op_phases;
    int *d_op_types, *d_lags_a, *d_lags_b, *d_lags_c;
    uint8_t *d_mask, *d_out_bytes;

    size_t max_rules = 32;

    cudaMalloc(&d_in_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_out_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_anchor_spins, FRAME_BITS * sizeof(int8_t));
    cudaMalloc(&d_mask, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_out_bytes, FRAME_BYTES * sizeof(uint8_t));
    cudaMalloc(&d_op_types, max_rules * sizeof(int));
    cudaMalloc(&d_lags_a, max_rules * sizeof(int));
    cudaMalloc(&d_lags_b, max_rules * sizeof(int));
    cudaMalloc(&d_lags_c, max_rules * sizeof(int));
    cudaMalloc(&d_op_phases, max_rules * sizeof(int8_t));

    cudaMemcpy(d_in_spins, spins_corrupted.data(), FRAME_BITS * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anchor_spins, spins_gt.data(), FRAME_BITS * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), FRAME_BYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int block_1d = 256;
    int grid_bits = (FRAME_BITS + block_1d - 1) / block_1d;
    int grid_bytes = (FRAME_BYTES + block_1d - 1) / block_1d;

    std::cout << "[*] Launching Neurosymbolic Wavefield Relaxation & Operator Genesis Loop...\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= 10; ++iter) {
        int num_rules = engine.active_rules.size();
        std::vector<int> h_types(num_rules), h_la(num_rules), h_lb(num_rules), h_lc(num_rules);
        std::vector<int8_t> h_phases(num_rules);

        for (int r = 0; r < num_rules; ++r) {
            h_types[r] = (int)engine.active_rules[r].type;
            h_la[r] = engine.active_rules[r].lag_a;
            h_lb[r] = engine.active_rules[r].lag_b;
            h_lc[r] = engine.active_rules[r].lag_c;
            h_phases[r] = engine.active_rules[r].phase;
        }

        cudaMemcpy(d_op_types, h_types.data(), num_rules * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lags_a, h_la.data(), num_rules * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lags_b, h_lb.data(), num_rules * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lags_c, h_lc.data(), num_rules * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_op_phases, h_phases.data(), num_rules * sizeof(int8_t), cudaMemcpyHostToDevice);

        // Forward Wavefield Propagation using Active Neurosymbolic Rules
        cuda_neurosymbolic_forward_kernel<<<grid_bits, block_1d>>>(
            d_in_spins, d_out_spins, d_anchor_spins, d_mask,
            d_op_types, d_lags_a, d_lags_b, d_lags_c, d_op_phases, num_rules
        );
        cudaDeviceSynchronize();

        std::vector<int8_t> h_spins(FRAME_BITS);
        cudaMemcpy(h_spins.data(), d_out_spins, FRAME_BITS * sizeof(int8_t), cudaMemcpyDeviceToHost);

        size_t bit_errors = 0;
        size_t total_mask_bits = masked_bytes * 8;
        for (size_t i = 0; i < FRAME_BITS; ++i) {
            if (mask[i / 8] == 1) {
                if (h_spins[i] != spins_gt[i]) bit_errors++;
            }
        }
        double bit_error_rate = (100.0 * bit_errors) / total_mask_bits;

        std::cout << "Iteration " << std::setw(2) << iter << " | Active Rules: " << std::setw(2) << num_rules 
                  << " | Mask BER: " << std::fixed << std::setprecision(2) << bit_error_rate << "%\n";

        // Neurosymbolic Genesis Step (Synthesize new rules at iteration 2)
        if (iter == 2) {
            engine.SynthesizeHighOrderRules(h_spins, mask);
        }

        std::swap(d_in_spins, d_out_spins);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Output Final Reconstruction
    cuda_spins_to_bytes_kernel<<<grid_bytes, block_1d>>>(d_in_spins, d_out_bytes);
    std::vector<uint8_t> frame_reconstructed(FRAME_BYTES);
    cudaMemcpy(frame_reconstructed.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);

    stbi_write_png("exp_963_neurosymbolic_reconstructed.png", WIDTH, HEIGHT, CHANNELS, frame_reconstructed.data(), WIDTH * CHANNELS);

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

    std::cout << "\n=== EXP-963 NEUROSYMBOLIC GENESIS TELEMETRY ===\n";
    std::cout << "[+] Total Active Rules Count        : " << engine.active_rules.size() << "\n";
    std::cout << "[+] Final Reconstructed Mask MAE   : " << std::fixed << std::setprecision(2) << final_mae << "\n";
    std::cout << "[+] Total Execution Latency        : " << std::setprecision(4) << duration_ms << " ms\n";
    std::cout << "[+] Per-Iteration Latency          : " << std::setprecision(4) << duration_ms / 10.0 << " ms\n";

    std::cout << "\n[*] Final Synthesized Neurosymbolic Rule Base:\n";
    for (const auto& r : engine.active_rules) {
        std::cout << "    - Rule: " << std::setw(38) << r.name << " | Phase: " 
                  << (r.phase > 0 ? "+1" : "-1") << " | Efficacy: " << r.efficacy << "\n";
    }

    cudaFree(d_in_spins);
    cudaFree(d_out_spins);
    cudaFree(d_anchor_spins);
    cudaFree(d_mask);
    cudaFree(d_out_bytes);
    cudaFree(d_op_types);
    cudaFree(d_lags_a);
    cudaFree(d_lags_b);
    cudaFree(d_lags_c);
    cudaFree(d_op_phases);

    return 0;
}
