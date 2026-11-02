#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ============================================================================
// KARYON-CORE EXP-957: 2D CONTINUOUS SPATIAL PHASE DIFFUSION & SOLITONS (CUDA)
// KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5, 7
// Substrate: Pure Continuous Bit-Spin Manifold (V=2), Zero Weight Matrices
// Target: Autonomous Boundary Stabilization via Physical Soliton Coupling
// ============================================================================

constexpr int WIDTH = 128;
constexpr int HEIGHT = 128;
constexpr int CHANNELS = 3;
constexpr int TOTAL_PIXELS = WIDTH * HEIGHT;                     // 16,384
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS;        // 49,152 bytes
constexpr size_t FRAME_BITS = FRAME_BYTES * 8;                   // 393,216 bits/spins
constexpr size_t WORDS_PER_FRAME = FRAME_BITS / 64;              // 6,144 uint64_t words

// CUDA Kernel 1: Bit-packed Multichannel Superposition with Continuous Phase Resonance
__global__ void cuda_phase_resonance_kernel(
    const uint64_t* __restrict__ d_packed_manifold,
    const size_t* __restrict__ d_ensemble_indices,
    int K,
    float* __restrict__ d_continuous_spins // Continuous floating-point phase field [-1.0, +1.0]
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
        
        // Continuous mean physical spin s \in [-1.0, +1.0]
        float mean_s = (2.0f * ones - (float)K) / (float)K;
        int spin_idx = w * 64 + bit;
        d_continuous_spins[spin_idx] = mean_s;
    }
}

// CUDA Kernel 2: 2D Physical Soliton Wavefield Diffusion & Boundary Sharpening
// Continuous Lagrangian field relaxation:
// ds/dtau = D * \nabla^2 s + \tanh(\beta * s) - \gamma * s
__global__ void cuda_2d_soliton_diffusion_kernel(
    const float* __restrict__ d_in_spins,
    float* __restrict__ d_out_spins,
    float D_diff,
    float beta,
    float gamma,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= WIDTH || y >= HEIGHT || c >= CHANNELS) return;

    int pixel_idx = y * WIDTH + x;
    int byte_idx = pixel_idx * CHANNELS + c;
    int spin_base = byte_idx * 8;

    // 4-neighbor spatial bounds (Neumann zero-flux boundary conditions)
    int xm = (x > 0) ? (x - 1) : 0;
    int xp = (x < WIDTH - 1) ? (x + 1) : (WIDTH - 1);
    int ym = (y > 0) ? (y - 1) : 0;
    int yp = (y < HEIGHT - 1) ? (y + 1) : (HEIGHT - 1);

    int byte_left  = (y * WIDTH + xm) * CHANNELS + c;
    int byte_right = (y * WIDTH + xp) * CHANNELS + c;
    int byte_up    = (ym * WIDTH + x) * CHANNELS + c;
    int byte_down  = (yp * WIDTH + x) * CHANNELS + c;

    // Process all 8 bit planes in continuous spin space
    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        float s_center = d_in_spins[spin_base + bit];
        float s_left   = d_in_spins[byte_left * 8 + bit];
        float s_right  = d_in_spins[byte_right * 8 + bit];
        float s_up     = d_in_spins[byte_up * 8 + bit];
        float s_down   = d_in_spins[byte_down * 8 + bit];

        // Discrete 2D Laplacian operator on continuous spin field: \nabla^2 s
        float laplacian = (s_left + s_right + s_up + s_down - 4.0f * s_center);

        // Continuous Soliton Reaction-Diffusion PDE Step
        float ds_dt = D_diff * laplacian + tanhf(beta * s_center) - gamma * s_center;
        float s_new = s_center + dt * ds_dt;

        // Physical clamping to spin manifold bound [-1.0, +1.0]
        s_new = fmaxf(-1.0f, fminf(1.0f, s_new));
        d_out_spins[spin_base + bit] = s_new;
    }
}

// CUDA Kernel 3: Project Continuous Spin Manifold to Bitstream & Output RGB Bytes
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
    std::cout << " KARYON-CORE EXP-957: 2D CONTINUOUS PHASE DIFFUSION & SOLITONS\n";
    std::cout << " KEP v40.0 Sovereign Master | Principles 1, 2, 4, 5, 7\n";
    std::cout << " Substrate: Bit-Packed 64-Spin Continuous Wavefield (V=2)\n";
    std::cout << " Target: Autonomous Boundary Stabilization via 2D Soliton PDE\n";
    std::cout << "===================================================================\n\n";

    std::ifstream file("data/cifar_vehicles_128x128.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Error: cannot open data/cifar_vehicles_128x128.bin\n";
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

    // Ingest and pack raw dataset into 64-bit words
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

    // Allocate GPU Device Tensors
    uint64_t* d_packed_manifold = nullptr;
    size_t* d_ensemble_indices = nullptr;
    float* d_continuous_spins_A = nullptr;
    float* d_continuous_spins_B = nullptr;
    uint8_t* d_out_bytes = nullptr;

    cudaMalloc(&d_packed_manifold, packed_manifold.size() * sizeof(uint64_t));
    cudaMemcpy(d_packed_manifold, packed_manifold.data(), packed_manifold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<size_t> active_ensemble = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
    int K = active_ensemble.size();

    cudaMalloc(&d_ensemble_indices, K * sizeof(size_t));
    cudaMemcpy(d_ensemble_indices, active_ensemble.data(), K * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_continuous_spins_A, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_continuous_spins_B, FRAME_BITS * sizeof(float));
    cudaMalloc(&d_out_bytes, FRAME_BYTES * sizeof(uint8_t));

    // Execution Grids
    int block_1d = 256;
    int grid_1d = (WORDS_PER_FRAME + block_1d - 1) / block_1d;

    dim3 block_2d(16, 16);
    dim3 grid_2d((WIDTH + 15) / 16, (HEIGHT + 15) / 16, CHANNELS);

    int grid_bytes = (FRAME_BYTES + block_1d - 1) / block_1d;

    // Baseline: Initial Resonance without Spatial Diffusion
    cuda_phase_resonance_kernel<<<grid_1d, block_1d>>>(
        d_packed_manifold, d_ensemble_indices, K, d_continuous_spins_A
    );
    cuda_spins_to_bytes_kernel<<<grid_bytes, block_1d>>>(d_continuous_spins_A, d_out_bytes);
    
    std::vector<uint8_t> baseline_bytes(FRAME_BYTES);
    cudaMemcpy(baseline_bytes.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);
    stbi_write_png("exp_957_baseline_no_diffusion.png", WIDTH, HEIGHT, CHANNELS, baseline_bytes.data(), WIDTH * CHANNELS);
    std::cout << "[+] Saved baseline: exp_957_baseline_no_diffusion.png\n";

    // Benchmarking 2D Soliton Diffusion PDE Iterations
    // PDE Parameters: Diffusion D=0.15, Beta=3.0, Gamma=0.8, dt=0.05
    float D_diff = 0.15f;
    float beta = 3.0f;
    float gamma = 0.8f;
    float dt = 0.05f;
    int diffusion_steps = 10;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Run 2D Soliton Relaxation Loop
    float* src_spins = d_continuous_spins_A;
    float* dst_spins = d_continuous_spins_B;

    for (int step = 0; step < diffusion_steps; ++step) {
        cuda_2d_soliton_diffusion_kernel<<<grid_2d, block_2d>>>(
            src_spins, dst_spins, D_diff, beta, gamma, dt
        );
        std::swap(src_spins, dst_spins);
    }
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Render Soliton Relaxed Image
    cuda_spins_to_bytes_kernel<<<grid_bytes, block_1d>>>(src_spins, d_out_bytes);
    std::vector<uint8_t> soliton_bytes(FRAME_BYTES);
    cudaMemcpy(soliton_bytes.data(), d_out_bytes, FRAME_BYTES, cudaMemcpyDeviceToHost);
    stbi_write_png("exp_957_soliton_diffused.png", WIDTH, HEIGHT, CHANNELS, soliton_bytes.data(), WIDTH * CHANNELS);
    std::cout << "[+] Saved soliton diffused frame: exp_957_soliton_diffused.png\n";

    // Quantitative Contrast & Coherence Telemetry
    double mean_base = 0.0, sq_base = 0.0;
    double mean_sol  = 0.0, sq_sol  = 0.0;
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        mean_base += baseline_bytes[i];
        sq_base += baseline_bytes[i] * baseline_bytes[i];

        mean_sol += soliton_bytes[i];
        sq_sol += soliton_bytes[i] * soliton_bytes[i];
    }
    mean_base /= FRAME_BYTES;
    mean_sol  /= FRAME_BYTES;
    double std_base = std::sqrt(sq_base / FRAME_BYTES - mean_base * mean_base);
    double std_sol  = std::sqrt(sq_sol / FRAME_BYTES - mean_sol * mean_sol);

    // High-Throughput FPS Benchmark (1000 full PDE passes)
    int bench_runs = 1000;
    auto bench_t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < bench_runs; ++r) {
        cuda_phase_resonance_kernel<<<grid_1d, block_1d>>>(
            d_packed_manifold, d_ensemble_indices, K, d_continuous_spins_A
        );
        src_spins = d_continuous_spins_A;
        dst_spins = d_continuous_spins_B;
        for (int step = 0; step < diffusion_steps; ++step) {
            cuda_2d_soliton_diffusion_kernel<<<grid_2d, block_2d>>>(
                src_spins, dst_spins, D_diff, beta, gamma, dt
            );
            std::swap(src_spins, dst_spins);
        }
    }
    cudaDeviceSynchronize();
    auto bench_t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(bench_t1 - bench_t0).count();

    double fps = bench_runs / total_sec;
    double per_pass_ms = (total_sec / bench_runs) * 1000.0;
    double mspins_sec = (bench_runs * diffusion_steps * FRAME_BITS) / (total_sec * 1e6);

    std::cout << "\n=== EXP-957 EMPIRICAL TELEMETRY ===\n";
    std::cout << "[+] Baseline Contrast (Std Dev) : " << std::fixed << std::setprecision(2) << std_base << "\n";
    std::cout << "[+] Soliton Diffused Contrast   : " << std::setprecision(2) << std_sol << "\n";
    std::cout << "[+] Contrast Delta              : " << std::showpos << (std_sol - std_base) << std::noshowpos << "\n";
    std::cout << "[+] 10-Step Soliton Latency     : " << std::setprecision(4) << latency_ms << " ms\n";
    std::cout << "[+] Full End-to-End Speed       : " << std::setprecision(2) << fps << " FPS (" << per_pass_ms << " ms/pass)\n";
    std::cout << "[+] Total Soliton Spin Throughput: " << std::setprecision(2) << mspins_sec << " M-Spins/sec (" << mspins_sec / 1000.0 << " G-Spins/s)\n";

    cudaFree(d_packed_manifold);
    cudaFree(d_ensemble_indices);
    cudaFree(d_continuous_spins_A);
    cudaFree(d_continuous_spins_B);
    cudaFree(d_out_bytes);

    return 0;
}
