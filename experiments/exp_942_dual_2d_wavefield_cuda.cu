// ============================================================================
// EXP-942: CUDA Continuous 2D Spatial Wavefield Synthesis Kernel
// Standard: KEP v40.0 Sovereign Master | Principle 1 & 2 (Dual Backend CUDA Engine)
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

// CUDA 2D Spatial Wavefield Kernel with 2D Spatial Stride Taps (x-1, y-1, x-1/y-1)
__global__ void cuda_2d_wavefield_synth_kernel(
    const float* __restrict__ d_text_phase_vector,
    const uint8_t* __restrict__ d_dataset_buffer,
    int num_frames,
    uint8_t* d_output_frame
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int pixel_idx = (y * WIDTH + x) * CHANNELS;

    float r_phase = 0.0f, g_phase = 0.0f, b_phase = 0.0f;
    float total_weight = 0.0001f;

    for (int f = 0; f < num_frames; ++f) {
        size_t offset = f * FRAME_BYTES + pixel_idx;
        uint8_t frame_r = d_dataset_buffer[offset + 0];
        uint8_t frame_g = d_dataset_buffer[offset + 1];
        uint8_t frame_b = d_dataset_buffer[offset + 2];

        float stride_coherence = 1.0f;

        if (x > 0) {
            size_t left_offset = f * FRAME_BYTES + (y * WIDTH + (x - 1)) * CHANNELS;
            float left_r = d_dataset_buffer[left_offset + 0];
            float left_g = d_dataset_buffer[left_offset + 1];
            float left_b = d_dataset_buffer[left_offset + 2];
            float diff_left = (fabsf((float)frame_r - left_r) + fabsf((float)frame_g - left_g) + fabsf((float)frame_b - left_b)) / 3.0f;
            stride_coherence += (255.0f - diff_left) / 255.0f;
        }

        if (y > 0) {
            size_t up_offset = f * FRAME_BYTES + ((y - 1) * WIDTH + x) * CHANNELS;
            float up_r = d_dataset_buffer[up_offset + 0];
            float up_g = d_dataset_buffer[up_offset + 1];
            float up_b = d_dataset_buffer[up_offset + 2];
            float diff_up = (fabsf((float)frame_r - up_r) + fabsf((float)frame_g - up_g) + fabsf((float)frame_b - up_b)) / 3.0f;
            stride_coherence += (255.0f - diff_up) / 255.0f;
        }

        if (x > 0 && y > 0) {
            size_t diag_offset = f * FRAME_BYTES + ((y - 1) * WIDTH + (x - 1)) * CHANNELS;
            float diag_r = d_dataset_buffer[diag_offset + 0];
            float diag_g = d_dataset_buffer[diag_offset + 1];
            float diag_b = d_dataset_buffer[diag_offset + 2];
            float diff_diag = (fabsf((float)frame_r - diag_r) + fabsf((float)frame_g - diag_g) + fabsf((float)frame_b - diag_b)) / 3.0f;
            stride_coherence += (255.0f - diff_diag) / 255.0f;
        }

        float text_match = d_text_phase_vector[f % 32];
        float weight = expf(text_match * 1.8f + (stride_coherence / 3.0f) * 0.8f);

        r_phase += frame_r * weight;
        g_phase += frame_g * weight;
        b_phase += frame_b * weight;
        total_weight += weight;
    }

    float norm_r = r_phase / total_weight;
    float norm_g = g_phase / total_weight;
    float norm_b = b_phase / total_weight;

    d_output_frame[pixel_idx + 0] = (uint8_t)fminf(fmaxf(norm_r, 0.0f), 255.0f);
    d_output_frame[pixel_idx + 1] = (uint8_t)fminf(fmaxf(norm_g, 0.0f), 255.0f);
    d_output_frame[pixel_idx + 2] = (uint8_t)fminf(fmaxf(norm_b, 0.0f), 255.0f);
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-942: CUDA Continuous 2D Spatial Wavefield Engine\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 1 & 2 (CUDA Backend)\n";
    std::cout << "==================================================================\n\n";

    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset " << dataset_path << "\n";
        return 1;
    }

    size_t file_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> h_dataset_buffer(file_bytes);
    file.read(reinterpret_cast<char*>(h_dataset_buffer.data()), file_bytes);
    file.close();

    int num_frames = file_bytes / FRAME_BYTES;

    std::vector<float> h_text_phase(32);
    for (int i = 0; i < 32; ++i) {
        h_text_phase[i] = sinf(i * 0.25f) * 0.8f + cosf(i * 0.15f) * 0.5f;
    }

    // CUDA Allocation
    float* d_text_phase = nullptr;
    uint8_t* d_dataset_buffer = nullptr;
    uint8_t* d_output_frame = nullptr;

    cudaMalloc(&d_text_phase, 32 * sizeof(float));
    cudaMalloc(&d_dataset_buffer, file_bytes);
    cudaMalloc(&d_output_frame, FRAME_BYTES);

    cudaMemcpy(d_text_phase, h_text_phase.data(), 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataset_buffer, h_dataset_buffer.data(), file_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Warmup
    for (int w = 0; w < 10; ++w) {
        cuda_2d_wavefield_synth_kernel<<<numBlocks, threadsPerBlock>>>(d_text_phase, d_dataset_buffer, num_frames, d_output_frame);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cuda_2d_wavefield_synth_kernel<<<numBlocks, threadsPerBlock>>>(d_text_phase, d_dataset_buffer, num_frames, d_output_frame);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    double avg_latency_ms = total_ms / iterations;
    double throughput_fps = 1000.0 / avg_latency_ms;

    std::vector<uint8_t> h_output_frame(FRAME_BYTES);
    cudaMemcpy(h_output_frame.data(), d_output_frame, FRAME_BYTES, cudaMemcpyDeviceToHost);

    // Get VRAM Usage
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    double vram_used_mb = (total_bytes - free_bytes) / 1024.0 / 1024.0;

    std::cout << "1. CUDA KERNEL BENCHMARK TELEMETRY:\n";
    std::cout << "   -> Evaluated Iterations: " << iterations << " frames\n";
    std::cout << "   -> Average CUDA Frame Latency: " << std::fixed << std::setprecision(4) << avg_latency_ms << " ms (" << (avg_latency_ms * 1000.0) << " us)\n";
    std::cout << "   -> CUDA Generation Throughput: " << std::fixed << std::setprecision(2) << throughput_fps << " FPS\n";
    std::cout << "   -> VRAM Allocated: " << std::fixed << std::setprecision(2) << vram_used_mb << " MB\n\n";

    // Save PNG output
    std::string out_png = "data/karyon_cuda_furry_128.png";
    int write_res = stbi_write_png(out_png.c_str(), WIDTH, HEIGHT, CHANNELS, h_output_frame.data(), WIDTH * CHANNELS);
    std::cout << "2. VISUAL OUTPUT SERIALIZATION:\n";
    std::cout << "   -> Saved CUDA Synthesized PNG: " << out_png << " (" << (write_res ? "SUCCESS" : "FAILED") << ")\n\n";

    cudaFree(d_text_phase);
    cudaFree(d_dataset_buffer);
    cudaFree(d_output_frame);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
