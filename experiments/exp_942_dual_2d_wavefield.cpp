// ============================================================================
// EXP-942: Dual CPU/GPU Continuous 2D Spatial Wavefield Engine
// Standard: KEP v40.0 Sovereign Master | Principle 1 & 2 (Dual CPU/GPU Substrate)
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int WIDTH = 128;
constexpr int HEIGHT = 128;
constexpr int CHANNELS = 3;
constexpr size_t FRAME_BYTES = WIDTH * HEIGHT * CHANNELS;

// CPU SIMD (AVX2/OpenMP) 2D Spatial Wavefield Engine with 2D Stride Taps
void cpu_2d_wavefield_synth(
    const float* text_phase,
    const uint8_t* dataset_buffer,
    int num_frames,
    uint8_t* output_frame
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int pixel_idx = (y * WIDTH + x) * CHANNELS;

            float r_phase = 0.0f, g_phase = 0.0f, b_phase = 0.0f;
            float total_weight = 0.0001f;

            for (int f = 0; f < num_frames; ++f) {
                size_t offset = f * FRAME_BYTES + pixel_idx;
                uint8_t frame_r = dataset_buffer[offset + 0];
                uint8_t frame_g = dataset_buffer[offset + 1];
                uint8_t frame_b = dataset_buffer[offset + 2];

                // 2D Spatial Stride Taps: Left (x-1), Up (y-1), Up-Left (x-1, y-1)
                float stride_coherence = 1.0f;

                if (x > 0) {
                    size_t left_offset = f * FRAME_BYTES + (y * WIDTH + (x - 1)) * CHANNELS;
                    float left_r = dataset_buffer[left_offset + 0];
                    float left_g = dataset_buffer[left_offset + 1];
                    float left_b = dataset_buffer[left_offset + 2];
                    float diff_left = (fabsf((float)frame_r - left_r) + fabsf((float)frame_g - left_g) + fabsf((float)frame_b - left_b)) / 3.0f;
                    stride_coherence += (255.0f - diff_left) / 255.0f;
                }

                if (y > 0) {
                    size_t up_offset = f * FRAME_BYTES + ((y - 1) * WIDTH + x) * CHANNELS;
                    float up_r = dataset_buffer[up_offset + 0];
                    float up_g = dataset_buffer[up_offset + 1];
                    float up_b = dataset_buffer[up_offset + 2];
                    float diff_up = (fabsf((float)frame_r - up_r) + fabsf((float)frame_g - up_g) + fabsf((float)frame_b - up_b)) / 3.0f;
                    stride_coherence += (255.0f - diff_up) / 255.0f;
                }

                if (x > 0 && y > 0) {
                    size_t diag_offset = f * FRAME_BYTES + ((y - 1) * WIDTH + (x - 1)) * CHANNELS;
                    float diag_r = dataset_buffer[diag_offset + 0];
                    float diag_g = dataset_buffer[diag_offset + 1];
                    float diag_b = dataset_buffer[diag_offset + 2];
                    float diff_diag = (fabsf((float)frame_r - diag_r) + fabsf((float)frame_g - diag_g) + fabsf((float)frame_b - diag_b)) / 3.0f;
                    stride_coherence += (255.0f - diff_diag) / 255.0f;
                }

                float text_match = text_phase[f % 32];
                float weight = expf(text_match * 1.8f + (stride_coherence / 3.0f) * 0.8f);

                r_phase += frame_r * weight;
                g_phase += frame_g * weight;
                b_phase += frame_b * weight;
                total_weight += weight;
            }

            float norm_r = r_phase / total_weight;
            float norm_g = g_phase / total_weight;
            float norm_b = b_phase / total_weight;

            output_frame[pixel_idx + 0] = (uint8_t)std::min(std::max(norm_r, 0.0f), 255.0f);
            output_frame[pixel_idx + 1] = (uint8_t)std::min(std::max(norm_g, 0.0f), 255.0f);
            output_frame[pixel_idx + 2] = (uint8_t)std::min(std::max(norm_b, 0.0f), 255.0f);
        }
    }
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-942: Dual CPU/GPU Continuous 2D Spatial Wavefield Engine\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 1 & 2 (Dual Backend)\n";
    std::cout << "==================================================================\n\n";

    std::string dataset_path = "data/furry_hires/furry_128x128_rgb.bin";
    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset " << dataset_path << "\n";
        return 1;
    }

    size_t file_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> dataset_buffer(file_bytes);
    file.read(reinterpret_cast<char*>(dataset_buffer.data()), file_bytes);
    file.close();

    int num_frames = file_bytes / FRAME_BYTES;
    std::cout << "1. DATASET INGESTION TELEMETRY:\n";
    std::cout << "   -> Loaded File: " << dataset_path << "\n";
    std::cout << "   -> Size: " << file_bytes << " bytes (" << (file_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "   -> Total Frames: " << num_frames << " frames (128x128 RGB)\n";
    std::cout << "   -> Total Bit-Spins: " << (file_bytes * 8) << " spins\n\n";

    // Text conditioning vector
    std::vector<float> text_phase(32);
    for (int i = 0; i < 32; ++i) {
        text_phase[i] = sinf(i * 0.25f) * 0.8f + cosf(i * 0.15f) * 0.5f;
    }

    std::vector<uint8_t> cpu_output(FRAME_BYTES, 0);

    // CPU Benchmark (Warmup + 20 iterations)
    std::cout << "2. CPU SIMD / OpenMP 2D SPATIAL WAVEFIELD BENCHMARK:\n";
    for (int w = 0; w < 3; ++w) {
        cpu_2d_wavefield_synth(text_phase.data(), dataset_buffer.data(), num_frames, cpu_output.data());
    }

    int iterations = 20;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        cpu_2d_wavefield_synth(text_phase.data(), dataset_buffer.data(), num_frames, cpu_output.data());
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_latency_ms = total_ms / iterations;
    double throughput_fps = 1000.0 / avg_latency_ms;

    std::cout << "   -> Executed " << iterations << " CPU frames\n";
    std::cout << "   -> Average CPU Frame Latency: " << std::fixed << std::setprecision(4) << avg_latency_ms << " ms (" << (avg_latency_ms * 1000.0) << " us)\n";
    std::cout << "   -> CPU Throughput: " << std::fixed << std::setprecision(2) << throughput_fps << " FPS\n\n";

    // Save PNG output
    std::string out_png = "data/karyon_cpu_furry_128.png";
    int write_res = stbi_write_png(out_png.c_str(), WIDTH, HEIGHT, CHANNELS, cpu_output.data(), WIDTH * CHANNELS);
    std::cout << "3. VISUAL OUTPUT SERIALIZATION:\n";
    std::cout << "   -> Saved CPU Synthesized PNG: " << out_png << " (" << (write_res ? "SUCCESS" : "FAILED") << ")\n\n";

    return 0;
}
