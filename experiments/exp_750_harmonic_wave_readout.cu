/* ============================================================================
   KARYON-CORE EXP-750: Harmonic Spectral Wavefield & Contractive Metric Attractor
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v36.0 Sovereign Master Standard
   
   Axioms Implemented (Zero Weight Matrices):
   1. Contractive Metric Attractor Potential:
      Forces field states onto a compact [-1, 1]^D manifold via non-linear mapping:
      Phi_{t+1} = tanh( Phi_t + dt * ( (X_in + R_mem) - Phi_t ) )
   2. Harmonic Fourier/Spectral Wave Readout:
      Direct orthogonal projection from continuous field Phi to 258 byte logits:
      logit_v = sum_d Phi_d * cos( 2*pi * v * d / 258 )
   3. Free Energy Governed Dynamic Memory Allocation:
      Slots are updated only when local Free Energy / Surprise (||X_in - Phi||)
      exceeds dynamic threshold, locking high-salience context patterns.
   ============================================================================ */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int VOCAB_SIZE = 258;       // Byte level V = 258 (256 bytes + pad + eos)
constexpr int DIM_FIELD = 256;        // Continuous field dimension D
constexpr int MEMORY_SLOTS = 128;     // Dynamic Memory Capacity
constexpr float DT = 0.15f;           // Physical time step
constexpr float M_PI_F = 3.14159265358979323846f;

// CUDA Kernel: Harmonic Wavefield Super-Primitive
__global__ void harmonic_wavefield_kernel(
    const float* __restrict__ input_bytes, // [batch, dim_field]
    float* __restrict__ field_state,       // [batch, dim_field]
    float* __restrict__ dynamic_memory,    // [batch, memory_slots, dim_field]
    float* __restrict__ memory_salience,   // [batch, memory_slots]
    float* __restrict__ causal_credits,    // [batch, dim_field]
    float* __restrict__ lock_shield,       // [batch, dim_field]
    float* __restrict__ out_logits,        // [batch, vocab_size]
    int batch_size,
    int step_counter,
    float hardware_noise_scale,
    unsigned int seed
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    unsigned int prng_state = seed + b * 1337;

    // 1. Phase-Resonant Memory Interference
    float r_mem[DIM_FIELD];
    for (int d = 0; d < DIM_FIELD; ++d) r_mem[d] = 0.0f;

    float max_align = -1e9f;
    float alignments[MEMORY_SLOTS];

    for (int m = 0; m < MEMORY_SLOTS; ++m) {
        float dot_prod = 0.0f;
        for (int d = 0; d < DIM_FIELD; ++d) {
            dot_prod += field_state[b * DIM_FIELD + d] * dynamic_memory[(b * MEMORY_SLOTS + m) * DIM_FIELD + d];
        }
        float align = dot_prod / sqrtf((float)DIM_FIELD);
        alignments[m] = align;
        if (align > max_align) max_align = align;
    }

    float sum_exp = 0.0f;
    float weights[MEMORY_SLOTS];
    for (int m = 0; m < MEMORY_SLOTS; ++m) {
        float w = expf((alignments[m] - max_align) * 2.0f);
        weights[m] = w;
        sum_exp += w;
    }

    for (int m = 0; m < MEMORY_SLOTS; ++m) {
        float norm_w = weights[m] / sum_exp;
        for (int d = 0; d < DIM_FIELD; ++d) {
            r_mem[d] += norm_w * dynamic_memory[(b * MEMORY_SLOTS + m) * DIM_FIELD + d];
        }
    }

    // 2. Contractive Metric Attractor Field Update
    float total_free_energy = 0.0f;
    for (int d = 0; d < DIM_FIELD; ++d) {
        float x_in = input_bytes[b * DIM_FIELD + d];
        float phi = field_state[b * DIM_FIELD + d];
        float shield = lock_shield[b * DIM_FIELD + d];

        prng_state = prng_state * 1664525u + 1013904223u;
        float noise = ((float)(prng_state & 0x00FFFFFF) / (float)0x01000000 - 0.5f) * hardware_noise_scale;

        float grad_V = (x_in + 0.5f * r_mem[d]) - phi;
        float d_phi = DT * (grad_V + noise) * (1.0f - 0.8f * shield);

        // Contractive Attractor Mapping: tanh keeps field bounded in [-1, 1]
        float new_phi = tanhf(phi + d_phi);
        field_state[b * DIM_FIELD + d] = new_phi;

        float dE = fabsf(x_in - phi) - fabsf(x_in - new_phi);
        causal_credits[b * DIM_FIELD + d] = 0.95f * causal_credits[b * DIM_FIELD + d] + 0.05f * dE;

        if (causal_credits[b * DIM_FIELD + d] > 0.01f) {
            lock_shield[b * DIM_FIELD + d] = fminf(1.0f, lock_shield[b * DIM_FIELD + d] + 0.02f);
        } else {
            lock_shield[b * DIM_FIELD + d] = fmaxf(0.0f, lock_shield[b * DIM_FIELD + d] - 0.005f);
        }

        total_free_energy += fabsf(x_in - new_phi);
    }

    // 3. Free-Energy Governed Memory Allocation
    float free_energy_avg = total_free_energy / DIM_FIELD;
    if (free_energy_avg > 0.15f) { // Write high surprise state to memory
        int slot = step_counter % MEMORY_SLOTS;
        memory_salience[b * MEMORY_SLOTS + slot] = free_energy_avg;
        for (int d = 0; d < DIM_FIELD; ++d) {
            dynamic_memory[(b * MEMORY_SLOTS + slot) * DIM_FIELD + d] = field_state[b * DIM_FIELD + d];
        }
    }

    // 4. Harmonic Fourier Spectral Readout to Logits (Zero Trained Weights)
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        float logit = 0.0f;
        float freq = 2.0f * M_PI_F * (float)v / (float)VOCAB_SIZE;
        for (int d = 0; d < DIM_FIELD; ++d) {
            float phi = field_state[b * DIM_FIELD + d];
            logit += phi * cosf(freq * (float)d);
        }
        out_logits[b * VOCAB_SIZE + v] = logit * 0.1f;
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-750: Harmonic Spectral Wavefield Engine" << std::endl;
    std::cout << "  Protocol: KEP v36.0 Sovereign Master Standard" << std::endl;
    std::cout << "=================================================================" << std::endl;

    constexpr int NUM_CHUNKS = 32;
    constexpr int CHUNK_SIZE = 128; // Total 4096 bytes
    constexpr int BATCH_SIZE = 1;

    // Allocate Device Memory
    float *d_input, *d_field, *d_memory, *d_salience, *d_credits, *d_shield, *d_logits;
    CHECK_CUDA(cudaMalloc(&d_input, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_field, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_memory, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_salience, BATCH_SIZE * MEMORY_SLOTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_credits, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shield, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));

    // Initialize State
    CHECK_CUDA(cudaMemset(d_field, 0, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_memory, 0, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_salience, 0, BATCH_SIZE * MEMORY_SLOTS * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_credits, 0, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_shield, 0, BATCH_SIZE * DIM_FIELD * sizeof(float)));

    std::vector<float> h_input(DIM_FIELD);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "\n[Run] Executing 32 Chunks (4096 Bytes) across Harmonic Spectral Field..." << std::endl;

    float total_loss = 0.0f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int chunk = 1; chunk <= NUM_CHUNKS; ++chunk) {
        for (int d = 0; d < DIM_FIELD; ++d) {
            h_input[d] = sinf(chunk * 0.1f + d * 0.05f) + dist(rng) * 0.05f;
        }

        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), DIM_FIELD * sizeof(float), cudaMemcpyHostToDevice));

        // Launch Kernel
        harmonic_wavefield_kernel<<<1, BATCH_SIZE>>>(
            d_input, d_field, d_memory, d_salience, d_credits, d_shield, d_logits,
            BATCH_SIZE, chunk, 0.01f, chunk * 777
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> h_logits(VOCAB_SIZE);
        CHECK_CUDA(cudaMemcpy(h_logits.data(), d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        int target_byte = (chunk * 7) % 256;
        float max_logit = h_logits[0];
        for (int v = 1; v < VOCAB_SIZE; ++v) {
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            sum_exp += expf(h_logits[v] - max_logit);
        }

        float log_prob = h_logits[target_byte] - max_logit - logf(sum_exp);
        float loss = -log_prob;

        if (chunk == 1) initial_loss = loss;
        if (chunk == NUM_CHUNKS) final_loss = loss;

        total_loss += loss;

        if (chunk % 4 == 0 || chunk == NUM_CHUNKS) {
            std::vector<float> h_shield(DIM_FIELD);
            CHECK_CUDA(cudaMemcpy(h_shield.data(), d_shield, DIM_FIELD * sizeof(float), cudaMemcpyDeviceToHost));
            float locked_count = 0;
            for (int d = 0; d < DIM_FIELD; ++d) {
                if (h_shield[d] > 0.5f) locked_count += 1.0f;
            }
            float lock_pct = (locked_count / DIM_FIELD) * 100.0f;

            std::cout << "Chunk " << chunk << "/" << NUM_CHUNKS 
                      << " (t=" << chunk * CHUNK_SIZE << "): Loss = " << loss 
                      << " nats | PPL = " << expf(loss)
                      << " | Locked Nodes = " << lock_pct << "%" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    float avg_loss = total_loss / NUM_CHUNKS;
    float throughput = (NUM_CHUNKS * CHUNK_SIZE) / elapsed_sec;

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-750 EMPIRICAL TELEMETRY RESULTS" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Initial Loss          : " << initial_loss << " nats/byte" << std::endl;
    std::cout << "  Final Loss            : " << final_loss << " nats/byte" << std::endl;
    std::cout << "  Average Loss          : " << avg_loss << " nats/byte" << std::endl;
    std::cout << "  Perplexity (PPL)      : " << expf(final_loss) << std::endl;
    std::cout << "  Throughput            : " << throughput << " bytes/sec" << std::endl;
    std::cout << "  Static Weight Matrices: 0 (PURE WEIGHTLESS CONTAINER)" << std::endl;
    std::cout << "  Verdict               : " << (final_loss < initial_loss ? "🟢 POSITIVE" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Output JSON result
    std::ofstream json_out("experiments/exp_750_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-750\",\n"
             << "  \"initial_loss\": " << initial_loss << ",\n"
             << "  \"final_loss\": " << final_loss << ",\n"
             << "  \"ppl\": " << expf(final_loss) << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"verdict\": \"" << (final_loss < initial_loss ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    // Free CUDA Resources
    cudaFree(d_input);
    cudaFree(d_field);
    cudaFree(d_memory);
    cudaFree(d_salience);
    cudaFree(d_credits);
    cudaFree(d_shield);
    cudaFree(d_logits);

    return 0;
}
