/* ============================================================================
   KARYON-CORE EXP-751: Autopoietic Plastic Conductance Field (APCF)
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v36.0 Sovereign Master Standard
   
   Axioms Implemented:
   1. Zero Pretrained Static Weight Matrices (Pure Weightless Container):
      Conductance tensor G in R^(V x D) starts at 0 and self-organizes on-the-fly.
   2. Local Online Autopoietic Update Rule (No Backprop / No Adam):
      Logits: y_v = sum_d G_{v, d} * Phi_d
      Probabilities: p_v = softmax(y_v)
      Error Vector: e_v = target_onehot_v - p_v
      Conductance Plasticity: Delta G_{v, d} = eta * e_v * Phi_d * (1 - 0.80 * Shield_d)
   3. Soft Epigenetic Shielding (AC-EVE from EXP-747):
      Tracks causal credit per dimension C_d. High C_d builds shield S_d <= 0.80,
      protecting established cognitive pathways while leaving 20%+ plastic headroom.
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
constexpr float DT = 0.20f;           // Physical time step
constexpr float ETA_PLASTICITY = 0.15f;// Conductance plasticity learning rate
constexpr float SHIELD_CEILING = 0.80f;// EXP-747 soft ceiling limit

// CUDA Kernel: Autopoietic Conductance Field Update & Readout
__global__ void apcf_wavefield_kernel(
    const float* __restrict__ input_bytes, // [batch, dim_field]
    float* __restrict__ field_state,       // [batch, dim_field]
    float* __restrict__ dynamic_memory,    // [batch, memory_slots, dim_field]
    float* __restrict__ conductance_G,     // [vocab_size, dim_field]
    float* __restrict__ causal_credits,    // [dim_field]
    float* __restrict__ lock_shield,       // [dim_field]
    float* __restrict__ out_logits,        // [batch, vocab_size]
    int target_byte,
    int batch_size,
    int step_counter,
    float hardware_noise_scale,
    unsigned int seed
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    unsigned int prng_state = seed + b * 1337;

    // 1. Phase-Resonant Dynamic Memory Interference
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

    // 2. Continuous Field State Integration with Contractive Attractor
    float free_energy = 0.0f;
    for (int d = 0; d < DIM_FIELD; ++d) {
        float x_in = input_bytes[b * DIM_FIELD + d];
        float phi = field_state[b * DIM_FIELD + d];
        float shield = lock_shield[d];

        prng_state = prng_state * 1664525u + 1013904223u;
        float noise = ((float)(prng_state & 0x00FFFFFF) / (float)0x01000000 - 0.5f) * hardware_noise_scale;

        float grad_V = (x_in + 0.5f * r_mem[d]) - phi;
        float d_phi = DT * (grad_V + noise) * (1.0f - shield);

        float new_phi = tanhf(phi + d_phi);
        field_state[b * DIM_FIELD + d] = new_phi;

        float dE = fabsf(x_in - phi) - fabsf(x_in - new_phi);
        causal_credits[d] = 0.95f * causal_credits[d] + 0.05f * dE;

        // Soft Epigenetic Shielding (capped at 0.80)
        if (causal_credits[d] > 0.01f) {
            lock_shield[d] = fminf(SHIELD_CEILING, lock_shield[d] + 0.02f);
        } else {
            lock_shield[d] = fmaxf(0.0f, lock_shield[d] - 0.005f);
        }

        free_energy += fabsf(x_in - new_phi);
    }

    // 3. Forward Readout: Logits via Conductance Field G
    float max_l = -1e9f;
    float probs[VOCAB_SIZE];

    for (int v = 0; v < VOCAB_SIZE; ++v) {
        float logit = 0.0f;
        for (int d = 0; d < DIM_FIELD; ++d) {
            logit += conductance_G[v * DIM_FIELD + d] * field_state[b * DIM_FIELD + d];
        }
        out_logits[b * VOCAB_SIZE + v] = logit;
        if (logit > max_l) max_l = logit;
    }

    // Softmax probabilities
    float sum_p = 0.0f;
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        probs[v] = expf(out_logits[b * VOCAB_SIZE + v] - max_l);
        sum_p += probs[v];
    }
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        probs[v] /= sum_p;
    }

    // 4. Local Autopoietic Plasticity Update (Delta G = eta * e_v * Phi_d * (1 - S_d))
    if (target_byte >= 0 && target_byte < VOCAB_SIZE) {
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float target_onehot = (v == target_byte) ? 1.0f : 0.0f;
            float error = target_onehot - probs[v];

            for (int d = 0; d < DIM_FIELD; ++d) {
                float phi = field_state[b * DIM_FIELD + d];
                float shield = lock_shield[d];
                float delta_g = ETA_PLASTICITY * error * phi * (1.0f - shield);
                
                // Update Conductance Tensor G in-place on GPU
                conductance_G[v * DIM_FIELD + d] += delta_g;
            }
        }
    }

    // 5. Salience Dynamic Memory Allocation
    if ((free_energy / DIM_FIELD) > 0.12f) {
        int slot = step_counter % MEMORY_SLOTS;
        for (int d = 0; d < DIM_FIELD; ++d) {
            dynamic_memory[(b * MEMORY_SLOTS + slot) * DIM_FIELD + d] = field_state[b * DIM_FIELD + d];
        }
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-751: Autopoietic Plastic Conductance Field" << std::endl;
    std::cout << "  Protocol: KEP v36.0 Sovereign Master Standard" << std::endl;
    std::cout << "=================================================================" << std::endl;

    constexpr int NUM_CHUNKS = 32;
    constexpr int CHUNK_SIZE = 128; // Total 4096 bytes
    constexpr int BATCH_SIZE = 1;

    // Allocate Device Memory
    float *d_input, *d_field, *d_memory, *d_conductance, *d_credits, *d_shield, *d_logits;
    CHECK_CUDA(cudaMalloc(&d_input, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_field, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_memory, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conductance, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_credits, DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shield, DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));

    // Initialize State (Zero Conductance Tensor G - Pure Autopoiesis)
    CHECK_CUDA(cudaMemset(d_field, 0, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_memory, 0, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_conductance, 0, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_credits, 0, DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_shield, 0, DIM_FIELD * sizeof(float)));

    std::vector<float> h_input(DIM_FIELD);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "\n[Run] Executing 32 Chunks (4096 Bytes) across Plastic Conductance Field..." << std::endl;

    float total_loss = 0.0f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int chunk = 1; chunk <= NUM_CHUNKS; ++chunk) {
        int target_byte = (chunk * 7) % 256;

        for (int d = 0; d < DIM_FIELD; ++d) {
            h_input[d] = sinf(chunk * 0.1f + d * 0.05f) + dist(rng) * 0.05f;
        }

        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), DIM_FIELD * sizeof(float), cudaMemcpyHostToDevice));

        // Launch APCF Kernel
        apcf_wavefield_kernel<<<1, BATCH_SIZE>>>(
            d_input, d_field, d_memory, d_conductance, d_credits, d_shield, d_logits,
            target_byte, BATCH_SIZE, chunk, 0.01f, chunk * 777
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> h_logits(VOCAB_SIZE);
        CHECK_CUDA(cudaMemcpy(h_logits.data(), d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

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
                if (h_shield[d] > 0.4f) locked_count += 1.0f;
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
    std::cout << "  EXP-751 EMPIRICAL TELEMETRY RESULTS" << std::endl;
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
    std::ofstream json_out("experiments/exp_751_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-751\",\n"
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
    cudaFree(d_conductance);
    cudaFree(d_credits);
    cudaFree(d_shield);
    cudaFree(d_logits);

    return 0;
}
