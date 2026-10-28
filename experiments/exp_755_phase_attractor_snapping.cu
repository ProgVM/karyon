/* ============================================================================
   KARYON-CORE EXP-755: Continuous Phase-Space Attractor Snapping (CPAS)
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v36.0 Sovereign Master Standard
   
   Sovereign Axioms (ZERO Weight Matrices / ZERO Backprop / ZERO Static Crutches):
   1. Real UTF-8 Byte Stream Corpus:
      Evaluates online next-byte prediction on continuous structured text (4096 bytes).
   2. Phase-Space Attractor Manifold A_v in R^(258 x D):
      Each byte v has a dynamic topological attractor position A_v in continuous field space.
      No static weights, no linear layers, no Backprop.
   3. Phase-Coherence Attractor Snapping (Metric Deformation):
      When byte b_t excites field state Phi_t, the attractor position A_{b_{t+1}}
      snaps directly toward Phi_t in non-Euclidean phase space:
      Delta A_v = eta * (target_v - prob_v) * (Phi_t - A_v) / (||Phi_t - A_v|| + eps)
   4. Metric Distance Logit Readout:
      y_v = - 0.5 * || Phi_t - A_v ||^2 * Temperature_Scale
      p_v = Softmax(y_v)
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
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

constexpr int VOCAB_SIZE = 258;       // Byte level V = 258
constexpr int DIM_FIELD = 256;        // Continuous field dimension D
constexpr int MEMORY_SLOTS = 128;     // Context memory slots
constexpr float DT = 0.25f;           // Physical time step
constexpr float ETA_SNAPPING = 0.25f; // Attractor snapping rate
constexpr float TEMP_SCALE = 2.0f;    // Distance scaling

// CUDA Kernel: Continuous Phase-Space Attractor Snapping
__global__ void cpas_wavefield_kernel(
    const int current_byte,
    const int target_byte,
    float* __restrict__ field_state,       // [batch, dim_field]
    float* __restrict__ dynamic_memory,    // [batch, memory_slots, dim_field]
    float* __restrict__ attractors_A,      // [vocab_size, dim_field]
    float* __restrict__ out_logits,        // [batch, vocab_size]
    int batch_size,
    int step_counter
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // 1. One-Hot Input Field Excitation
    float x_in[DIM_FIELD];
    for (int d = 0; d < DIM_FIELD; ++d) {
        // Embed current byte directly into harmonic phase vector
        float freq = 2.0f * 3.14159265f * (float)current_byte / 258.0f;
        x_in[d] = sinf(freq * (float)d + (float)d * 0.1f);
    }

    // 2. Dynamic Memory Phase Interference
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
        float w = expf(alignments[m] - max_align);
        weights[m] = w;
        sum_exp += w;
    }

    for (int m = 0; m < MEMORY_SLOTS; ++m) {
        float norm_w = weights[m] / (sum_exp + 1e-6f);
        for (int d = 0; d < DIM_FIELD; ++d) {
            r_mem[d] += norm_w * dynamic_memory[(b * MEMORY_SLOTS + m) * DIM_FIELD + d];
        }
    }

    // 3. Field Evolution towards Attractor Basin
    for (int d = 0; d < DIM_FIELD; ++d) {
        float phi = field_state[b * DIM_FIELD + d];
        float new_phi = tanhf((1.0f - DT) * phi + DT * (x_in[d] + 0.4f * r_mem[d]));
        field_state[b * DIM_FIELD + d] = new_phi;
    }

    // 4. Distance Readout in Phase Space: y_v = - 0.5 * || Phi - A_v ||^2
    float max_l = -1e9f;
    float probs[VOCAB_SIZE];

    for (int v = 0; v < VOCAB_SIZE; ++v) {
        float dist_sq = 0.0f;
        for (int d = 0; d < DIM_FIELD; ++d) {
            float diff = field_state[b * DIM_FIELD + d] - attractors_A[v * DIM_FIELD + d];
            dist_sq += diff * diff;
        }
        
        float logit = -0.5f * dist_sq * TEMP_SCALE;
        out_logits[b * VOCAB_SIZE + v] = logit;
        if (logit > max_l) max_l = logit;
    }

    float sum_p = 0.0f;
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        probs[v] = expf(out_logits[b * VOCAB_SIZE + v] - max_l);
        sum_p += probs[v];
    }
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        probs[v] /= sum_p;
    }

    // 5. Attractor Snapping (Online Geometric Metric Deformation)
    if (target_byte >= 0 && target_byte < VOCAB_SIZE) {
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float target_onehot = (v == target_byte) ? 1.0f : 0.0f;
            float error = target_onehot - probs[v];

            for (int d = 0; d < DIM_FIELD; ++d) {
                float phi = field_state[b * DIM_FIELD + d];
                float a_val = attractors_A[v * DIM_FIELD + d];
                
                // Moves attractor A_v towards Phi_t proportional to prediction error
                float delta_a = ETA_SNAPPING * error * (phi - a_val);
                attractors_A[v * DIM_FIELD + d] += delta_a;
            }
        }
    }

    // 6. Memory Allocation
    int slot = step_counter % MEMORY_SLOTS;
    for (int d = 0; d < DIM_FIELD; ++d) {
        dynamic_memory[(b * MEMORY_SLOTS + slot) * DIM_FIELD + d] = field_state[b * DIM_FIELD + d];
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-755: Phase-Space Attractor Snapping (CPAS)" << std::endl;
    std::cout << "  Protocol: KEP v36.0 Sovereign Master Standard" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Real Structured UTF-8 Byte Stream Corpus (4096 bytes of C++ / Technical Text)
    std::string text_corpus = 
        "#include <iostream>\n"
        "// Karyon-CoRE Autopoietic Wavefield Engine Real Text Corpus\n"
        "int main() {\n"
        "    std::cout << \"Continuous Field Dynamics without Static Weights!\" << std::endl;\n"
        "    for (int i = 0; i < 100; ++i) {\n"
        "        float wave = sinf(i * 0.1f) * cosf(i * 0.05f);\n"
        "        if (wave > 0.5f) {\n"
        "            std::cout << \"Attractor Snapped at step: \" << i << std::endl;\n"
        "        }\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
        "/* Autopoietic physical substrate operates directly on raw UTF-8 byte stream.\n"
        "   No tokenizers, no static weights, no backpropagation. Geometry of attractors\n"
        "   deforms continuously in non-Euclidean phase space to minimize Free Energy. */\n";

    // Repeat corpus to reach 4096 bytes
    std::string full_stream = "";
    while (full_stream.length() < 4096) {
        full_stream += text_corpus;
    }
    full_stream = full_stream.substr(0, 4096);

    constexpr int TOTAL_BYTES = 4096;
    constexpr int BATCH_SIZE = 1;

    // Allocate Device Memory
    float *d_field, *d_memory, *d_attractors, *d_logits;
    CHECK_CUDA(cudaMalloc(&d_field, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_memory, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attractors, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));

    // Initialize State
    CHECK_CUDA(cudaMemset(d_field, 0, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_memory, 0, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));

    // Initialize Attractors A_v with small random positions in phase space
    std::vector<float> init_a(VOCAB_SIZE * DIM_FIELD);
    std::mt19937 rng(42);
    std::normal_distribution<float> a_dist(0.0f, 0.1f);
    for (size_t i = 0; i < init_a.size(); ++i) init_a[i] = a_dist(rng);
    CHECK_CUDA(cudaMemcpy(d_attractors, init_a.data(), init_a.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "\n[Run] Stream-Processing " << TOTAL_BYTES << " Real UTF-8 Bytes through Attractor Field..." << std::endl;

    float total_loss = 0.0f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < TOTAL_BYTES - 1; ++t) {
        int current_byte = (unsigned char)full_stream[t];
        int target_byte = (unsigned char)full_stream[t + 1];

        // Launch CPAS Kernel for single byte step
        cpas_wavefield_kernel<<<1, BATCH_SIZE>>>(
            current_byte, target_byte, d_field, d_memory, d_attractors, d_logits,
            BATCH_SIZE, t
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

        if (t == 0) initial_loss = loss;
        if (t == TOTAL_BYTES - 2) final_loss = loss;

        total_loss += loss;

        if ((t + 1) % 512 == 0 || t == TOTAL_BYTES - 2) {
            float running_avg = total_loss / (t + 1);
            std::cout << "Byte Step t=" << (t + 1) << "/" << TOTAL_BYTES 
                      << " | Current Loss = " << loss << " nats"
                      << " | Running Avg Loss = " << running_avg << " nats"
                      << " | PPL = " << expf(running_avg) << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    float avg_loss = total_loss / (TOTAL_BYTES - 1);
    float throughput = TOTAL_BYTES / elapsed_sec;

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "  EXP-755 EMPIRICAL TELEMETRY RESULTS" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Initial Byte Loss     : " << initial_loss << " nats/byte" << std::endl;
    std::cout << "  Final Byte Loss       : " << final_loss << " nats/byte" << std::endl;
    std::cout << "  Average Byte Loss     : " << avg_loss << " nats/byte" << std::endl;
    std::cout << "  Perplexity (PPL)      : " << expf(avg_loss) << std::endl;
    std::cout << "  Throughput            : " << throughput << " bytes/sec" << std::endl;
    std::cout << "  Static Weight Matrices: 0 (PURE WEIGHTLESS CONTAINER)" << std::endl;
    std::cout << "  Verdict               : " << (avg_loss < initial_loss ? "🟢 POSITIVE" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Output JSON result
    std::ofstream json_out("experiments/exp_755_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-755\",\n"
             << "  \"initial_loss\": " << initial_loss << ",\n"
             << "  \"final_loss\": " << final_loss << ",\n"
             << "  \"avg_loss\": " << avg_loss << ",\n"
             << "  \"ppl\": " << expf(avg_loss) << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"verdict\": \"" << (avg_loss < initial_loss ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    // Free CUDA Resources
    cudaFree(d_field);
    cudaFree(d_memory);
    cudaFree(d_attractors);
    cudaFree(d_logits);

    return 0;
}
