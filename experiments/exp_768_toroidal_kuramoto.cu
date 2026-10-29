/* ============================================================================
   KARYON-CORE EXP-768: True Toroidal Kuramoto Phase-Order Substrate (TT-KPOS)
   ============================================================================
   Author: Bazilevs (ProgVM) & Lead AI Cyberneticist Gema
   Protocol: KEP v30.0 Sovereign Master Standard (Principles 2, 27 & 30 Compliant)
   
   THE DETECTIVE BREAKTHROUGH OVER EXP-767 PARALYSIS:
   
   Why EXP-767 halted at Loss = 5.47 nats (ln(258) ~ 5.55):
   1. In EXP-767, Phi was artificially clamped by tanh in [-1, +1], while target phases
      theta(d) evolved on the circle [0, 2*pi]. This mismatched geometry caused
      sin(theta_target - Phi) to act as incoherent white noise, cancelling out updates.
   2. Sum of cosines without sharpness/inverse temperature gave PLV ~ 0.01 across all bytes,
      yielding flat uniform entropy ~ ln(258).
      
   THE EXP-768 SOLUTION:
   1. Real Toroidal Phase Angle on S^1:
      Phi_d in [-pi, +pi] evolves via unconstrained angular momentum:
      d Phi_d / dt = omega_d + K_coup * Phase_Interference
      Phi_d = atan2(sin(Phi_d), cos(Phi_d))  <-- Strict Toroidal Topology S^1
      
   2. Complex Order Parameter & Sharpness:
      Each candidate byte v defines a test phase wave.
      Order parameter:
      Z_v = (1/D) sum_d exp( i * (Phi_d - theta_v(d)) ) = R_v * e^{i * psi_v}
      Real Resonance Energy: E_v = beta_sharp * R_v * cos(psi_v)
      Where R_v in [0, 1] is the Kuramoto coherence magnitude!
      When phases align with byte v, R_v -> 1.0, creating a sharp, massive energy peak.
      
   3. True Kuramoto Phase-Torque Plasticity:
      The potential field V_{d, k} stores resonant phase biases on the torus:
      Delta V_{d, k} = eta * sin( theta_target(d) - Phi_d - V_{d, k} )
      This acts as genuine mechanical torque on S^1, pulling the attractor basin
      directly into alignment with the incoming byte stream!
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

constexpr int VOCAB_SIZE = 258;       // Raw Byte V = 258
constexpr int DIM_FIELD = 128;        // Phase Oscillators D = 128
constexpr int HARMONIC_MODES = 8;     // Toroidal Harmonics K = 8
constexpr int MEMORY_SLOTS = 32;      // Memory Wave Traces M = 32
constexpr float BETA_SHARP = 8.0f;    // Kuramoto Resonance Sharpness Inverse Temperature
constexpr float ETA_PLASTICITY = 0.20f; // Angular Torque Plasticity Rate
constexpr float PI = 3.14159265358979323846f;

// CUDA Kernel: True Toroidal Kuramoto Phase-Order Substrate
__global__ void ttkpos_toroidal_kernel(
    const int current_byte,
    const int target_byte,
    float* __restrict__ phase_phi,       // [batch, dim_field] on [-pi, +pi]
    float* __restrict__ phase_memory,    // [batch, memory_slots, dim_field] on [-pi, +pi]
    float* __restrict__ potential_V,     // [dim_field, harmonic_modes] on [-pi, +pi]
    float* __restrict__ out_logits,      // [batch, vocab_size]
    int batch_size,
    int step_counter
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // 1. Input Physical Wave Phase Angle on S^1
    float theta_in[DIM_FIELD];
    for (int d = 0; d < DIM_FIELD; ++d) {
        float angle = 2.0f * PI * ((float)current_byte / (float)VOCAB_SIZE) * (float)(d + 1);
        // Wrap to [-pi, +pi]
        theta_in[d] = atan2f(sinf(angle), cosf(angle));
    }

    // 2. Toroidal Phase Memory Interference (Phase Alignment Vector)
    float mem_cos[DIM_FIELD];
    float mem_sin[DIM_FIELD];
    for (int d = 0; d < DIM_FIELD; ++d) {
        mem_cos[d] = 0.0f;
        mem_sin[d] = 0.0f;
        for (int m = 0; m < MEMORY_SLOTS; ++m) {
            float m_phi = phase_memory[(b * MEMORY_SLOTS + m) * DIM_FIELD + d];
            mem_cos[d] += cosf(m_phi);
            mem_sin[d] += sinf(m_phi);
        }
    }

    // 3. True Toroidal Phase Dynamics on S^1 (Kuramoto Phase Equation)
    for (int d = 0; d < DIM_FIELD; ++d) {
        float phi = phase_phi[b * DIM_FIELD + d];

        // Intrinsic Memory Phase Angle
        float phi_mem = atan2f(mem_sin[d], mem_cos[d]);

        // Input Torque + Memory Torque
        float torque_in = sinf(theta_in[d] - phi);
        float torque_mem = sinf(phi_mem - phi);

        // Potential Field Phase Coupling on Harmonics
        float torque_pot = 0.0f;
        for (int k = 0; k < HARMONIC_MODES; ++k) {
            float v_angle = potential_V[d * HARMONIC_MODES + k];
            torque_pot += sinf(v_angle - phi * (float)(k + 1));
        }

        // Spatial Kuramoto Nearest-Neighbor Coupling
        int d_next = (d + 1) % DIM_FIELD;
        int d_prev = (d - 1 + DIM_FIELD) % DIM_FIELD;
        float torque_spatial = sinf(phase_phi[b * DIM_FIELD + d_next] - phi) + 
                               sinf(phase_phi[b * DIM_FIELD + d_prev] - phi);

        // Net Phase Velocity dPhi/dt
        float d_phi = 0.35f * torque_in + 0.20f * torque_mem + 0.15f * torque_pot + 0.10f * torque_spatial;

        // Unconstrained Evolution mapped strictly to Torus S^1
        float new_phi = phi + d_phi;
        phase_phi[b * DIM_FIELD + d] = atan2f(sinf(new_phi), cosf(new_phi));
    }

    // Update Phase Memory Buffer
    int mem_slot = step_counter % MEMORY_SLOTS;
    for (int d = 0; d < DIM_FIELD; ++d) {
        phase_memory[(b * MEMORY_SLOTS + mem_slot) * DIM_FIELD + d] = phase_phi[b * DIM_FIELD + d];
    }

    // 4. Kuramoto Complex Order Parameter Readout (R_v * cos(psi_v))
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        float v_base = 2.0f * PI * ((float)v / (float)VOCAB_SIZE);

        for (int d = 0; d < DIM_FIELD; ++d) {
            float phi = phase_phi[b * DIM_FIELD + d];
            float test_angle = v_base * (float)(d + 1);
            float diff = phi - test_angle;

            real_sum += cosf(diff);
            imag_sum += sinf(diff);
        }

        // Order parameter R_v in [0, 1]
        float R_v = sqrtf(real_sum * real_sum + imag_sum * imag_sum) / (float)DIM_FIELD;
        float psi_v = atan2f(imag_sum, real_sum);

        // Coherent projection amplified by sharpness beta
        out_logits[b * VOCAB_SIZE + v] = BETA_SHARP * R_v * cosf(psi_v);
    }

    // 5. True Angular Torque Plasticity on S^1
    if (target_byte >= 0 && target_byte < VOCAB_SIZE) {
        float target_base = 2.0f * PI * ((float)target_byte / (float)VOCAB_SIZE);

        for (int d = 0; d < DIM_FIELD; ++d) {
            float phi = phase_phi[b * DIM_FIELD + d];
            float target_angle = target_base * (float)(d + 1);
            float target_wrapped = atan2f(sinf(target_angle), cosf(target_angle));

            for (int k = 0; k < HARMONIC_MODES; ++k) {
                float v_curr = potential_V[d * HARMONIC_MODES + k];
                // Kuramoto torque pulling V directly toward target phase difference
                float torque = sinf(target_wrapped - phi * (float)(k + 1) - v_curr);
                float new_v = v_curr + ETA_PLASTICITY * torque;
                potential_V[d * HARMONIC_MODES + k] = atan2f(sinf(new_v), cosf(new_v));
            }
        }
    }
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  KARYON-CORE EXP-768: True Toroidal Kuramoto Phase-Order Substrate" << std::endl;
    std::cout << "  S^1 Topology atan2() | Complex Order Parameter R*cos(psi)" << std::endl;
    std::cout << "  Resolving EXP-767 Phase Paralysis & ln(258) Barrier" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Structured Corpus (4096 bytes)
    std::string text_corpus = 
        "#include <iostream>\n"
        "// Karyon-CoRE True Toroidal Kuramoto Phase-Order Substrate\n"
        "int main() {\n"
        "    std::cout << \"S1 Toroidal Phase Resonance and Kuramoto Order Parameter!\" << std::endl;\n"
        "    for (int t = 0; t < 1000; ++t) {\n"
        "        if (t % 10 == 0) {\n"
        "            std::cout << \"Attractor Phase Basin Alignment: \" << t << std::endl;\n"
        "        }\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
        "/* The system operates strictly on the circle S^1 via atan2 wrap-around,\n"
        "   computing genuine Kuramoto order parameters R_v to generate razor-sharp\n"
        "   contrast and shatter the uniform entropy barrier. */\n";

    std::string full_stream = "";
    while (full_stream.length() < 4096) {
        full_stream += text_corpus;
    }
    full_stream = full_stream.substr(0, 4096);

    constexpr int TOTAL_BYTES = 4096;
    constexpr int BATCH_SIZE = 1;

    // Allocate Device Memory
    float *d_phi, *d_memory, *d_potential, *d_logits;
    CHECK_CUDA(cudaMalloc(&d_phi, BATCH_SIZE * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_memory, BATCH_SIZE * MEMORY_SLOTS * DIM_FIELD * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_potential, DIM_FIELD * HARMONIC_MODES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));

    // Initialize State randomly on [-pi, +pi]
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(-PI, PI);

    std::vector<float> init_phi(DIM_FIELD);
    for (size_t i = 0; i < init_phi.size(); ++i) init_phi[i] = angle_dist(rng);
    CHECK_CUDA(cudaMemcpy(d_phi, init_phi.data(), init_phi.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> init_mem(MEMORY_SLOTS * DIM_FIELD);
    for (size_t i = 0; i < init_mem.size(); ++i) init_mem[i] = angle_dist(rng);
    CHECK_CUDA(cudaMemcpy(d_memory, init_mem.data(), init_mem.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> init_v(DIM_FIELD * HARMONIC_MODES);
    for (size_t i = 0; i < init_v.size(); ++i) init_v[i] = angle_dist(rng);
    CHECK_CUDA(cudaMemcpy(d_potential, init_v.data(), init_v.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "\n[Run] Streaming " << TOTAL_BYTES << " Bytes through Toroidal Substrate..." << std::endl;

    float total_loss = 0.0f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < TOTAL_BYTES - 1; ++t) {
        int current_byte = (unsigned char)full_stream[t];
        int target_byte = (unsigned char)full_stream[t + 1];

        ttkpos_toroidal_kernel<<<1, BATCH_SIZE>>>(
            current_byte, target_byte, d_phi, d_memory, d_potential, d_logits,
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
            std::cout << "Step t=" << (t + 1) << "/" << TOTAL_BYTES 
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
    std::cout << "  EXP-768 EMPIRICAL TELEMETRY RESULTS" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Initial Byte Loss       : " << initial_loss << " nats/byte" << std::endl;
    std::cout << "  Final Byte Loss         : " << final_loss << " nats/byte" << std::endl;
    std::cout << "  Average Byte Loss       : " << avg_loss << " nats/byte" << std::endl;
    std::cout << "  Perplexity (PPL)        : " << expf(avg_loss) << std::endl;
    std::cout << "  Throughput              : " << throughput << " bytes/sec" << std::endl;
    std::cout << "  Topology                : Strict S^1 Torus (atan2f)" << std::endl;
    std::cout << "  Order Parameter         : Kuramoto R_v * cos(psi_v)" << std::endl;
    std::cout << "  Plasticity Torque       : Kuramoto S^1 Torque (sin(Delta theta))" << std::endl;
    std::cout << "  Static Weight Matrices  : 0" << std::endl;
    std::cout << "  Verdict                 : " << (avg_loss < initial_loss && avg_loss < 5.40f ? "🟢 POSITIVE" : "🔴 REJECTED") << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Output JSON result
    std::ofstream json_out("experiments/exp_768_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-768\",\n"
             << "  \"initial_loss\": " << initial_loss << ",\n"
             << "  \"final_loss\": " << final_loss << ",\n"
             << "  \"avg_loss\": " << avg_loss << ",\n"
             << "  \"ppl\": " << expf(avg_loss) << ",\n"
             << "  \"throughput_bytes_per_sec\": " << throughput << ",\n"
             << "  \"verdict\": \"" << (avg_loss < initial_loss && avg_loss < 5.40f ? "POSITIVE" : "REJECTED") << "\"\n"
             << "}\n";
    json_out.close();

    // Free CUDA Resources
    cudaFree(d_phi);
    cudaFree(d_memory);
    cudaFree(d_potential);
    cudaFree(d_logits);

    return 0;
}
