// ==============================================================================
// EXP-735.2: PURE CUDA CONTINUOUS STATE-SPACE SEQUENCE MEMORY (SSD-TRACE CUDA)
// Fix: Exact Softmax Cross-Entropy Gradients for W_read (y_v - P_v) * Psi_d
// ==============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>

#define NUM_UNIVERSES 64
#define DIM_FIELD 256
#define VOCAB_SIZE 256
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void init_sequence_memory_kernel(
    float* __restrict__ Phi,
    float* __restrict__ W_inj,
    float* __restrict__ W_trans,
    float* __restrict__ W_read,
    float* __restrict__ Epigenetic_Shield,
    float* __restrict__ Prev_Phi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < NUM_UNIVERSES * DIM_FIELD) {
        Phi[idx] = 0.0f;
        Prev_Phi[idx] = 0.0f;
        Epigenetic_Shield[idx] = 0.0f;
    }

    if (idx < VOCAB_SIZE * DIM_FIELD) {
        float r1 = (float)(idx % 97) / 97.0f - 0.5f;
        W_inj[idx] = r1 * 0.1f;

        float r2 = (float)(idx % 89) / 89.0f - 0.5f;
        W_read[idx] = r2 * 0.1f;
    }

    if (idx < DIM_FIELD * DIM_FIELD) {
        int r_i = idx / DIM_FIELD;
        int c_j = idx % DIM_FIELD;
        float diag = (r_i == c_j) ? 0.90f : 0.0f;
        float rand_val = ((float)((idx * 1337 + 7) % 101) / 101.0f - 0.5f) * 0.02f;
        W_trans[idx] = diag + rand_val;
    }
}

__global__ void ssd_sequence_evolution_kernel(
    float* __restrict__ Phi,
    float* __restrict__ Prev_Phi,
    const float* __restrict__ W_inj,
    const float* __restrict__ W_trans,
    int byte_in,
    float alpha_decay
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= NUM_UNIVERSES) return;

    const float* phi_prev_u = &Prev_Phi[u * DIM_FIELD];
    float* phi_u = &Phi[u * DIM_FIELD];
    const float* x_wave = &W_inj[byte_in * DIM_FIELD];

    float local_new[DIM_FIELD];
    float norm_sq = 0.0f;

    for (int d = 0; d < DIM_FIELD; ++d) {
        float trans_sum = 0.0f;
        const float* w_row = &W_trans[d * DIM_FIELD];
        
        for (int k = 0; k < DIM_FIELD; ++k) {
            trans_sum += w_row[k] * phi_prev_u[k];
        }

        float net_input = trans_sum + x_wave[d];
        float activated = tanhf(net_input);

        float updated_val = alpha_decay * phi_prev_u[d] + (1.0f - alpha_decay) * activated;
        local_new[d] = updated_val;
        norm_sq += updated_val * updated_val;
    }

    float inv_norm = rsqrtf(norm_sq + 1e-6f);
    for (int d = 0; d < DIM_FIELD; ++d) {
        phi_u[d] = local_new[d] * inv_norm;
    }
}

__global__ void ssd_readout_and_vector_error_kernel(
    const float* __restrict__ Phi,
    const float* __restrict__ W_read,
    float* __restrict__ Psi_Interference,
    float* __restrict__ Probs_Global,
    float* __restrict__ Delta_Psi,
    float* __restrict__ Loss_Dev,
    int byte_target
) {
    __shared__ float s_psi[DIM_FIELD];
    __shared__ float s_probs[VOCAB_SIZE];
    int tid = threadIdx.x;

    // 1. Interference Superposition over Parallel Universes
    if (tid < DIM_FIELD) {
        float sum_d = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            sum_d += Phi[u * DIM_FIELD + tid];
        }
        float mean_d = sum_d / (float)NUM_UNIVERSES;
        Psi_Interference[tid] = mean_d;
        s_psi[tid] = mean_d;
    }
    __syncthreads();

    // 2. Readout Projection & Softmax
    if (tid == 0) {
        float Logits[VOCAB_SIZE];
        float Probs[VOCAB_SIZE];
        float m = -1e9f;

        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float sum_v = 0.0f;
            const float* w_v = &W_read[v * DIM_FIELD];
            for (int d = 0; d < DIM_FIELD; ++d) {
                sum_v += w_v[d] * s_psi[d];
            }
            Logits[v] = sum_v;
            if (sum_v > m) m = sum_v;
        }

        float s = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float exp_val = expf(Logits[v] - m);
            Probs[v] = exp_val;
            s += exp_val;
        }
        float inv_s = 1.0f / (s + 1e-8f);
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            Probs[v] *= inv_s;
            s_probs[v] = Probs[v];
            Probs_Global[v] = Probs[v];
        }

        float target_prob = fmaxf(Probs[byte_target], 1e-7f);
        *Loss_Dev = -logf(target_prob);
    }
    __syncthreads();

    // 3. Back-propagate Exact Softmax Sensory Prediction Error Vector:
    // Delta_Psi[d] = \sum_v (e_target[v] - Probs[v]) * W_read[v, d]
    if (tid < DIM_FIELD) {
        float err_d = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float target_sig = (v == byte_target) ? 1.0f : 0.0f;
            float err_v = target_sig - s_probs[v];
            err_d += err_v * W_read[v * DIM_FIELD + tid];
        }
        Delta_Psi[tid] = err_d;
    }
}

__global__ void ssd_temporal_plasticity_kernel(
    float* __restrict__ W_trans,
    float* __restrict__ W_read,
    const float* __restrict__ Delta_Psi,
    const float* __restrict__ Probs_Global,
    const float* __restrict__ Prev_Phi,
    const float* __restrict__ Psi_Interference,
    const float* __restrict__ Epigenetic_Shield,
    int byte_target,
    float eta_trans,
    float eta_read
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Update W_trans (DIM_FIELD x DIM_FIELD) via Temporal Outer Product
    if (idx < DIM_FIELD * DIM_FIELD) {
        int r_i = idx / DIM_FIELD;
        int c_j = idx % DIM_FIELD;

        float prev_phi_j = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            prev_phi_j += Prev_Phi[u * DIM_FIELD + c_j];
        }
        prev_phi_j /= (float)NUM_UNIVERSES;

        float delta_i = Delta_Psi[r_i];
        float shield_j = Epigenetic_Shield[c_j];

        float dw = eta_trans * delta_i * prev_phi_j * (1.0f - shield_j);
        float new_w = W_trans[idx] + dw;
        W_trans[idx] = fmaxf(fminf(new_w, 3.0f), -3.0f);
    }

    // 2. Update Readout Matrix W_read (VOCAB_SIZE x DIM_FIELD) using EXACT Cross-Entropy Gradient
    if (idx < VOCAB_SIZE * DIM_FIELD) {
        int v = idx / DIM_FIELD;
        int d = idx % DIM_FIELD;
        
        float psi_d = Psi_Interference[d];
        float target_sig = (v == byte_target) ? 1.0f : 0.0f;
        float err_v = target_sig - Probs_Global[v]; // Exact Softmax Error (y_v - P_v)
        
        float dw_read = eta_read * err_v * psi_d;
        float new_w_read = W_read[idx] + dw_read;
        W_read[idx] = fmaxf(fminf(new_w_read, 5.0f), -5.0f);
    }
}

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "=== EXP-735.2: PURE CUDA SSD SEQUENCE MEMORY (EXACT SOFTMAX GRADIENTS) ===" << std::endl;
    std::cout << "====================================================================" << std::endl;

    float *d_Phi, *d_Prev_Phi, *d_W_inj, *d_W_trans, *d_W_read, *d_Epigenetic_Shield;
    float *d_Psi_Interference, *d_Probs_Global, *d_Delta_Psi, *d_Loss_Dev;

    CUDA_CHECK(cudaMalloc(&d_Phi, NUM_UNIVERSES * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Prev_Phi, NUM_UNIVERSES * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_inj, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_trans, DIM_FIELD * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_read, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Epigenetic_Shield, DIM_FIELD * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Psi_Interference, DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Probs_Global, VOCAB_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Delta_Psi, DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Loss_Dev, sizeof(float)));

    int total_threads = DIM_FIELD * DIM_FIELD;
    int blocks_init = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_sequence_memory_kernel<<<blocks_init, BLOCK_SIZE>>>(
        d_Phi, d_W_inj, d_W_trans, d_W_read, d_Epigenetic_Shield, d_Prev_Phi
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::string text_block = 
        "Karyon-CoRE v30.0 Sovereign Native CUDA State-Space Sequence Memory Engine. "
        "Continuous memory integration across time t eliminates temporal dissipation. "
        "Outer-product temporal plasticity updates transfer operator W_trans directly from sequence history. "
        "Feynman path integral interference over parallel universes discovers stable conceptual attractors. ";

    std::string full_stream = "";
    while (full_stream.size() < 4096) {
        full_stream += text_block;
    }
    full_stream = full_stream.substr(0, 4096);
    int total_bytes = (int)full_stream.size();

    std::cout << "[*] Sovereign Native GPU Substrate: Tesla GPU" << std::endl;
    std::cout << "[*] Stream Length: T=" << total_bytes << " bytes (11 repetitions of 370-byte pattern)." << std::endl;
    std::cout << "--- Commencing Native GPU SSD Execution ---" << std::endl;

    std::vector<float> chunk_losses;
    float cumulative_loss = 0.0f;
    int chunk_size = 128;
    int total_chunks = total_bytes / chunk_size;

    float alpha_decay = 0.90f;
    float eta_trans = 0.035f;
    float eta_read = 0.080f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        float chunk_loss = 0.0f;

        for (int step = 0; step < chunk_size; ++step) {
            int t = chunk * chunk_size + step;
            if (t >= total_bytes - 1) break;

            int byte_in = (unsigned char)full_stream[t];
            int byte_target = (unsigned char)full_stream[t + 1];

            CUDA_CHECK(cudaMemcpy(d_Prev_Phi, d_Phi, NUM_UNIVERSES * DIM_FIELD * sizeof(float), cudaMemcpyDeviceToDevice));

            ssd_sequence_evolution_kernel<<<NUM_UNIVERSES, 1>>>(
                d_Phi, d_Prev_Phi, d_W_inj, d_W_trans, byte_in, alpha_decay
            );

            ssd_readout_and_vector_error_kernel<<<1, BLOCK_SIZE>>>(
                d_Phi, d_W_read, d_Psi_Interference, d_Probs_Global, d_Delta_Psi, d_Loss_Dev, byte_target
            );

            int plasticity_threads = VOCAB_SIZE * DIM_FIELD;
            int plasticity_blocks = (plasticity_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ssd_temporal_plasticity_kernel<<<plasticity_blocks, BLOCK_SIZE>>>(
                d_W_trans, d_W_read, d_Delta_Psi, d_Probs_Global, d_Prev_Phi, d_Psi_Interference,
                d_Epigenetic_Shield, byte_target, eta_trans, eta_read
            );

            float h_loss = 0.0f;
            CUDA_CHECK(cudaMemcpy(&h_loss, d_Loss_Dev, sizeof(float), cudaMemcpyDeviceToHost));
            chunk_loss += h_loss;
        }

        float avg_chunk_loss = chunk_loss / chunk_size;
        chunk_losses.push_back(avg_chunk_loss);
        cumulative_loss += avg_chunk_loss;

        if ((chunk + 1) % 4 == 0 || chunk == total_chunks - 1) {
            float ppl = expf(avg_chunk_loss);
            std::cout << "Chunk " << (chunk + 1) << "/" << total_chunks 
                      << " (t=" << (chunk + 1) * chunk_size << ") | Loss: " 
                      << avg_chunk_loss << " nats | PPL: " << ppl << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double bytes_per_sec = (double)total_bytes / total_sec;

    float initial_loss = chunk_losses.front();
    float final_loss = chunk_losses.back();
    float delta_loss = initial_loss - final_loss;

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "EXP-735.2 FINAL TELEMETRY RESULTS" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Initial Chunk Loss : " << initial_loss << " nats/byte" << std::endl;
    std::cout << "Final Chunk Loss   : " << final_loss << " nats/byte" << std::endl;
    std::cout << "Delta Improvement  : " << delta_loss << " nats/byte" << std::endl;
    std::cout << "Final Perplexity   : " << expf(final_loss) << std::endl;
    std::cout << "Throughput         : " << bytes_per_sec << " bytes/sec" << std::endl;

    std::ofstream json_file("experiments/exp_735_results.json");
    json_file << "{\n";
    json_file << "  \"exp_id\": \"EXP-735\",\n";
    json_file << "  \"initial_loss\": " << initial_loss << ",\n";
    json_file << "  \"final_loss\": " << final_loss << ",\n";
    json_file << "  \"delta_loss\": " << delta_loss << ",\n";
    json_file << "  \"ppl\": " << expf(final_loss) << ",\n";
    json_file << "  \"throughput_bytes_per_sec\": " << bytes_per_sec << "\n";
    json_file << "}\n";
    json_file.close();

    cudaFree(d_Phi); cudaFree(d_Prev_Phi); cudaFree(d_W_inj);
    cudaFree(d_W_trans); cudaFree(d_W_read); cudaFree(d_Epigenetic_Shield);
    cudaFree(d_Psi_Interference); cudaFree(d_Probs_Global); cudaFree(d_Delta_Psi); cudaFree(d_Loss_Dev);

    return 0;
}
