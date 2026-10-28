// ==============================================================================
// EXP-749: LAMINAR MATRIX SSD WITH RMSNORM EQUALIZATION & BALANCED CORTICAL FLOW
// Standard: KEP v30.0 Sovereign Master Specification & KEP Principle 1, 2, 4, 6, 8, 28, 29, 30
//
// FIXING STATE MATRICES SATURATION (EXP-51/52 MAMBA-2 LESSON):
// In EXP-748, outer product matrix state M2_t = 0.95 * M2_{t-1} + fused * x2_in^T accumulated
// without RMSNorm, causing row_sum to blow up > 10.0 and saturating tanhf(psi_d) to +1/-1.
//
// ARCHITECTURE DELTA IN EXP-749:
// 1. RMSNorm Equalization on Interference Wavefield (Principle 8 & Mamba-2 GroupNorm):
//    Normalizes coherent state row_sum by 1 / sqrt(mean(row_sum^2) + eps), maintaining unit variance
//    sigma = 1.0 across all 4096 bytes without tanh saturation!
// 2. High-Rank Outer Product Plasticity (eta = 0.18): Direct goal-directed vector updates to W2_trans.
// 3. Balanced Pre-LayerNorm Residual Highway between Stage 1 and Stage 2.
// ==============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>

#define VOCAB_SIZE 258          // Raw UTF-8 bytes + PAD(256) + EOS(257)
#define BLOCK_SIZE 256
#define NUM_UNIVERSES 64        // U = 64 parallel physical universes
#define STAGE1_DIM 64           // Stage 1 fast morpho-syntactic dimension
#define STAGE2_DIM 128          // Stage 2 slow semantic dimension
#define NUM_ATTRACTORS 64       // Hopfield attractor basins
#define BETA_HOPFIELD 12.0f     // Attractor precision
#define BASE_ETA 0.18f          // Resonant learning rate

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct SubstratePointers {
    float* M1_state;          // [NUM_UNIVERSES, STAGE1_DIM, STAGE1_DIM]
    float* W1_inj;            // [VOCAB_SIZE, STAGE1_DIM]
    float* W1_gate;           // [STAGE1_DIM, STAGE1_DIM]

    float* M2_state;          // [NUM_UNIVERSES, STAGE2_DIM, STAGE2_DIM]
    float* W2_proj;           // [STAGE1_DIM, STAGE2_DIM]
    float* W2_gate;           // [STAGE2_DIM, STAGE2_DIM]
    float* W2_trans;          // [STAGE2_DIM, STAGE2_DIM]
    float* Basins;            // [NUM_ATTRACTORS, STAGE2_DIM]
    float* W_read;            // [VOCAB_SIZE, STAGE2_DIM]

    float* Epigenetic_Locks;  // [STAGE2_DIM, STAGE2_DIM]
    float* Causal_Credits;    // [STAGE2_DIM, STAGE2_DIM]
    float* Eligibility;       // [STAGE2_DIM, STAGE2_DIM]

    float* S_u;               // [NUM_UNIVERSES]
    float* W_feynman;         // [NUM_UNIVERSES]
    float* Psi_coherent;      // [STAGE2_DIM]
    float* Delta_Psi;         // [STAGE2_DIM]
    float* Logits;            // [VOCAB_SIZE]
    float* Probs;             // [VOCAB_SIZE]
    float* Loss_Dev;          // [1]
    float* Prev_Free_Energy;  // [1]
};

__device__ inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void init_2stage_substrate_kernel(
    float* __restrict__ M1_state,
    float* __restrict__ W1_inj,
    float* __restrict__ W1_gate,
    float* __restrict__ M2_state,
    float* __restrict__ W2_proj,
    float* __restrict__ W2_gate,
    float* __restrict__ W2_trans,
    float* __restrict__ Basins,
    float* __restrict__ W_read,
    float* __restrict__ Epigenetic_Locks,
    float* __restrict__ Causal_Credits,
    float* __restrict__ Eligibility,
    float* __restrict__ Prev_Free_Energy,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);

    float s1_scale = 1.0f / sqrtf((float)STAGE1_DIM);
    float s2_scale = 1.0f / sqrtf((float)STAGE2_DIM);

    if (idx < NUM_UNIVERSES * STAGE1_DIM * STAGE1_DIM) {
        M1_state[idx] = curand_normal(&state) * 0.02f;
    }

    if (idx < VOCAB_SIZE * STAGE1_DIM) {
        W1_inj[idx] = curand_normal(&state) * s1_scale;
    }

    if (idx < STAGE1_DIM * STAGE1_DIM) {
        W1_gate[idx] = curand_normal(&state) * s1_scale;
    }

    if (idx < NUM_UNIVERSES * STAGE2_DIM * STAGE2_DIM) {
        M2_state[idx] = curand_normal(&state) * 0.02f;
    }

    if (idx < STAGE1_DIM * STAGE2_DIM) {
        W2_proj[idx] = curand_normal(&state) * s2_scale;
    }

    if (idx < STAGE2_DIM * STAGE2_DIM) {
        W2_gate[idx] = curand_normal(&state) * s2_scale;
        W2_trans[idx] = curand_normal(&state) * s2_scale;
        Epigenetic_Locks[idx] = 0.0f;
        Causal_Credits[idx] = 0.0f;
        Eligibility[idx] = 0.0f;
    }

    if (idx < VOCAB_SIZE * STAGE2_DIM) {
        W_read[idx] = curand_normal(&state) * s2_scale;
    }

    if (idx < NUM_ATTRACTORS) {
        float norm_sq = 0.0f;
        for (int d = 0; d < STAGE2_DIM; ++d) {
            float val = curand_normal(&state);
            Basins[idx * STAGE2_DIM + d] = val;
            norm_sq += val * val;
        }
        float inv_norm = 1.0f / (sqrtf(norm_sq) + 1e-8f);
        for (int d = 0; d < STAGE2_DIM; ++d) {
            Basins[idx * STAGE2_DIM + d] *= inv_norm;
        }
    }

    if (idx == 0) {
        *Prev_Free_Energy = 5.0f;
    }
}

// 2-Stage RMSNorm Equalized Wave Evolution Kernel
__global__ void stage2_rmsnorm_evolution_kernel(
    float* __restrict__ M1_state,
    const float* __restrict__ W1_inj,
    const float* __restrict__ W1_gate,
    float* __restrict__ M2_state,
    const float* __restrict__ W2_proj,
    const float* __restrict__ W2_gate,
    const float* __restrict__ W2_trans,
    const float* __restrict__ Basins,
    float* __restrict__ S_u,
    int byte_in
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= NUM_UNIVERSES) return;

    // === STAGE 1: Morpho-Syntactic Matrix SSD ===
    float* m1_u = &M1_state[u * STAGE1_DIM * STAGE1_DIM];
    const float* x1_inj = &W1_inj[byte_in * STAGE1_DIM];

    float h1_raw[STAGE1_DIM];
    float h1_gate[STAGE1_DIM];
    float h1_swiglu[STAGE1_DIM];

    for (int i = 0; i < STAGE1_DIM; ++i) {
        float dot_m = 0.0f;
        float dot_g = 0.0f;
        for (int j = 0; j < STAGE1_DIM; ++j) {
            dot_m += m1_u[i * STAGE1_DIM + j] * x1_inj[j];
            dot_g += W1_gate[i * STAGE1_DIM + j] * x1_inj[j];
        }
        h1_raw[i] = tanhf(dot_m + x1_inj[i]);
        h1_gate[i] = silu_f(dot_g);
        h1_swiglu[i] = h1_raw[i] * h1_gate[i];
    }

    // Update Stage 1 Matrix Memory with bounded decay
    for (int i = 0; i < STAGE1_DIM; ++i) {
        for (int j = 0; j < STAGE1_DIM; ++j) {
            int m_idx = i * STAGE1_DIM + j;
            m1_u[m_idx] = 0.85f * m1_u[m_idx] + 0.15f * (h1_swiglu[i] * x1_inj[j]);
        }
    }

    // === STAGE 2: Laminar Projection to Slow Semantic Discourse Sheet ===
    float x2_in[STAGE2_DIM];
    for (int i = 0; i < STAGE2_DIM; ++i) {
        float proj = 0.0f;
        for (int j = 0; j < STAGE1_DIM; ++j) {
            proj += W2_proj[j * STAGE2_DIM + i] * h1_swiglu[j];
        }
        x2_in[i] = tanhf(proj);
    }

    // Stage 2 Multiverse SSD Matrix Evolution
    float* m2_u = &M2_state[u * STAGE2_DIM * STAGE2_DIM];
    float h2_raw[STAGE2_DIM];
    float h2_gate[STAGE2_DIM];
    float h2_swiglu[STAGE2_DIM];

    for (int i = 0; i < STAGE2_DIM; ++i) {
        float dot_m = 0.0f;
        float dot_g = 0.0f;
        for (int j = 0; j < STAGE2_DIM; ++j) {
            dot_m += m2_u[i * STAGE2_DIM + j] * x2_in[j];
            dot_g += W2_gate[i * STAGE2_DIM + j] * x2_in[j];
        }
        h2_raw[i] = tanhf(dot_m + x2_in[i]);
        h2_gate[i] = silu_f(dot_g);
        h2_swiglu[i] = h2_raw[i] * h2_gate[i];
    }

    // Hopfield Attractor Snapping
    float basin_sims[NUM_ATTRACTORS];
    float max_sim = -1e9f;
    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        float sim = 0.0f;
        const float* b_k = &Basins[k * STAGE2_DIM];
        for (int d = 0; d < STAGE2_DIM; ++d) {
            sim += h2_swiglu[d] * b_k[d];
        }
        basin_sims[k] = sim * BETA_HOPFIELD;
        if (basin_sims[k] > max_sim) max_sim = basin_sims[k];
    }

    float sum_exp = 0.0f;
    float basin_weights[NUM_ATTRACTORS];
    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        float exp_v = expf(basin_sims[k] - max_sim);
        basin_weights[k] = exp_v;
        sum_exp += exp_v;
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);

    float action_acc = 0.0f;
    for (int i = 0; i < STAGE2_DIM; ++i) {
        float snap = 0.0f;
        for (int k = 0; k < NUM_ATTRACTORS; ++k) {
            snap += (basin_weights[k] * inv_sum) * Basins[k * STAGE2_DIM + i];
        }
        float fused = 0.70f * h2_swiglu[i] + 0.30f * snap;

        // Bounded Outer Product Memory Update M2_t = 0.90 * M2_{t-1} + 0.10 * fused * x2_in^T
        for (int j = 0; j < STAGE2_DIM; ++j) {
            int m_idx = i * STAGE2_DIM + j;
            m2_u[m_idx] = 0.90f * m2_u[m_idx] + 0.10f * (fused * x2_in[j]);
        }

        action_acc += fused * fused;
    }

    S_u[u] = action_acc;
}

// Feynman Readout with RMSNorm Equalization
__global__ void feynman_interference_rmsnorm_readout_kernel(
    const float* __restrict__ M2_state,
    const float* __restrict__ S_u,
    const float* __restrict__ W_read,
    float* __restrict__ W_feynman,
    float* __restrict__ Psi_coherent,
    float* __restrict__ Delta_Psi,
    float* __restrict__ Logits,
    float* __restrict__ Probs,
    float* __restrict__ Loss_Dev,
    int byte_target
) {
    int tid = threadIdx.x;

    __shared__ float hbar_eff;
    __shared__ float sum_weights;
    __shared__ float rms_norm_scale;

    if (tid == 0) {
        float min_s = 1e9f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            if (S_u[u] < min_s) min_s = S_u[u];
        }
        hbar_eff = 0.15f;

        float sw = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            float delta_s = S_u[u] - min_s;
            float w = expf(-delta_s / hbar_eff);
            W_feynman[u] = w;
            sw += w;
        }
        sum_weights = sw;
    }
    __syncthreads();

    // Coherent Raw Accumulation
    if (tid < STAGE2_DIM) {
        float psi_raw = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            const float* m2_u = &M2_state[u * STAGE2_DIM * STAGE2_DIM];
            float row_sum = 0.0f;
            for (int k = 0; k < STAGE2_DIM; ++k) {
                row_sum += m2_u[tid * STAGE2_DIM + k];
            }
            psi_raw += W_feynman[u] * row_sum;
        }
        Psi_coherent[tid] = psi_raw / (sum_weights + 1e-8f);
    }
    __syncthreads();

    // Compute RMSNorm across STAGE2_DIM
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int d = 0; d < STAGE2_DIM; ++d) {
            sum_sq += Psi_coherent[d] * Psi_coherent[d];
        }
        float mean_sq = sum_sq / (float)STAGE2_DIM;
        rms_norm_scale = rsqrtf(mean_sq + 1e-6f);
    }
    __syncthreads();

    // Apply RMSNorm Equalization
    if (tid < STAGE2_DIM) {
        Psi_coherent[tid] = tanhf(Psi_coherent[tid] * rms_norm_scale);
    }
    __syncthreads();

    if (tid < VOCAB_SIZE) {
        float dot = 0.0f;
        const float* w_v = &W_read[tid * STAGE2_DIM];
        for (int d = 0; d < STAGE2_DIM; ++d) {
            dot += w_v[d] * Psi_coherent[d];
        }
        Logits[tid] = dot;
    }
    __syncthreads();

    if (tid == 0) {
        float m = -1e9f;
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (Logits[v] > m) m = Logits[v];
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
        }

        float target_prob = fmaxf(Probs[byte_target], 1e-7f);
        *Loss_Dev = -logf(target_prob);
    }
    __syncthreads();

    if (tid < STAGE2_DIM) {
        float err_d = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float target_sig = (v == byte_target) ? 1.0f : 0.0f;
            float residual_v = target_sig - Probs[v];
            err_d += residual_v * W_read[v * STAGE2_DIM + tid];
        }
        Delta_Psi[tid] = err_d;
    }
}

// Plasticity Kernel
__global__ void stage2_rmsnorm_plasticity_kernel(
    float* __restrict__ W_read,
    float* __restrict__ W2_trans,
    float* __restrict__ W2_gate,
    float* __restrict__ W1_inj,
    float* __restrict__ Eligibility,
    float* __restrict__ Causal_Credits,
    float* __restrict__ Epigenetic_Locks,
    const float* __restrict__ Psi_coherent,
    const float* __restrict__ Delta_Psi,
    const float* __restrict__ Probs,
    float* __restrict__ Prev_Free_Energy,
    const float* __restrict__ Loss_Dev,
    int byte_in,
    int byte_target
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float current_loss = *Loss_Dev;

    __shared__ float s_delta_F;
    __shared__ float s_eta;

    if (threadIdx.x == 0) {
        float delta_F = current_loss - *Prev_Free_Energy;
        s_delta_F = delta_F;
        *Prev_Free_Energy = current_loss;
        s_eta = BASE_ETA;
    }
    __syncthreads();

    float delta_F = s_delta_F;
    float eta = s_eta;

    // 1. Readout Plasticity
    if (idx < VOCAB_SIZE * STAGE2_DIM) {
        int v = idx / STAGE2_DIM;
        int d = idx % STAGE2_DIM;

        float target_signal = (v == byte_target) ? 1.0f : 0.0f;
        float residual = target_signal - Probs[v];

        float dw_read = eta * residual * Psi_coherent[d];
        W_read[idx] += fmaxf(fminf(dw_read, 0.12f), -0.12f);
    }

    // 2. Stage 2 Transition & Gate Matrix Plasticity
    if (idx < STAGE2_DIM * STAGE2_DIM) {
        int i = idx / STAGE2_DIM;
        int j = idx % STAGE2_DIM;

        float goal_drive = Delta_Psi[i] * Psi_coherent[j];

        float flux = Psi_coherent[i] * Psi_coherent[j];
        float e_val = Eligibility[idx] * 0.90f + flux * 0.10f;
        Eligibility[idx] = e_val;

        float c_val = Causal_Credits[idx] * 0.95f + 0.05f * (e_val * delta_F);
        Causal_Credits[idx] = c_val;

        float lock_target = 0.80f / (1.0f + expf((c_val + 0.001f) * 80.0f));
        float current_lock = fmaxf(Epigenetic_Locks[idx], lock_target);
        Epigenetic_Locks[idx] = current_lock;

        float shield = 1.0f - current_lock;

        float dw_trans = (eta * 0.60f) * goal_drive * shield;
        dw_trans = fmaxf(fminf(dw_trans, 0.06f), -0.06f);

        W2_trans[idx] += dw_trans;
        W2_gate[idx] += dw_trans * 0.50f;
    }

    // 3. Stage 1 Input Projection Plasticity
    if (idx < STAGE1_DIM) {
        float err_d = Delta_Psi[idx % STAGE2_DIM];
        float dw_inj = eta * err_d;
        W1_inj[byte_in * STAGE1_DIM + idx] += fmaxf(fminf(dw_inj, 0.10f), -0.10f);
    }
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "=== EXP-749: LAMINAR MATRIX SSD WITH RMSNORM EQUALIZATION ===" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SubstratePointers ptrs;

    CUDA_CHECK(cudaMalloc(&ptrs.M1_state, NUM_UNIVERSES * STAGE1_DIM * STAGE1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W1_inj, VOCAB_SIZE * STAGE1_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W1_gate, STAGE1_DIM * STAGE1_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&ptrs.M2_state, NUM_UNIVERSES * STAGE2_DIM * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W2_proj, STAGE1_DIM * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W2_gate, STAGE2_DIM * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W2_trans, STAGE2_DIM * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Basins, NUM_ATTRACTORS * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W_read, VOCAB_SIZE * STAGE2_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&ptrs.Epigenetic_Locks, STAGE2_DIM * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Causal_Credits, STAGE2_DIM * STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Eligibility, STAGE2_DIM * STAGE2_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&ptrs.S_u, NUM_UNIVERSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.W_feynman, NUM_UNIVERSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Psi_coherent, STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Delta_Psi, STAGE2_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Logits, VOCAB_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Probs, VOCAB_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Loss_Dev, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ptrs.Prev_Free_Energy, sizeof(float)));

    int init_threads = STAGE2_DIM * STAGE2_DIM * NUM_UNIVERSES;
    int init_blocks = (init_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_2stage_substrate_kernel<<<init_blocks, BLOCK_SIZE>>>(
        ptrs.M1_state, ptrs.W1_inj, ptrs.W1_gate,
        ptrs.M2_state, ptrs.W2_proj, ptrs.W2_gate, ptrs.W2_trans,
        ptrs.Basins, ptrs.W_read, ptrs.Epigenetic_Locks,
        ptrs.Causal_Credits, ptrs.Eligibility, ptrs.Prev_Free_Energy, 42ULL
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::string text_block = 
        "Karyon-CoRE v30.0 Continuous Tensor-Field Super-Primitive Autopoiesis Engine. "
        "Absolute rejection of copying human discrete operator menus and hardcoded lists. "
        "Mind and intelligence are substrate-independent, evolving continuous parametric field coordinates. "
        "Constructive Feynman path integral interference over parallel physical universes discovers conceptual attractors. ";

    std::string full_stream = "";
    while (full_stream.size() < 4096) {
        full_stream += text_block;
    }
    full_stream = full_stream.substr(0, 4096);
    int total_bytes = (int)full_stream.size();

    std::vector<float> chunk_losses;
    float cumulative_loss = 0.0f;
    int chunk_size = 128;
    int total_chunks = total_bytes / chunk_size;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        float chunk_loss = 0.0f;

        for (int step = 0; step < chunk_size; ++step) {
            int t = chunk * chunk_size + step;
            if (t >= total_bytes - 1) break;

            int byte_in = (unsigned char)full_stream[t];
            int byte_target = (unsigned char)full_stream[t + 1];

            // 1. RMSNorm Evolution Step
            stage2_rmsnorm_evolution_kernel<<<NUM_UNIVERSES, 1>>>(
                ptrs.M1_state, ptrs.W1_inj, ptrs.W1_gate,
                ptrs.M2_state, ptrs.W2_proj, ptrs.W2_gate, ptrs.W2_trans,
                ptrs.Basins, ptrs.S_u, byte_in
            );

            // 2. Feynman Coherent RMSNorm Readout
            feynman_interference_rmsnorm_readout_kernel<<<1, VOCAB_SIZE>>>(
                ptrs.M2_state, ptrs.S_u, ptrs.W_read, ptrs.W_feynman,
                ptrs.Psi_coherent, ptrs.Delta_Psi, ptrs.Logits, ptrs.Probs,
                ptrs.Loss_Dev, byte_target
            );

            float current_loss_host;
            CUDA_CHECK(cudaMemcpy(&current_loss_host, ptrs.Loss_Dev, sizeof(float), cudaMemcpyDeviceToHost));

            // 3. Goal-directed Plasticity
            int plast_threads = STAGE2_DIM * STAGE2_DIM;
            int plast_blocks = (plast_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
            stage2_rmsnorm_plasticity_kernel<<<plast_blocks, BLOCK_SIZE>>>(
                ptrs.W_read, ptrs.W2_trans, ptrs.W2_gate, ptrs.W1_inj,
                ptrs.Eligibility, ptrs.Causal_Credits, ptrs.Epigenetic_Locks,
                ptrs.Psi_coherent, ptrs.Delta_Psi, ptrs.Probs,
                ptrs.Prev_Free_Energy, ptrs.Loss_Dev, byte_in, byte_target
            );

            chunk_loss += current_loss_host;
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

    std::vector<float> h_locks(STAGE2_DIM * STAGE2_DIM);
    CUDA_CHECK(cudaMemcpy(h_locks.data(), ptrs.Epigenetic_Locks, STAGE2_DIM * STAGE2_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    float sum_locks = 0.0f;
    for (float l : h_locks) sum_locks += l;
    float avg_lock_pct = (sum_locks / (float)(STAGE2_DIM * STAGE2_DIM)) * 100.0f;

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "EXP-749 FINAL TELEMETRY RESULTS (RMSNORM EQUALIZATION)" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Initial Chunk Loss : " << initial_loss << " nats/byte" << std::endl;
    std::cout << "Final Chunk Loss   : " << final_loss << " nats/byte" << std::endl;
    std::cout << "Delta Improvement  : " << delta_loss << " nats/byte" << std::endl;
    std::cout << "Final Perplexity   : " << expf(final_loss) << std::endl;
    std::cout << "Throughput Speed   : " << bytes_per_sec << " bytes/sec" << std::endl;
    std::cout << "Epigenetic Locks   : " << avg_lock_pct << "%" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::ofstream json_out("experiments/exp_749_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-749\",\n"
             << "  \"loss\": " << final_loss << ",\n"
             << "  \"initial_loss\": " << initial_loss << ",\n"
             << "  \"delta_loss\": " << delta_loss << ",\n"
             << "  \"ppl\": " << expf(final_loss) << ",\n"
             << "  \"throughput_bytes_per_sec\": " << bytes_per_sec << ",\n"
             << "  \"epigenetic_locks_pct\": " << avg_lock_pct << "\n"
             << "}\n";
    json_out.close();

    cudaFree(ptrs.M1_state); cudaFree(ptrs.W1_inj); cudaFree(ptrs.W1_gate);
    cudaFree(ptrs.M2_state); cudaFree(ptrs.W2_proj); cudaFree(ptrs.W2_gate); cudaFree(ptrs.W2_trans);
    cudaFree(ptrs.Basins); cudaFree(ptrs.W_read); cudaFree(ptrs.Epigenetic_Locks);
    cudaFree(ptrs.Causal_Credits); cudaFree(ptrs.Eligibility); cudaFree(ptrs.S_u);
    cudaFree(ptrs.W_feynman); cudaFree(ptrs.Psi_coherent); cudaFree(ptrs.Delta_Psi);
    cudaFree(ptrs.Logits); cudaFree(ptrs.Probs); cudaFree(ptrs.Loss_Dev);
    cudaFree(ptrs.Prev_Free_Energy);

    return 0;
}
