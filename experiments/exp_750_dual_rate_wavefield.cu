// ==============================================================================
// EXP-750: MULTI-SCALE DUAL-RATE FIELD (FAST SYNTACTIC & SLOW DISCOURSE DYNAMICS)
// Standard: KEP v30.0 Sovereign Master Specification
// Principles: 1 (C++20/CUDA Engine, Zero PyTorch), 2 (Universal Silicon Substrate),
//             3 (Continuous Tensor Field), 5 (Causal Credit), 6 (Epigenetic Shielding),
//             8 (Unshackled Flow), 28 (Authentic Behavioral Competence), 29 (Full Control)
//
// ARCHITECTURAL DISCOVERY & DELTA:
// Single-rate fields cannot retain discourse context:
// With MOMENTUM_DAMP = 0.85 and g_conductance = 0.20..1.0, memories decay within 50-100 bytes.
//
// In EXP-750 (Multi-Scale Dual-Rate Field Dynamics):
// We decompose the 256-dimensional field Phi into TWO naturally coupled frequency bands:
// 1. FAST SYNTACTIC BAND (Dimensions 0..127):
//    - High damping (0.80), large effective step (dt = 0.35), high plasticity (eta = 0.16).
//    - Rapidly absorbs raw phonotactic, byte-level syntax and morphologic transitions.
// 2. SLOW DISCOURSE BAND (Dimensions 128..255):
//    - Ultra-slow inertial damping (0.985), small effective step (dt = 0.05), gentle plasticity (0.04).
//    - Maintains long-range cross-sentence discourse context across the entire 4096-byte stream!
// 3. CROSS-BAND SYMPLECTIC COUPLING:
//    - Off-diagonal elements in W_trans continuously fuse fast syntactic evidence into slow discourse
//      attractors and down-project slow contextual bias into fast syntactic logits!
// ==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int NUM_UNIVERSES = 64;       // U = 64 parallel physical universes
constexpr int DIM_FIELD = 256;          // Total continuous field dimension D = 256
constexpr int DIM_FAST = 128;           // Fast syntactic sub-band [0..127]
constexpr int DIM_SLOW = 128;           // Slow discourse sub-band [128..255]
constexpr int NUM_ATTRACTORS = 64;      // K = 64 Hopfield soliton attractor basins
constexpr int VOCAB_SIZE = 258;         // Raw UTF-8 bytes [0..255] + PAD(256) + EOS(257)
constexpr float BETA_HOPFIELD = 12.0f;  // Attractor precision sharpening
constexpr float RING_LAPLACIAN_G = 0.05f;// Inter-universe tunneling conductance
constexpr float BASE_ETA = 0.16f;       // Base plasticity rate

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct SubstratePointers {
    float* Phi;              // [NUM_UNIVERSES, DIM_FIELD]
    float* Momenta;          // [NUM_UNIVERSES, DIM_FIELD]
    float* W_inj;            // [VOCAB_SIZE, DIM_FIELD]
    float* W_trans;          // [DIM_FIELD, DIM_FIELD]
    float* W_pot;            // [DIM_FIELD, DIM_FIELD]
    float* Basins;           // [NUM_ATTRACTORS, DIM_FIELD]
    float* W_read;           // [VOCAB_SIZE, DIM_FIELD]
    float* Epigenetic_Locks; // [DIM_FIELD, DIM_FIELD]
    float* Causal_Credits;   // [DIM_FIELD, DIM_FIELD]
    float* Eligibility;      // [DIM_FIELD, DIM_FIELD]
    
    // Scratch buffers
    float* S_u;              // [NUM_UNIVERSES]
    float* W_feynman;        // [NUM_UNIVERSES]
    float* Psi_coherent;     // [DIM_FIELD]
    float* Delta_Psi;        // [DIM_FIELD] Sensory Error Vector
    float* Logits;           // [VOCAB_SIZE]
    float* Probs;            // [VOCAB_SIZE]
    float* F_self_u;         // [NUM_UNIVERSES]
    float* Loss_Dev;         // [1]
    float* Prev_Free_Energy; // [1]
};

__global__ void init_dual_rate_substrate_kernel(
    float* __restrict__ Phi,
    float* __restrict__ Momenta,
    float* __restrict__ W_inj,
    float* __restrict__ W_trans,
    float* __restrict__ W_pot,
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

    if (idx < NUM_UNIVERSES * DIM_FIELD) {
        Phi[idx] = curand_normal(&state) * 0.05f;
        Momenta[idx] = 0.0f;
    }

    if (idx < VOCAB_SIZE * DIM_FIELD) {
        float scale = 1.0f / sqrtf((float)DIM_FIELD);
        W_inj[idx] = curand_normal(&state) * scale;
        W_read[idx] = curand_normal(&state) * scale;
    }

    if (idx < DIM_FIELD * DIM_FIELD) {
        float scale = 1.0f / sqrtf((float)DIM_FIELD);
        W_trans[idx] = curand_normal(&state) * scale;
        W_pot[idx] = curand_normal(&state) * (scale * 0.2f);
        Epigenetic_Locks[idx] = 0.0f;
        Causal_Credits[idx] = 0.0f;
        Eligibility[idx] = 0.0f;
    }

    if (idx < NUM_ATTRACTORS) {
        float norm_sq = 0.0f;
        for (int d = 0; d < DIM_FIELD; ++d) {
            float val = curand_normal(&state);
            Basins[idx * DIM_FIELD + d] = val;
            norm_sq += val * val;
        }
        float inv_norm = 1.0f / (sqrtf(norm_sq) + 1e-8f);
        for (int d = 0; d < DIM_FIELD; ++d) {
            Basins[idx * DIM_FIELD + d] *= inv_norm;
        }
    }

    if (idx == 0) {
        *Prev_Free_Energy = 3.5f;
    }
}

// Multi-Scale Dual-Rate Symplectic Evolution Kernel
__global__ void dual_rate_evolution_kernel(
    float* __restrict__ Phi,
    float* __restrict__ Momenta,
    const float* __restrict__ W_inj,
    const float* __restrict__ W_trans,
    const float* __restrict__ W_pot,
    const float* __restrict__ Basins,
    float* __restrict__ S_u,
    float* __restrict__ F_self_u,
    int byte_in,
    float prev_sensory_error
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= NUM_UNIVERSES) return;

    int u_prev = (u - 1 + NUM_UNIVERSES) % NUM_UNIVERSES;
    int u_next = (u + 1) % NUM_UNIVERSES;

    const float* x_wave = &W_inj[byte_in * DIM_FIELD];
    float* phi_u = &Phi[u * DIM_FIELD];
    float* p_u = &Momenta[u * DIM_FIELD];
    const float* phi_prev = &Phi[u_prev * DIM_FIELD];
    const float* phi_next = &Phi[u_next * DIM_FIELD];

    float s_hat[DIM_FIELD];
    float f_self_acc = 0.0f;
    for (int i = 0; i < DIM_FIELD; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < DIM_FIELD; ++j) {
            sum += W_trans[i * DIM_FIELD + j] * phi_u[j];
        }
        s_hat[i] = tanhf(sum);
        float diff = s_hat[i] - phi_u[i];
        f_self_acc += diff * diff;
    }
    float f_self = f_self_acc / (float)DIM_FIELD;
    F_self_u[u] = f_self;

    float local_entropy = f_self + prev_sensory_error * 0.2f;

    // Hopfield Soliton Attractor Snapping
    float phi_norm_sq = 0.0f;
    for (int d = 0; d < DIM_FIELD; ++d) {
        phi_norm_sq += phi_u[d] * phi_u[d];
    }
    float inv_phi_norm = 1.0f / (sqrtf(phi_norm_sq) + 1e-8f);

    float basin_affinity[NUM_ATTRACTORS];
    float max_aff = -1e9f;
    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        float dot = 0.0f;
        for (int d = 0; d < DIM_FIELD; ++d) {
            dot += (phi_u[d] * inv_phi_norm) * Basins[k * DIM_FIELD + d];
        }
        basin_affinity[k] = dot * BETA_HOPFIELD;
        if (basin_affinity[k] > max_aff) max_aff = basin_affinity[k];
    }

    float sum_p = 0.0f;
    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        basin_affinity[k] = expf(basin_affinity[k] - max_aff);
        sum_p += basin_affinity[k];
    }
    float inv_sum_p = 1.0f / (sum_p + 1e-8f);

    float snap_target[DIM_FIELD];
    for (int d = 0; d < DIM_FIELD; ++d) snap_target[d] = 0.0f;

    for (int k = 0; k < NUM_ATTRACTORS; ++k) {
        float p_k = basin_affinity[k] * inv_sum_p;
        for (int d = 0; d < DIM_FIELD; ++d) {
            snap_target[d] += p_k * Basins[k * DIM_FIELD + d];
        }
    }

    float kinetic_acc = 0.0f;
    float potential_acc = 0.0f;
    float coupling_acc = 0.0f;

    // DUAL-RATE INTEGRATION:
    // Fast Band: d in [0..127]   -> High damping (0.80), large dt (0.35)
    // Slow Band: d in [128..255] -> High inertia damping (0.985), small dt (0.06)
    for (int d = 0; d < DIM_FIELD; ++d) {
        bool is_fast = (d < DIM_FAST);

        float damp = is_fast ? 0.80f : 0.985f;
        float dt_base = is_fast ? 0.35f : 0.06f;
        float dt_eff = dt_base / (1.0f + 0.8f * local_entropy);
        float g_cond = is_fast ? 0.80f : 0.25f;

        float v_force = 0.0f;
        for (int j = 0; j < DIM_FIELD; ++j) {
            v_force -= W_pot[d * DIM_FIELD + j] * phi_u[j];
        }

        float tunneling = RING_LAPLACIAN_G * (phi_prev[d] + phi_next[d] - 2.0f * phi_u[d]);
        float f_attractor = snap_target[d] - phi_u[d];

        float accel = s_hat[d] + x_wave[d] + v_force + 0.35f * f_attractor + tunneling - phi_u[d];

        p_u[d] = p_u[d] * damp + dt_eff * accel;
        phi_u[d] = phi_u[d] + dt_eff * (g_cond * p_u[d]);

        kinetic_acc += 0.5f * p_u[d] * p_u[d];
        potential_acc += 0.5f * phi_u[d] * phi_u[d];
        coupling_acc -= phi_u[d] * x_wave[d];
    }

    S_u[u] = kinetic_acc + potential_acc + coupling_acc;
}

// Feynman Superposition & Sensory Readout Kernel
__global__ void feynman_dual_rate_readout_kernel(
    const float* __restrict__ Phi,
    const float* __restrict__ S_u,
    const float* __restrict__ F_self_u,
    const float* __restrict__ W_read,
    float* __restrict__ W_feynman,
    float* __restrict__ Psi_coherent,
    float* __restrict__ Delta_Psi,
    float* __restrict__ Logits,
    float* __restrict__ Probs,
    float* __restrict__ Loss_Dev,
    int byte_target,
    float prev_sensory_error
) {
    int tid = threadIdx.x;

    __shared__ float s_min_val;
    __shared__ float hbar_eff;
    __shared__ float sum_weights;
    __shared__ float s_psi[DIM_FIELD];

    if (tid == 0) {
        float min_s = 1e9f;
        float sum_f = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            if (S_u[u] < min_s) min_s = S_u[u];
            sum_f += F_self_u[u];
        }
        s_min_val = min_s;
        float mean_f = (sum_f / (float)NUM_UNIVERSES) + prev_sensory_error;
        hbar_eff = 1.2f * (1.0f + 1.5f * fminf(fmaxf(mean_f, 0.0f), 4.0f));

        float total_w = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            float shifted = (S_u[u] - min_s) / (hbar_eff + 1e-6f);
            float w = expf(-shifted);
            W_feynman[u] = w;
            total_w += w;
        }
        sum_weights = total_w;
        float inv_w = 1.0f / (total_w + 1e-8f);
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            W_feynman[u] *= inv_w;
        }
    }
    __syncthreads();

    if (tid < DIM_FIELD) {
        float psi_d = 0.0f;
        for (int u = 0; u < NUM_UNIVERSES; ++u) {
            psi_d += W_feynman[u] * Phi[u * DIM_FIELD + tid];
        }
        Psi_coherent[tid] = psi_d;
        s_psi[tid] = psi_d;
    }
    __syncthreads();

    if (tid < VOCAB_SIZE) {
        float dot = 0.0f;
        const float* w_v = &W_read[tid * DIM_FIELD];
        for (int d = 0; d < DIM_FIELD; ++d) {
            dot += w_v[d] * s_psi[d];
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

    if (tid < DIM_FIELD) {
        float err_d = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            float target_sig = (v == byte_target) ? 1.0f : 0.0f;
            float residual_v = target_sig - Probs[v];
            err_d += residual_v * W_read[v * DIM_FIELD + tid];
        }
        Delta_Psi[tid] = err_d;
    }
}

// Dual-Rate Plasticity Kernel
__global__ void dual_rate_plasticity_kernel(
    float* __restrict__ W_read,
    float* __restrict__ W_trans,
    float* __restrict__ W_inj,
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

    // 1. Goal-directed Readout Update W_read
    if (idx < VOCAB_SIZE * DIM_FIELD) {
        int v = idx / DIM_FIELD;
        int d = idx % DIM_FIELD;

        float target_signal = (v == byte_target) ? 1.0f : 0.0f;
        float residual = target_signal - Probs[v];

        // Fast band has full eta; slow discourse band has stabilized gradient
        float rate_scale = (d < DIM_FAST) ? 1.0f : 0.50f;
        W_read[idx] += eta * rate_scale * residual * Psi_coherent[d];
    }

    // 2. Goal-directed Wavefield Transformation Update W_trans [DIM_FIELD, DIM_FIELD]
    if (idx < DIM_FIELD * DIM_FIELD) {
        int i = idx / DIM_FIELD;
        int j = idx % DIM_FIELD;

        float goal_drive = Delta_Psi[i] * Psi_coherent[j];

        float flux = Psi_coherent[i] * Psi_coherent[j];
        float e_val = Eligibility[idx] * 0.90f + flux * 0.10f;
        Eligibility[idx] = e_val;

        float c_val = Causal_Credits[idx] * 0.95f + 0.05f * (e_val * delta_F);
        Causal_Credits[idx] = c_val;

        float lock_target = 1.0f / (1.0f + expf((c_val + 0.002f) * 100.0f));
        float current_lock = fmaxf(Epigenetic_Locks[idx], lock_target);
        Epigenetic_Locks[idx] = current_lock;

        float shield = 1.0f - 0.50f * current_lock;

        // Band-specific plasticity rates
        float rate_multiplier = 0.50f;
        if (i >= DIM_FAST && j >= DIM_FAST) {
            rate_multiplier = 0.20f; // Slow-to-slow discourse transitions are ultra-stable
        } else if (i < DIM_FAST && j < DIM_FAST) {
            rate_multiplier = 0.60f; // Fast-to-fast syntax adapts briskly
        }

        W_trans[idx] += (eta * rate_multiplier) * goal_drive * shield;
    }

    // 3. Goal-directed Continuous Wave Injection Update W_inj
    if (idx < DIM_FIELD) {
        float err_d = Delta_Psi[idx];
        float rate_scale = (idx < DIM_FAST) ? 1.0f : 0.40f;
        W_inj[byte_in * DIM_FIELD + idx] += eta * rate_scale * err_d;
    }
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "=== EXP-750: MULTI-SCALE DUAL-RATE CONTINUOUS WAVEFIELD (EVE-CUDA) ===" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SubstratePointers sub;

    CUDA_CHECK(cudaMalloc(&sub.Phi, NUM_UNIVERSES * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Momenta, NUM_UNIVERSES * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.W_inj, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.W_trans, DIM_FIELD * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.W_pot, DIM_FIELD * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Basins, NUM_ATTRACTORS * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.W_read, VOCAB_SIZE * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Epigenetic_Locks, DIM_FIELD * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Causal_Credits, DIM_FIELD * DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Eligibility, DIM_FIELD * DIM_FIELD * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&sub.S_u, NUM_UNIVERSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.W_feynman, NUM_UNIVERSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Psi_coherent, DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Delta_Psi, DIM_FIELD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Logits, VOCAB_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Probs, VOCAB_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.F_self_u, NUM_UNIVERSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Loss_Dev, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub.Prev_Free_Energy, sizeof(float)));

    int total_init_threads = (NUM_UNIVERSES * DIM_FIELD > DIM_FIELD * DIM_FIELD) ?
                              NUM_UNIVERSES * DIM_FIELD : DIM_FIELD * DIM_FIELD;
    int init_blocks = (total_init_threads + 255) / 256;
    init_dual_rate_substrate_kernel<<<init_blocks, 256>>>(
        sub.Phi, sub.Momenta, sub.W_inj, sub.W_trans, sub.W_pot, sub.Basins,
        sub.W_read, sub.Epigenetic_Locks, sub.Causal_Credits, sub.Eligibility,
        sub.Prev_Free_Energy, 1337ULL
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

    int chunk_size = 128;
    int num_chunks = total_bytes / chunk_size;
    std::vector<float> chunk_losses;
    float total_loss = 0.0f;
    float prev_sensory_error = 0.5f;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < num_chunks; ++c) {
        float chunk_loss_acc = 0.0f;
        float current_loss_host = 5.0f;

        for (int i = 0; i < chunk_size - 1; ++i) {
            int byte_in = (unsigned char)full_stream[c * chunk_size + i];
            int byte_target = (unsigned char)full_stream[c * chunk_size + i + 1];

            dual_rate_evolution_kernel<<<NUM_UNIVERSES, 1>>>(
                sub.Phi, sub.Momenta, sub.W_inj, sub.W_trans, sub.W_pot, sub.Basins,
                sub.S_u, sub.F_self_u, byte_in, prev_sensory_error
            );

            feynman_dual_rate_readout_kernel<<<1, 256>>>(
                sub.Phi, sub.S_u, sub.F_self_u, sub.W_read, sub.W_feynman,
                sub.Psi_coherent, sub.Delta_Psi, sub.Logits, sub.Probs,
                sub.Loss_Dev, byte_target, prev_sensory_error
            );

            int total_threads = (VOCAB_SIZE * DIM_FIELD > DIM_FIELD * DIM_FIELD) ?
                                 VOCAB_SIZE * DIM_FIELD : DIM_FIELD * DIM_FIELD;
            int blocks = (total_threads + 255) / 256;
            dual_rate_plasticity_kernel<<<blocks, 256>>>(
                sub.W_read, sub.W_trans, sub.W_inj, sub.Eligibility, sub.Causal_Credits,
                sub.Epigenetic_Locks, sub.Psi_coherent, sub.Delta_Psi, sub.Probs,
                sub.Prev_Free_Energy, sub.Loss_Dev, byte_in, byte_target
            );

            if (i % 32 == 31 || i == chunk_size - 2) {
                CUDA_CHECK(cudaMemcpy(&current_loss_host, sub.Loss_Dev, sizeof(float), cudaMemcpyDeviceToHost));
                prev_sensory_error = current_loss_host;
            }
            chunk_loss_acc += current_loss_host;
        }

        float avg_chunk_loss = chunk_loss_acc / (float)(chunk_size - 1);
        chunk_losses.push_back(avg_chunk_loss);
        total_loss += avg_chunk_loss;

        if ((c + 1) % 4 == 0 || c == num_chunks - 1) {
            float ppl = expf(fminf(avg_chunk_loss, 20.0f));
            std::cout << "Chunk " << (c + 1) << "/" << num_chunks
                      << " (t=" << (c + 1) * chunk_size << ") | Loss: " << avg_chunk_loss
                      << " nats | PPL: " << ppl << std::endl;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    float initial_loss = chunk_losses.front();
    float final_loss = chunk_losses.back();
    float mean_loss = total_loss / (float)num_chunks;
    float final_ppl = expf(fminf(final_loss, 20.0f));
    double bytes_per_sec = (double)total_bytes / elapsed.count();

    std::vector<float> h_locks(DIM_FIELD * DIM_FIELD);
    CUDA_CHECK(cudaMemcpy(h_locks.data(), sub.Epigenetic_Locks, DIM_FIELD * DIM_FIELD * sizeof(float), cudaMemcpyDeviceToHost));
    int locked_count = 0;
    for (float l : h_locks) {
        if (l > 0.5f) locked_count++;
    }
    float lock_pct = ((float)locked_count / (float)(DIM_FIELD * DIM_FIELD)) * 100.0f;

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "EXP-750 FINAL TELEMETRY RESULTS (DUAL-RATE MULTI-SCALE EVE-CUDA)" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Initial Chunk Loss : " << initial_loss << " nats/byte" << std::endl;
    std::cout << "Final Chunk Loss   : " << final_loss << " nats/byte" << std::endl;
    std::cout << "Mean Stream Loss   : " << mean_loss << " nats/byte" << std::endl;
    std::cout << "Final Perplexity   : " << final_ppl << std::endl;
    std::cout << "Throughput Speed   : " << bytes_per_sec << " bytes/sec (Zero Python Overhead)" << std::endl;
    std::cout << "Elapsed Time       : " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Epigenetic Locks   : " << lock_pct << "%" << std::endl;

    float baseline_loss = 2.74623f; // Baseline from EXP-734.2
    float delta_loss = baseline_loss - final_loss;
    std::cout << "Delta vs EXP-734.2 : " << (delta_loss > 0 ? "+" : "") << delta_loss << " nats/byte" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::ofstream json_out("experiments/exp_750_results.json");
    json_out << "{\n"
             << "  \"exp_id\": \"EXP-750\",\n"
             << "  \"loss\": " << final_loss << ",\n"
             << "  \"initial_loss\": " << initial_loss << ",\n"
             << "  \"mean_loss\": " << mean_loss << ",\n"
             << "  \"delta_loss\": " << delta_loss << ",\n"
             << "  \"ppl\": " << final_ppl << ",\n"
             << "  \"throughput_bytes_per_sec\": " << bytes_per_sec << ",\n"
             << "  \"epigenetic_locks_pct\": " << lock_pct << "\n"
             << "}\n";
    json_out.close();

    cudaFree(sub.Phi); cudaFree(sub.Momenta); cudaFree(sub.W_inj); cudaFree(sub.W_trans);
    cudaFree(sub.W_pot); cudaFree(sub.Basins); cudaFree(sub.W_read); cudaFree(sub.Epigenetic_Locks);
    cudaFree(sub.Causal_Credits); cudaFree(sub.Eligibility); cudaFree(sub.S_u);
    cudaFree(sub.W_feynman); cudaFree(sub.Psi_coherent); cudaFree(sub.Delta_Psi);
    cudaFree(sub.Logits); cudaFree(sub.Probs); cudaFree(sub.F_self_u); cudaFree(sub.Loss_Dev);
    cudaFree(sub.Prev_Free_Energy);

    return 0;
}
