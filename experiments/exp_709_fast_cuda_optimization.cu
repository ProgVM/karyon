// ============================================================================
// EXP-709: CUDA Kernel Compilation & Execution Optimization (CKEO)
// Master Protocol: KEP v30.0 Substrate Specification (Principle 1)
//
// Objective: Eliminate the 2000s compilation/execution bottleneck in NVCC.
// Root Causes Identified:
// 1. `#pragma unroll` on inner 64x16 double loops forced NVCC to unroll 1024 spatial
//    iterations per tick, generating massive register spill and taking 30 minutes to compile!
// 2. Linear scan of 24,757 stream obstacles per thread was causing uncoalesced memory stalls.
//
// Optimizations Applied in EXP-709:
// 1. Removed aggressive `#pragma unroll` from D1 (64) and MAX_D2 (16) loops.
// 2. Added `-use_fast_math -Xptxas -O3` flags for ultra-fast GPU assembly generation.
// 3. Fast binary search indexing for Bloodbath stream obstacles.
// ============================================================================

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int MAX_UNIVERSES = 512;        // Multiverse capacity
constexpr int INITIAL_UNIVERSES = 128;    // Unshackled start
constexpr int D1 = 64;                    // Spatial longitudinal horizon X
constexpr int MAX_D2 = 16;                // Transverse spatial height capacity Y
constexpr int TOTAL_FIELD_DIM = D1 * MAX_D2; // 1024 degrees of freedom
constexpr int LAW_SPECTRAL_DIM = 4 * MAX_D2; // 64 parameters per universe
constexpr int METRIC_TENSOR_DIM = 4;      // [g_xx, g_yy, g_xy, alpha_dim]
constexpr int NUM_SYMBOLS = 4;            // S0: Vacuum, S1: Spike, S2: Platform, S3: Ceiling
constexpr int TICKS = 8192;               // Asymptotic Horizon
constexpr int TOTAL_OBSTACLES = 24757;    // Bloodbath level obstacle count
constexpr float BASE_DT = 0.005f;         // Base proper time
constexpr float SYMPLECTIC_GAMMA = 0.05f; // Inter-universe transport strength
constexpr int LOOKAHEAD_HORIZON = 12;     // ~60 ms future window
constexpr float DX_HORIZON = 2.0f;        // Continuous spatial step size in X

// Physical Agent Parameters:
constexpr float GRAVITY = -450.0f;        
constexpr float JUMP_IMPULSE = +320.0f;   
constexpr float AGENT_RADIUS = 0.45f;     

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Obstacle {
    float x;
    float y;
    float oid;
};

__device__ inline int map_y_to_k_dev(float y) {
    float dy = 0.02f * (y - 100.0f);
    float norm = asinhf(dy) * 0.26178f; // 1.0 / 3.82f
    int k = (int)((norm + 0.5f) * 10.0f); // 15.0 / 1.5
    if (k < 0) k = 0;
    if (k >= MAX_D2) k = MAX_D2 - 1;
    return k;
}

__device__ inline float map_k_to_y_dev(int k) {
    float norm = ((float)k * 0.066667f) * 1.5f - 0.5f;
    float dy = sinhf(norm * 3.82f);
    return 100.0f + dy * 50.0f;
}

__device__ inline void sample_2d_level_geometry_fast(
    const Obstacle* __restrict__ StreamData,
    float target_x_start,
    float g_u,
    float j_2d[D1][MAX_D2]
) {
    for (int i = 0; i < D1; ++i) {
        for (int k = 0; k < MAX_D2; ++k) {
            j_2d[i][k] = 0.0f;
        }
    }

    for (int i = 0; i < D1; ++i) {
        float sample_x = target_x_start + i * DX_HORIZON;

        int low = 0;
        int high = TOTAL_OBSTACLES - 1;

        while (low < high - 1) {
            int mid = (low + high) >> 1;
            if (StreamData[mid].x <= sample_x) {
                low = mid;
            } else {
                high = mid;
            }
        }

        int scan_start = (low >= 3) ? (low - 3) : 0;
        int scan_end = (low + 3 < TOTAL_OBSTACLES) ? (low + 3) : (TOTAL_OBSTACLES - 1);

        for (int idx = scan_start; idx <= scan_end; ++idx) {
            Obstacle obs = StreamData[idx];
            float dist_x = fabsf(obs.x - sample_x);

            if (dist_x < 3.0f) {
                int k_layer = map_y_to_k_dev(obs.y);
                float dy = 0.02f * (obs.y - 100.0f);
                float intensity = g_u * (asinhf(dy) + 0.001f * obs.oid);
                float x_weight = expf(-0.5f * dist_x * dist_x);

                j_2d[i][k_layer] += intensity * x_weight;
            }
        }
    }
}

__device__ inline float wrap_angle_dev(float angle) {
    while (angle > 3.14159265f) angle -= 6.28318530f;
    while (angle < -3.14159265f) angle += 6.28318530f;
    return angle;
}

// ----------------------------------------------------------------------------
// Optimized CUDA Kernel: No aggressive pragma unrolls on 1024 spatial loops!
// ----------------------------------------------------------------------------
__global__ void fast_topological_symbolic_genesis_kernel(
    float* __restrict__ Field2D,               
    float* __restrict__ Momenta2D,             
    float* __restrict__ AntiGaugeField2D,      
    float* __restrict__ ScreeningField2D,      
    float* __restrict__ AnticipationField2D,   
    float* __restrict__ SpectralLaws,          
    float* __restrict__ MetricTensor,          
    float* __restrict__ MetricScale,           
    float* __restrict__ UniverseVelocity,      
    float* __restrict__ EndogenousClock,       
    const Obstacle* __restrict__ StreamData,   
    int* __restrict__ ActiveCountIO,           
    float* __restrict__ AgentXOut,             
    float* __restrict__ AgentYOut,             
    float* __restrict__ EvasionRateOut,        
    float* __restrict__ DiscoveredSymbolsOut,  
    float* __restrict__ FreeEnergyOut,         
    float* __restrict__ OrderIntegrityOut,     
    curandState* __restrict__ RandStates,
    int ticks
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int current_active = *ActiveCountIO;
    if (u >= current_active) return;

    curandState local_rand = RandStates[u];
    int offset = u * TOTAL_FIELD_DIM;
    int law_offset = u * LAW_SPECTRAL_DIM;
    int metric_offset = u * METRIC_TENSOR_DIM;

    float phi[D1][MAX_D2];
    float pi[D1][MAX_D2];
    float phi_bar[D1][MAX_D2];
    float chi_screen[D1][MAX_D2];
    float zeta_antic[D1][MAX_D2];

    for (int i = 0; i < D1; ++i) {
        for (int k = 0; k < MAX_D2; ++k) {
            int idx = offset + i * MAX_D2 + k;
            phi[i][k] = Field2D[idx];
            pi[i][k] = Momenta2D[idx];
            phi_bar[i][k] = AntiGaugeField2D[idx];
            chi_screen[i][k] = ScreeningField2D[idx];
            zeta_antic[i][k] = AnticipationField2D[idx];
        }
    }

    float k_u[MAX_D2];
    float gamma_u[MAX_D2];
    float c2_u[MAX_D2];
    float beta_u[MAX_D2];

    for (int k = 0; k < MAX_D2; ++k) {
        k_u[k]     = SpectralLaws[law_offset + 0 * MAX_D2 + k];
        gamma_u[k] = SpectralLaws[law_offset + 1 * MAX_D2 + k];
        c2_u[k]    = SpectralLaws[law_offset + 2 * MAX_D2 + k];
        beta_u[k]  = SpectralLaws[law_offset + 3 * MAX_D2 + k];
    }

    float g_xx     = MetricTensor[metric_offset + 0];
    float g_yy     = MetricTensor[metric_offset + 1];
    float g_xy     = MetricTensor[metric_offset + 2];

    float g_u = MetricScale[u];
    if (g_u <= 0.1f) g_u = 1.0f;

    float v_u = UniverseVelocity[u];
    if (v_u <= 5.0f) v_u = 30.0f;

    float dt_eff = BASE_DT;
    float accumulated_free_energy = 0.0f;
    float accumulated_order = 0.0f;

    float agent_x = (float)((u * 320) % 35000);
    float agent_y = 100.0f;        
    float agent_vy = 0.0f;         
    bool is_grounded = true;       

    int total_hazards_encountered = 0;
    int total_collisions = 0;
    int symbol_histogram[NUM_SYMBOLS] = {0, 0, 0, 0};

    float tau_fut = LOOKAHEAD_HORIZON * dt_eff;

    for (int t = 0; t < ticks; ++t) {
        agent_x += v_u * dt_eff;

        float j_2d[D1][MAX_D2];
        float j_2d_future[D1][MAX_D2];

        sample_2d_level_geometry_fast(StreamData, agent_x, g_u, j_2d);
        sample_2d_level_geometry_fast(StreamData, agent_x + v_u * tau_fut, g_u, j_2d_future);

        // 1. AUTOPOIETIC SCREENING & COMPACTON WAVE FIELD DYNAMICS
        float mob[D1][MAX_D2];
        for (int i = 0; i < D1; ++i) {
            for (int k = 0; k < MAX_D2; ++k) {
                float ext = j_2d[i][k];
                float ext_mag = fabsf(ext);
                float field_err = fabsf(phi[i][k] - ext);

                float desired_chi = (ext_mag < 0.05f) ? (2.0f + 6.0f * field_err) : 0.0f;
                chi_screen[i][k] += 0.05f * (desired_chi - chi_screen[i][k]);

                float fut_err = fabsf(phi[i][k] - j_2d_future[i][k]);
                zeta_antic[i][k] += 0.02f * ((0.05f + 0.35f * tanhf(fut_err)) - zeta_antic[i][k]);

                float empty_weight = 1.0f - tanhf(4.0f * ext_mag);
                phi_bar[i][k] += 0.10f * (phi[i][k] * empty_weight - phi_bar[i][k]);

                mob[i][k] = tanhf(4.0f * ext_mag) + 0.005f;
            }
        }

        for (int i = 0; i < D1; ++i) {
            int left = (i == 0) ? (D1 - 1) : (i - 1);
            int right = (i == D1 - 1) ? 0 : (i + 1);

            for (int k = 0; k < MAX_D2; ++k) {
                int up_k = (k == 0) ? 0 : (k - 1);
                int down_k = (k == MAX_D2 - 1) ? (MAX_D2 - 1) : (k + 1);

                float p = phi[i][k];
                float m = pi[i][k];

                float mob_r = 0.5f * (mob[i][k] + mob[right][k]);
                float mob_l = 0.5f * (mob[i][k] + mob[left][k]);
                float div_flux_xx = mob_r * (phi[right][k] - p) - mob_l * (p - phi[left][k]);

                float mob_d = 0.5f * (mob[i][k] + mob[i][down_k]);
                float mob_u = 0.5f * (mob[i][k] + mob[i][up_k]);
                float div_flux_yy = mob_d * (phi[i][down_k] - p) - mob_u * (p - phi[i][up_k]);

                float lap_xy = (phi[right][down_k] - phi[right][up_k] - phi[left][down_k] + phi[left][up_k]) * 0.25f;
                float lap_metric = g_xx * div_flux_xx + g_yy * div_flux_yy + 2.0f * g_xy * lap_xy;

                float ext_2d = j_2d[i][k];
                float c2_eff = c2_u[k] / (1.0f + 0.5f * chi_screen[i][k]);

                float accel_present = c2_eff * lap_metric 
                                    - k_u[k] * (p - ext_2d) 
                                    - chi_screen[i][k] * p
                                    - beta_u[k] * p * p * p 
                                    - 0.01f * p * p * p * p * p 
                                    - gamma_u[k] * m;

                float p_future = p + m * tau_fut + 0.5f * accel_present * tau_fut * tau_fut;
                float future_error = p_future - j_2d_future[i][k];

                pi[i][k] += (accel_present - zeta_antic[i][k] * future_error) * dt_eff;
                phi[i][k] += pi[i][k] * dt_eff;
            }
        }

        // 2. TOPOLOGICAL PHASE VORTEX CHARGE EXTRACTION (Q_{u,i,k}):
        float phase[D1][MAX_D2];
        for (int i = 0; i < D1; ++i) {
            for (int k = 0; k < MAX_D2; ++k) {
                phase[i][k] = atan2f(pi[i][k], phi[i][k] + 1e-6f);
            }
        }

        float max_vortex_q = 0.0f;
        int active_symbol = 0; 

        for (int step = 1; step <= 4; ++step) {
            int look_i = (step < D1 - 1) ? step : (D1 - 2);

            for (int k = 1; k < MAX_D2 - 1; ++k) {
                float d1_x = wrap_angle_dev(phase[look_i + 1][k] - phase[look_i][k]);
                float d2_y = wrap_angle_dev(phase[look_i + 1][k + 1] - phase[look_i + 1][k]);
                float d3_x = wrap_angle_dev(phase[look_i][k + 1] - phase[look_i + 1][k + 1]);
                float d4_y = wrap_angle_dev(phase[look_i][k] - phase[look_i][k + 1]);

                float Q_vortex = (d1_x + d2_y + d3_x + d4_y) * 0.15915494f; // 1 / 2\pi
                float abs_q = fabsf(Q_vortex);

                if (abs_q > fabsf(max_vortex_q)) {
                    max_vortex_q = Q_vortex;

                    float obstacle_y = map_k_to_y_dev(k);
                    if (abs_q > 0.40f && obstacle_y > 115.0f) {
                        active_symbol = 1; 
                    } else if (abs_q > 0.20f && obstacle_y <= 115.0f) {
                        active_symbol = 2; 
                    } else if (Q_vortex < -0.30f && obstacle_y > 135.0f) {
                        active_symbol = 3; 
                    }
                }
            }
        }

        symbol_histogram[active_symbol]++;

        // 3. FEYNMAN PATH ACTION MOTOR CONTROL:
        float h_eff = 0.50f;
        if (active_symbol == 1) {
            h_eff = 0.10f; 
        } else if (active_symbol == 2) {
            h_eff = 0.30f; 
        } else if (active_symbol == 3) {
            h_eff = 0.80f; 
        }

        int agent_k = map_y_to_k_dev(agent_y);
        float action_glide = 0.0f;
        float action_jump  = 0.0f;

        for (int step = 1; step <= 8; ++step) {
            int look_i = (step < D1) ? step : (D1 - 1);

            float t_sim = step * dt_eff * 2.0f;
            float y_glide_sim = agent_y + agent_vy * t_sim + 0.5f * GRAVITY * t_sim * t_sim;
            int k_glide = map_y_to_k_dev(y_glide_sim);

            float vy_jump_sim = agent_vy + JUMP_IMPULSE;
            float y_jump_sim = agent_y + vy_jump_sim * t_sim + 0.5f * GRAVITY * t_sim * t_sim;
            int k_jump = map_y_to_k_dev(y_jump_sim);

            float phi_eff_glide = phi[look_i][k_glide] - phi_bar[look_i][k_glide] * (1.0f - tanhf(4.0f * fabsf(j_2d[look_i][k_glide])));
            float phi_eff_jump  = phi[look_i][k_jump]  - phi_bar[look_i][k_jump]  * (1.0f - tanhf(4.0f * fabsf(j_2d[look_i][k_jump])));

            action_glide += fabsf(phi_eff_glide) * 12.0f + 0.001f * (y_glide_sim - 100.0f) * (y_glide_sim - 100.0f);
            action_jump  += fabsf(phi_eff_jump)  * 12.0f + 0.001f * (y_jump_sim - 100.0f)  * (y_jump_sim - 100.0f);
        }

        float exp_glide = expf(-action_glide / h_eff);
        float exp_jump  = expf(-action_jump / h_eff);
        float p_jump = exp_jump / (exp_glide + exp_jump + 1e-8f);

        if (p_jump > 0.52f && is_grounded && active_symbol != 3) {
            agent_vy = JUMP_IMPULSE;
            is_grounded = false;
        }

        agent_vy += GRAVITY * dt_eff;
        agent_y  += agent_vy * dt_eff;

        if (agent_y <= 100.0f) {
            agent_y = 100.0f;
            agent_vy = 0.0f;
            is_grounded = true;
        }

        float current_hazard_density = j_2d[0][agent_k];
        if (fabsf(current_hazard_density) > 0.15f) {
            total_hazards_encountered++;
            float hazard_y = map_k_to_y_dev(agent_k);
            if (fabsf(agent_y - hazard_y) < (AGENT_RADIUS * 22.0f)) {
                total_collisions++;
            }
        }

        float step_free_energy = fabsf(phi[0][agent_k] - j_2d[0][agent_k]);
        accumulated_free_energy += step_free_energy;
        accumulated_order += expf(-0.20f * step_free_energy);
    }

    for (int i = 0; i < D1; ++i) {
        for (int k = 0; k < MAX_D2; ++k) {
            int idx = offset + i * MAX_D2 + k;
            Field2D[idx] = phi[i][k];
            Momenta2D[idx] = pi[i][k];
            AntiGaugeField2D[idx] = phi_bar[i][k];
            ScreeningField2D[idx] = chi_screen[i][k];
            AnticipationField2D[idx] = zeta_antic[i][k];
        }
    }

    AgentXOut[u] = agent_x;
    AgentYOut[u] = agent_y;

    float evasion_rate = (total_hazards_encountered > 0) 
        ? (100.0f * (1.0f - (float)total_collisions / (float)total_hazards_encountered)) 
        : 100.0f;

    int unique_symbols = 0;
    for (int s = 0; s < NUM_SYMBOLS; ++s) {
        if (symbol_histogram[s] > 10) unique_symbols++;
    }

    EvasionRateOut[u] = evasion_rate;
    DiscoveredSymbolsOut[u] = (float)unique_symbols;
    FreeEnergyOut[u] = accumulated_free_energy / ticks;
    OrderIntegrityOut[u] = accumulated_order / ticks;
    RandStates[u] = local_rand;
}

__global__ void init_rand_kernel_32(curandState* states, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &states[id]);
    }
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "EXP-709: CUDA Kernel Compilation & Execution Optimization (CKEO)\n";
    std::cout << "Ultra-Fast NVCC Build & Execution Benchmark\n";
    std::cout << "====================================================================\n\n";

    std::ifstream infile("data/bloodbath_stream.bin", std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error opening data/bloodbath_stream.bin!" << std::endl;
        return 1;
    }

    std::vector<Obstacle> h_Stream(TOTAL_OBSTACLES);
    infile.read(reinterpret_cast<char*>(h_Stream.data()), TOTAL_OBSTACLES * sizeof(Obstacle));
    infile.close();

    size_t field_bytes = MAX_UNIVERSES * TOTAL_FIELD_DIM * sizeof(float);
    size_t law_bytes = MAX_UNIVERSES * LAW_SPECTRAL_DIM * sizeof(float);
    size_t metric_bytes = MAX_UNIVERSES * METRIC_TENSOR_DIM * sizeof(float);
    size_t obstacle_bytes = TOTAL_OBSTACLES * sizeof(Obstacle);
    size_t score_bytes = MAX_UNIVERSES * sizeof(float);
    size_t rand_bytes = MAX_UNIVERSES * sizeof(curandState);

    std::vector<float> h_Field(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_Momenta(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_AntiGauge(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_ScreenField(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_AnticField(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.15f);
    std::vector<float> h_SpectralLaws(MAX_UNIVERSES * LAW_SPECTRAL_DIM);
    std::vector<float> h_MetricTensor(MAX_UNIVERSES * METRIC_TENSOR_DIM);
    std::vector<float> h_MetricScale(MAX_UNIVERSES, 1.0f);
    std::vector<float> h_Velocity(MAX_UNIVERSES, 30.0f);
    std::vector<float> h_Clock(MAX_UNIVERSES, BASE_DT);

    for (int u = 0; u < MAX_UNIVERSES; ++u) {
        for (int k = 0; k < MAX_D2; ++k) {
            float mode_scale = 1.0f + 0.05f * k;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 0 * MAX_D2 + k] = 1.0f * mode_scale;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 1 * MAX_D2 + k] = 2.0f * mode_scale;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 2 * MAX_D2 + k] = 0.35f + 0.02f * (u % 3);
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 3 * MAX_D2 + k] = 0.01f;
        }

        h_MetricTensor[u * METRIC_TENSOR_DIM + 0] = 1.0f;  
        h_MetricTensor[u * METRIC_TENSOR_DIM + 1] = 0.35f; 
        h_MetricTensor[u * METRIC_TENSOR_DIM + 2] = 0.0f;  
        h_MetricTensor[u * METRIC_TENSOR_DIM + 3] = 0.20f; 
    }

    std::vector<float> h_AgentX(MAX_UNIVERSES);
    std::vector<float> h_AgentY(MAX_UNIVERSES);
    std::vector<float> h_EvasionRate(MAX_UNIVERSES);
    std::vector<float> h_DiscoveredSymbols(MAX_UNIVERSES);
    std::vector<float> h_FreeEnergy(MAX_UNIVERSES);
    std::vector<float> h_OrderIntegrity(MAX_UNIVERSES);

    int active_universes = INITIAL_UNIVERSES;

    float *d_Field, *d_Momenta, *d_AntiGauge, *d_ScreenField, *d_AnticField, *d_SpectralLaws, *d_MetricTensor, *d_MetricScale, *d_Velocity, *d_Clock;
    Obstacle *d_StreamData;
    int *d_ActiveCount;
    float *d_AgentX, *d_AgentY, *d_EvasionRate, *d_DiscoveredSymbols, *d_FreeEnergy, *d_OrderIntegrity;
    curandState *d_RandStates;

    CUDA_CHECK(cudaMalloc(&d_Field, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_Momenta, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_AntiGauge, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_ScreenField, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_AnticField, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_SpectralLaws, law_bytes));
    CUDA_CHECK(cudaMalloc(&d_MetricTensor, metric_bytes));
    CUDA_CHECK(cudaMalloc(&d_MetricScale, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Velocity, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Clock, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_StreamData, obstacle_bytes));
    CUDA_CHECK(cudaMalloc(&d_ActiveCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_AgentX, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_AgentY, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_EvasionRate, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_DiscoveredSymbols, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_FreeEnergy, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_OrderIntegrity, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_RandStates, rand_bytes));

    CUDA_CHECK(cudaMemcpy(d_Field, h_Field.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Momenta, h_Momenta.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AntiGauge, h_AntiGauge.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ScreenField, h_ScreenField.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AnticField, h_AnticField.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SpectralLaws, h_SpectralLaws.data(), law_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MetricTensor, h_MetricTensor.data(), metric_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MetricScale, h_MetricScale.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Velocity, h_Velocity.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Clock, h_Clock.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_StreamData, h_Stream.data(), obstacle_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ActiveCount, &active_universes, sizeof(int), cudaMemcpyHostToDevice));

    dim3 threads(32);
    dim3 blocks((MAX_UNIVERSES + threads.x - 1) / threads.x);

    init_rand_kernel_32<<<blocks, threads>>>(d_RandStates, 2026, MAX_UNIVERSES);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    fast_topological_symbolic_genesis_kernel<<<blocks, threads>>>(
        d_Field, d_Momenta, d_AntiGauge, d_ScreenField, d_AnticField, d_SpectralLaws, d_MetricTensor, d_MetricScale, d_Velocity, d_Clock, d_StreamData, d_ActiveCount,
        d_AgentX, d_AgentY, d_EvasionRate, d_DiscoveredSymbols, d_FreeEnergy, d_OrderIntegrity,
        d_RandStates, TICKS
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    CUDA_CHECK(cudaMemcpy(h_AgentX.data(), d_AgentX, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_AgentY.data(), d_AgentY, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_EvasionRate.data(), d_EvasionRate, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_DiscoveredSymbols.data(), d_DiscoveredSymbols, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_FreeEnergy.data(), d_FreeEnergy, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_OrderIntegrity.data(), d_OrderIntegrity, score_bytes, cudaMemcpyDeviceToHost));

    float mean_agent_x = 0.0f;
    float mean_evasion_rate = 0.0f;
    float mean_symbols = 0.0f;
    float mean_free_energy = 0.0f;
    float mean_order = 0.0f;

    for (int u = 0; u < active_universes; ++u) {
        mean_agent_x += h_AgentX[u];
        mean_evasion_rate += h_EvasionRate[u];
        mean_symbols += h_DiscoveredSymbols[u];
        mean_free_energy += h_FreeEnergy[u];
        mean_order += h_OrderIntegrity[u];
    }
    mean_agent_x /= active_universes;
    mean_evasion_rate /= active_universes;
    mean_symbols /= active_universes;
    mean_free_energy /= active_universes;
    mean_order /= active_universes;

    std::cout << "[EXP-709 CKEO Fast Build & Execution Benchmark Summary]:\n";
    std::cout << "  - Active Universes in Multiverse  : " << active_universes << "\n";
    std::cout << "  - Symplectic Horizon Ticks        : " << TICKS << " ticks\n";
    std::cout << "  - GPU Kernel Execution Time       : " << ms << " ms (" << ms / 1000.0 << " s)\n";
    std::cout << "  - Mean Navigated Distance X_agent : " << mean_agent_x << " level units\n";
    std::cout << "  - Hazard Evasion Success Rate %   : " << mean_evasion_rate << " %\n";
    std::cout << "  - Closed-Loop Free Energy F       : " << mean_free_energy << " nats\n";
    std::cout << "  - Intrinsic Order Integrity O_u   : " << mean_order << "\n";

    CUDA_CHECK(cudaFree(d_Field));
    CUDA_CHECK(cudaFree(d_Momenta));
    CUDA_CHECK(cudaFree(d_AntiGauge));
    CUDA_CHECK(cudaFree(d_ScreenField));
    CUDA_CHECK(cudaFree(d_AnticField));
    CUDA_CHECK(cudaFree(d_SpectralLaws));
    CUDA_CHECK(cudaFree(d_MetricTensor));
    CUDA_CHECK(cudaFree(d_MetricScale));
    CUDA_CHECK(cudaFree(d_Velocity));
    CUDA_CHECK(cudaFree(d_Clock));
    CUDA_CHECK(cudaFree(d_StreamData));
    CUDA_CHECK(cudaFree(d_ActiveCount));
    CUDA_CHECK(cudaFree(d_AgentX));
    CUDA_CHECK(cudaFree(d_AgentY));
    CUDA_CHECK(cudaFree(d_EvasionRate));
    CUDA_CHECK(cudaFree(d_DiscoveredSymbols));
    CUDA_CHECK(cudaFree(d_FreeEnergy));
    CUDA_CHECK(cudaFree(d_OrderIntegrity));
    CUDA_CHECK(cudaFree(d_RandStates));

    return 0;
}
