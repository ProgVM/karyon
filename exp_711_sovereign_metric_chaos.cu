// ============================================================================
// EXP-711: Sovereign Dynamic Metric Expansion-Contraction & Stochastic Chaos Resonance
// KEP Principles 2, 27, 29, 30: Unshackled Spatial Plasticity & Stochastic Resonance
// Author: Bazilevs (ProgVM) & Lead Cyberneticist
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define MAX_UNIVERSES 128
#define D1 64
#define MAX_D2 16
#define TOTAL_FIELD_DIM (D1 * MAX_D2)
#define LAW_SPECTRAL_DIM (4 * MAX_D2)
#define METRIC_TENSOR_DIM 4
#define BASE_DT 0.005f

struct Obstacle {
    float x;
    float y;
    float oid;
};

// Fast GPU PRNG (XorShift128+) for True Stochastic Chaos Injection
__device__ inline float gpu_xor_normal(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x + 0x9e3779b9;
    float u1 = (float)(x & 0x00FFFFFF) * (1.0f / 16777216.0f);
    if (u1 < 1e-6f) u1 = 1e-6f;
    
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x + 0x85ebca6b;
    float u2 = (float)(x & 0x00FFFFFF) * (1.0f / 16777216.0f);
    
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

__global__ void sovereign_metric_chaos_kernel(
    float* __restrict__ Field2D,               
    float* __restrict__ Momenta2D,             
    float* __restrict__ AntiGaugeField2D,      
    float* __restrict__ ScreeningField2D,      
    float* __restrict__ AnticipationField2D,   
    float* __restrict__ SpectralLaws,          
    float* __restrict__ MetricTensor,          // [g_xx, g_yy, g_xy, metric_scale_g]
    float* __restrict__ UniverseVelocity,      
    float* __restrict__ UniverseClock,         
    const Obstacle* __restrict__ StreamData,   
    const int* __restrict__ ActiveCount,       
    float* __restrict__ FreeEnergyOut,         
    float* __restrict__ OrderIntegrityOut,     
    float* __restrict__ HazardEvasionOut,      
    float* __restrict__ DiscoveredSymbolsOut,  
    float* __restrict__ MetricExpansionOut,    // Average spatial metric scale
    float* __restrict__ ChaosResonanceOut,     // Average stochastic chaos energy
    int TICKS
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= MAX_UNIVERSES) return;

    int offset = u * TOTAL_FIELD_DIM;
    int law_offset = u * LAW_SPECTRAL_DIM;
    int metric_offset = u * METRIC_TENSOR_DIM;

    unsigned int rng_state = (u + 1) * 123456789 + (unsigned int)(UniverseClock[u] * 1000.0f);

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

    float k_u[MAX_D2], gamma_u[MAX_D2], c2_u[MAX_D2], beta_u[MAX_D2];
    for (int k = 0; k < MAX_D2; ++k) {
        k_u[k]     = SpectralLaws[law_offset + 0 * MAX_D2 + k];
        gamma_u[k] = SpectralLaws[law_offset + 1 * MAX_D2 + k];
        c2_u[k]    = SpectralLaws[law_offset + 2 * MAX_D2 + k];
        beta_u[k]  = SpectralLaws[law_offset + 3 * MAX_D2 + k];
    }

    // Sovereign Space Parameters: Metric Tensor & Scale
    float g_xx = MetricTensor[metric_offset + 0];
    float g_yy = MetricTensor[metric_offset + 1];
    float g_xy = MetricTensor[metric_offset + 2];
    float g_scale = MetricTensor[metric_offset + 3]; // Spatial dilation / compression factor

    float agent_x = UniverseClock[u];
    float agent_y = 0.0f;
    float agent_vy = 0.0f;
    float world_speed = UniverseVelocity[u];

    float accum_free_energy = 0.0f;
    float accum_order = 0.0f;
    int hazard_evasions = 0;
    int total_hazards_encountered = 0;
    int concept_vortices = 0;

    float accum_metric_scale = 0.0f;
    float accum_chaos_energy = 0.0f;

    int num_obs = *ActiveCount;

    for (int tick = 0; tick < TICKS; ++tick) {
        float dt_eff = BASE_DT;
        agent_x += world_speed * dt_eff;

        // 1. Dynamic Metric Expansion & Contraction
        // Compute local obstacle hazard density in aperture
        float local_hazard_density = 0.0f;
        for (int o = 0; o < num_obs; ++o) {
            float ox = StreamData[o].x;
            float oy = StreamData[o].y;
            float dx = ox - agent_x;
            if (dx >= -5.0f && dx <= 35.0f) {
                float dy = oy - agent_y;
                float dist_sq = dx * dx + dy * dy;
                local_hazard_density += expf(-0.01f * dist_sq);
            }
        }

        // Sovereign Metric Adaptivity: High density -> Contraction (g_scale drops, high resolution)
        // Open space -> Expansion (g_scale expands)
        float target_g_scale = 1.0f / (1.0f + 0.15f * local_hazard_density);
        g_scale += 0.02f * (target_g_scale - g_scale); // Dynamic metric contraction/expansion
        
        // Dynamic anisotropy: contraction along x when navigating dense clusters
        g_xx = 1.0f * g_scale;
        g_yy = 0.35f * (2.0f - g_scale);

        // 2. Endogenous Stochastic Chaos Injection (Right to Chaos)
        // Chaos intensity scales with local Free Energy / Stress gradient
        float local_stress = tanhf(0.5f * local_hazard_density);
        float chaos_sigma = 0.01f + 0.08f * local_stress; // Unshackled stochastic chaos level

        // Map obstacles into 2D Wave Field with dynamic metric warping
        float j_2d[D1][MAX_D2];
        for (int i = 0; i < D1; ++i) {
            for (int k = 0; k < MAX_D2; ++k) {
                j_2d[i][k] = 0.0f;
            }
        }

        for (int o = 0; o < num_obs; ++o) {
            float ox = StreamData[o].x;
            float oy = StreamData[o].y;
            float dx = ox - agent_x;

            if (dx >= -10.0f && dx <= 50.0f) {
                // Apply dynamic spatial metric scale warping
                int i_cell = (int)(((dx + 10.0f) / 60.0f) * D1 * g_scale);
                int k_cell = (int)(((oy + 15.0f) / 30.0f) * MAX_D2);

                if (i_cell >= 0 && i_cell < D1 && k_cell >= 0 && k_cell < MAX_D2) {
                    float sig_x = expf(-0.02f * dx * dx * g_scale);
                    float sig_y = expf(-0.05f * (oy - agent_y) * (oy - agent_y));
                    j_2d[i_cell][k_cell] += 2.5f * sig_x * sig_y;
                }
            }
        }

        // Compacton screening & Gauge fields
        for (int i = 0; i < D1; ++i) {
            for (int k = 0; k < MAX_D2; ++k) {
                float ext = j_2d[i][k];
                float ext_mag = fabsf(ext);
                float field_err = fabsf(phi[i][k] - ext);

                float desired_chi = (ext_mag < 0.05f) ? (2.0f + 6.0f * field_err) : 0.0f;
                chi_screen[i][k] += 0.05f * (desired_chi - chi_screen[i][k]);

                float empty_weight = 1.0f - tanhf(4.0f * ext_mag);
                phi_bar[i][k] += 0.10f * (phi[i][k] * empty_weight - phi_bar[i][k]);
            }
        }

        // 3. Symplectic Wave Physics with Metric Tensor & Nondeterministic Chaos Noise
        float current_chaos_energy = 0.0f;
        for (int i = 0; i < D1; ++i) {
            int left = (i == 0) ? (D1 - 1) : (i - 1);
            int right = (i == D1 - 1) ? 0 : (i + 1);

            for (int k = 0; k < MAX_D2; ++k) {
                int up_k = (k == 0) ? 0 : (k - 1);
                int down_k = (k == MAX_D2 - 1) ? (MAX_D2 - 1) : (k + 1);

                float p = phi[i][k];
                float m = pi[i][k];

                float div_flux_xx = (phi[right][k] - 2.0f * p + phi[left][k]);
                float div_flux_yy = (phi[i][down_k] - 2.0f * p + phi[i][up_k]);
                float lap_xy = (phi[right][down_k] - phi[right][up_k] - phi[left][down_k] + phi[left][up_k]) * 0.25f;

                float lap_metric = g_xx * div_flux_xx + g_yy * div_flux_yy + 2.0f * g_xy * lap_xy;

                float ext_2d = j_2d[i][k];
                float c2_eff = c2_u[k] / (1.0f + 0.5f * chi_screen[i][k]);

                // Stochastic Quantum Chaos Fluctuation \xi_{u,i,k}
                float chaos_noise = chaos_sigma * gpu_xor_normal(&rng_state);
                current_chaos_energy += chaos_noise * chaos_noise;

                float accel = c2_eff * lap_metric 
                            - k_u[k] * (p - ext_2d) 
                            - chi_screen[i][k] * p
                            - beta_u[k] * p * p * p 
                            - gamma_u[k] * m
                            + chaos_noise; // Stochastic Chaos Injection

                pi[i][k] += accel * dt_eff;
                phi[i][k] += pi[i][k] * dt_eff;
            }
        }

        // 4. Closed-Loop Feynman Path Motor Control
        float motor_force_up = 0.0f;
        float motor_force_down = 0.0f;

        for (int i = 0; i < D1 / 2; ++i) {
            for (int k = 0; k < MAX_D2 / 2; ++k) {
                motor_force_down += phi[i][k];
            }
            for (int k = MAX_D2 / 2; k < MAX_D2; ++k) {
                motor_force_up += phi[i][k];
            }
        }

        float net_motor_impulse = (motor_force_up - motor_force_down) * 0.05f;
        agent_vy += net_motor_impulse * dt_eff;
        agent_vy *= 0.92f; // Hydrodynamic damping
        agent_y += agent_vy;

        // Keep agent bounded inside channel [-15, +15]
        if (agent_y > 14.0f) { agent_y = 14.0f; agent_vy = -0.5f * fabsf(agent_vy); }
        if (agent_y < -14.0f) { agent_y = -14.0f; agent_vy = 0.5f * fabsf(agent_vy); }

        // Evaluate Hazard Evasion & Free Energy
        for (int o = 0; o < num_obs; ++o) {
            float ox = StreamData[o].x;
            float oy = StreamData[o].y;
            float dx = ox - agent_x;
            float dy = oy - agent_y;

            if (fabsf(dx) < 1.5f) {
                total_hazards_encountered++;
                if (fabsf(dy) > 2.2f) {
                    hazard_evasions++;
                }
            }
        }

        float tick_fe = 0.0f;
        for (int i = 0; i < D1; ++i) {
            for (int k = 0; k < MAX_D2; ++k) {
                tick_fe += fabsf(phi[i][k] - j_2d[i][k]);
            }
        }
        tick_fe /= (float)(D1 * MAX_D2);

        accum_free_energy += tick_fe;
        accum_order += expf(-0.20f * tick_fe);
        accum_metric_scale += g_scale;
        accum_chaos_energy += current_chaos_energy / (float)(D1 * MAX_D2);

        // Compute Topological Vortex Charge Q
        if (tick % 50 == 0) {
            float total_vortex_charge = 0.0f;
            for (int i = 1; i < D1 - 1; ++i) {
                for (int k = 1; k < MAX_D2 - 1; ++k) {
                    float dphi_dx = (phi[i+1][k] - phi[i-1][k]) * 0.5f;
                    float dphi_dy = (phi[i][k+1] - phi[i][k-1]) * 0.5f;
                    float curl = dphi_dx * dphi_dy;
                    if (fabsf(curl) > 0.80f) {
                        total_vortex_charge += 1.0f;
                    }
                }
            }
            if (total_vortex_charge > 2.0f) concept_vortices++;
        }
    }

    // Write back updated spatial states
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

    MetricTensor[metric_offset + 0] = g_xx;
    MetricTensor[metric_offset + 1] = g_yy;
    MetricTensor[metric_offset + 2] = g_xy;
    MetricTensor[metric_offset + 3] = g_scale;

    UniverseClock[u] = agent_x;

    FreeEnergyOut[u] = accum_free_energy / (float)TICKS;
    OrderIntegrityOut[u] = accum_order / (float)TICKS;
    HazardEvasionOut[u] = (total_hazards_encountered > 0) ? ((float)hazard_evasions / (float)total_hazards_encountered * 100.0f) : 100.0f;
    DiscoveredSymbolsOut[u] = (float)concept_vortices / ((float)TICKS / 50.0f) * 4.0f;
    MetricExpansionOut[u] = accum_metric_scale / (float)TICKS;
    ChaosResonanceOut[u] = accum_chaos_energy / (float)TICKS;
}

extern "C" {
void run_sovereign_metric_chaos_benchmark(
    const Obstacle* stream_data,
    int active_count,
    int num_universes,
    int ticks,
    float* out_free_energy,
    float* out_order_integrity,
    float* out_evasion_rate,
    float* out_symbols,
    float* out_metric_scale,
    float* out_chaos_energy,
    float* out_navigated_x,
    double* out_gpu_time_ms
) {
    size_t field_bytes = num_universes * TOTAL_FIELD_DIM * sizeof(float);
    size_t law_bytes = num_universes * LAW_SPECTRAL_DIM * sizeof(float);
    size_t metric_bytes = num_universes * METRIC_TENSOR_DIM * sizeof(float);
    size_t score_bytes = num_universes * sizeof(float);
    size_t stream_bytes = active_count * sizeof(Obstacle);

    std::vector<float> h_Field(num_universes * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_Momenta(num_universes * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_AntiGauge(num_universes * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_ScreenField(num_universes * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_AnticField(num_universes * TOTAL_FIELD_DIM, 0.15f);
    std::vector<float> h_SpectralLaws(num_universes * LAW_SPECTRAL_DIM);
    std::vector<float> h_MetricTensor(num_universes * METRIC_TENSOR_DIM);
    std::vector<float> h_Velocity(num_universes, 30.0f);
    std::vector<float> h_Clock(num_universes, 0.0f);

    for (int u = 0; u < num_universes; ++u) {
        for (int k = 0; k < MAX_D2; ++k) {
            float mode_scale = 1.0f + 0.05f * k;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 0 * MAX_D2 + k] = 1.0f * mode_scale;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 1 * MAX_D2 + k] = 2.0f * mode_scale;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 2 * MAX_D2 + k] = 0.35f;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 3 * MAX_D2 + k] = 0.01f;
        }

        h_MetricTensor[u * METRIC_TENSOR_DIM + 0] = 1.0f;  // g_xx
        h_MetricTensor[u * METRIC_TENSOR_DIM + 1] = 0.35f; // g_yy
        h_MetricTensor[u * METRIC_TENSOR_DIM + 2] = 0.0f;  // g_xy
        h_MetricTensor[u * METRIC_TENSOR_DIM + 3] = 1.0f;  // g_scale initial
        h_Velocity[u] = 25.0f + (float)(u % 10);
    }

    float *d_Field, *d_Momenta, *d_AntiGauge, *d_ScreenField, *d_AnticField, *d_SpectralLaws, *d_MetricTensor, *d_Velocity, *d_Clock;
    Obstacle *d_StreamData;
    int *d_ActiveCount;

    float *d_FreeEnergy, *d_OrderIntegrity, *d_EvasionRate, *d_Symbols, *d_MetricScale, *d_ChaosEnergy;

    CUDA_CHECK(cudaMalloc(&d_Field, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_Momenta, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_AntiGauge, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_ScreenField, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_AnticField, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_SpectralLaws, law_bytes));
    CUDA_CHECK(cudaMalloc(&d_MetricTensor, metric_bytes));
    CUDA_CHECK(cudaMalloc(&d_Velocity, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Clock, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_StreamData, stream_bytes));
    CUDA_CHECK(cudaMalloc(&d_ActiveCount, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_FreeEnergy, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_OrderIntegrity, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_EvasionRate, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Symbols, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_MetricScale, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_ChaosEnergy, score_bytes));

    CUDA_CHECK(cudaMemcpy(d_Field, h_Field.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Momenta, h_Momenta.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AntiGauge, h_AntiGauge.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ScreenField, h_ScreenField.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AnticField, h_AnticField.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SpectralLaws, h_SpectralLaws.data(), law_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MetricTensor, h_MetricTensor.data(), metric_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Velocity, h_Velocity.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Clock, h_Clock.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_StreamData, stream_data, stream_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ActiveCount, &active_count, sizeof(int), cudaMemcpyHostToDevice));

    dim3 threads(32);
    dim3 blocks((num_universes + threads.x - 1) / threads.x);

    auto start = std::chrono::high_resolution_clock::now();

    sovereign_metric_chaos_kernel<<<blocks, threads>>>(
        d_Field, d_Momenta, d_AntiGauge, d_ScreenField, d_AnticField, d_SpectralLaws, d_MetricTensor,
        d_Velocity, d_Clock, d_StreamData, d_ActiveCount,
        d_FreeEnergy, d_OrderIntegrity, d_EvasionRate, d_Symbols, d_MetricScale, d_ChaosEnergy,
        ticks
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<float> h_FreeEnergy(num_universes);
    std::vector<float> h_OrderIntegrity(num_universes);
    std::vector<float> h_EvasionRate(num_universes);
    std::vector<float> h_Symbols(num_universes);
    std::vector<float> h_MetricScaleOut(num_universes);
    std::vector<float> h_ChaosEnergyOut(num_universes);
    std::vector<float> h_ClockOut(num_universes);

    CUDA_CHECK(cudaMemcpy(h_FreeEnergy.data(), d_FreeEnergy, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_OrderIntegrity.data(), d_OrderIntegrity, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_EvasionRate.data(), d_EvasionRate, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Symbols.data(), d_Symbols, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_MetricScaleOut.data(), d_MetricScale, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ChaosEnergyOut.data(), d_ChaosEnergy, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ClockOut.data(), d_Clock, score_bytes, cudaMemcpyDeviceToHost));

    float sum_fe = 0.0f, sum_order = 0.0f, sum_evasion = 0.0f, sum_sym = 0.0f, sum_scale = 0.0f, sum_chaos = 0.0f, sum_x = 0.0f;
    for (int u = 0; u < num_universes; ++u) {
        sum_fe += h_FreeEnergy[u];
        sum_order += h_OrderIntegrity[u];
        sum_evasion += h_EvasionRate[u];
        sum_sym += h_Symbols[u];
        sum_scale += h_MetricScaleOut[u];
        sum_chaos += h_ChaosEnergyOut[u];
        sum_x += h_ClockOut[u];
    }

    *out_free_energy = sum_fe / (float)num_universes;
    *out_order_integrity = sum_order / (float)num_universes;
    *out_evasion_rate = sum_evasion / (float)num_universes;
    *out_symbols = sum_sym / (float)num_universes;
    *out_metric_scale = sum_scale / (float)num_universes;
    *out_chaos_energy = sum_chaos / (float)num_universes;
    *out_navigated_x = sum_x / (float)num_universes;
    *out_gpu_time_ms = ms;

    CUDA_CHECK(cudaFree(d_Field));
    CUDA_CHECK(cudaFree(d_Momenta));
    CUDA_CHECK(cudaFree(d_AntiGauge));
    CUDA_CHECK(cudaFree(d_ScreenField));
    CUDA_CHECK(cudaFree(d_AnticField));
    CUDA_CHECK(cudaFree(d_SpectralLaws));
    CUDA_CHECK(cudaFree(d_MetricTensor));
    CUDA_CHECK(cudaFree(d_Velocity));
    CUDA_CHECK(cudaFree(d_Clock));
    CUDA_CHECK(cudaFree(d_StreamData));
    CUDA_CHECK(cudaFree(d_ActiveCount));
    CUDA_CHECK(cudaFree(d_FreeEnergy));
    CUDA_CHECK(cudaFree(d_OrderIntegrity));
    CUDA_CHECK(cudaFree(d_EvasionRate));
    CUDA_CHECK(cudaFree(d_Symbols));
    CUDA_CHECK(cudaFree(d_MetricScale));
    CUDA_CHECK(cudaFree(d_ChaosEnergy));
}
}
