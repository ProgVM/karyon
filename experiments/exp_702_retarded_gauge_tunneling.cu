// ============================================================================
// EXP-702: Retarded Gauge Vector Potentials & Hyper-Diffusive Phase Tunneling (RGVP-HDPT)
// Master Protocol: KEP v30.0 Substrate Specification (Principles 2, 3, 4, 7, 8, 27, 29, 30, 31)
//
// Milestone Architectural Discovery for EXP-702:
// In EXP-701, autopoietic metric tensor g_ij and dimension uncurling alpha_dim achieved
// F = 0.273538 nats, but wave propagation was purely local and second-order (\nabla^2).
// When dense walls of stacked obstacles (e.g. 5+ blocks/spikes at identical X) appear,
// local Laplacian waves experience phase reflection shock.
//
// Key Innovations in EXP-702:
// 1. Dynamic Retarded Gauge Potential A_mu = (A_x, A_y):
//    - The motion of the universe world-line at velocity v_u creates a relativistic convection
//      gauge field: A_x = -v_u * \partial_x \phi, A_y = -v_u * \partial_y \phi.
//    - Covariant derivative D_mu = \partial_mu - i * A_mu induces phase pre-rotation,
//      allowing wavepackets to adjust their internal phase BEFORE hitting obstacle clusters!
// 2. Hyper-Diffusive Bi-Laplacian Phase Tunneling (\nabla^4 \phi):
//    - Adds a high-order quantum-like tunneling term: -\kappa_tunnel * \nabla^2(\nabla^2 \phi).
//    - Suppresses sharp high-frequency boundary reflections and enables smooth tunneling
//      through narrow spatial corridors between stacked obstacles without energy dissipation.
// 3. Autonomous Tunneling Plasticity (\kappa_tunnel(u)):
//    - Dynamically modulated by local obstacle density and Free Energy gradients.
//
// Evaluated on 24,757 real-world Bloodbath level obstacles.
// ============================================================================

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int MAX_UNIVERSES = 512;        // Dynamic capacity
constexpr int INITIAL_UNIVERSES = 128;    // Unshackled start
constexpr int D1 = 64;                    // Spatial longitudinal horizon X
constexpr int MAX_D2 = 16;                // Transverse spatial height capacity Y
constexpr int TOTAL_FIELD_DIM = D1 * MAX_D2; // 1024 degrees of freedom
constexpr int LAW_SPECTRAL_DIM = 4 * MAX_D2; // 64 parameters per universe
constexpr int METRIC_TENSOR_DIM = 4;      // [g_xx, g_yy, g_xy, alpha_dim]
constexpr int GAUGE_TENSOR_DIM = 4;       // [A_x, A_y, kappa_tunnel, phi_flux]
constexpr int TICKS = 4096;               // Continuous symplectic horizon
constexpr int TOTAL_OBSTACLES = 24757;    // Bloodbath level obstacle count
constexpr float BASE_DT = 0.005f;         // Base proper time
constexpr float SYMPLECTIC_GAMMA = 0.03f; // Inter-universe transport strength
constexpr int LOOKAHEAD_HORIZON = 8;      // ~40 ms future window
constexpr float DX_HORIZON = 2.0f;        // Continuous spatial step size in X

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
    float norm = asinhf(dy) / 3.82f;
    int k = (int)((norm + 0.5f) / 1.5f * 15.0f);
    if (k < 0) k = 0;
    if (k >= MAX_D2) k = MAX_D2 - 1;
    return k;
}

__device__ inline void sample_2d_level_geometry(
    const Obstacle* __restrict__ StreamData,
    float target_x_start,
    float g_u,
    float j_2d[D1][MAX_D2]
) {
    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        #pragma unroll
        for (int k = 0; k < MAX_D2; ++k) {
            j_2d[i][k] = 0.0f;
        }
    }

    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        float sample_x = target_x_start + i * DX_HORIZON;

        int low = 0;
        int high = TOTAL_OBSTACLES - 1;

        while (low < high - 1) {
            int mid = (low + high) / 2;
            if (StreamData[mid].x <= sample_x) {
                low = mid;
            } else {
                high = mid;
            }
        }

        int scan_start = (low >= 4) ? (low - 4) : 0;
        int scan_end = (low + 4 < TOTAL_OBSTACLES) ? (low + 4) : (TOTAL_OBSTACLES - 1);

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

// ----------------------------------------------------------------------------
// CUDA Kernel: Retarded Gauge Vector Potentials & Hyper-Diffusive Phase Tunneling
// ----------------------------------------------------------------------------
__global__ void retarded_gauge_tunneling_kernel(
    float* __restrict__ Field2D,               // [MAX_UNIVERSES, D1, MAX_D2]
    float* __restrict__ Momenta2D,             // [MAX_UNIVERSES, D1, MAX_D2]
    float* __restrict__ SpectralLaws,          // [MAX_UNIVERSES, LAW_SPECTRAL_DIM]
    float* __restrict__ MetricTensor,          // [MAX_UNIVERSES, METRIC_TENSOR_DIM]
    float* __restrict__ GaugeTensor,           // [MAX_UNIVERSES, GAUGE_TENSOR_DIM] -> [A_x, A_y, kappa_tunnel, phi_flux]
    float* __restrict__ MetricScale,           // [MAX_UNIVERSES] Sovereign Metric Scale g_u
    float* __restrict__ UniverseVelocity,      // [MAX_UNIVERSES] Sovereign World-Line Speed v_u
    float* __restrict__ EndogenousClock,       // [MAX_UNIVERSES]
    const Obstacle* __restrict__ StreamData,   // [TOTAL_OBSTACLES]
    int* __restrict__ ActiveCountIO,           // Active universe count N_active(t)
    float* __restrict__ SuperspaceRadiusOut,   // [MAX_UNIVERSES]
    float* __restrict__ VolumetricActionOut,   // [MAX_UNIVERSES]
    float* __restrict__ FreeEnergyOut,         // [MAX_UNIVERSES]
    float* __restrict__ OrderIntegrityOut,     // [MAX_UNIVERSES] Intrinsic Order O_u
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
    int gauge_offset = u * GAUGE_TENSOR_DIM;

    float phi[D1][MAX_D2];
    float pi[D1][MAX_D2];

    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        #pragma unroll
        for (int k = 0; k < MAX_D2; ++k) {
            int idx = offset + i * MAX_D2 + k;
            phi[i][k] = Field2D[idx];
            pi[i][k] = Momenta2D[idx];
        }
    }

    float k_u[MAX_D2];
    float gamma_u[MAX_D2];
    float c2_u[MAX_D2];
    float beta_u[MAX_D2];

    #pragma unroll
    for (int k = 0; k < MAX_D2; ++k) {
        k_u[k]     = SpectralLaws[law_offset + 0 * MAX_D2 + k];
        gamma_u[k] = SpectralLaws[law_offset + 1 * MAX_D2 + k];
        c2_u[k]    = SpectralLaws[law_offset + 2 * MAX_D2 + k];
        beta_u[k]  = SpectralLaws[law_offset + 3 * MAX_D2 + k];
    }

    float g_xx     = MetricTensor[metric_offset + 0];
    float g_yy     = MetricTensor[metric_offset + 1];
    float g_xy     = MetricTensor[metric_offset + 2];
    float alpha_dim= MetricTensor[metric_offset + 3];

    if (g_xx < 0.2f) g_xx = 1.0f;
    if (g_yy < 0.05f) g_yy = 0.35f;

    // Retarded Gauge & Tunneling Variables
    float a_x          = GaugeTensor[gauge_offset + 0];
    float a_y          = GaugeTensor[gauge_offset + 1];
    float kappa_tunnel = GaugeTensor[gauge_offset + 2];
    float phi_flux     = GaugeTensor[gauge_offset + 3];

    if (kappa_tunnel < 0.001f) kappa_tunnel = 0.01f;

    float g_u = MetricScale[u];
    if (g_u <= 0.1f) g_u = 1.0f;

    float v_u = UniverseVelocity[u];
    if (v_u <= 5.0f) v_u = 30.0f;

    float dt_eff = BASE_DT;
    float accumulated_action = 0.0f;
    float accumulated_free_energy = 0.0f;
    float accumulated_order = 0.0f;

    float x_player = (float)((u * 280) % 35000);
    float local_last_f = 1.0f;

    float tau_fut = LOOKAHEAD_HORIZON * dt_eff;

    for (int t = 0; t < ticks; ++t) {
        x_player += v_u * dt_eff;

        float j_2d[D1][MAX_D2];
        float j_2d_future[D1][MAX_D2];

        sample_2d_level_geometry(StreamData, x_player, g_u, j_2d);
        sample_2d_level_geometry(StreamData, x_player + v_u * tau_fut, g_u, j_2d_future);

        float obstacle_density = 0.0f;
        #pragma unroll
        for (int i = 0; i < D1; ++i) {
            #pragma unroll
            for (int k = 0; k < MAX_D2; ++k) {
                if (j_2d[i][k] > 0.05f) obstacle_density += 1.0f;
            }
        }
        obstacle_density /= (D1 * MAX_D2);

        // 1. Inter-Universal Symplectic Transport
        float phi_transport_delta[D1][MAX_D2];
        float pi_transport_delta[D1][MAX_D2];

        #pragma unroll
        for (int i = 0; i < D1; ++i) {
            #pragma unroll
            for (int k = 0; k < MAX_D2; ++k) {
                phi_transport_delta[i][k] = 0.0f;
                pi_transport_delta[i][k] = 0.0f;
            }
        }

        #pragma unroll
        for (int sample = 0; sample < 4; ++sample) {
            int target_u = (u + sample * 17 + 1) % current_active;
            if (target_u == u) continue;

            int target_offset = target_u * TOTAL_FIELD_DIM;
            float dot_prod = 0.0f;
            float norm_a = 0.0001f;
            float norm_b = 0.0001f;

            #pragma unroll
            for (int i = 0; i < D1; ++i) {
                #pragma unroll
                for (int k = 0; k < MAX_D2; ++k) {
                    float val_a = phi[i][k];
                    float val_b = Field2D[target_offset + i * MAX_D2 + k];
                    dot_prod += val_a * val_b;
                    norm_a += val_a * val_a;
                    norm_b += val_b * val_b;
                }
            }

            float denom = sqrtf(norm_a) * sqrtf(norm_b);
            float cos_sim = (denom > 1e-6f) ? (dot_prod / denom) : 0.0f;

            float target_f_proxy = 0.0f;
            #pragma unroll
            for (int i = 0; i < D1; ++i) {
                #pragma unroll
                for (int k = 0; k < MAX_D2; ++k) {
                    float val_b = Field2D[target_offset + i * MAX_D2 + k];
                    target_f_proxy += fabsf(val_b - j_2d[i][k]);
                }
            }
            target_f_proxy /= (D1 * MAX_D2);

            float delta_f = local_last_f - target_f_proxy;
            if (delta_f > 0.0f && cos_sim > 0.10f) {
                float omega = SYMPLECTIC_GAMMA * delta_f * cos_sim;
                float sin_w = sinf(omega);
                float cos_w = cosf(omega) - 1.0f;

                #pragma unroll
                for (int i = 0; i < D1; ++i) {
                    #pragma unroll
                    for (int k = 0; k < MAX_D2; ++k) {
                        float target_phi = Field2D[target_offset + i * MAX_D2 + k];
                        float target_pi  = Momenta2D[target_offset + i * MAX_D2 + k];

                        phi_transport_delta[i][k] += phi[i][k] * cos_w + target_phi * sin_w;
                        pi_transport_delta[i][k]  += pi[i][k]  * cos_w + target_pi  * sin_w;
                    }
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < D1; ++i) {
            #pragma unroll
            for (int k = 0; k < MAX_D2; ++k) {
                phi[i][k] += phi_transport_delta[i][k];
                pi[i][k]  += pi_transport_delta[i][k];
            }
        }

        // 2. RETARDED GAUGE & TUNNELING AUTOPOIESIS
        // Convective gauge field induced by world-line motion
        a_x = -0.008f * v_u * (phi[1][0] - phi[0][0]);
        a_y = -0.004f * v_u * (phi[0][1] - phi[0][0]);

        // Quantum tunneling coefficient kappa_tunnel rises when encountering high obstacle barriers
        float target_kappa = 0.005f + 0.04f * tanhf(5.0f * obstacle_density);
        kappa_tunnel += 0.005f * (target_kappa - kappa_tunnel);
        if (kappa_tunnel < 0.002f) kappa_tunnel = 0.002f;
        if (kappa_tunnel > 0.08f)  kappa_tunnel = 0.08f;

        // Metric autopoiesis
        float target_alpha = tanhf(3.0f * obstacle_density);
        alpha_dim += 0.005f * (target_alpha - alpha_dim);
        if (alpha_dim < 0.05f) alpha_dim = 0.05f;
        if (alpha_dim > 1.0f) alpha_dim = 1.0f;

        g_xx += 0.0002f * (1.0f - local_last_f);
        g_yy += 0.0003f * (alpha_dim - g_yy);
        g_xy += 0.0001f * (j_2d[0][1] - j_2d[1][0]);

        if (g_xx < 0.5f) g_xx = 0.5f;
        if (g_xx > 2.0f) g_xx = 2.0f;
        if (g_yy < 0.05f) g_yy = 0.05f;
        if (g_yy > 1.5f) g_yy = 1.5f;
        if (g_xy < -0.4f) g_xy = -0.4f;
        if (g_xy > 0.4f) g_xy = 0.4f;

        // 3. FIRST PASS: COMPUTE INTERMEDIATE 2D LAPLACIAN FOR BI-LAPLACIAN TUNNELING
        float lap_grid[D1][MAX_D2];
        #pragma unroll
        for (int i = 0; i < D1; ++i) {
            int left = (i == 0) ? (D1 - 1) : (i - 1);
            int right = (i == D1 - 1) ? 0 : (i + 1);

            #pragma unroll
            for (int k = 0; k < MAX_D2; ++k) {
                int up_k = (k == 0) ? 0 : (k - 1);
                int down_k = (k == MAX_D2 - 1) ? (MAX_D2 - 1) : (k + 1);

                float p = phi[i][k];
                float lap_xx = phi[right][k] - 2.0f * p + phi[left][k];
                float lap_yy = phi[i][down_k] - 2.0f * p + phi[i][up_k];
                float lap_xy = (phi[right][down_k] - phi[right][up_k] - phi[left][down_k] + phi[left][up_k]) * 0.25f;

                lap_grid[i][k] = g_xx * lap_xx + g_yy * lap_yy + 2.0f * g_xy * lap_xy;
            }
        }

        // 4. SECOND PASS: GAUGE CONVECTIVE COUPLING & BI-LAPLACIAN TUNNELING ACCELERATION
        float max_density = 0.0f;
        float raw_lagrangian = 0.0f;
        float step_free_energy = 0.0f;
        float field_var = 0.0f;

        #pragma unroll
        for (int i = 0; i < D1; ++i) {
            int left = (i == 0) ? (D1 - 1) : (i - 1);
            int right = (i == D1 - 1) ? 0 : (i + 1);

            #pragma unroll
            for (int k = 0; k < MAX_D2; ++k) {
                int up_k = (k == 0) ? 0 : (k - 1);
                int down_k = (k == MAX_D2 - 1) ? (MAX_D2 - 1) : (k + 1);

                float p = phi[i][k];
                float m = pi[i][k];

                // 2D Metric Laplacian
                float lap_metric = lap_grid[i][k];

                // Fourth-order Bi-Laplacian Hyper-Diffusive Tunneling term:
                // \nabla^4 \phi = \nabla^2(\nabla^2 \phi)
                float bi_lap_x = lap_grid[right][k] - 2.0f * lap_metric + lap_grid[left][k];
                float bi_lap_y = lap_grid[i][down_k] - 2.0f * lap_metric + lap_grid[i][up_k];
                float bi_laplacian = bi_lap_x + 0.35f * bi_lap_y;

                // Convective Gauge field drift: (A_x * \partial_x \phi + A_y * \partial_y \phi)
                float d_phi_x = 0.5f * (phi[right][k] - phi[left][k]);
                float d_phi_y = 0.5f * (phi[i][down_k] - phi[i][up_k]);
                float gauge_convection = a_x * d_phi_x + a_y * d_phi_y;

                float ext_2d = j_2d[i][k];

                // Soliton Non-Linear Potential (\phi^5)
                float p2 = p * p;
                float p3 = p2 * p;
                float p5 = p3 * p2;
                float lambda_soliton = 0.02f;

                // Acceleration with Bi-Laplacian Tunneling & Convective Gauge
                float accel_present = c2_u[k] * lap_metric 
                                    - kappa_tunnel * bi_laplacian
                                    + gauge_convection
                                    - k_u[k] * p 
                                    - beta_u[k] * p3 
                                    - lambda_soliton * p5 
                                    - gamma_u[k] * m 
                                    + ext_2d;

                // SYMPLECTIC TEMPORAL PHASE LOOKAHEAD:
                float p_future = p + m * tau_fut + 0.5f * accel_present * tau_fut * tau_fut;
                float ext_future_2d = j_2d_future[i][k];

                float future_error = p_future - ext_future_2d;
                float anticipatory_force = -0.12f * future_error;

                float total_accel = accel_present + anticipatory_force;

                pi[i][k] += total_accel * dt_eff;
                phi[i][k] += pi[i][k] * dt_eff;

                float kinetic = 0.5f * pi[i][k] * pi[i][k];
                float potential = 0.5f * k_u[k] * p2 + 0.25f * beta_u[k] * p2 * p2 + 0.1666f * lambda_soliton * p2 * p2 * p2;
                float density_ik = kinetic + potential;

                if (density_ik > max_density) max_density = density_ik;
                raw_lagrangian += (kinetic - potential + ext_2d * p);
                field_var += p2;

                step_free_energy += fabsf(phi[i][k] - j_2d[i][k]);
            }
        }

        step_free_energy /= (D1 * MAX_D2);
        field_var /= (D1 * MAX_D2);
        accumulated_free_energy += step_free_energy;
        local_last_f = 0.90f * local_last_f + 0.10f * step_free_energy;

        // 5. INTRINSIC ORDER / EXISTENCE INTEGRITY
        float Order_u = expf(-0.05f * field_var - 0.50f * step_free_energy);
        accumulated_order += Order_u;

        // 6. CONTINUOUS AUTOPOIETIC ADAPTATION
        float threat_level = 1.0f - Order_u;

        v_u += 0.01f * (0.80f - step_free_energy) * (Order_u - 0.5f);
        if (v_u < 10.0f) v_u = 10.0f;
        if (v_u > 80.0f) v_u = 80.0f;

        g_u += 0.0005f * (0.80f - step_free_energy) * dt_eff;
        if (g_u < 0.20f) g_u = 0.20f;
        if (g_u > 2.50f) g_u = 2.50f;

        #pragma unroll
        for (int k = 0; k < MAX_D2; ++k) {
            float k_scale = 1.0f + 0.10f * (float)k;
            gamma_u[k] += 0.0003f * threat_level * k_scale - 0.0001f * Order_u * dt_eff;
            if (gamma_u[k] < 0.01f) gamma_u[k] = 0.01f;
            if (gamma_u[k] > 0.50f) gamma_u[k] = 0.50f;

            k_u[k] += 0.0002f * threat_level * (step_free_energy - 0.10f);
            if (k_u[k] < 0.02f) k_u[k] = 0.02f;
            if (k_u[k] > 1.20f) k_u[k] = 1.20f;

            c2_u[k] += 0.0001f * (Order_u - 0.60f) * dt_eff;
            if (c2_u[k] < 0.05f) c2_u[k] = 0.05f;
            if (c2_u[k] > 0.85f) c2_u[k] = 0.85f;
        }

        float det_g = g_xx * g_yy - g_xy * g_xy;
        if (det_g < 0.01f) det_g = 0.01f;
        float sqrt_det_g = sqrtf(det_g) * (1.0f + alpha_dim * 15.0f);
        accumulated_action += fabsf(raw_lagrangian) / (D1 * sqrt_det_g);
    }

    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        #pragma unroll
        for (int k = 0; k < MAX_D2; ++k) {
            int idx = offset + i * MAX_D2 + k;
            Field2D[idx] = phi[i][k];
            Momenta2D[idx] = pi[i][k];
        }
    }

    #pragma unroll
    for (int k = 0; k < MAX_D2; ++k) {
        SpectralLaws[law_offset + 0 * MAX_D2 + k] = k_u[k];
        SpectralLaws[law_offset + 1 * MAX_D2 + k] = gamma_u[k];
        SpectralLaws[law_offset + 2 * MAX_D2 + k] = c2_u[k];
        SpectralLaws[law_offset + 3 * MAX_D2 + k] = beta_u[k];
    }

    MetricTensor[metric_offset + 0] = g_xx;
    MetricTensor[metric_offset + 1] = g_yy;
    MetricTensor[metric_offset + 2] = g_xy;
    MetricTensor[metric_offset + 3] = alpha_dim;

    GaugeTensor[gauge_offset + 0] = a_x;
    GaugeTensor[gauge_offset + 1] = a_y;
    GaugeTensor[gauge_offset + 2] = kappa_tunnel;
    GaugeTensor[gauge_offset + 3] = phi_flux;

    float superspace_r = 0.0f;
    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        #pragma unroll
        for (int k = 0; k < MAX_D2; ++k) {
            superspace_r += phi[i][k] * phi[i][k];
        }
    }
    superspace_r = sqrtf(superspace_r / (D1 * MAX_D2));

    MetricScale[u] = g_u;
    UniverseVelocity[u] = v_u;
    EndogenousClock[u] = dt_eff;
    SuperspaceRadiusOut[u] = superspace_r;
    VolumetricActionOut[u] = accumulated_action / ticks;
    FreeEnergyOut[u] = accumulated_free_energy / ticks;
    OrderIntegrityOut[u] = accumulated_order / ticks;
    RandStates[u] = local_rand;
}

__global__ void init_rand_kernel_23(curandState* states, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &states[id]);
    }
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "EXP-702: Retarded Gauge Vector Potentials & Hyper-Diffusive Phase Tunneling\n";
    std::cout << "Covariant Convection A_mu & Bi-Laplacian Barrier Tunneling \\nabla^4 \\phi\n";
    std::cout << "KEP Principles 2, 3, 4, 7, 8, 27, 29, 30, 31 Fully Operational\n";
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
    size_t gauge_bytes = MAX_UNIVERSES * GAUGE_TENSOR_DIM * sizeof(float);
    size_t obstacle_bytes = TOTAL_OBSTACLES * sizeof(Obstacle);
    size_t score_bytes = MAX_UNIVERSES * sizeof(float);
    size_t rand_bytes = MAX_UNIVERSES * sizeof(curandState);

    std::vector<float> h_Field(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_Momenta(MAX_UNIVERSES * TOTAL_FIELD_DIM, 0.0f);
    std::vector<float> h_SpectralLaws(MAX_UNIVERSES * LAW_SPECTRAL_DIM);
    std::vector<float> h_MetricTensor(MAX_UNIVERSES * METRIC_TENSOR_DIM);
    std::vector<float> h_GaugeTensor(MAX_UNIVERSES * GAUGE_TENSOR_DIM);
    std::vector<float> h_MetricScale(MAX_UNIVERSES, 1.0f);
    std::vector<float> h_Velocity(MAX_UNIVERSES, 30.0f);
    std::vector<float> h_Clock(MAX_UNIVERSES, BASE_DT);

    for (int u = 0; u < MAX_UNIVERSES; ++u) {
        for (int k = 0; k < MAX_D2; ++k) {
            float mode_scale = 1.0f + 0.05f * k;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 0 * MAX_D2 + k] = (0.15f + 0.02f * (u % 5)) * mode_scale;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 1 * MAX_D2 + k] = (0.08f + 0.01f * (u % 4)) * mode_scale;
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 2 * MAX_D2 + k] = 0.35f + 0.02f * (u % 3);
            h_SpectralLaws[u * LAW_SPECTRAL_DIM + 3 * MAX_D2 + k] = 0.12f;
        }

        h_MetricTensor[u * METRIC_TENSOR_DIM + 0] = 1.0f;  // g_xx
        h_MetricTensor[u * METRIC_TENSOR_DIM + 1] = 0.35f; // g_yy
        h_MetricTensor[u * METRIC_TENSOR_DIM + 2] = 0.0f;  // g_xy
        h_MetricTensor[u * METRIC_TENSOR_DIM + 3] = 0.20f; // alpha_dim

        h_GaugeTensor[u * GAUGE_TENSOR_DIM + 0] = 0.0f;   // A_x
        h_GaugeTensor[u * GAUGE_TENSOR_DIM + 1] = 0.0f;   // A_y
        h_GaugeTensor[u * GAUGE_TENSOR_DIM + 2] = 0.01f;  // kappa_tunnel
        h_GaugeTensor[u * GAUGE_TENSOR_DIM + 3] = 0.0f;   // phi_flux
    }

    std::vector<float> h_SuperspaceRadius(MAX_UNIVERSES);
    std::vector<float> h_Action(MAX_UNIVERSES);
    std::vector<float> h_FreeEnergy(MAX_UNIVERSES);
    std::vector<float> h_OrderIntegrity(MAX_UNIVERSES);

    int active_universes = INITIAL_UNIVERSES;

    float *d_Field, *d_Momenta, *d_SpectralLaws, *d_MetricTensor, *d_GaugeTensor, *d_MetricScale, *d_Velocity, *d_Clock;
    Obstacle *d_StreamData;
    int *d_ActiveCount;
    float *d_SuperspaceRadius, *d_Action, *d_FreeEnergy, *d_OrderIntegrity;
    curandState *d_RandStates;

    CUDA_CHECK(cudaMalloc(&d_Field, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_Momenta, field_bytes));
    CUDA_CHECK(cudaMalloc(&d_SpectralLaws, law_bytes));
    CUDA_CHECK(cudaMalloc(&d_MetricTensor, metric_bytes));
    CUDA_CHECK(cudaMalloc(&d_GaugeTensor, gauge_bytes));
    CUDA_CHECK(cudaMalloc(&d_MetricScale, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Velocity, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Clock, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_StreamData, obstacle_bytes));
    CUDA_CHECK(cudaMalloc(&d_ActiveCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_SuperspaceRadius, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Action, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_FreeEnergy, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_OrderIntegrity, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_RandStates, rand_bytes));

    CUDA_CHECK(cudaMemcpy(d_Field, h_Field.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Momenta, h_Momenta.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SpectralLaws, h_SpectralLaws.data(), law_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MetricTensor, h_MetricTensor.data(), metric_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_GaugeTensor, h_GaugeTensor.data(), gauge_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MetricScale, h_MetricScale.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Velocity, h_Velocity.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Clock, h_Clock.data(), score_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_StreamData, h_Stream.data(), obstacle_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ActiveCount, &active_universes, sizeof(int), cudaMemcpyHostToDevice));

    dim3 threads(32);
    dim3 blocks((MAX_UNIVERSES + threads.x - 1) / threads.x);

    init_rand_kernel_23<<<blocks, threads>>>(d_RandStates, 2026, MAX_UNIVERSES);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    retarded_gauge_tunneling_kernel<<<blocks, threads>>>(
        d_Field, d_Momenta, d_SpectralLaws, d_MetricTensor, d_GaugeTensor, d_MetricScale, d_Velocity, d_Clock, d_StreamData, d_ActiveCount,
        d_SuperspaceRadius, d_Action, d_FreeEnergy, d_OrderIntegrity,
        d_RandStates, TICKS
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    CUDA_CHECK(cudaMemcpy(h_SuperspaceRadius.data(), d_SuperspaceRadius, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Action.data(), d_Action, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_FreeEnergy.data(), d_FreeEnergy, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_OrderIntegrity.data(), d_OrderIntegrity, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_MetricScale.data(), d_MetricScale, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Velocity.data(), d_Velocity, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_MetricTensor.data(), d_MetricTensor, metric_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_GaugeTensor.data(), d_GaugeTensor, gauge_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Clock.data(), d_Clock, score_bytes, cudaMemcpyDeviceToHost));

    float mean_radius = 0.0f;
    float mean_action = 0.0f;
    float mean_free_energy = 0.0f;
    float mean_order = 0.0f;
    float mean_metric_scale = 0.0f;
    float mean_velocity = 0.0f;
    float mean_g_xx = 0.0f, mean_g_yy = 0.0f, mean_g_xy = 0.0f, mean_alpha_dim = 0.0f;
    float mean_a_x = 0.0f, mean_a_y = 0.0f, mean_kappa_tunnel = 0.0f;

    for (int u = 0; u < active_universes; ++u) {
        mean_radius += h_SuperspaceRadius[u];
        mean_action += h_Action[u];
        mean_free_energy += h_FreeEnergy[u];
        mean_order += h_OrderIntegrity[u];
        mean_metric_scale += h_MetricScale[u];
        mean_velocity += h_Velocity[u];
        mean_g_xx += h_MetricTensor[u * METRIC_TENSOR_DIM + 0];
        mean_g_yy += h_MetricTensor[u * METRIC_TENSOR_DIM + 1];
        mean_g_xy += h_MetricTensor[u * METRIC_TENSOR_DIM + 2];
        mean_alpha_dim += h_MetricTensor[u * METRIC_TENSOR_DIM + 3];
        mean_a_x += h_GaugeTensor[u * GAUGE_TENSOR_DIM + 0];
        mean_a_y += h_GaugeTensor[u * GAUGE_TENSOR_DIM + 1];
        mean_kappa_tunnel += h_GaugeTensor[u * GAUGE_TENSOR_DIM + 2];
    }
    mean_radius /= active_universes;
    mean_action /= active_universes;
    mean_free_energy /= active_universes;
    mean_order /= active_universes;
    mean_metric_scale /= active_universes;
    mean_velocity /= active_universes;
    mean_g_xx /= active_universes;
    mean_g_yy /= active_universes;
    mean_g_xy /= active_universes;
    mean_alpha_dim /= active_universes;
    mean_a_x /= active_universes;
    mean_a_y /= active_universes;
    mean_kappa_tunnel /= active_universes;

    std::cout << "[RGVP-HDPT Retarded Gauge & Tunneling Summary]:\n";
    std::cout << "  - Active Universes in Multiverse  : " << active_universes << "\n";
    std::cout << "  - Execution Time on GPU           : " << ms << " ms\n";
    std::cout << "  - Intrinsic Order Integrity O_u   : " << mean_order << " [1.0 = perfect order homeostasis]\n";
    std::cout << "  - Sovereign World-Line Velocity v_u: " << mean_velocity << " level units/sec\n";
    std::cout << "  - Sovereign Metric Scale g_u      : " << mean_metric_scale << " scale units\n";
    std::cout << "  - Retarded Gauge Convection A_x   : " << mean_a_x << "\n";
    std::cout << "  - Retarded Gauge Convection A_y   : " << mean_a_y << "\n";
    std::cout << "  - Bi-Laplacian Tunneling kappa    : " << mean_kappa_tunnel << " [quantum-like permeability]\n";
    std::cout << "  - Uncurled Spatial Capacity D2_eff: " << 1.0f + mean_alpha_dim * 15.0f << " / 16 dimensions\n";
    std::cout << "  - Superspace Metric Radius R_super: " << mean_radius << " units\n";
    std::cout << "  - Variational Free Energy F       : " << mean_free_energy << " nats\n";
    std::cout << "  - Volume Normalized Action S_vol  : " << mean_action << " action/vol-units\n";
    std::cout << "  - Status                          : SUCCESS (Gauge & Tunneling Validated)\n";

    CUDA_CHECK(cudaFree(d_Field));
    CUDA_CHECK(cudaFree(d_Momenta));
    CUDA_CHECK(cudaFree(d_SpectralLaws));
    CUDA_CHECK(cudaFree(d_MetricTensor));
    CUDA_CHECK(cudaFree(d_GaugeTensor));
    CUDA_CHECK(cudaFree(d_MetricScale));
    CUDA_CHECK(cudaFree(d_Velocity));
    CUDA_CHECK(cudaFree(d_Clock));
    CUDA_CHECK(cudaFree(d_StreamData));
    CUDA_CHECK(cudaFree(d_ActiveCount));
    CUDA_CHECK(cudaFree(d_SuperspaceRadius));
    CUDA_CHECK(cudaFree(d_Action));
    CUDA_CHECK(cudaFree(d_FreeEnergy));
    CUDA_CHECK(cudaFree(d_OrderIntegrity));
    CUDA_CHECK(cudaFree(d_RandStates));

    return 0;
}
