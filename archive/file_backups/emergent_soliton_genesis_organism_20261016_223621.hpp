// ============================================================================
// EXP-518: Emergent Soliton Condensation & Autopoietic Particle Genesis
// Karyon-CoRE v28.0 Sovereign Architecture
//
// Root Cause of EXP-518 initial divergence:
// Quadratic strain (strain^2) is always positive: \int x * strain^2 dx is asymmetric
// and doesn't preserve sign under coordinate reflection when unpolarized!
// The true linear stress tensor flux is T_{01} = T(x) * \partial_x \Psi * \partial_t \Psi,
// and the restoring boundary torque is the odd-parity spatial momentum flux:
// \tau = \int_{-1}^{1} \partial_x \Psi(x, t) dx = \Psi(1, t) - \Psi(-1, t).
// By Stokes' Theorem, the boundary traction of a continuous 1D string field is
// simply the net difference between the right and left endpoint displacements!
// ============================================================================

#ifndef EMERGENCE_SOLITON_GENESIS_HPP
#define EMERGENCE_SOLITON_GENESIS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>
#include <string>

namespace karyon {

// Quantitative detector of emergent structures from pure field telemetry
struct EmergentStructuresTelemetry {
    size_t emergent_solitons = 0;       // Localized high-density energy packets (hadrons/atoms)
    size_t emergent_bound_states = 0;   // Coherent phase-locked soliton pairs (molecules)
    size_t emergent_domain_walls = 0;   // Invariant topological phase boundaries (cells/membranes)
    float field_coherence = 0.0f;
    float total_field_energy = 0.0f;
    float self_synthesized_stiffness = 0.0f;
    float self_synthesized_damping = 0.0f;
    float free_energy = 0.0f;
};

struct AutopoieticGenesisConfig {
    size_t spatial_resolution = 64;   // Discrete points of continuous spatial manifold x \in [-1, 1]
    float spatial_dx = 2.0f / 63.0f;
    float eta_autopoiesis = 0.08f;    // Sovereign plasticity rate
    size_t symplectic_substeps = 8;
};

class AutopoieticSolitonContinuum {
public:
    explicit AutopoieticSolitonContinuum(const AutopoieticGenesisConfig& cfg = AutopoieticGenesisConfig())
        : cfg_(cfg),
          x_(cfg.spatial_resolution),
          psi_(cfg.spatial_resolution, 0.0f),
          psi_dot_(cfg.spatial_resolution, 0.0f),
          metric_stiffness_(cfg.spatial_resolution, 5.0f),
          metric_damping_(cfg.spatial_resolution, 1.0f)
    {
        for (size_t i = 0; i < cfg_.spatial_resolution; ++i) {
            x_[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(cfg_.spatial_resolution - 1);
        }
    }

    // Single unified physical step: pure field dynamics without top-down structs
    float perceive_act_and_condense(float error, float error_rate, float dt) {
        size_t N = cfg_.spatial_resolution;
        float dx = cfg_.spatial_dx;
        float sub_dt = dt / static_cast<float>(cfg_.symplectic_substeps);

        // 1. Variational Free Energy: F = 1/2 * e^2 + 0.02 * de^2
        float fe = 0.5f * error * error + 0.02f * error_rate * error_rate;

        // 2. Continuous Spatial Excitation (Odd-parity coordinate plucking):
        // External mismatch polarizes the continuous field: S(x) = (error + 0.15 * error_rate) * x
        std::vector<float> field_source(N, 0.0f);
        for (size_t i = 0; i < N; ++i) {
            field_source[i] = (error * x_[i] + 0.15f * error_rate * x_[i]) * 20.0f;
        }

        // 3. Symplectic Non-Linear Klein-Gordon / Soliton Wave Evolution:
        // \partial_t^2 \Psi = \partial_x (T(x) \partial_x \Psi) - \lambda \sin(\Psi) - \Gamma(x) \partial_t \Psi + S(x)
        for (size_t sub = 0; sub < cfg_.symplectic_substeps; ++sub) {
            for (size_t i = 0; i < N; ++i) {
                // Continuum Laplacian flux
                float flux = 0.0f;
                if (i > 0 && i + 1 < N) {
                    float T_l = 0.5f * (metric_stiffness_[i] + metric_stiffness_[i - 1]);
                    float T_r = 0.5f * (metric_stiffness_[i] + metric_stiffness_[i + 1]);
                    flux = (T_r * (psi_[i + 1] - psi_[i]) - T_l * (psi_[i] - psi_[i - 1])) / (dx * dx);
                } else if (i == 0) {
                    flux = metric_stiffness_[0] * (psi_[1] - psi_[0]) / (dx * dx);
                } else {
                    flux = metric_stiffness_[N - 1] * (psi_[N - 2] - psi_[N - 1]) / (dx * dx);
                }

                // Non-linear cubic self-focusing potential (NLS / Solitons): - \lambda * \Psi^3
                float nonlin_potential = -0.5f * psi_[i] * psi_[i] * psi_[i];
                float damping_force    = -metric_damping_[i] * psi_dot_[i];

                float acc = flux + nonlin_potential + damping_force + field_source[i];

                psi_dot_[i] = std::clamp(psi_dot_[i] + acc * sub_dt, -40.0f, 40.0f);
                psi_[i]     = std::clamp(psi_[i] + psi_dot_[i] * sub_dt, -15.0f, 15.0f);
            }
        }

        // 4. Sovereign Autopoietic Self-Tuning (Principle 29):
        // When free energy exists, the metric tensor autonomously morphs its local stiffness and damping
        if (fe > 1e-12f) {
            float d_k = cfg_.eta_autopoiesis * std::abs(error) * dt;
            float d_gamma = cfg_.eta_autopoiesis * std::abs(error_rate) * dt;

            for (size_t i = 0; i < N; ++i) {
                metric_stiffness_[i] = std::clamp(metric_stiffness_[i] + d_k * std::abs(x_[i]), 0.1f, 80.0f);
                metric_damping_[i]   = std::clamp(metric_damping_[i] + d_gamma, 0.1f, 30.0f);
            }
        }

        // 5. Boundary Traction via Stokes' Theorem:
        // The net boundary torque exerted by the 1D field is the difference in boundary displacements
        // weighted by boundary metric tension:
        // \tau = T(1) * \Psi(1) - T(-1) * \Psi(-1)
        float left_traction  = metric_stiffness_[0] * psi_[0];
        float right_traction = metric_stiffness_[N - 1] * psi_[N - 1];
        float net_torque = (right_traction - left_traction) * 2.5f;

        latest_fe_ = fe;
        latest_torque_ = net_torque;

        return latest_torque_;
    }

    // Endoscopic probe: analyze emergent physical structures in the continuous field
    EmergentStructuresTelemetry analyze_emergent_structures() const {
        EmergentStructuresTelemetry telem;
        size_t N = cfg_.spatial_resolution;
        float dx = cfg_.spatial_dx;

        float total_energy = 0.0f;
        float k_sum = 0.0f;
        float d_sum = 0.0f;

        std::vector<float> energy_density(N, 0.0f);

        for (size_t i = 0; i < N; ++i) {
            float strain = (i + 1 < N) ? ((psi_[i + 1] - psi_[i]) / dx) : ((psi_[i] - psi_[i - 1]) / dx);
            float u = 0.5f * metric_stiffness_[i] * (strain * strain) + 0.5f * (psi_dot_[i] * psi_dot_[i]) + 0.125f * (psi_[i] * psi_[i] * psi_[i] * psi_[i]);
            energy_density[i] = u;
            total_energy += u * dx;
            k_sum += metric_stiffness_[i];
            d_sum += metric_damping_[i];
        }

        // 1. Detect Solitons (Localized peaks in energy density > 1.8x average)
        float avg_energy = (total_energy / 2.0f) + 1e-6f;
        std::vector<size_t> peak_indices;
        for (size_t i = 1; i + 1 < N; ++i) {
            if (energy_density[i] > 1.8f * avg_energy &&
                energy_density[i] > energy_density[i - 1] &&
                energy_density[i] > energy_density[i + 1]) {
                telem.emergent_solitons++;
                peak_indices.push_back(i);
            }
        }

        // 2. Detect Bound Breather States (Pairs of solitons separated by delta_x < 0.45)
        if (peak_indices.size() >= 2) {
            for (size_t p = 0; p + 1 < peak_indices.size(); ++p) {
                float dist = std::abs(x_[peak_indices[p + 1]] - x_[peak_indices[p]]);
                if (dist < 0.45f) {
                    telem.emergent_bound_states++;
                }
            }
        }

        // 3. Detect Domain Walls (Topological phase crossings where field crosses zero with steep gradient)
        for (size_t i = 0; i + 1 < N; ++i) {
            if ((psi_[i] < 0.0f && psi_[i + 1] > 0.0f) || (psi_[i] > 0.0f && psi_[i + 1] < 0.0f)) {
                float grad = std::abs(psi_[i + 1] - psi_[i]) / dx;
                if (grad > 0.5f) {
                    telem.emergent_domain_walls++;
                }
            }
        }

        telem.total_field_energy = total_energy;
        telem.self_synthesized_stiffness = k_sum / static_cast<float>(N);
        telem.self_synthesized_damping = d_sum / static_cast<float>(N);
        telem.free_energy = latest_fe_;

        return telem;
    }

private:
    AutopoieticGenesisConfig cfg_;
    std::vector<float> x_;
    std::vector<float> psi_;
    std::vector<float> psi_dot_;
    std::vector<float> metric_stiffness_;
    std::vector<float> metric_damping_;

    float latest_fe_ = 0.0f;
    float latest_torque_ = 0.0f;
};

} // namespace karyon

#endif // EMERGENCE_SOLITON_GENESIS_HPP
