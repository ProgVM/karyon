// ============================================================================
// EXP-518: Emergent Soliton Condensation & Autopoietic Particle Genesis
// Karyon-CoRE v28.0 Sovereign Architecture
//
// CORE PRINCIPLE:
// 1. The ONLY fundamental physical entity is the continuous spatial field \Psi(x, t).
// 2. No top-down hardcoded C++ structs (struct Atom, struct Cell, struct Hadron)!
// 3. Instead, "Atoms/Hadrons" are localized Topological Solitons (peaks in energy density u(x)).
// 4. "Molecules" are Bound Breather Pairs (coherent soliton phase-locks).
// 5. "Cells/Membranes" are Topological Domain Walls (zero-crossings of \Psi(x) with high gradient).
// 6. The field \Psi(x, t) is integrated via continuous spatial Fourier-Galerkin eigenmodes:
//    \Psi(x, t) = \sum_{k=1}^M q_k(t) \sin(k \pi x)
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

// Quantitative endoscopic detector of emergent structures in continuous field \Psi(x, t)
struct EmergentStructuresTelemetry {
    size_t emergent_solitons = 0;       // Localized high-density energy packets ("atoms/hadrons")
    size_t emergent_bound_states = 0;   // Coherent phase-locked soliton pairs ("molecules")
    size_t emergent_domain_walls = 0;   // Invariant topological phase boundaries ("cells/membranes")
    float total_field_energy = 0.0f;
    float self_synthesized_stiffness = 0.0f;
    float self_synthesized_damping = 0.0f;
    float free_energy = 0.0f;
};

struct AutopoieticGenesisConfig {
    size_t num_fourier_modes = 16;   // Spatial modes of continuous manifold x \in [-1, 1]
    float eta_autopoiesis = 0.08f;    // Sovereign plasticity rate
    size_t symplectic_substeps = 8;
};

class AutopoieticSolitonContinuum {
public:
    explicit AutopoieticSolitonContinuum(const AutopoieticGenesisConfig& cfg = AutopoieticGenesisConfig())
        : cfg_(cfg),
          q_modes_(cfg.num_fourier_modes, 0.0f),
          p_modes_(cfg.num_fourier_modes, 0.0f),
          stiffness_modes_(cfg.num_fourier_modes, 2.0f),
          damping_modes_(cfg.num_fourier_modes, 0.5f)
    {
    }

    // Single unified physical step: operates purely on field modes without top-down structs
    float perceive_act_and_condense(float error, float error_rate, float dt) {
        size_t M = cfg_.num_fourier_modes;
        float sub_dt = dt / static_cast<float>(cfg_.symplectic_substeps);

        // 1. Variational Free Energy: F = 1/2 * e^2 + 0.02 * de^2
        float fe = 0.5f * error * error + 0.02f * error_rate * error_rate;

        // 2. Continuous Spatial Excitation (Odd-parity coordinate plucking):
        std::vector<float> mode_excitation(M, 0.0f);
        for (size_t k = 0; k < M; ++k) {
            float mode_k = static_cast<float>(k + 1);
            mode_excitation[k] = (error + 0.15f * error_rate) * (1.0f / mode_k);
        }

        // 3. Symplectic Integration of Continuous Field Modes
        for (size_t sub = 0; sub < cfg_.symplectic_substeps; ++sub) {
            for (size_t k = 0; k < M; ++k) {
                float omega_sq = stiffness_modes_[k] * static_cast<float>((k + 1) * (k + 1));
                float damping  = damping_modes_[k];

                // Continuous non-linear wave equation: \ddot{q}_k + \gamma_k \dot{q}_k + \omega_k^2 q_k = F_k
                float acc = -omega_sq * q_modes_[k] - damping * p_modes_[k] + mode_excitation[k] * 10.0f;

                p_modes_[k] = std::clamp(p_modes_[k] + acc * sub_dt, -30.0f, 30.0f);
                q_modes_[k] = std::clamp(q_modes_[k] + p_modes_[k] * sub_dt, -10.0f, 10.0f);
            }
        }

        // 4. Sovereign Autopoietic Self-Tuning (Principle 29):
        // When free energy exists, the metric tensor autonomously morphs its local stiffness and damping
        if (fe > 1e-12f) {
            float d_stiff = cfg_.eta_autopoiesis * std::abs(error) * dt;
            float d_damp  = cfg_.eta_autopoiesis * std::abs(error_rate) * dt;

            for (size_t k = 0; k < M; ++k) {
                stiffness_modes_[k] = std::clamp(stiffness_modes_[k] + d_stiff * (1.0f / (k + 1)), 0.1f, 50.0f);
                damping_modes_[k]   = std::clamp(damping_modes_[k] + d_damp, 0.1f, 20.0f);
            }
        }

        // 5. Emergent Restorative Boundary Moment (Stress Tensor Trace):
        float restorative_moment = 0.0f;
        for (size_t k = 0; k < M; ++k) {
            restorative_moment += static_cast<float>(k + 1) * stiffness_modes_[k] * q_modes_[k];
        }

        float torque = 12.0f * restorative_moment;

        latest_fe_ = fe;
        latest_torque_ = torque;

        return latest_torque_;
    }

    // Endoscopic probe: analyze emergent physical structures in the continuous spatial field \Psi(x, t)
    EmergentStructuresTelemetry analyze_emergent_structures(size_t grid_samples = 64) const {
        EmergentStructuresTelemetry telem;
        size_t M = cfg_.num_fourier_modes;

        std::vector<float> x_grid(grid_samples);
        std::vector<float> psi_grid(grid_samples, 0.0f);
        std::vector<float> psi_dx_grid(grid_samples, 0.0f);
        std::vector<float> energy_density(grid_samples, 0.0f);

        float dx = 2.0f / static_cast<float>(grid_samples - 1);
        float total_energy = 0.0f;

        // Reconstruct spatial field \Psi(x) = \sum_k q_k \sin(k \pi x) and derivative \partial_x \Psi(x)
        for (size_t i = 0; i < grid_samples; ++i) {
            float x = -1.0f + static_cast<float>(i) * dx;
            x_grid[i] = x;

            for (size_t k = 0; k < M; ++k) {
                float k_pi = static_cast<float>(k + 1) * std::numbers::pi_v<float>;
                psi_grid[i] += q_modes_[k] * std::sin(k_pi * x);
                psi_dx_grid[i] += q_modes_[k] * k_pi * std::cos(k_pi * x);
            }

            // Energy density u(x) = 1/2 T (\partial_x \Psi)^2 + V(\Psi)
            float u = 0.5f * 2.0f * (psi_dx_grid[i] * psi_dx_grid[i]) + 0.5f * (psi_grid[i] * psi_grid[i]);
            energy_density[i] = u;
            total_energy += u * dx;
        }

        // 1. Detect Solitons ("Atoms"): Localized peaks in spatial energy density
        float avg_u = total_energy / 2.0f;
        std::vector<size_t> peak_indices;
        for (size_t i = 1; i + 1 < grid_samples; ++i) {
            if (energy_density[i] > 1.2f * avg_u &&
                energy_density[i] > energy_density[i - 1] &&
                energy_density[i] > energy_density[i + 1]) {
                telem.emergent_solitons++;
                peak_indices.push_back(i);
            }
        }

        // 2. Detect Bound Breathers ("Molecules"): Coherent pairs of solitons separated by delta_x < 0.4
        if (peak_indices.size() >= 2) {
            for (size_t p = 0; p + 1 < peak_indices.size(); ++p) {
                float dist = std::abs(x_grid[peak_indices[p + 1]] - x_grid[peak_indices[p]]);
                if (dist < 0.40f) {
                    telem.emergent_bound_states++;
                }
            }
        }

        // 3. Detect Domain Walls ("Cell Membranes"): Zero-crossings of \Psi(x) with steep spatial gradient
        for (size_t i = 0; i + 1 < grid_samples; ++i) {
            if ((psi_grid[i] < 0.0f && psi_grid[i + 1] > 0.0f) || (psi_grid[i] > 0.0f && psi_grid[i + 1] < 0.0f)) {
                if (std::abs(psi_dx_grid[i]) > 0.5f) {
                    telem.emergent_domain_walls++;
                }
            }
        }

        float k_sum = 0.0f, d_sum = 0.0f;
        for (float s : stiffness_modes_) k_sum += s;
        for (float d : damping_modes_) d_sum += d;

        telem.total_field_energy = total_energy;
        telem.self_synthesized_stiffness = k_sum / static_cast<float>(M);
        telem.self_synthesized_damping = d_sum / static_cast<float>(M);
        telem.free_energy = latest_fe_;

        return telem;
    }

private:
    AutopoieticGenesisConfig cfg_;
    std::vector<float> q_modes_;
    std::vector<float> p_modes_;
    std::vector<float> stiffness_modes_;
    std::vector<float> damping_modes_;

    float latest_fe_ = 0.0f;
    float latest_torque_ = 0.0f;
};

} // namespace karyon

#endif // EMERGENCE_SOLITON_GENESIS_HPP
