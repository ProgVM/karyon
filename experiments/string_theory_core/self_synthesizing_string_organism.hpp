// ============================================================================
// SelfSynthesizingStringOrganism: Zero Hand-Crafted Rules, Pure String Dynamics
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-514)
//
// "Rules are not hardcoded; rules are self-synthesized by the vibrating strings."
//
// ZERO TOP-DOWN MACRO LAWS:
// - No hardcoded Hodgkin-Huxley equations
// - No hardcoded Nernst formulas
// - No hardcoded tissue elasticity or muscle constants
//
// All macroscopic observables emerge as field projections of vibrating strings:
// 1. Local Potential V(x, t) = Re(Psi^* \partial_t Psi)   [String Phase Velocity]
// 2. Local Flux / Currents J(x, t) = Im(Psi^* \nabla Psi) [String Noether Current]
// 3. Mechanical Stress T_mu_nu = \nabla_mu Psi \nabla_nu Psi^* [Stress-Energy Tensor]
// 4. Actuation Force F = \int T_ij dS [Emergent Boundary Traction]
// ============================================================================

#ifndef SELF_SYNTHESIZING_STRING_ORGANISM_HPP
#define SELF_SYNTHESIZING_STRING_ORGANISM_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>
#include <string>

namespace karyon {

struct StringMeshNode {
    float x = 0.0f; // Spatial coordinate [-1, 1]
    float y = 0.0f;

    // Local string field state: sum of harmonic vibrational modes
    // Psi(x) = u(x) + i * v(x)
    float u = 0.0f;
    float v = 0.0f;
    float u_dot = 0.0f;
    float v_dot = 0.0f;

    // Emergent field observables synthesized purely by strings:
    float emergent_potential = 0.0f; // V(x) = Re(Psi^* Psi_dot)
    float emergent_flux_x = 0.0f;    // J_x = Im(Psi^* \partial_x Psi)
    float emergent_flux_y = 0.0f;    // J_y = Im(Psi^* \partial_y Psi)
    float emergent_stress_yy = 0.0f; // T_yy = (\partial_y Psi)^2
};

struct StringContinuumTelemetry {
    size_t total_nodes = 0;
    float total_string_action = 0.0f;
    float avg_emergent_potential = 0.0f;
    float left_boundary_traction = 0.0f;
    float right_boundary_traction = 0.0f;
    float net_physical_torque = 0.0f;
    float free_energy = 0.0f;
    std::vector<float> motor_efferent_actions;
};

struct StringContinuumConfig {
    size_t sensory_dim = 256;
    size_t action_dim = 4;
    size_t num_string_modes = 48; // Fundamental harmonic modes
    size_t grid_nx = 4;           // Spatial discrete mesh 4x6 = 24 nodes
    size_t grid_ny = 6;
    size_t relaxation_substeps = 8;
    float string_tension = 1.8f;  // Fundamental string tension T_0
    float string_coupling_beta = 0.02f; // Fermi-Pasta-Ulam non-linear interaction
    float dt = 0.04f;
};

class SelfSynthesizingStringOrganism {
public:
    explicit SelfSynthesizingStringOrganism(const StringContinuumConfig& cfg)
        : config_(cfg),
          mode_frequencies_(cfg.num_string_modes),
          nodes_(cfg.grid_nx * cfg.grid_ny)
    {
        // 1. Dispersion relation of strings: omega_n = c * sqrt(k_n^2 + m_0^2)
        for (size_t n = 0; n < config_.num_string_modes; ++n) {
            float mode_idx = static_cast<float>(n + 1);
            mode_frequencies_[n] = 0.15f * std::sqrt(mode_idx * mode_idx + 0.25f);
        }

        // 2. Spatial layout of the physical continuum mesh
        size_t idx = 0;
        for (size_t iy = 0; iy < config_.grid_ny; ++iy) {
            float y = -0.85f + 1.7f * static_cast<float>(iy) / static_cast<float>(config_.grid_ny - 1);
            for (size_t ix = 0; ix < config_.grid_nx; ++ix) {
                float x = -0.75f + 1.5f * static_cast<float>(ix) / static_cast<float>(config_.grid_nx - 1);
                nodes_[idx].x = x;
                nodes_[idx].y = y;
                idx++;
            }
        }

        // 3. Precompute topological boundary string coupling
        mode_eigenmodes_.resize(config_.num_string_modes, std::vector<float>(nodes_.size()));
        for (size_t n = 0; n < config_.num_string_modes; ++n) {
            float kx = std::numbers::pi_v<float> * static_cast<float>((n % 6) + 1);
            float ky = std::numbers::pi_v<float> * static_cast<float>((n / 6) + 1);
            for (size_t i = 0; i < nodes_.size(); ++i) {
                // Dirichlet-Neumann standing waves on manifold
                mode_eigenmodes_[n][i] = std::sin(kx * (nodes_[i].x + 1.0f) * 0.5f) *
                                         std::cos(ky * (nodes_[i].y + 1.0f) * 0.5f);
            }
        }

        // 4. Continuous Fourier sensor plucking basis
        fourier_plucking_.resize(config_.num_string_modes, std::vector<float>(config_.sensory_dim));
        for (size_t n = 0; n < config_.num_string_modes; ++n) {
            float mode_idx = static_cast<float>(n + 1);
            for (size_t k = 0; k < config_.sensory_dim; ++k) {
                float angle = 2.0f * std::numbers::pi_v<float> * mode_idx * static_cast<float>(k) / static_cast<float>(config_.sensory_dim);
                fourier_plucking_[n][k] = std::cos(angle) / std::sqrt(static_cast<float>(config_.sensory_dim));
            }
        }
    }

    StringContinuumTelemetry perceive_and_act(const std::vector<float>& sensory_state,
                                              const std::vector<float>& target_goal,
                                              float external_valence)
    {
        StringContinuumTelemetry state;
        state.motor_efferent_actions.assign(config_.action_dim, 0.0f);

        float macro_dt = config_.dt;
        float sub_dt = macro_dt / static_cast<float>(config_.relaxation_substeps);
        size_t M = config_.num_string_modes;
        size_t N = nodes_.size();
        size_t S = config_.sensory_dim;
        size_t nx = config_.grid_nx;
        size_t ny = config_.grid_ny;

        // 1. Sensory Field Plucking Force: F_sensory(x) = \sum_k (s_k - g_k) * Pluck(n, k)
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];
        }

        std::vector<float> mode_plucking_force(M, 0.0f);
        for (size_t n = 0; n < M; ++n) {
            float force = 0.0f;
            for (size_t k = 0; k < S; ++k) {
                force += err_s[k] * fourier_plucking_[n][k];
            }
            mode_plucking_force[n] = force * 2.5f;
        }

        // ====================================================================
        // PURE STRING FIELD RELAXATION (NO HANDCODED CELL EQUATIONS)
        // Wave Action: S = \int dt d^2x [ 1/2 (\partial_t \Psi)^2 - 1/2 (\nabla \Psi)^2 - V(\Psi) ]
        // ====================================================================
        float total_string_action = 0.0f;

        for (size_t substep = 0; substep < config_.relaxation_substeps; ++substep) {

            // Step A: Spatial Laplacian \nabla^2 \Psi across the mesh manifold
            std::vector<float> laplacian_u(N, 0.0f);
            std::vector<float> laplacian_v(N, 0.0f);

            for (size_t iy = 0; iy < ny; ++iy) {
                for (size_t ix = 0; ix < nx; ++ix) {
                    size_t idx = iy * nx + ix;
                    float u_c = nodes_[idx].u;
                    float v_c = nodes_[idx].v;

                    float sum_diff_u = 0.0f;
                    float sum_diff_v = 0.0f;

                    // 4-neighbor spatial coupling on the continuum
                    if (ix > 0) {
                        sum_diff_u += (nodes_[iy * nx + (ix - 1)].u - u_c);
                        sum_diff_v += (nodes_[iy * nx + (ix - 1)].v - v_c);
                    }
                    if (ix + 1 < nx) {
                        sum_diff_u += (nodes_[iy * nx + (ix + 1)].u - u_c);
                        sum_diff_v += (nodes_[iy * nx + (ix + 1)].v - v_c);
                    }
                    if (iy > 0) {
                        sum_diff_u += (nodes_[(iy - 1) * nx + ix].u - u_c);
                        sum_diff_v += (nodes_[(iy - 1) * nx + ix].v - v_c);
                    }
                    if (iy + 1 < ny) {
                        sum_diff_u += (nodes_[(iy + 1) * nx + ix].u - u_c);
                        sum_diff_v += (nodes_[(iy + 1) * nx + ix].v - v_c);
                    }

                    laplacian_u[idx] = sum_diff_u;
                    laplacian_v[idx] = sum_diff_v;
                }
            }

            // Step B: Update string field with non-linear self-interaction
            // Equation of motion: \partial_{tt} \Psi = T_0 \nabla^2 \Psi - \gamma \partial_t \Psi - \beta |\Psi|^2 \Psi + F_external
            for (size_t i = 0; i < N; ++i) {
                auto& node = nodes_[i];

                // External plucking is coupled to the top receptive boundary (y > 0.4)
                float external_drive_u = 0.0f;
                float external_drive_v = 0.0f;
                if (node.y > 0.2f) {
                    for (size_t n = 0; n < M; ++n) {
                        float mode_val = mode_eigenmodes_[n][i] * mode_plucking_force[n];
                        external_drive_u += mode_val;
                        external_drive_v += mode_val * 0.5f;
                    }
                }

                // Non-linear cubic self-interaction (FPU beta)
                float psi_sq = node.u * node.u + node.v * node.v;
                float nonlin_u = -config_.string_coupling_beta * psi_sq * node.u;
                float nonlin_v = -config_.string_coupling_beta * psi_sq * node.v;

                // Natural radiation damping
                float damping_u = -0.08f * node.u_dot;
                float damping_v = -0.08f * node.v_dot;

                // String acceleration
                float u_acc = config_.string_tension * laplacian_u[i] + nonlin_u + damping_u + external_drive_u;
                float v_acc = config_.string_tension * laplacian_v[i] + nonlin_v + damping_v + external_drive_v;

                node.u_dot = std::clamp(node.u_dot + u_acc * sub_dt, -8.0f, 8.0f);
                node.v_dot = std::clamp(node.v_dot + v_acc * sub_dt, -8.0f, 8.0f);

                node.u = std::clamp(node.u + node.u_dot * sub_dt, -5.0f, 5.0f);
                node.v = std::clamp(node.v + node.v_dot * sub_dt, -5.0f, 5.0f);

                total_string_action += (node.u_dot * node.u_dot + node.v_dot * node.v_dot);
            }

            // Step C: Self-Synthesized Field Projections (Emergent Observable Laws)
            for (size_t iy = 0; iy < ny; ++iy) {
                for (size_t ix = 0; ix < nx; ++ix) {
                    size_t idx = iy * nx + ix;
                    auto& node = nodes_[idx];

                    // 1. Emergent Potential V = Re(Psi^* \partial_t Psi) = u * u_dot + v * v_dot
                    node.emergent_potential = node.u * node.u_dot + node.v * node.v_dot;

                    // 2. Emergent Spatial Fluxes J_x, J_y (Noether currents)
                    float du_dx = (ix + 1 < nx ? nodes_[iy * nx + (ix + 1)].u : node.u) -
                                  (ix > 0 ? nodes_[iy * nx + (ix - 1)].u : node.u);
                    float dv_dx = (ix + 1 < nx ? nodes_[iy * nx + (ix + 1)].v : node.v) -
                                  (ix > 0 ? nodes_[iy * nx + (ix - 1)].v : node.v);

                    float du_dy = (iy + 1 < ny ? nodes_[(iy + 1) * nx + ix].u : node.u) -
                                  (iy > 0 ? nodes_[(iy - 1) * nx + ix].u : node.u);
                    float dv_dy = (iy + 1 < ny ? nodes_[(iy + 1) * nx + ix].v : node.v) -
                                  (iy > 0 ? nodes_[(iy - 1) * nx + ix].v : node.u);

                    node.emergent_flux_x = node.u * dv_dx - node.v * du_dx;
                    node.emergent_flux_y = node.u * dv_dy - node.v * du_dy;

                    // 3. Emergent Longitudinal Stress T_yy = (\partial_y u)^2 + (\partial_y v)^2
                    node.emergent_stress_yy = du_dy * du_dy + dv_dy * dv_dy;
                }
            }

        } // END OF SUB-STEPS

        // --------------------------------------------------------------------
        // MACROSCOPIC BOUNDARY TRACTION (EMERGENT FROM BOTTOM STRING MANIFOLD)
        // Traction = \int T_yy dS along left (x < 0) vs right (x > 0) bottom boundary
        // --------------------------------------------------------------------
        float left_traction = 0.0f;
        float right_traction = 0.0f;

        // Bottom row of nodes (iy = 0)
        for (size_t ix = 0; ix < nx; ++ix) {
            size_t idx = ix; // row 0
            if (nodes_[idx].x < 0.0f) {
                left_traction += nodes_[idx].emergent_stress_yy;
            } else {
                right_traction += nodes_[idx].emergent_stress_yy;
            }
        }

        // Bounded macroscopic actuation force via hyperbolic saturation
        float left_force  = std::tanh(left_traction * 0.8f);
        float right_force = std::tanh(right_traction * 0.8f);

        // Emergent Physical Torque = Left Pull - Right Pull
        float net_torque = left_force - right_force;

        state.motor_efferent_actions[0] = left_force;
        state.motor_efferent_actions[1] = right_force;
        state.motor_efferent_actions[2] = std::sin(net_torque * std::numbers::pi_v<float>);
        state.motor_efferent_actions[3] = std::cos(net_torque * std::numbers::pi_v<float>);

        // --------------------------------------------------------------------
        // Variational Free Energy & Telemetry
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0002f * total_string_action - 
                   0.02f * (left_force + right_force);

        float avg_potential = 0.0f;
        for (const auto& nd : nodes_) avg_potential += nd.emergent_potential;
        avg_potential /= static_cast<float>(N);

        state.total_nodes = N;
        state.total_string_action = total_string_action;
        state.avg_emergent_potential = avg_potential;
        state.left_boundary_traction = left_force;
        state.right_boundary_traction = right_force;
        state.net_physical_torque = net_torque;
        state.free_energy = fe;

        return state;
    }

private:
    StringContinuumConfig config_;
    std::vector<float> mode_frequencies_;
    std::vector<StringMeshNode> nodes_;
    std::vector<std::vector<float>> mode_eigenmodes_;
    std::vector<std::vector<float>> fourier_plucking_;
};

} // namespace karyon

#endif // SELF_SYNTHESIZING_STRING_ORGANISM_HPP
