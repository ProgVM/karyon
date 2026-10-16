// ============================================================================
// EmergentMolecularStringCore: Hierarchical Particle & Molecule Assembly
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-502)
//
// 100% WEIGHTLESS (No W matrices).
//
// HIERARCHY:
// 1. Strings / Quarks (Fundamental Modes n=1..N):
//    Symplectic unitary oscillators in multi-dimensional space.
// 2. Gluon String Flux Tubes (Hadron Fusion):
//    Non-linear phase-locking creates bound states (hadrons/solitons).
//    Energy of binding protects skills against external noise.
// 3. Atoms (Orbital Resonance Basins):
//    Quantized discrete energy shells representing distinct behavioral concepts.
// 4. Molecules (Combinatorial Poly-Atomic Assemblies):
//    Dynamic macro-molecular programs driving physical action terminals.
// ============================================================================

#ifndef EMERGENT_MOLECULAR_STRING_CORE_HPP
#define EMERGENT_MOLECULAR_STRING_CORE_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>

namespace karyon {

struct ParticleConfig {
    size_t sensory_dim = 256;
    size_t action_dim = 4;
    size_t num_quarks = 32;       // Fundamental harmonic modes n = 1..32
    size_t spatial_dim = 16;      // Internal topological dimensions D
    size_t max_hadrons = 24;      // Composite bound particle capacity
    size_t max_molecules = 8;     // High-order behavioral molecular compounds
    float fundamental_w0 = 0.2f;  // Fundamental string tension frequency
    float dt = 0.04f;             // Continuous symplectic time step
    float binding_affinity = 0.15f;// Strong interaction gluon flux strength
};

// A composite Hadron (e.g. Proton / Meson): A bound phase-locked pair/triplet of quarks
struct Hadron {
    bool active = false;
    size_t quark_i = 0;
    size_t quark_j = 0;
    float phase_lock = 0.0f;      // Bound phase difference delta_phi
    float binding_energy = 0.0f;  // Stability barrier preventing destruction
    float mass = 1.0f;            // Rest mass / inertia
    float valence_affinity = 0.0f;// Valence trace
};

// An Atomic/Molecular Compound: A poly-hadronic stable assembly generating macro-actions
struct Molecule {
    bool active = false;
    std::vector<size_t> constituent_hadrons;
    float activation = 0.0f;      // Chemical potential / concentration
    std::vector<float> action_profile; // Intrinsic polar motor charge
};

struct MolecularTelemetry {
    float total_field_energy = 0.0f;
    float binding_energy_total = 0.0f;
    size_t active_hadrons = 0;
    size_t active_molecules = 0;
    float free_energy = 0.0f;
    std::vector<float> action_outputs;
};

class EmergentMolecularStringCore {
public:
    explicit EmergentMolecularStringCore(const ParticleConfig& cfg)
        : config_(cfg),
          quarks_u_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          quarks_v_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          gluon_flux_(cfg.num_quarks, std::vector<float>(cfg.num_quarks, 0.0f)),
          hadrons_(cfg.max_hadrons),
          molecules_(cfg.max_molecules)
    {
        // Fixed spatial Fourier basis matrices for gauge sensory plucking
        fourier_cos_.resize(config_.num_quarks, std::vector<float>(config_.sensory_dim));
        fourier_sin_.resize(config_.num_quarks, std::vector<float>(config_.sensory_dim));
        for (size_t n = 0; n < config_.num_quarks; ++n) {
            float mode_idx = static_cast<float>(n + 1);
            for (size_t k = 0; k < config_.sensory_dim; ++k) {
                float angle = 2.0f * std::numbers::pi_v<float> * mode_idx * static_cast<float>(k) / static_cast<float>(config_.sensory_dim);
                fourier_cos_[n][k] = std::cos(angle) / std::sqrt(static_cast<float>(config_.sensory_dim));
                fourier_sin_[n][k] = std::sin(angle) / std::sqrt(static_cast<float>(config_.sensory_dim));
            }
        }

        // Initialize primordial baseline molecules representing polar coordinate directions
        for (size_t m = 0; m < config_.max_molecules; ++m) {
            molecules_[m].action_profile.assign(config_.action_dim, 0.0f);
            if (m < config_.action_dim) {
                molecules_[m].active = true;
                molecules_[m].action_profile[m] = 1.0f;
            }
        }
    }

    MolecularTelemetry step(const std::vector<float>& sensory_state,
                           const std::vector<float>& target_goal,
                           float external_valence,
                           std::vector<float>& out_actions)
    {
        out_actions.assign(config_.action_dim, 0.0f);
        MolecularTelemetry telem;

        float dt = config_.dt;
        size_t N = config_.num_quarks;
        size_t D = config_.spatial_dim;
        size_t S = config_.sensory_dim;

        // --------------------------------------------------------------------
        // 1. GAUGE SENSORY PLUCKING (Quark Wave Excitations)
        // --------------------------------------------------------------------
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];
        }

        std::vector<std::vector<float>> force_u(N, std::vector<float>(D, 0.0f));
        std::vector<std::vector<float>> force_v(N, std::vector<float>(D, 0.0f));

        for (size_t n = 0; n < N; ++n) {
            float proj_cos = 0.0f, proj_sin = 0.0f;
            for (size_t k = 0; k < S; ++k) {
                proj_cos += err_s[k] * fourier_cos_[n][k];
                proj_sin += err_s[k] * fourier_sin_[n][k];
            }
            for (size_t d = 0; d < D; ++d) {
                float phase_d = 2.0f * std::numbers::pi_v<float> * static_cast<float>(d) / static_cast<float>(D);
                force_u[n][d] = std::clamp(proj_cos * std::cos(phase_d) - proj_sin * std::sin(phase_d), -1.5f, 1.5f);
                force_v[n][d] = std::clamp(proj_cos * std::sin(phase_d) + proj_sin * std::cos(phase_d), -1.5f, 1.5f);
            }
        }

        // --------------------------------------------------------------------
        // 2. SYMPLECTIC EVOLUTION & GLUON TUBE CONFINEMENT
        // --------------------------------------------------------------------
        float total_quark_energy = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            float mode_num = static_cast<float>(n + 1);
            float w_n = config_.fundamental_w0 * std::sqrt(mode_num);
            float cos_w = std::cos(w_n * dt);
            float sin_w = std::sin(w_n * dt);
            float damping = std::exp(-0.015f * dt * mode_num);

            for (size_t d = 0; d < D; ++d) {
                float u = quarks_u_[n][d];
                float v = quarks_v_[n][d];

                // Gluon confinement force from bound hadrons
                float gluon_pull_u = 0.0f;
                float gluon_pull_v = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (gluon_flux_[n][j] > 1e-4f) {
                        gluon_pull_u += gluon_flux_[n][j] * (quarks_u_[j][d] - u);
                        gluon_pull_v += gluon_flux_[n][j] * (quarks_v_[j][d] - v);
                    }
                }

                // Symplectic rotation
                float u_rot = u * cos_w - v * sin_w;
                float v_rot = u * sin_w + v * cos_w;

                quarks_u_[n][d] = std::clamp((u_rot + (force_u[n][d] + gluon_pull_u) * dt) * damping, -4.0f, 4.0f);
                quarks_v_[n][d] = std::clamp((v_rot + (force_v[n][d] + gluon_pull_v) * dt) * damping, -4.0f, 4.0f);

                total_quark_energy += quarks_u_[n][d] * quarks_u_[n][d] + quarks_v_[n][d] * quarks_v_[n][d];
            }
        }

        // --------------------------------------------------------------------
        // 3. HADRON FUSION (Phase-Locking & Gluon Flux Tube Formation)
        // --------------------------------------------------------------------
        // Detect resonant phase co-occurrence between quark modes
        float total_binding_energy = 0.0f;
        size_t active_hadrons_count = 0;

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                // Compute spatial inner product (phase overlap)
                float overlap = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    overlap += quarks_u_[i][d] * quarks_u_[j][d] + quarks_v_[i][d] * quarks_v_[j][d];
                }

                // Relaxed fusion threshold (0.2f instead of 0.8f) and valence tolerance
                if (std::abs(overlap) > 0.2f) {
                    float fusion_delta = (external_valence >= 0.0f) ? 0.05f * (external_valence + 0.1f) : 0.01f;
                    gluon_flux_[i][j] = std::clamp(gluon_flux_[i][j] + fusion_delta, 0.0f, 1.0f);
                    gluon_flux_[j][i] = gluon_flux_[i][j];

                    bool already_bound = false;
                    for (auto& h : hadrons_) {
                        if (h.active && ((h.quark_i == i && h.quark_j == j) || (h.quark_i == j && h.quark_j == i))) {
                            h.binding_energy = std::min(h.binding_energy + 0.1f * std::max(external_valence, 0.1f), 10.0f);
                            already_bound = true;
                            break;
                        }
                    }
                    if (!already_bound) {
                        for (auto& h : hadrons_) {
                            if (!h.active) {
                                h.active = true;
                                h.quark_i = i;
                                h.quark_j = j;
                                h.binding_energy = 1.0f;
                                h.mass = static_cast<float>(i + j + 2) * 0.5f;
                                break;
                            }
                        }
                    }
                } else if (external_valence < -0.5f && gluon_flux_[i][j] > 0.0f) {
                    gluon_flux_[i][j] = std::max(gluon_flux_[i][j] - 0.01f, 0.0f);
                    gluon_flux_[j][i] = gluon_flux_[i][j];
                }
            }
        }

        // Count and protect active hadrons
        for (auto& h : hadrons_) {
            if (h.active) {
                if (gluon_flux_[h.quark_i][h.quark_j] < 0.005f) {
                    h.active = false; // Hadron decays
                } else {
                    active_hadrons_count++;
                    total_binding_energy += h.binding_energy;
                }
            }
        }

        // --------------------------------------------------------------------
        // 4. MOLECULAR ASSEMBLY & MACRO-ACTION EMERGENCE
        // --------------------------------------------------------------------
        size_t active_molecules_count = 0;
        for (size_t m = 0; m < config_.max_molecules; ++m) {
            auto& mol = molecules_[m];
            if (!mol.active) continue;

            float mol_energy = 0.0f;
            size_t target_mode = m % N;
            for (size_t d = 0; d < D; ++d) {
                mol_energy += quarks_u_[target_mode][d] * quarks_u_[target_mode][d] +
                              quarks_v_[target_mode][d] * quarks_v_[target_mode][d];
            }

            mol.activation = std::tanh(mol_energy * 0.15f);
            if (mol.activation > 0.01f) {
                active_molecules_count++;
                for (size_t a = 0; a < config_.action_dim; ++a) {
                    out_actions[a] += mol.activation * mol.action_profile[a];
                }
            }
        }

        // Normalization / non-linear saturation of motor terminals
        for (size_t a = 0; a < config_.action_dim; ++a) {
            out_actions[a] = std::tanh(out_actions[a] * 1.5f);
        }

        // --------------------------------------------------------------------
        // 5. FREE ENERGY EVALUATION
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0005f * total_quark_energy - 
                   0.001f * total_binding_energy;

        telem.total_field_energy = total_quark_energy;
        telem.binding_energy_total = total_binding_energy;
        telem.active_hadrons = active_hadrons_count;
        telem.active_molecules = active_molecules_count;
        telem.free_energy = fe;
        telem.action_outputs = out_actions;

        return telem;
    }

private:
    ParticleConfig config_;

    std::vector<std::vector<float>> quarks_u_;
    std::vector<std::vector<float>> quarks_v_;

    std::vector<std::vector<float>> gluon_flux_;

    std::vector<Hadron> hadrons_;

    std::vector<Molecule> molecules_;

    std::vector<std::vector<float>> fourier_cos_;
    std::vector<std::vector<float>> fourier_sin_;
};

} // namespace karyon

#endif // EMERGENT_MOLECULAR_STRING_CORE_HPP
