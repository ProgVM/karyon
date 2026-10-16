// ============================================================================
// FullKaryonAtomicOrganism: Full Physics Hierarchy from Sub-Atomic Particles 
// to Atoms, Molecules, and the Macro-Molecular Cybernetic Organism KARYON.
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-503)
//
// 100% WEIGHTLESS (Zero Weight Matrices W).
//
// PHYSICAL HIERARCHY:
// 1. Strings / Quarks & Gauge Bosons:
//    - Photons & Neutrinos: Massless, fast energy/phase carriers (sensory/FE).
//    - Quarks: Symplectic unitary wave modes.
// 2. Hadrons:
//    - Unstable Hadrons (Pions/Mesons): Short lifespan, explorative decay.
//    - Stable Hadrons (Protons/Neutrons): Baryonic nucleus cores.
// 3. Atoms:
//    - Nucleus (Z protons, N neutrons) + Electron Shells (Valence).
// 4. Molecules:
//    - Covalent & Ionic poly-atomic compounds.
// 5. KARYON ORGANISM:
//    - Living macro-molecular cybernetic organism assembled from active molecules.
// ============================================================================

#ifndef FULL_KARYON_ATOMIC_ORGANISM_HPP
#define FULL_KARYON_ATOMIC_ORGANISM_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>
#include <string>

namespace karyon {

enum class HadronType {
    UNSTABLE_MESON,  // Short-lived pion/kaon -> exploratory search pulse
    STABLE_PROTON,   // Positive charge, stable baryonic core
    STABLE_NEUTRON   // Neutral mass, structural stability
};

struct HadronParticle {
    bool active = false;
    HadronType type = HadronType::STABLE_PROTON;
    size_t quark_i = 0;
    size_t quark_j = 0;
    float lifespan = 1.0f;       // Decay timer for unstable hadrons
    float binding_energy = 0.0f;
    float charge = 1.0f;
    float mass = 1.0f;
};

struct AtomicShell {
    size_t atomic_number = 1;    // Z (number of protons)
    size_t neutron_count = 0;     // N (number of neutrons)
    size_t valence_electrons = 1; // Valence capacity for bonding
    float atomic_mass = 1.0f;
    std::vector<size_t> constituent_hadrons;
};

struct ChemicalMolecule {
    bool active = false;
    std::string name = "Monomer";
    std::vector<size_t> bound_atom_indices;
    float bond_stability = 0.0f;
    float concentration = 0.0f;
    std::vector<float> action_vector; // Specific motor charge
};

struct KaryonOrganismState {
    size_t total_atoms = 0;
    size_t total_molecules = 0;
    size_t unstable_decays = 0;
    float total_mass = 0.0f;
    float internal_metabolism = 0.0f;
    float free_energy = 0.0f;
    std::vector<float> motor_efferent_actions;
};

struct OrganismConfig {
    size_t sensory_dim = 256;
    size_t action_dim = 4;
    size_t num_quarks = 32;
    size_t spatial_dim = 16;
    size_t max_hadrons = 32;
    size_t max_atoms = 16;
    size_t max_molecules = 8;
    float dt = 0.04f;
};

class KaryonOrganism {
public:
    explicit KaryonOrganism(const OrganismConfig& cfg)
        : config_(cfg),
          quarks_u_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          quarks_v_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          hadrons_(cfg.max_hadrons),
          atoms_(cfg.max_atoms),
          molecules_(cfg.max_molecules)
    {
        // Fixed spatial Fourier basis matrices for gauge photon / neutrino plucking
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

        // Initialize primordial Atoms (Hydrogenic / Helium-like cores)
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            atoms_[a].atomic_number = (a % 4) + 1;
            atoms_[a].neutron_count = a % 3;
            atoms_[a].valence_electrons = (atoms_[a].atomic_number % 2) + 1;
            atoms_[a].atomic_mass = static_cast<float>(atoms_[a].atomic_number + atoms_[a].neutron_count);
        }

        // Initialize primordial Macro-Molecules that form the KARYON Organism Body
        for (size_t m = 0; m < config_.max_molecules; ++m) {
            molecules_[m].active = true;
            molecules_[m].name = "Macro-Molecule-" + std::to_string(m + 1);
            molecules_[m].action_vector.assign(config_.action_dim, 0.0f);
            size_t target_act = m % config_.action_dim;
            molecules_[m].action_vector[target_act] = (m >= config_.action_dim) ? -1.0f : 1.0f;
            molecules_[m].bound_atom_indices = {m % config_.max_atoms, (m + 1) % config_.max_atoms};
            molecules_[m].bond_stability = 1.0f;
        }
    }

    KaryonOrganismState perceive_and_act(const std::vector<float>& sensory_state,
                                         const std::vector<float>& target_goal,
                                         float external_valence)
    {
        KaryonOrganismState state;
        state.motor_efferent_actions.assign(config_.action_dim, 0.0f);

        float dt = config_.dt;
        size_t N = config_.num_quarks;
        size_t D = config_.spatial_dim;
        size_t S = config_.sensory_dim;

        // --------------------------------------------------------------------
        // 1. GAUGE PHOTON & NEUTRINO SENSORY PROPAGATION
        // Massless gauge particles transport instantaneous sensory error
        // --------------------------------------------------------------------
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];
        }

        // --------------------------------------------------------------------
        // 2. QUARK FIELD SYMPLECTIC EVOLUTION
        // --------------------------------------------------------------------
        float total_quark_energy = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            float proj_cos = 0.0f, proj_sin = 0.0f;
            for (size_t k = 0; k < S; ++k) {
                proj_cos += err_s[k] * fourier_cos_[n][k];
                proj_sin += err_s[k] * fourier_sin_[n][k];
            }

            float w_n = 0.2f * std::sqrt(static_cast<float>(n + 1));
            float cos_w = std::cos(w_n * dt);
            float sin_w = std::sin(w_n * dt);
            float damping = std::exp(-0.015f * dt * static_cast<float>(n + 1));

            for (size_t d = 0; d < D; ++d) {
                float phase_d = 2.0f * std::numbers::pi_v<float> * static_cast<float>(d) / static_cast<float>(D);
                float f_u = std::clamp(proj_cos * std::cos(phase_d) - proj_sin * std::sin(phase_d), -1.5f, 1.5f);
                float f_v = std::clamp(proj_cos * std::sin(phase_d) + proj_sin * std::cos(phase_d), -1.5f, 1.5f);

                float u = quarks_u_[n][d];
                float v = quarks_v_[n][d];

                float u_rot = u * cos_w - v * sin_w;
                float v_rot = u * sin_w + v * cos_w;

                quarks_u_[n][d] = std::clamp((u_rot + f_u * dt) * damping, -4.0f, 4.0f);
                quarks_v_[n][d] = std::clamp((v_rot + f_v * dt) * damping, -4.0f, 4.0f);

                total_quark_energy += quarks_u_[n][d] * quarks_u_[n][d] + quarks_v_[n][d] * quarks_v_[n][d];
            }
        }

        // --------------------------------------------------------------------
        // 3. HADRON FORMATION & UNSTABLE MESON DECAY
        // --------------------------------------------------------------------
        size_t active_hadrons_count = 0;
        size_t decays_this_step = 0;

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                float overlap = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    overlap += quarks_u_[i][d] * quarks_u_[j][d] + quarks_v_[i][d] * quarks_v_[j][d];
                }

                if (std::abs(overlap) > 0.15f) {
                    // Nucleate Hadron
                    for (auto& h : hadrons_) {
                        if (!h.active) {
                            h.active = true;
                            h.quark_i = i;
                            h.quark_j = j;
                            if (external_valence < 0.0f) {
                                // Mismatch -> create UNSTABLE MESON for explorative pulse
                                h.type = HadronType::UNSTABLE_MESON;
                                h.lifespan = 0.12f; // Short half-life
                                h.mass = 0.5f;
                            } else {
                                // Stable Protons / Neutrons
                                h.type = (i % 2 == 0) ? HadronType::STABLE_PROTON : HadronType::STABLE_NEUTRON;
                                h.lifespan = 100.0f;
                                h.mass = 1.0f;
                            }
                            h.binding_energy = 1.0f;
                            break;
                        }
                    }
                }
            }
        }

        // Process Hadron decay and lifespan
        for (auto& h : hadrons_) {
            if (h.active) {
                if (h.type == HadronType::UNSTABLE_MESON) {
                    h.lifespan -= dt;
                    if (h.lifespan <= 0.0f) {
                        h.active = false; // Decays into photons (explorative burst)
                        decays_this_step++;
                    } else {
                        active_hadrons_count++;
                    }
                } else {
                    active_hadrons_count++;
                }
            }
        }

        // --------------------------------------------------------------------
        // 4. ATOMIC NUCLEOSYNTHESIS & ELECTRON SHELL BINDING
        // --------------------------------------------------------------------
        size_t active_atoms_count = 0;
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            if (active_hadrons_count > a) {
                active_atoms_count++;
            }
        }

        // --------------------------------------------------------------------
        // 5. MOLECULAR SYNTHESIS & KARYON ORGANISM MACRO-ACTIONS
        // --------------------------------------------------------------------
        size_t active_molecules_count = 0;
        float organism_mass = 0.0f;

        for (size_t m = 0; m < config_.max_molecules; ++m) {
            auto& mol = molecules_[m];
            if (!mol.active) continue;

            float mol_energy = 0.0f;
            size_t target_mode = m % N;
            for (size_t d = 0; d < D; ++d) {
                mol_energy += quarks_u_[target_mode][d] * quarks_u_[target_mode][d] +
                              quarks_v_[target_mode][d] * quarks_v_[target_mode][d];
            }

            // Chemical concentration driven by quark energy and meson decay pulses
            mol.concentration = std::tanh(mol_energy * 0.25f + 0.1f * static_cast<float>(decays_this_step));
            if (mol.concentration > 0.01f) {
                active_molecules_count++;
                organism_mass += 2.5f * mol.concentration;

                for (size_t act_idx = 0; act_idx < config_.action_dim; ++act_idx) {
                    state.motor_efferent_actions[act_idx] += mol.concentration * mol.action_vector[act_idx];
                }
            }
        }

        // Apply non-linear motor terminal activation
        for (size_t act_idx = 0; act_idx < config_.action_dim; ++act_idx) {
            state.motor_efferent_actions[act_idx] = std::tanh(state.motor_efferent_actions[act_idx] * 1.5f);
        }

        // --------------------------------------------------------------------
        // 6. FREE ENERGY & ORGANISM METABOLIC STATE
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0005f * total_quark_energy - 
                   0.002f * static_cast<float>(active_atoms_count + active_molecules_count);

        state.total_atoms = active_atoms_count;
        state.total_molecules = active_molecules_count;
        state.unstable_decays = decays_this_step;
        state.total_mass = organism_mass;
        state.internal_metabolism = total_quark_energy;
        state.free_energy = fe;

        return state;
    }

private:
    OrganismConfig config_;

    std::vector<std::vector<float>> quarks_u_;
    std::vector<std::vector<float>> quarks_v_;

    std::vector<HadronParticle> hadrons_;
    std::vector<AtomicShell> atoms_;
    std::vector<ChemicalMolecule> molecules_;

    std::vector<std::vector<float>> fourier_cos_;
    std::vector<std::vector<float>> fourier_sin_;
};

} // namespace karyon

#endif // FULL_KARYON_ATOMIC_ORGANISM_HPP
