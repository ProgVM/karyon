// ============================================================================
// CompleteKaryonMacroBodyOrganism: Full Physical & Anatomical Hierarchy
// From Sub-Atomic Particles to Atoms, Molecules, Macroscopic Molecular Bodies/Tissues,
// and the Living Cybernetic Organism KARYON.
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-504)
//
// 100% WEIGHTLESS (Zero Weight Matrices W).
//
// PHYSICAL & ANATOMICAL HIERARCHY:
// 1. Strings / Quarks & Gauge Bosons:
//    - Photons & Neutrinos: Massless, fast energy/phase carriers.
//    - Quarks: Symplectic unitary wave modes.
// 2. Hadrons:
//    - Unstable Hadrons (Pions/Mesons): Explorative decay pulses.
//    - Stable Hadrons (Protons/Neutrons): Baryonic nucleus cores.
// 3. Atoms:
//    - Nucleus (Z protons, N neutrons) + Electron Shells.
// 4. Molecules:
//    - Covalent & Ionic chemical compounds.
// 5. Macroscopic Molecular Bodies / Tissues / Organelles (Molecular Structures):
//    - Sensory Receptor Body (Sensory Membrane Organelle)
//    - Actuator Muscle Tissue (Motor Efferent Body)
//    - Metabolic Homeostatic Core Body (Energy Organelle)
// 6. KARYON ORGANISM:
//    - Sovereign living cybernetic organism composed of macroscopic molecular bodies.
// ============================================================================

#ifndef COMPLETE_KARYON_MACRO_BODY_ORGANISM_HPP
#define COMPLETE_KARYON_MACRO_BODY_ORGANISM_HPP

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
    float lifespan = 1.0f;
    float binding_energy = 0.0f;
    float charge = 1.0f;
    float mass = 1.0f;
};

struct AtomicShell {
    size_t atomic_number = 1;    // Z (number of protons)
    size_t neutron_count = 0;     // N (number of neutrons)
    size_t valence_electrons = 1; // Valence capacity
    float atomic_mass = 1.0f;
    std::vector<size_t> constituent_hadrons;
};

struct ChemicalMolecule {
    bool active = false;
    std::string name = "Monomer";
    std::vector<size_t> bound_atom_indices;
    float bond_stability = 0.0f;
    float concentration = 0.0f;
    std::vector<float> action_vector;
};

// Macroscopic Molecular Body / Tissue / Organelle
enum class MolecularBodyType {
    SENSORY_MEMBRANE,   // Perception Tissue
    ACTUATOR_MUSCLE,    // Efferent Motor Tissue
    METABOLIC_CORE      // Homeostatic Energy Organelle
};

struct MacroscopicMolecularBody {
    bool active = false;
    std::string body_name = "Tissue";
    MolecularBodyType type = MolecularBodyType::ACTUATOR_MUSCLE;
    std::vector<size_t> constituent_molecules;
    float tissue_density = 0.0f;
    float functional_capacity = 0.0f;
    std::vector<float> macro_effector_signal;
};

struct KaryonMacroOrganismState {
    size_t total_atoms = 0;
    size_t total_molecules = 0;
    size_t total_molecular_bodies = 0;
    size_t unstable_decays = 0;
    float total_body_mass = 0.0f;
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
    size_t max_bodies = 3; // 3 Core Macroscopic Molecular Bodies
    float dt = 0.04f;
};

class KaryonMacroOrganism {
public:
    explicit KaryonMacroOrganism(const OrganismConfig& cfg)
        : config_(cfg),
          quarks_u_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          quarks_v_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          hadrons_(cfg.max_hadrons),
          atoms_(cfg.max_atoms),
          molecules_(cfg.max_molecules),
          macro_bodies_(cfg.max_bodies)
    {
        // Fixed spatial Fourier basis matrices
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

        // Initialize Atoms
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            atoms_[a].atomic_number = (a % 4) + 1;
            atoms_[a].neutron_count = a % 3;
            atoms_[a].valence_electrons = (atoms_[a].atomic_number % 2) + 1;
            atoms_[a].atomic_mass = static_cast<float>(atoms_[a].atomic_number + atoms_[a].neutron_count);
        }

        // Initialize Molecules
        for (size_t m = 0; m < config_.max_molecules; ++m) {
            molecules_[m].active = true;
            molecules_[m].name = "Molecule-" + std::to_string(m + 1);
            molecules_[m].action_vector.assign(config_.action_dim, 0.0f);
            size_t target_act = m % config_.action_dim;
            molecules_[m].action_vector[target_act] = (m >= config_.action_dim) ? -1.0f : 1.0f;
            molecules_[m].bound_atom_indices = {m % config_.max_atoms, (m + 1) % config_.max_atoms};
            molecules_[m].bond_stability = 1.0f;
        }

        // Initialize 3 Macroscopic Molecular Bodies / Tissues forming KARYON
        macro_bodies_[0].active = true;
        macro_bodies_[0].body_name = "Sensory Membrane Organelle";
        macro_bodies_[0].type = MolecularBodyType::SENSORY_MEMBRANE;
        macro_bodies_[0].constituent_molecules = {0, 1, 2};

        macro_bodies_[1].active = true;
        macro_bodies_[1].body_name = "Actuator Muscle Tissue Body";
        macro_bodies_[1].type = MolecularBodyType::ACTUATOR_MUSCLE;
        macro_bodies_[1].constituent_molecules = {3, 4, 5, 6, 7};
        macro_bodies_[1].macro_effector_signal.assign(config_.action_dim, 0.0f);

        macro_bodies_[2].active = true;
        macro_bodies_[2].body_name = "Metabolic Homeostatic Core Body";
        macro_bodies_[2].type = MolecularBodyType::METABOLIC_CORE;
        macro_bodies_[2].constituent_molecules = {0, 3, 7};
    }

    KaryonMacroOrganismState perceive_and_act(const std::vector<float>& sensory_state,
                                              const std::vector<float>& target_goal,
                                              float external_valence)
    {
        KaryonMacroOrganismState state;
        state.motor_efferent_actions.assign(config_.action_dim, 0.0f);

        float dt = config_.dt;
        size_t N = config_.num_quarks;
        size_t D = config_.spatial_dim;
        size_t S = config_.sensory_dim;

        // --------------------------------------------------------------------
        // 1. GAUGE PHOTON / NEUTRINO PROPAGATION
        // --------------------------------------------------------------------
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];
        }

        // --------------------------------------------------------------------
        // 2. QUARK WAVE FIELD EVOLUTION
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
        // 3. HADRON FORMATION & MESON DECAY
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
                    for (auto& h : hadrons_) {
                        if (!h.active) {
                            h.active = true;
                            h.quark_i = i;
                            h.quark_j = j;
                            if (external_valence < 0.0f) {
                                h.type = HadronType::UNSTABLE_MESON;
                                h.lifespan = 0.12f;
                                h.mass = 0.5f;
                            } else {
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

        for (auto& h : hadrons_) {
            if (h.active) {
                if (h.type == HadronType::UNSTABLE_MESON) {
                    h.lifespan -= dt;
                    if (h.lifespan <= 0.0f) {
                        h.active = false;
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
        // 4. ATOMIC NUCLEOSYNTHESIS
        // --------------------------------------------------------------------
        size_t active_atoms_count = 0;
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            if (active_hadrons_count > a) {
                active_atoms_count++;
            }
        }

        // --------------------------------------------------------------------
        // 5. MOLECULAR SYNTHESIS
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

            mol.concentration = std::tanh(mol_energy * 0.25f + 0.1f * static_cast<float>(decays_this_step));
            if (mol.concentration > 0.01f) {
                active_molecules_count++;
            }
        }

        // --------------------------------------------------------------------
        // 6. MACROSCOPIC MOLECULAR BODIES & TISSUES ASSEMBLY (LEVEL 5)
        // Molecules aggregate into functional bodies (Sensory, Muscle, Core)
        // --------------------------------------------------------------------
        size_t active_bodies_count = 0;
        float total_organism_mass = 0.0f;

        for (auto& body : macro_bodies_) {
            if (!body.active) continue;

            float body_density = 0.0f;
            for (size_t mol_idx : body.constituent_molecules) {
                if (mol_idx < config_.max_molecules) {
                    body_density += molecules_[mol_idx].concentration;
                }
            }

            body.tissue_density = std::tanh(body_density * 0.5f);
            if (body.tissue_density > 0.01f) {
                active_bodies_count++;
                total_organism_mass += 5.0f * body.tissue_density;

                // Actuator Muscle Tissue drives macro-effector actions
                if (body.type == MolecularBodyType::ACTUATOR_MUSCLE) {
                    body.macro_effector_signal.assign(config_.action_dim, 0.0f);
                    for (size_t mol_idx : body.constituent_molecules) {
                        if (mol_idx < config_.max_molecules) {
                            const auto& mol = molecules_[mol_idx];
                            for (size_t act_idx = 0; act_idx < config_.action_dim; ++act_idx) {
                                body.macro_effector_signal[act_idx] += mol.concentration * mol.action_vector[act_idx];
                            }
                        }
                    }

                    for (size_t act_idx = 0; act_idx < config_.action_dim; ++act_idx) {
                        state.motor_efferent_actions[act_idx] += body.macro_effector_signal[act_idx] * body.tissue_density;
                    }
                }
            }
        }

        // Saturate motor output
        for (size_t act_idx = 0; act_idx < config_.action_dim; ++act_idx) {
            state.motor_efferent_actions[act_idx] = std::tanh(state.motor_efferent_actions[act_idx] * 1.5f);
        }

        // --------------------------------------------------------------------
        // 7. FREE ENERGY & LIVING ORGANISM STATE (LEVEL 6 - KARYON)
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0005f * total_quark_energy - 
                   0.003f * static_cast<float>(active_atoms_count + active_molecules_count + active_bodies_count * 5);

        state.total_atoms = active_atoms_count;
        state.total_molecules = active_molecules_count;
        state.total_molecular_bodies = active_bodies_count;
        state.unstable_decays = decays_this_step;
        state.total_body_mass = total_organism_mass;
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
    std::vector<MacroscopicMolecularBody> macro_bodies_;

    std::vector<std::vector<float>> fourier_cos_;
    std::vector<std::vector<float>> fourier_sin_;
};

} // namespace karyon

#endif // COMPLETE_KARYON_MACRO_BODY_ORGANISM_HPP
