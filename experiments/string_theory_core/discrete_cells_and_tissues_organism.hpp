// ============================================================================
// DiscreteCellsAndTissuesKaryon: Complete 8-Level Hierarchical Physics Core
// From Sub-Atomic Particles, Atoms & Free Ions, Molecules,
// Discrete Single Cells (incorporating both molecules and atomic ions),
// Collective Tissue Aggregates (higher hierarchy), to the Cybernetic Organism KARYON.
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-505)
//
// 100% WEIGHTLESS (Zero Weight Matrices W).
//
// PHYSICAL HIERARCHY:
// 1. Strings / Quarks & Gauge Bosons (Photons, Neutrinos, Symplectic Quarks).
// 2. Hadrons (Unstable Mesons for search pulses, Stable Baryons).
// 3. Atoms & Free Atomic Ions (Z protons, N neutrons, charge +/- for polarity).
// 4. Molecules (Covalent & Ionic chemical compounds).
// 5. Discrete Single Bodies / Cells (SingleCellBody):
//    - Composed of BOTH Molecules AND Free Atomic Ions.
//    - Receptor Cells, Myocyte Cells, Metabolic Organelles with membrane potentials.
// 6. Collective Tissue Aggregates & Organs (TissueAggregate):
//    - Higher hierarchical ensembles of synchronized cells and intercellular ionic fluid.
//    - Sensory Epithelium, Motor Muscle Fascicle, Coherent Coordination Ganglion.
// 7. Organism KARYON:
//    - Sovereign autopoietic cybernetic macro-organism orchestrating tissues.
// ============================================================================

#ifndef DISCRETE_CELLS_AND_TISSUES_ORGANISM_HPP
#define DISCRETE_CELLS_AND_TISSUES_ORGANISM_HPP

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

// Level 3: Atoms & Free Ions
struct AtomicUnit {
    size_t atomic_number = 1;     // Z (protons)
    size_t neutron_count = 0;      // N (neutrons)
    int net_ionic_charge = 0;      // Ion charge: e.g. +1 (Na+), +2 (Ca2+), -1 (Cl-)
    float ion_potential = 0.0f;    // Dynamic ionic polarization
    float atomic_mass = 1.0f;
};

// Level 4: Molecules
struct ChemicalMolecule {
    bool active = false;
    std::string name = "Monomer";
    std::vector<size_t> bound_atom_indices;
    float bond_stability = 0.0f;
    float concentration = 0.0f;
    std::vector<float> molecular_polarity;
};

// Level 5: Discrete Single Cells / Organelles (incorporating BOTH Molecules and Atomic Ions)
enum class SingleCellType {
    RECEPTOR_CELL,     // Sensory Transducer Cell
    MYOCYTE_CELL,      // Contractile Motor Cell
    METABOLIC_ORGANELLE // Mitochondrion-like Energy Node
};

struct SingleCellBody {
    bool active = false;
    std::string cell_id = "Cell-0";
    SingleCellType type = SingleCellType::MYOCYTE_CELL;
    std::vector<size_t> molecule_indices;   // Incorporated molecules
    std::vector<size_t> direct_atom_indices; // Directly bound atomic ions
    float membrane_potential = -70.0f;       // Cell potential in mV
    float activation = 0.0f;                 // Action potential firing state
    float mass = 1.0f;
};

// Level 6: Collective Tissue Aggregates & Organs (Groups of Cells standing higher in hierarchy)
enum class TissueType {
    SENSORY_EPITHELIUM, // Perception Tissue (cluster of receptor cells)
    MUSCLE_FASCICLE,    // Coherent Actuator Tissue (bundle of myocytes)
    NEURAL_GANGLION     // Inter-cellular Coordination Tissue
};

struct TissueAggregate {
    bool active = false;
    std::string tissue_name = "Tissue-Aggregate";
    TissueType type = TissueType::MUSCLE_FASCICLE;
    std::vector<size_t> constituent_cells;  // Higher hierarchy: bundle of cells
    std::vector<size_t> matrix_free_ions;    // Extracellular fluid free ions
    float tissue_coherence = 0.0f;          // Synchrony across cells
    float tissue_mass = 0.0f;
    std::vector<float> tissue_motor_effort; // Coherent macro force
};

// Organism Level Telemetry State
struct CompleteOrganismState {
    size_t total_atoms = 0;
    size_t total_molecules = 0;
    size_t total_discrete_cells = 0;
    size_t total_tissue_aggregates = 0;
    size_t meson_decays = 0;
    float total_organism_mass = 0.0f;
    float avg_membrane_potential = 0.0f;
    float tissue_coherence_factor = 0.0f;
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
    size_t max_cells = 12;      // 12 discrete single cells
    size_t max_tissues = 3;     // 3 high-level collective tissue organs
    float dt = 0.04f;
};

class CompleteKaryonOrganism {
public:
    explicit CompleteKaryonOrganism(const OrganismConfig& cfg)
        : config_(cfg),
          quarks_u_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          quarks_v_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          hadrons_(cfg.max_hadrons),
          atoms_(cfg.max_atoms),
          molecules_(cfg.max_molecules),
          cells_(cfg.max_cells),
          tissues_(cfg.max_tissues)
    {
        // 1. Fourier matrices for Gauge Photons / Neutrinos
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

        // 2. Initialize Atoms and Free Ions (Level 3)
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            atoms_[a].atomic_number = (a % 4) + 1;
            atoms_[a].neutron_count = a % 3;
            // Introduce both neutral atoms and direct ionic charge (Na+, Ca2+, Cl-)
            atoms_[a].net_ionic_charge = (a % 3 == 0) ? 1 : ((a % 3 == 1) ? 2 : -1);
            atoms_[a].ion_potential = static_cast<float>(atoms_[a].net_ionic_charge) * 15.0f;
            atoms_[a].atomic_mass = static_cast<float>(atoms_[a].atomic_number + atoms_[a].neutron_count);
        }

        // 3. Initialize Chemical Molecules (Level 4)
        for (size_t m = 0; m < config_.max_molecules; ++m) {
            molecules_[m].active = true;
            molecules_[m].name = "Bio-Molecule-" + std::to_string(m + 1);
            molecules_[m].molecular_polarity.assign(config_.action_dim, 0.0f);
            size_t target_act = m % config_.action_dim;
            molecules_[m].molecular_polarity[target_act] = (m >= config_.action_dim) ? -1.0f : 1.0f;
            molecules_[m].bound_atom_indices = {m % config_.max_atoms, (m + 1) % config_.max_atoms};
            molecules_[m].bond_stability = 1.0f;
        }

        // 4. Initialize Discrete Single Cells / Organelles (Level 5)
        // Note: Incorporating BOTH Molecules AND Free Atomic Ions!
        for (size_t c = 0; c < config_.max_cells; ++c) {
            cells_[c].active = true;
            cells_[c].cell_id = "SingleCell-" + std::to_string(c);
            cells_[c].mass = 1.2f;

            if (c < 4) {
                cells_[c].type = SingleCellType::RECEPTOR_CELL;
                cells_[c].molecule_indices = {0, 1};
                cells_[c].direct_atom_indices = {0, 1, 2}; // Direct Na+ / Ca2+ ions
            } else if (c < 10) {
                cells_[c].type = SingleCellType::MYOCYTE_CELL;
                cells_[c].molecule_indices = {2, 3, 4, 5, 6, 7};
                cells_[c].direct_atom_indices = {3, 4, 5}; // Ca2+ ions trigger contraction
            } else {
                cells_[c].type = SingleCellType::METABOLIC_ORGANELLE;
                cells_[c].molecule_indices = {0, 7};
                cells_[c].direct_atom_indices = {6, 7}; // Proton gradient ions
            }
        }

        // 5. Initialize Collective Tissue Aggregates & Organs (Level 6)
        // Standing higher in the hierarchy: bundles of cells + matrix free ions
        tissues_[0].active = true;
        tissues_[0].tissue_name = "Sensory Epithelium Tissue";
        tissues_[0].type = TissueType::SENSORY_EPITHELIUM;
        tissues_[0].constituent_cells = {0, 1, 2, 3};
        tissues_[0].matrix_free_ions = {0, 1};

        tissues_[1].active = true;
        tissues_[1].tissue_name = "Actuator Muscle Fascicle Organ";
        tissues_[1].type = TissueType::MUSCLE_FASCICLE;
        tissues_[1].constituent_cells = {4, 5, 6, 7, 8, 9};
        tissues_[1].matrix_free_ions = {2, 3, 4};
        tissues_[1].tissue_motor_effort.assign(config_.action_dim, 0.0f);

        tissues_[2].active = true;
        tissues_[2].tissue_name = "Homeostatic Coordination Ganglion";
        tissues_[2].type = TissueType::NEURAL_GANGLION;
        tissues_[2].constituent_cells = {10, 11};
        tissues_[2].matrix_free_ions = {5, 6, 7};
    }

    CompleteOrganismState perceive_and_act(const std::vector<float>& sensory_state,
                                           const std::vector<float>& target_goal,
                                           float external_valence)
    {
        CompleteOrganismState state;
        state.motor_efferent_actions.assign(config_.action_dim, 0.0f);

        float dt = config_.dt;
        size_t N = config_.num_quarks;
        size_t D = config_.spatial_dim;
        size_t S = config_.sensory_dim;

        // --------------------------------------------------------------------
        // 1. GAUGE PHOTONS & NEUTRINOS (Sensory Error Carrier)
        // --------------------------------------------------------------------
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];
        }

        // --------------------------------------------------------------------
        // 2. QUARK WAVE FIELD
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
        // 3. HADRONS (Unstable Mesons vs Stable Baryons)
        // --------------------------------------------------------------------
        size_t active_hadrons_count = 0;
        size_t meson_decays_count = 0;

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
                                h.lifespan = 0.10f;
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
                        meson_decays_count++;
                    } else {
                        active_hadrons_count++;
                    }
                } else {
                    active_hadrons_count++;
                }
            }
        }

        // --------------------------------------------------------------------
        // 4. ATOMS & FREE ATOMIC IONS (Level 3)
        // --------------------------------------------------------------------
        size_t active_atoms_count = 0;
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            if (active_hadrons_count > a) {
                active_atoms_count++;
                // Dynamic ionic fluctuation driven by meson decay bursts
                atoms_[a].ion_potential += 0.05f * static_cast<float>(meson_decays_count) * static_cast<float>(atoms_[a].net_ionic_charge);
                atoms_[a].ion_potential = std::clamp(atoms_[a].ion_potential, -40.0f, 40.0f);
            }
        }

        // --------------------------------------------------------------------
        // 5. CHEMICAL MOLECULES (Level 4)
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

            mol.concentration = std::tanh(mol_energy * 0.25f + 0.05f * static_cast<float>(meson_decays_count));
            if (mol.concentration > 0.01f) {
                active_molecules_count++;
            }
        }

        // --------------------------------------------------------------------
        // 6. DISCRETE SINGLE CELLS (Level 5)
        // Incorporating BOTH molecules and direct atomic ions!
        // --------------------------------------------------------------------
        size_t active_cells_count = 0;
        float total_cell_potential = 0.0f;

        for (auto& cell : cells_) {
            if (!cell.active) continue;

            // Compute cell driving force from incorporated molecules
            float mol_drive = 0.0f;
            for (size_t midx : cell.molecule_indices) {
                if (midx < config_.max_molecules) {
                    mol_drive += molecules_[midx].concentration;
                }
            }

            // Compute direct atomic ion contribution to membrane potential
            float ionic_drive = 0.0f;
            for (size_t aidx : cell.direct_atom_indices) {
                if (aidx < config_.max_atoms) {
                    ionic_drive += atoms_[aidx].ion_potential;
                }
            }

            // Cell membrane equation: dV/dt = ionic_drive + mol_drive
            cell.membrane_potential = -70.0f + 25.0f * std::tanh((ionic_drive * 0.05f) + mol_drive * 1.5f);
            cell.activation = std::tanh((cell.membrane_potential + 50.0f) * 0.1f); // Depolarization above -50mV

            if (cell.activation > 0.01f) {
                active_cells_count++;
            }
            total_cell_potential += cell.membrane_potential;
        }

        // --------------------------------------------------------------------
        // 7. COLLECTIVE TISSUE AGGREGATES & ORGANS (Level 6)
        // Synchronizing groups of cells + extracellular matrix free ions
        // --------------------------------------------------------------------
        size_t active_tissues_count = 0;
        float total_organism_mass = 0.0f;
        float aggregate_coherence = 0.0f;

        for (auto& tissue : tissues_) {
            if (!tissue.active) continue;

            float cell_activation_sum = 0.0f;
            for (size_t cidx : tissue.constituent_cells) {
                if (cidx < config_.max_cells) {
                    cell_activation_sum += cells_[cidx].activation;
                }
            }

            // Influence of extracellular matrix free ions
            float matrix_ion_boost = 0.0f;
            for (size_t ion_idx : tissue.matrix_free_ions) {
                if (ion_idx < config_.max_atoms) {
                    matrix_ion_boost += std::abs(atoms_[ion_idx].ion_potential);
                }
            }

            tissue.tissue_coherence = std::tanh((cell_activation_sum / static_cast<float>(tissue.constituent_cells.size())) * 
                                                (1.0f + 0.02f * matrix_ion_boost));
            tissue.tissue_mass = 4.0f * static_cast<float>(tissue.constituent_cells.size()) * tissue.tissue_coherence;

            if (tissue.tissue_coherence > 0.01f) {
                active_tissues_count++;
                total_organism_mass += tissue.tissue_mass;
                aggregate_coherence += tissue.tissue_coherence;

                // Muscle Fascicle generates coherent macro effort
                if (tissue.type == TissueType::MUSCLE_FASCICLE) {
                    tissue.tissue_motor_effort.assign(config_.action_dim, 0.0f);
                    for (size_t cidx : tissue.constituent_cells) {
                        const auto& cell = cells_[cidx];
                        for (size_t midx : cell.molecule_indices) {
                            if (midx < config_.max_molecules) {
                                const auto& mol = molecules_[midx];
                                for (size_t act = 0; act < config_.action_dim; ++act) {
                                    tissue.tissue_motor_effort[act] += cell.activation * mol.concentration * mol.molecular_polarity[act];
                                }
                            }
                        }
                    }

                    for (size_t act = 0; act < config_.action_dim; ++act) {
                        state.motor_efferent_actions[act] += tissue.tissue_motor_effort[act] * tissue.tissue_coherence;
                    }
                }
            }
        }

        // Motor output non-linear saturation
        for (size_t act = 0; act < config_.action_dim; ++act) {
            state.motor_efferent_actions[act] = std::tanh(state.motor_efferent_actions[act] * 1.5f);
        }

        // --------------------------------------------------------------------
        // 8. FREE ENERGY & KARYON ORGANISM STATE (Level 7)
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0005f * total_quark_energy - 
                   0.002f * static_cast<float>(active_atoms_count + active_molecules_count) -
                   0.005f * static_cast<float>(active_cells_count + active_tissues_count * 4);

        state.total_atoms = active_atoms_count;
        state.total_molecules = active_molecules_count;
        state.total_discrete_cells = active_cells_count;
        state.total_tissue_aggregates = active_tissues_count;
        state.meson_decays = meson_decays_count;
        state.total_organism_mass = total_organism_mass;
        state.avg_membrane_potential = total_cell_potential / static_cast<float>(config_.max_cells);
        state.tissue_coherence_factor = aggregate_coherence / static_cast<float>(config_.max_tissues);
        state.free_energy = fe;

        return state;
    }

private:
    OrganismConfig config_;

    std::vector<std::vector<float>> quarks_u_;
    std::vector<std::vector<float>> quarks_v_;

    std::vector<HadronParticle> hadrons_;
    std::vector<AtomicUnit> atoms_;
    std::vector<ChemicalMolecule> molecules_;
    std::vector<SingleCellBody> cells_;
    std::vector<TissueAggregate> tissues_;

    std::vector<std::vector<float>> fourier_cos_;
    std::vector<std::vector<float>> fourier_sin_;
};

} // namespace karyon

#endif // DISCRETE_CELLS_AND_TISSUES_ORGANISM_HPP
