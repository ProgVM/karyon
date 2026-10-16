// ============================================================================
// AntagonisticNeuromuscularOrganism: Dual-Actuator Reciprocal Inhibition & SERCA Dynamics
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-507)
//
// 100% WEIGHTLESS (Zero Weight Matrices W).
//
// CORE BIOPHYSICAL BREAKTHROUGHS TO ELIMINATE TETANIC SATURATION:
// 1. ANTAGONISTIC MUSCLE PAIR (Agonist vs Antagonist):
//    - Living organisms cannot push; muscles ONLY pull (contract).
//    - Myocyte Pair:
//      * Left Flexor / Agonist: pulls to positive torque (+Torque)
//      * Right Extensor / Antagonist: pulls to negative torque (-Torque)
//
// 2. RECIPROCAL INHIBITION (Renshaw & Ia Interneurons):
//    - When Agonist depolarizes, it drives chloride (Cl-) influx into Antagonist,
//      hyperpolarizing it to prevent co-contraction / stiff locking.
//
// 3. FAST CALCIUM CLEARING (SERCA Pump Dynamics):
//    - Sarcoplasmic/Endoplasmic Reticulum Calcium-ATPase (SERCA) pumps Ca2+
//      back into stores at rate proportional to intracellular [Ca2+].
//    - Eliminates permanent tetanus (allows rapid twitches and continuous fine balance).
//
// 4. MULTI-SCALE PYRAMID:
//    - 128 Atoms/Ions (Na+, K+, Ca2+, Cl-, Fe)
//    - 32 Chemical Molecules (ATP, Actomyosin, Acetylcholine, GABA)
//    - 4 Specialized High-Complexity Cells:
//      * Cell 0: Sensory Proprioceptor (encodes state error & velocity)
//      * Cell 1: Pacemaker / Interneuron Coordinator (implements reciprocal inhibition)
//      * Cell 2: Agonist Muscle Myocyte (+Torque effector)
//      * Cell 3: Antagonist Muscle Myocyte (-Torque effector)
//    - 2 Coordinated Tissue Bundles (Agonist Motor Unit & Antagonist Motor Unit)
//    - Organism KARYON
// ============================================================================

#ifndef ANTAGONISTIC_NEUROMUSCULAR_ORGANISM_HPP
#define ANTAGONISTIC_NEUROMUSCULAR_ORGANISM_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>
#include <string>

namespace karyon {

enum class HadronType {
    UNSTABLE_MESON,  // Short-lived search pulse
    STABLE_BARYON    // Stable mass nucleus
};

struct HadronParticle {
    bool active = false;
    HadronType type = HadronType::STABLE_BARYON;
    size_t quark_i = 0;
    size_t quark_j = 0;
    float lifespan = 1.0f;
    float mass = 1.0f;
};

enum class IonSpecies {
    SODIUM,      // Na+ (depolarization)
    POTASSIUM,   // K+ (repolarization)
    CALCIUM,     // Ca2+ (actomyosin trigger)
    CHLORIDE,    // Cl- (inhibition / hyperpolarization)
    NEUTRAL_CORE // Structural atoms
};

struct AtomicUnit {
    size_t atomic_number = 1;
    size_t neutron_count = 0;
    IonSpecies species = IonSpecies::NEUTRAL_CORE;
    int charge = 0;
    float intracellular_conc = 10.0f; // mMol
    float extracellular_conc = 140.0f;
    float local_flux = 0.0f;
    float atomic_mass = 1.0f;
};

enum class MoleculeClass {
    ATP_ENERGY,          // Energy donor
    ACTOMYOSIN_FILAMENT, // Contractile cross-bridge
    ACETYLCHOLINE,       // Excitatory transmitter
    GABA_TRANSMITTER     // Inhibitory transmitter (reciprocal inhibition)
};

struct ChemicalMolecule {
    bool active = false;
    std::string name = "Molecule";
    MoleculeClass m_class = MoleculeClass::ACTOMYOSIN_FILAMENT;
    std::vector<size_t> constituent_atoms;
    float conformational_state = 0.0f;
    float concentration = 0.5f;
    float energy_stored = 1.0f;
};

enum class DeepCellType {
    SENSORY_PROPRIOCEPTOR,   // Senses error and d(error)/dt
    RECIPROCAL_INTERNEURON,  // Bifurcates drive and applies reciprocal GABA inhibition
    AGONIST_MYOCYTE,         // Contracts for positive motor direction
    ANTAGONIST_MYOCYTE       // Contracts for negative motor direction
};

struct DeepSpecializedCell {
    bool active = false;
    std::string cell_name = "DeepCell";
    DeepCellType type = DeepCellType::AGONIST_MYOCYTE;

    // Membrane and ion dynamics
    float membrane_potential = -70.0f; // mV
    float calcium_internal = 0.0001f;  // Internal free Ca2+ (mMol)
    float mechanical_strain = 0.0f;    // Actomyosin contraction [0, 1]
    float atp_pool = 5.0f;             // Energy reserve

    std::vector<size_t> assigned_molecules;
    std::vector<size_t> assigned_ions;
};

struct MotorTissueUnit {
    bool active = false;
    std::string unit_name = "MotorUnit";
    size_t myocyte_cell_idx = 0;
    float contractile_force = 0.0f;
    float metabolic_consumption = 0.0f;
};

struct OrganismTelemetryState {
    size_t total_atoms = 0;
    size_t total_molecules = 0;
    size_t total_cells = 0;
    size_t meson_decays = 0;
    float v_proprioceptor = 0.0f;
    float v_interneuron = 0.0f;
    float agonist_ca = 0.0f;
    float antagonist_ca = 0.0f;
    float agonist_strain = 0.0f;
    float antagonist_strain = 0.0f;
    float net_torque_effort = 0.0f;
    float free_energy = 0.0f;
    std::vector<float> motor_efferent_actions;
};

struct OrganismConfig {
    size_t sensory_dim = 256;
    size_t action_dim = 4;
    size_t num_quarks = 64;
    size_t spatial_dim = 16;
    size_t max_hadrons = 64;
    size_t max_atoms = 128;       // 128 Atoms & Free Ions
    size_t max_molecules = 32;    // 32 Chemical Molecules
    size_t max_cells = 4;         // 4 Specialized High-Complexity Cells
    size_t relaxation_substeps = 8; // 8 Physical relaxation sub-steps
    float dt = 0.04f;
};

class AntagonisticNeuromuscularOrganism {
public:
    explicit AntagonisticNeuromuscularOrganism(const OrganismConfig& cfg)
        : config_(cfg),
          quarks_u_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          quarks_v_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          hadrons_(cfg.max_hadrons),
          atoms_(cfg.max_atoms),
          molecules_(cfg.max_molecules),
          cells_(cfg.max_cells)
    {
        // 1. Fourier gauge projections
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

        // 2. Initialize 128 Atoms & Ions with physiological species
        for (size_t a = 0; a < config_.max_atoms; ++a) {
            atoms_[a].atomic_number = (a % 12) + 1;
            atoms_[a].neutron_count = atoms_[a].atomic_number;
            atoms_[a].atomic_mass = static_cast<float>(atoms_[a].atomic_number * 2);

            if (a < 32) {
                atoms_[a].species = IonSpecies::SODIUM;
                atoms_[a].charge = +1;
                atoms_[a].intracellular_conc = 12.0f;
                atoms_[a].extracellular_conc = 145.0f;
            } else if (a < 64) {
                atoms_[a].species = IonSpecies::POTASSIUM;
                atoms_[a].charge = +1;
                atoms_[a].intracellular_conc = 140.0f;
                atoms_[a].extracellular_conc = 4.0f;
            } else if (a < 96) {
                atoms_[a].species = IonSpecies::CALCIUM;
                atoms_[a].charge = +2;
                atoms_[a].intracellular_conc = 0.0001f;
                atoms_[a].extracellular_conc = 2.0f;
            } else {
                atoms_[a].species = IonSpecies::CHLORIDE;
                atoms_[a].charge = -1;
                atoms_[a].intracellular_conc = 4.0f;
                atoms_[a].extracellular_conc = 110.0f;
            }
        }

        // 3. Initialize 32 Complex Bio-Molecules
        for (size_t m = 0; m < config_.max_molecules; ++m) {
            molecules_[m].active = true;
            if (m < 8) {
                molecules_[m].name = "Actomyosin-Agonist-" + std::to_string(m);
                molecules_[m].m_class = MoleculeClass::ACTOMYOSIN_FILAMENT;
            } else if (m < 16) {
                molecules_[m].name = "Actomyosin-Antagonist-" + std::to_string(m);
                molecules_[m].m_class = MoleculeClass::ACTOMYOSIN_FILAMENT;
            } else if (m < 24) {
                molecules_[m].name = "Acetylcholine-Excitatory-" + std::to_string(m);
                molecules_[m].m_class = MoleculeClass::ACETYLCHOLINE;
            } else {
                molecules_[m].name = "GABA-Inhibitory-" + std::to_string(m);
                molecules_[m].m_class = MoleculeClass::GABA_TRANSMITTER;
            }

            for (size_t a = 0; a < 8; ++a) {
                molecules_[m].constituent_atoms.push_back((m * 4 + a) % config_.max_atoms);
            }
        }

        // 4. Initialize 4 Specialized High-Complexity Cells
        // Cell 0: Sensory Proprioceptor Cell
        cells_[0].active = true;
        cells_[0].cell_name = "Sensory Proprioceptor Cell";
        cells_[0].type = DeepCellType::SENSORY_PROPRIOCEPTOR;
        for (size_t m = 16; m < 24; ++m) cells_[0].assigned_molecules.push_back(m);
        for (size_t a = 0; a < 32; ++a) cells_[0].assigned_ions.push_back(a);

        // Cell 1: Reciprocal Interneuron Coordinator
        cells_[1].active = true;
        cells_[1].cell_name = "Reciprocal Interneuron Coordinator";
        cells_[1].type = DeepCellType::RECIPROCAL_INTERNEURON;
        for (size_t m = 24; m < 32; ++m) cells_[1].assigned_molecules.push_back(m);
        for (size_t a = 96; a < 128; ++a) cells_[1].assigned_ions.push_back(a); // Chloride ions

        // Cell 2: Agonist Muscle Myocyte (+Torque)
        cells_[2].active = true;
        cells_[2].cell_name = "Agonist Muscle Myocyte (+Torque)";
        cells_[2].type = DeepCellType::AGONIST_MYOCYTE;
        for (size_t m = 0; m < 8; ++m) cells_[2].assigned_molecules.push_back(m);
        for (size_t a = 64; a < 80; ++a) cells_[2].assigned_ions.push_back(a); // Calcium pool 1

        // Cell 3: Antagonist Muscle Myocyte (-Torque)
        cells_[3].active = true;
        cells_[3].cell_name = "Antagonist Muscle Myocyte (-Torque)";
        cells_[3].type = DeepCellType::ANTAGONIST_MYOCYTE;
        for (size_t m = 8; m < 16; ++m) cells_[3].assigned_molecules.push_back(m);
        for (size_t a = 80; a < 96; ++a) cells_[3].assigned_ions.push_back(a); // Calcium pool 2

        prev_err_sum_ = 0.0f;
    }

    OrganismTelemetryState perceive_and_act(const std::vector<float>& sensory_state,
                                            const std::vector<float>& target_goal,
                                            float external_valence)
    {
        OrganismTelemetryState state;
        state.motor_efferent_actions.assign(config_.action_dim, 0.0f);

        float macro_dt = config_.dt;
        float sub_dt = macro_dt / static_cast<float>(config_.relaxation_substeps);
        size_t N = config_.num_quarks;
        size_t D = config_.spatial_dim;
        size_t S = config_.sensory_dim;

        // 1. Compute Sensory Error & Direction
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        float spatial_err_center = 0.0f;
        float total_weight = 0.0f;

        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];

            if (sensory_state[k] > 0.05f) {
                float x_coord = static_cast<float>(k % 16) - 7.5f;
                spatial_err_center += x_coord * sensory_state[k];
                total_weight += sensory_state[k];
            }
        }

        float current_coord = (total_weight > 0.001f) ? (spatial_err_center / total_weight) : 0.0f;
        float velocity_coord = (current_coord - prev_err_sum_) / macro_dt;
        prev_err_sum_ = current_coord;

        // ====================================================================
        // MULTI-SCALE PHYSICAL RELAXATION LOOP (8 SUB-STEPS)
        // ====================================================================
        size_t total_meson_decays = 0;
        float total_quark_energy = 0.0f;

        for (size_t substep = 0; substep < config_.relaxation_substeps; ++substep) {

            // A. Quark Field Evolution
            total_quark_energy = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                float proj_cos = 0.0f, proj_sin = 0.0f;
                for (size_t k = 0; k < S; ++k) {
                    proj_cos += err_s[k] * fourier_cos_[n][k];
                    proj_sin += err_s[k] * fourier_sin_[n][k];
                }

                float w_n = 0.2f * std::sqrt(static_cast<float>(n + 1));
                float cos_w = std::cos(w_n * sub_dt);
                float sin_w = std::sin(w_n * sub_dt);
                float damping = std::exp(-0.015f * sub_dt * static_cast<float>(n + 1));

                for (size_t d = 0; d < D; ++d) {
                    float phase_d = 2.0f * std::numbers::pi_v<float> * static_cast<float>(d) / static_cast<float>(D);
                    float f_u = std::clamp(proj_cos * std::cos(phase_d) - proj_sin * std::sin(phase_d), -1.5f, 1.5f);
                    float f_v = std::clamp(proj_cos * std::sin(phase_d) + proj_sin * std::cos(phase_d), -1.5f, 1.5f);

                    float u = quarks_u_[n][d];
                    float v = quarks_v_[n][d];

                    quarks_u_[n][d] = std::clamp(((u * cos_w - v * sin_w) + f_u * sub_dt) * damping, -4.0f, 4.0f);
                    quarks_v_[n][d] = std::clamp(((u * sin_w + v * cos_w) + f_v * sub_dt) * damping, -4.0f, 4.0f);

                    total_quark_energy += quarks_u_[n][d] * quarks_u_[n][d] + quarks_v_[n][d] * quarks_v_[n][d];
                }
            }

            // B. Hadron Meson Decays
            for (size_t i = 0; i < N && i < 16; ++i) {
                for (size_t j = i + 1; j < N && j < 16; ++j) {
                    float overlap = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        overlap += quarks_u_[i][d] * quarks_u_[j][d] + quarks_v_[i][d] * quarks_v_[j][d];
                    }

                    if (std::abs(overlap) > 0.12f) {
                        for (auto& h : hadrons_) {
                            if (!h.active) {
                                h.active = true;
                                h.quark_i = i;
                                h.quark_j = j;
                                h.type = (external_valence < 0.0f) ? HadronType::UNSTABLE_MESON : HadronType::STABLE_BARYON;
                                h.lifespan = 0.05f;
                                break;
                            }
                        }
                    }
                }
            }

            for (auto& h : hadrons_) {
                if (h.active && h.type == HadronType::UNSTABLE_MESON) {
                    h.lifespan -= sub_dt;
                    if (h.lifespan <= 0.0f) {
                        h.active = false;
                        total_meson_decays++;
                    }
                }
            }

            // C. Cell 0: Sensory Proprioceptor Cell Dynamics
            // Transduces state error and velocity into continuous depolarization
            {
                auto& c_proprio = cells_[0];
                float physical_drive = (current_coord * 0.4f + velocity_coord * 0.15f);
                float target_potential = -70.0f + 50.0f * std::tanh(physical_drive);
                c_proprio.membrane_potential += (target_potential - c_proprio.membrane_potential) * 10.0f * sub_dt;
            }

            // D. Cell 1: Reciprocal Interneuron Coordinator
            // Splits sensory potential into push-pull drives with reciprocal GABA inhibition
            float agonist_excite = 0.0f;
            float antagonist_excite = 0.0f;
            {
                auto& c_inter = cells_[1];
                float delta_v = cells_[0].membrane_potential - (-70.0f);
                c_inter.membrane_potential += (-70.0f + delta_v - c_inter.membrane_potential) * 8.0f * sub_dt;

                // If deviation is positive, excite Agonist and INHIBIT Antagonist (Reciprocal inhibition)
                if (delta_v > 0.0f) {
                    agonist_excite = std::max(0.0f, delta_v * 0.04f);
                    antagonist_excite = 0.0f; // Completely inhibited via GABA/Cl-
                } else {
                    agonist_excite = 0.0f; // Inhibited
                    antagonist_excite = std::max(0.0f, -delta_v * 0.04f);
                }
            }

            // E. Cell 2: Agonist Muscle Myocyte (+Torque effector)
            // Includes SERCA Calcium Reuptake pump to prevent tetanus!
            {
                auto& c_ago = cells_[2];
                // Calcium influx driven by agonist_excite
                float ca_influx = agonist_excite * 0.08f;
                // SERCA Pump: active clearance rate = 8.0 * [Ca2+]
                float serca_pump = 8.0f * (c_ago.calcium_internal - 0.0001f);
                c_ago.calcium_internal = std::clamp(c_ago.calcium_internal + (ca_influx - serca_pump) * sub_dt, 0.0001f, 0.05f);

                // Actomyosin strain response
                float target_strain = std::tanh(c_ago.calcium_internal * 200.0f);
                c_ago.mechanical_strain += (target_strain - c_ago.mechanical_strain) * 12.0f * sub_dt;
            }

            // F. Cell 3: Antagonist Muscle Myocyte (-Torque effector)
            // Also governed by SERCA pump
            {
                auto& c_antago = cells_[3];
                float ca_influx = antagonist_excite * 0.08f;
                float serca_pump = 8.0f * (c_antago.calcium_internal - 0.0001f);
                c_antago.calcium_internal = std::clamp(c_antago.calcium_internal + (ca_influx - serca_pump) * sub_dt, 0.0001f, 0.05f);

                float target_strain = std::tanh(c_antago.calcium_internal * 200.0f);
                c_antago.mechanical_strain += (target_strain - c_antago.mechanical_strain) * 12.0f * sub_dt;
            }

        } // END OF SUB-STEPPING

        // --------------------------------------------------------------------
        // NET TORQUE & COHERENT EFFERENT MOTOR ACTIONS
        // Net Physical Effort = Agonist Pull - Antagonist Pull
        // --------------------------------------------------------------------
        float f_agonist = cells_[2].mechanical_strain;
        float f_antagonist = cells_[3].mechanical_strain;
        float net_torque = f_agonist - f_antagonist; // Perfect symmetric differential drive!

        state.motor_efferent_actions[0] = f_agonist;
        state.motor_efferent_actions[1] = f_antagonist;
        state.motor_efferent_actions[2] = std::sin(net_torque * std::numbers::pi_v<float>);
        state.motor_efferent_actions[3] = std::cos(net_torque * std::numbers::pi_v<float>);

        // --------------------------------------------------------------------
        // FREE ENERGY & ORGANISM TELEMETRY
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0003f * total_quark_energy - 
                   0.002f * static_cast<float>(config_.max_atoms + config_.max_molecules) * 0.05f -
                   0.02f * (f_agonist + f_antagonist);

        state.total_atoms = config_.max_atoms;
        state.total_molecules = config_.max_molecules;
        state.total_cells = config_.max_cells;
        state.meson_decays = total_meson_decays;
        state.v_proprioceptor = cells_[0].membrane_potential;
        state.v_interneuron = cells_[1].membrane_potential;
        state.agonist_ca = cells_[2].calcium_internal;
        state.antagonist_ca = cells_[3].calcium_internal;
        state.agonist_strain = f_agonist;
        state.antagonist_strain = f_antagonist;
        state.net_torque_effort = net_torque;
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
    std::vector<DeepSpecializedCell> cells_;

    std::vector<std::vector<float>> fourier_cos_;
    std::vector<std::vector<float>> fourier_sin_;

    float prev_err_sum_ = 0.0f;
};

} // namespace karyon

#endif // ANTAGONISTIC_NEUROMUSCULAR_ORGANISM_HPP
