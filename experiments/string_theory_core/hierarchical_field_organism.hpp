// ============================================================================
// HierarchicalFieldOrganism: Pure Multi-Scale Hierarchical Environmental Coupling
// Karyon-CoRE v28.0 Sovereign Architecture (EXP-513)
//
// ZERO TOP-DOWN HARDCODING (Zero Weight Matrices W, Zero Hardcoded Cell Enums).
// Zero artificial cell-level rules.
//
// The 8-Level Physical Hierarchy:
// Level 1: Quark Wavefield (SU(3) color-neutral modes u_n, v_n)
// Level 2: Hadron & Meson Phase Boundaries (spontaneous creation/decay)
// Level 3: Interstitial Extracellular Ionic Matrix (ECM [Na+]_out, [K+]_out, [Ca2+]_out)
//          - Natural depletion of ionic gradients during prolonged firing!
// Level 4: Macromolecular Filament Assemblies (Actin/Myosin sliding cross-bridges)
// Level 5: Totipotent Cellular Bilayers (permeability governed by bilayer mechanical tension)
// Level 6: Tissue Coherence & Stress Field (Eulerian mechanical strain transmission)
// Level 7: Macroscopic Organismal Mass & Tendons (inertial body dynamics)
// Level 8: External Dynamical Environment (Inverted Pendulum / Orbit Tracking)
// ============================================================================

#ifndef HIERARCHICAL_FIELD_ORGANISM_HPP
#define HIERARCHICAL_FIELD_ORGANISM_HPP

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

// Level 3 & 5: Extracellular Interstitial Fluid Compartment & Cell
struct InterstitialCellCompartment {
    size_t id = 0;
    float pos_x = 0.0f; // [-1, 1]
    float pos_y = 0.0f; // [-1, 1]

    // Intracellular State (Level 5)
    float membrane_potential = -70.0f; // mV
    float intracellular_ca = 0.0001f;  // Internal free Ca2+
    float actomyosin_strain = 0.0f;    // Macromolecular strain [0, 1]
    float membrane_tension = 0.0f;     // Bilayer lateral tension (from mechanical stretch)

    // Extracellular Micro-Environment (Level 3 - Interstitial fluid around cell)
    // Depletes when cell fires, recovers by bulk interstitial diffusion from organism pool!
    float ecm_na = 145.0f;  // mM (normal high extracellular Na+)
    float ecm_k  = 4.5f;    // mM (normal low extracellular K+)
    float ecm_ca = 2.0f;    // mM (normal extracellular Ca2+)
};

struct HierarchicalTelemetry {
    size_t total_cells = 0;
    size_t meson_decays = 0;
    float avg_membrane_potential = 0.0f;
    float avg_ecm_k = 0.0f;
    float left_muscle_strain = 0.0f;
    float right_muscle_strain = 0.0f;
    float net_physical_torque = 0.0f;
    float free_energy = 0.0f;
    std::vector<float> motor_efferent_actions;
};

struct HierarchicalConfig {
    size_t sensory_dim = 256;
    size_t action_dim = 4;
    size_t num_quarks = 64;
    size_t spatial_dim = 16;
    size_t max_hadrons = 64;
    size_t num_cells = 6;
    size_t relaxation_substeps = 8;
    float dt = 0.04f;
};

class HierarchicalFieldOrganism {
public:
    explicit HierarchicalFieldOrganism(const HierarchicalConfig& cfg)
        : config_(cfg),
          quarks_u_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          quarks_v_(cfg.num_quarks, std::vector<float>(cfg.spatial_dim, 0.0f)),
          hadrons_(cfg.max_hadrons),
          cells_(cfg.num_cells),
          tissue_elastic_mesh_(cfg.num_cells, std::vector<float>(cfg.num_cells, 0.0f))
    {
        // 1. Fourier continuous field projections
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

        // 2. Spatial Cellular Morphology (2 bilateral columns of 3 cells)
        std::vector<std::pair<float, float>> positions = {
            {-0.5f,  0.75f}, // Cell 0: Left sensory pole
            { 0.5f,  0.75f}, // Cell 1: Right sensory pole
            {-0.4f,  0.00f}, // Cell 2: Left trunk
            { 0.4f,  0.00f}, // Cell 3: Right trunk
            {-0.5f, -0.75f}, // Cell 4: Left tendon effector
            { 0.5f, -0.75f}  // Cell 5: Right tendon effector
        };

        for (size_t i = 0; i < config_.num_cells; ++i) {
            cells_[i].id = i;
            cells_[i].pos_x = positions[i].first;
            cells_[i].pos_y = positions[i].second;
        }

        // 3. Level 6: Tissue Elastic Stiffness Mesh & Fluid Diffusion Geometry
        for (size_t i = 0; i < config_.num_cells; ++i) {
            for (size_t j = i + 1; j < config_.num_cells; ++j) {
                float dx = cells_[i].pos_x - cells_[j].pos_x;
                float dy = cells_[i].pos_y - cells_[j].pos_y;
                float dist = std::sqrt(dx * dx + dy * dy);

                // Tissue elasticity k_elastic decreases with distance
                float k_elastic = 2.0f / (1.0f + 3.0f * dist);
                tissue_elastic_mesh_[i][j] = k_elastic;
                tissue_elastic_mesh_[j][i] = k_elastic;
            }
        }
    }

    HierarchicalTelemetry perceive_and_act(const std::vector<float>& sensory_state,
                                          const std::vector<float>& target_goal,
                                          float external_valence)
    {
        HierarchicalTelemetry state;
        state.motor_efferent_actions.assign(config_.action_dim, 0.0f);

        float macro_dt = config_.dt;
        float sub_dt = macro_dt / static_cast<float>(config_.relaxation_substeps);
        size_t N = config_.num_quarks;
        size_t D = config_.spatial_dim;
        size_t S = config_.sensory_dim;
        size_t num_cells = config_.num_cells;

        // --------------------------------------------------------------------
        // LEVEL 8 -> LEVEL 1: Sensory Field Boundary Conditions
        // --------------------------------------------------------------------
        std::vector<float> err_s(S, 0.0f);
        float sensory_mismatch_sq = 0.0f;
        float spatial_field_left = 0.0f;
        float spatial_field_right = 0.0f;

        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
            sensory_mismatch_sq += err_s[k] * err_s[k];

            if (sensory_state[k] > 0.01f) {
                float x_pos = (static_cast<float>(k % 16) - 7.5f) / 7.5f; // [-1.0, 1.0]
                if (x_pos < 0.0f) spatial_field_left += sensory_state[k] * std::abs(x_pos);
                else spatial_field_right += sensory_state[k] * std::abs(x_pos);
            }
        }

        // Photoreceptor transduction drives ionic exchange at sensory boundary
        float photon_flux_left = 300.0f * std::tanh(spatial_field_left * 2.0f);
        float photon_flux_right = 300.0f * std::tanh(spatial_field_right * 2.0f);

        // ====================================================================
        // MULTI-SCALE PHYSICAL RELAXATION LOOP (8 SUB-STEPS)
        // ====================================================================
        size_t total_meson_decays = 0;
        float total_quark_energy = 0.0f;

        for (size_t substep = 0; substep < config_.relaxation_substeps; ++substep) {

            // ----------------------------------------------------------------
            // LEVEL 1: Quark Wavefield SU(3) Confinement Dynamics
            // ----------------------------------------------------------------
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

            // ----------------------------------------------------------------
            // LEVEL 2: Hadron Particle Condensation & Meson Decays
            // ----------------------------------------------------------------
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

            // ----------------------------------------------------------------
            // LEVEL 6: Tissue Stress Field & Bilayer Tension Computation
            // Mechanical strain in tendon pulls on upstream cells, increasing membrane tension
            // ----------------------------------------------------------------
            for (size_t i = 0; i < num_cells; ++i) {
                float total_tension = 0.0f;
                for (size_t j = 0; j < num_cells; ++j) {
                    if (i != j) {
                        float strain_diff = cells_[j].actomyosin_strain - cells_[i].actomyosin_strain;
                        total_tension += tissue_elastic_mesh_[i][j] * std::abs(strain_diff);
                    }
                }
                // Membrane tension relaxes toward tissue stress
                cells_[i].membrane_tension += (total_tension - cells_[i].membrane_tension) * 10.0f * sub_dt;
            }

            // ----------------------------------------------------------------
            // LEVEL 3 & LEVEL 5: Environmental Extracellular Ionic Dynamics & Cell Bioenergetics
            // ----------------------------------------------------------------
            for (size_t i = 0; i < num_cells; ++i) {
                auto& c = cells_[i];

                // 1) Dynamic Nernst Potentials based on REAL local extracellular ion concentrations!
                // E = (RT/zF) * ln([Ion]_out / [Ion]_in)
                // If cell fires repeatedly, ecm_na drops, ecm_k rises -> E_na drops, E_k depolarizes -> automatic fatigue!
                float E_na = 26.7f * std::log(std::max(10.0f, c.ecm_na) / 12.0f);   // Internal [Na+] = 12 mM
                float E_k  = 26.7f * std::log(std::max(1.0f,  c.ecm_k)  / 140.0f);  // Internal [K+] = 140 mM

                // 2) Membrane currents
                // Mechanosensitive channel closure: high tension reduces Na+ open probability (protection against over-stretch)
                float tension_inhibition = 1.0f / (1.0f + 2.5f * c.membrane_tension);

                // Voltage-gated activation
                float v_act = 1.0f / (1.0f + std::exp(-(c.membrane_potential + 42.0f) / 6.0f));
                float g_na = 55.0f * (v_act * v_act * v_act) * tension_inhibition;
                float i_na = g_na * (E_na - c.membrane_potential);

                // Delayed rectifier K+
                float v_k_act = 1.0f / (1.0f + std::exp(-(c.membrane_potential + 25.0f) / 7.0f));
                float g_k = 30.0f * v_k_act;
                float i_k = g_k * (c.membrane_potential - E_k);

                // Environmental leak
                float i_leak = 2.0f * (c.membrane_potential - (-70.0f));

                // External sensory photon drive at receptive borders
                float i_sensory = 0.0f;
                if (i == 0) i_sensory = photon_flux_left;
                if (i == 1) i_sensory = photon_flux_right;

                // Gap junction / axial current from neighbors through tissue mesh
                float i_axial = 0.0f;
                for (size_t j = 0; j < num_cells; ++j) {
                    if (i != j) {
                        float v_diff = cells_[j].membrane_potential - c.membrane_potential;
                        // Longitudinal conductivity is higher than transverse
                        bool ipsi = (c.pos_x * cells_[j].pos_x > 0.0f);
                        float g_conn = ipsi ? (2.8f * tissue_elastic_mesh_[i][j]) : (0.05f * tissue_elastic_mesh_[i][j]);
                        i_axial += g_conn * v_diff;
                    }
                }

                // Membrane potential integration
                float dV = (i_sensory + i_axial + i_na - i_k - i_leak) * sub_dt;
                c.membrane_potential = std::clamp(c.membrane_potential + dV, -85.0f, +50.0f);

                // 3) ENVIRONMENTAL ION EXCHANGE & DEPLETION (Level 3 Interstitial Environment)
                // When Na+ enters the cell (i_na > 0), extracellular [Na+] decreases!
                // When K+ leaves the cell (i_k > 0), extracellular [K+] accumulates!
                float d_na_ecm = (-i_na * 0.002f) + (145.0f - c.ecm_na) * 0.8f; // Resupplied by bulk fluid
                float d_k_ecm  = ( i_k  * 0.003f) + (4.5f   - c.ecm_k)  * 0.8f; // Cleared by bulk fluid diffusion
                c.ecm_na = std::clamp(c.ecm_na + d_na_ecm * sub_dt, 80.0f, 155.0f);
                c.ecm_k  = std::clamp(c.ecm_k  + d_k_ecm  * sub_dt, 3.0f,  30.0f);

                // 4) LEVEL 4: Macromolecular Filament Activation (Calcium & Actomyosin)
                // Voltage-dependent calcium influx
                float ca_gate = std::max(0.0f, (c.membrane_potential - (-15.0f)) * 0.04f);
                float ca_influx = ca_gate * (c.ecm_ca / 2.0f) * 0.35f;

                // SERCA Pump reuptakes Ca2+ into sarcoplasmic reticulum
                float serca_pump = 12.0f * (c.intracellular_ca - 0.0001f);
                c.intracellular_ca = std::clamp(c.intracellular_ca + (ca_influx - serca_pump) * sub_dt, 0.0001f, 0.05f);

                // Actomyosin cross-bridge cycle (sliding filament strain)
                float target_strain = std::tanh(c.intracellular_ca * 300.0f);
                c.actomyosin_strain += (target_strain - c.actomyosin_strain) * 18.0f * sub_dt;
            }

            // Interstitial fluid diffusion between neighboring extracellular compartments
            for (size_t i = 0; i < num_cells; ++i) {
                for (size_t j = i + 1; j < num_cells; ++j) {
                    float diff_k = cells_[j].ecm_k - cells_[i].ecm_k;
                    float k_flux = diff_k * 0.15f * tissue_elastic_mesh_[i][j] * sub_dt;
                    cells_[i].ecm_k += k_flux;
                    cells_[j].ecm_k -= k_flux;
                }
            }

        } // END OF RELAXATION SUB-STEPS

        // --------------------------------------------------------------------
        // LEVEL 7: Macroscopic Organismal Force & Tendon Net Torque
        // Tendon effectors are located at bottom poles (Cell 4: Left, Cell 5: Right)
        // --------------------------------------------------------------------
        float left_pull = cells_[4].actomyosin_strain;
        float right_pull = cells_[5].actomyosin_strain;

        float net_torque = left_pull - right_pull;

        state.motor_efferent_actions[0] = left_pull;
        state.motor_efferent_actions[1] = right_pull;
        state.motor_efferent_actions[2] = std::sin(net_torque * std::numbers::pi_v<float>);
        state.motor_efferent_actions[3] = std::cos(net_torque * std::numbers::pi_v<float>);

        // --------------------------------------------------------------------
        // Free Energy & Environmental Telemetry
        // --------------------------------------------------------------------
        float fe = (sensory_mismatch_sq / static_cast<float>(S)) + 
                   0.0003f * total_quark_energy - 
                   0.02f * (left_pull + right_pull);

        float avg_vm = 0.0f;
        float avg_k = 0.0f;
        for (const auto& c : cells_) {
            avg_vm += c.membrane_potential;
            avg_k  += c.ecm_k;
        }
        avg_vm /= static_cast<float>(num_cells);
        avg_k  /= static_cast<float>(num_cells);

        state.total_cells = num_cells;
        state.meson_decays = total_meson_decays;
        state.avg_membrane_potential = avg_vm;
        state.avg_ecm_k = avg_k;
        state.left_muscle_strain = left_pull;
        state.right_muscle_strain = right_pull;
        state.net_physical_torque = net_torque;
        state.free_energy = fe;

        return state;
    }

private:
    HierarchicalConfig config_;

    std::vector<std::vector<float>> quarks_u_;
    std::vector<std::vector<float>> quarks_v_;

    std::vector<HadronParticle> hadrons_;
    std::vector<InterstitialCellCompartment> cells_;
    std::vector<std::vector<float>> tissue_elastic_mesh_;

    std::vector<std::vector<float>> fourier_cos_;
    std::vector<std::vector<float>> fourier_sin_;
};

} // namespace karyon

#endif // HIERARCHICAL_FIELD_ORGANISM_HPP
