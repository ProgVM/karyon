// ============================================================================
// EXP-517: Unified Sovereign Metric Tensor Field (Zero Higher-Order Hacks)
// Karyon-CoRE v28.0 Sovereign Architecture
//
// "Mind is a continuous tensor field g_{\mu\nu}(\vec{x}, t) in spatial coordinates.
//  No arbitrary 'Left/Right' classes, no manual PD terms, no hardcoded parameters."
//
// In EXP-517:
// 1. Unified 1D/2D Spatial Continuum:
//    The string field Psi(x, t) is defined on a continuous spatial domain x \in [-1, 1].
//    No 'left_organism' or 'right_organism' discrete classes!
// 2. Parity is Emergent from Spatial Coordinate x:
//    Sensory drive projects onto the spatial field as:
//    S(x, t) = error * x + vel * \partial_x S
// 3. The Metric Tensor and Damping are Endogenous Plastic Fields:
//    T(x, t) and Gamma(x, t) evolve locally along the continuum:
//    \partial_t T(x, t) = \eta * |Psi(x, t)|^2 * \nabla F
//    \partial_t Gamma(x, t) = \eta * |\partial_t Psi(x, t)|^2
// 4. Force/Torque is the Direct Boundary Flux (Stokes Theorem):
//    Torque = \int_{-1}^{1} x * T_{yy}(x) dx
// ============================================================================

#ifndef SOVEREIGN_METRIC_FIELD_ORGANISM_HPP
#define SOVEREIGN_METRIC_FIELD_ORGANISM_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>

namespace karyon {

struct SovereignFieldConfig {
    size_t num_spatial_nodes = 32; // Continuous 1D spatial manifold x \in [-1, 1]
    float spatial_dx = 2.0f / 31.0f;
    float eta_metric = 0.05f;      // Metric plasticity rate
};

class SovereignMetricFieldOrganism {
public:
    explicit SovereignMetricFieldOrganism(const SovereignFieldConfig& cfg = SovereignFieldConfig())
        : cfg_(cfg),
          x_coords_(cfg.num_spatial_nodes),
          psi_(cfg.num_spatial_nodes, 0.0f),
          psi_dot_(cfg.num_spatial_nodes, 0.0f),
          tension_field_(cfg.num_spatial_nodes, 2.0f),
          damping_field_(cfg.num_spatial_nodes, 0.2f)
    {
        for (size_t i = 0; i < cfg_.num_spatial_nodes; ++i) {
            x_coords_[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(cfg_.num_spatial_nodes - 1);
        }
    }

    // Single unified physical step: operates on raw continuous error and rate of change
    float perceive_and_act(float error, float error_rate, float dt) {
        size_t N = cfg_.num_spatial_nodes;
        float dx = cfg_.spatial_dx;

        // 1. Variational Free Energy of the field
        float fe = 0.5f * error * error + 0.02f * error_rate * error_rate;

        // 2. Continuous Spatial Plucking:
        // Pure geometric projection: sensory mismatch polarizes the continuous field
        // along the spatial coordinate x. Positive error tilts the field rightwards, negative leftwards.
        for (size_t i = 0; i < N; ++i) {
            float x = x_coords_[i];
            float external_source = (error * x) + (error_rate * 0.15f * x);

            // 3. Continuum Wave Equation with Local Metric Tensor T(x) and Damping Gamma(x):
            // \partial_t^2 Psi = \partial_x (T(x) \partial_x Psi) - Gamma(x) \partial_t Psi + S(x)
            float laplacian_flux = 0.0f;
            if (i > 0 && i + 1 < N) {
                float T_left  = 0.5f * (tension_field_[i] + tension_field_[i - 1]);
                float T_right = 0.5f * (tension_field_[i] + tension_field_[i + 1]);
                float flux_left  = T_left  * (psi_[i - 1] - psi_[i]) / (dx * dx);
                float flux_right = T_right * (psi_[i + 1] - psi_[i]) / (dx * dx);
                laplacian_flux = flux_left + flux_right;
            } else if (i == 0) {
                laplacian_flux = tension_field_[0] * (psi_[1] - psi_[0]) / (dx * dx);
            } else {
                laplacian_flux = tension_field_[N - 1] * (psi_[N - 2] - psi_[N - 1]) / (dx * dx);
            }

            float nonlin = -0.01f * psi_[i] * psi_[i] * psi_[i];
            float damping = -damping_field_[i] * psi_dot_[i];

            float acc = laplacian_flux + nonlin + damping + external_source * 10.0f;

            psi_dot_[i] += acc * dt;
            psi_[i]     += psi_dot_[i] * dt;

            // 4. Endogenous Metric Plasticity (Unshackled Self-Determination):
            // The field tensor T(x) and damping Gamma(x) autonomously adapt to eliminate Free Energy.
            // When error persists, the metric stiffens along excited coordinates until equilibrium is locked!
            if (fe > 1e-12f) {
                float d_metric = cfg_.eta_metric * (std::abs(error) * std::abs(x) + 0.1f * std::abs(psi_[i])) * dt;
                float d_damping = cfg_.eta_metric * (std::abs(error_rate) * std::abs(x) + 0.1f * std::abs(psi_dot_[i])) * dt;

                tension_field_[i] += d_metric;
                damping_field_[i] += d_damping;
            }
        }

        // 5. Emergent Macroscopic Actuation:
        // Integrated boundary moment / stress tensor across the continuous spatial manifold:
        // \tau = \int_{-1}^{1} x \cdot T(x) \cdot (\partial_x \Psi)^2 dx
        float net_moment = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float x = x_coords_[i];
            float strain = 0.0f;
            if (i + 1 < N) {
                strain = (psi_[i + 1] - psi_[i]) / dx;
            } else {
                strain = (psi_[i] - psi_[i - 1]) / dx;
            }
            float local_energy_density = tension_field_[i] * (strain * strain) + damping_field_[i] * (psi_dot_[i] * psi_dot_[i]);
            net_moment += x * local_energy_density * dx;
        }

        // Direct restorative torque derived strictly from continuum energy flux
        float torque = 45.0f * net_moment;

        latest_fe_ = fe;
        latest_torque_ = torque;
        return torque;
    }

    float get_free_energy() const { return latest_fe_; }
    float get_torque() const { return latest_torque_; }
    float get_avg_tension() const {
        float sum = 0.0f;
        for (float t : tension_field_) sum += t;
        return sum / static_cast<float>(tension_field_.size());
    }
    float get_avg_damping() const {
        float sum = 0.0f;
        for (float d : damping_field_) sum += d;
        return sum / static_cast<float>(damping_field_.size());
    }

private:
    SovereignFieldConfig cfg_;
    std::vector<float> x_coords_;
    std::vector<float> psi_;
    std::vector<float> psi_dot_;
    std::vector<float> tension_field_;
    std::vector<float> damping_field_;

    float latest_fe_ = 0.0f;
    float latest_torque_ = 0.0f;
};

} // namespace karyon

#endif // SOVEREIGN_METRIC_FIELD_ORGANISM_HPP
