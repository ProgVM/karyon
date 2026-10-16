// ============================================================================
// ResonantStringCore: Weightless Multi-Dimensional String/Manifold Substrate
// Karyon-CoRE v28.0 - KEP Sovereign Architecture
// No Weight Matrices W. Dynamics emerge purely from harmonic mode resonance,
// spatial Fourier projections, and non-linear Fermi-Pasta-Ulam wave interactions.
// Uses Unitary Symplectic Rotation for Unconditionally Stable Wave Dynamics.
// ============================================================================

#ifndef RESONANT_STRING_CORE_HPP
#define RESONANT_STRING_CORE_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <iostream>

namespace karyon {

struct StringConfig {
    size_t sensory_dim = 256;
    size_t action_dim = 4;
    size_t num_modes = 32;       // Number of harmonic modes n = 1..N
    size_t mode_spatial_dim = 16;// Spatial dimension D per harmonic mode
    float base_frequency = 0.1f; // Fundamental frequency w_0
    float damping_base = 0.02f;  // Dissipation / damping gamma_0
    float dt = 0.05f;            // Time step
    float nonlinearity_beta = 0.01f; // FPU-beta non-linear coupling constant
};

struct StringTelemetry {
    float total_string_energy = 0.0f;
    float spectral_entropy = 0.0f;
    float free_energy = 0.0f;
    std::vector<float> mode_energies;
    std::vector<float> action_outputs;
};

class ResonantStringCore {
public:
    explicit ResonantStringCore(const StringConfig& cfg)
        : config_(cfg),
          modes_u_(cfg.num_modes, std::vector<float>(cfg.mode_spatial_dim, 0.0f)),
          modes_v_(cfg.num_modes, std::vector<float>(cfg.mode_spatial_dim, 0.0f)),
          mode_phases_(cfg.num_modes, 0.0f),
          mode_tensions_(cfg.num_modes, 1.0f)
    {
        // Precompute fixed spatial Fourier basis matrices for sensory injection
        sensory_fourier_cos_.resize(config_.num_modes, std::vector<float>(config_.sensory_dim));
        sensory_fourier_sin_.resize(config_.num_modes, std::vector<float>(config_.sensory_dim));
        for (size_t n = 0; n < config_.num_modes; ++n) {
            float mode_idx = static_cast<float>(n + 1);
            for (size_t k = 0; k < config_.sensory_dim; ++k) {
                float angle = 2.0f * std::numbers::pi_v<float> * mode_idx * static_cast<float>(k) / static_cast<float>(config_.sensory_dim);
                sensory_fourier_cos_[n][k] = std::cos(angle) / std::sqrt(static_cast<float>(config_.sensory_dim));
                sensory_fourier_sin_[n][k] = std::sin(angle) / std::sqrt(static_cast<float>(config_.sensory_dim));
            }
        }
    }

    // Step environment without any weight matrix multiplications W*x
    StringTelemetry step(const std::vector<float>& sensory_state,
                         const std::vector<float>& target_goal,
                         float external_valence,
                         std::vector<float>& out_actions)
    {
        out_actions.assign(config_.action_dim, 0.0f);
        StringTelemetry telem;

        float dt = config_.dt;
        size_t N = config_.num_modes;
        size_t D = config_.mode_spatial_dim;
        size_t S = config_.sensory_dim;

        // 1. Fixed Fourier Spatial Projection of Sensory Error / Input onto Mode Forces (Plucking)
        std::vector<float> err_s(S, 0.0f);
        for (size_t k = 0; k < S && k < sensory_state.size(); ++k) {
            float goal_val = (k < target_goal.size()) ? target_goal[k] : 0.0f;
            err_s[k] = sensory_state[k] - goal_val;
        }

        std::vector<std::vector<float>> force_u(N, std::vector<float>(D, 0.0f));
        std::vector<std::vector<float>> force_v(N, std::vector<float>(D, 0.0f));

        for (size_t n = 0; n < N; ++n) {
            float proj_cos = 0.0f;
            float proj_sin = 0.0f;
            for (size_t k = 0; k < S; ++k) {
                proj_cos += err_s[k] * sensory_fourier_cos_[n][k];
                proj_sin += err_s[k] * sensory_fourier_sin_[n][k];
            }
            for (size_t d = 0; d < D; ++d) {
                float phase_d = 2.0f * std::numbers::pi_v<float> * static_cast<float>(d) / static_cast<float>(D);
                force_u[n][d] = std::clamp(proj_cos * std::cos(phase_d) - proj_sin * std::sin(phase_d), -1.0f, 1.0f);
                force_v[n][d] = std::clamp(proj_cos * std::sin(phase_d) + proj_sin * std::cos(phase_d), -1.0f, 1.0f);
            }
        }

        // 2. Unconditionally Stable Unitary Phase Rotation + Dissipation + Coupling
        std::vector<float> mode_energies(N, 0.0f);
        float total_energy = 0.0f;

        for (size_t n = 0; n < N; ++n) {
            float mode_num = static_cast<float>(n + 1);
            float w_n = config_.base_frequency * mode_num * mode_tensions_[n];
            float cos_w = std::cos(w_n * dt);
            float sin_w = std::sin(w_n * dt);
            float damping = std::exp(-config_.damping_base * dt * mode_num);

            for (size_t d = 0; d < D; ++d) {
                float u_curr = modes_u_[n][d];
                float v_curr = modes_v_[n][d];

                // Symplectic exact rotation matrix (unitary evolution)
                float u_rot = u_curr * cos_w - v_curr * sin_w;
                float v_rot = u_curr * sin_w + v_curr * cos_w;

                // FPU non-linear inter-mode coupling (harmonic energy transfer)
                float fpu_u = 0.0f;
                float fpu_v = 0.0f;
                if (n > 0) {
                    fpu_u += config_.nonlinearity_beta * (modes_u_[n-1][d] - u_curr);
                    fpu_v += config_.nonlinearity_beta * (modes_v_[n-1][d] - v_curr);
                }
                if (n + 1 < N) {
                    fpu_u += config_.nonlinearity_beta * (modes_u_[n+1][d] - u_curr);
                    fpu_v += config_.nonlinearity_beta * (modes_v_[n+1][d] - v_curr);
                }

                // Update with damping, FPU coupling, and sensory force impulse
                modes_u_[n][d] = std::clamp((u_rot + (fpu_u + force_u[n][d]) * dt) * damping, -3.0f, 3.0f);
                modes_v_[n][d] = std::clamp((v_rot + (fpu_v + force_v[n][d]) * dt) * damping, -3.0f, 3.0f);

                float energy_nd = modes_u_[n][d] * modes_u_[n][d] + modes_v_[n][d] * modes_v_[n][d];
                mode_energies[n] += energy_nd;
            }
            total_energy += mode_energies[n];
        }

        // 3. Adaptive Tension Tuning (Epigenetic Allostatic Pitch Shift) based on Valence
        for (size_t n = 0; n < N; ++n) {
            float tension_drift = -0.005f * external_valence * (mode_energies[n] / (total_energy + 1e-6f));
            if (external_valence < 0.0f) {
                tension_drift += 0.002f * std::sin(static_cast<float>(n + 1) * mode_phases_[n]);
            }
            mode_tensions_[n] = std::clamp(mode_tensions_[n] + tension_drift, 0.5f, 2.0f);
            mode_phases_[n] += 0.02f;
        }

        // 4. Action Extraction from Spatial Interference / Wave Phase Boundary
        size_t modes_per_action = N / config_.action_dim;
        for (size_t a = 0; a < config_.action_dim; ++a) {
            float action_drive = 0.0f;
            size_t start_mode = a * modes_per_action;
            size_t end_mode = (a + 1 == config_.action_dim) ? N : (a + 1) * modes_per_action;

            for (size_t n = start_mode; n < end_mode; ++n) {
                for (size_t d = 0; d < D; ++d) {
                    action_drive += modes_u_[n][d] * std::cos(mode_phases_[n]) - modes_v_[n][d] * std::sin(mode_phases_[n]);
                }
            }
            out_actions[a] = std::tanh(action_drive * 0.05f); // Smooth bounded action drive
        }

        // 5. Compute Free Energy
        float fe = 0.0f;
        for (size_t k = 0; k < S; ++k) {
            fe += err_s[k] * err_s[k];
        }
        fe = (fe / static_cast<float>(S)) + 0.001f * total_energy;

        telem.total_string_energy = total_energy;
        telem.free_energy = fe;
        telem.mode_energies = mode_energies;
        telem.action_outputs = out_actions;

        return telem;
    }

private:
    StringConfig config_;
    std::vector<std::vector<float>> modes_u_;
    std::vector<std::vector<float>> modes_v_;
    std::vector<float> mode_phases_;
    std::vector<float> mode_tensions_;

    std::vector<std::vector<float>> sensory_fourier_cos_;
    std::vector<std::vector<float>> sensory_fourier_sin_;
};

} // namespace karyon

#endif // RESONANT_STRING_CORE_HPP
