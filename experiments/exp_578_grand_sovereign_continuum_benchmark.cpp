// ============================================================================
// EXP-578: GRAND SOVEREIGN CONTINUUM BENCHMARK
// Comparing 5 Non-Biological Sovereign Substrates Side-by-Side:
// 1. Wavefield Continuum (512D Coupled SDE, EXP-572)
// 2. Symplectic String Continuum (256D Autopoietic Strings, EXP-575/576)
// 3. Budding Hyper-Operator Continuum (Continuous Super-Primitives, EXP-577)
// 4. Topo-Geometric Fluid Lattice (Vorticity Fluid Dynamics) [NEW - EXP-578a]
// 5. Non-Commutative Quantum Phase Space (Symplectic Weyl-Wigner) [NEW - EXP-578b]
//
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 29, 30)
// 300,000 continuous learning steps on Alpaca corpus per substrate under
// 100% sovereign physical self-control (zero top-down human clamps).
// ============================================================================

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>
#include <numbers>
#include <omp.h>
#include <map>

#include "karyon_core.h"

namespace karyon {

// Shared text metric evaluation
struct SubstrateSpeechMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
};

SubstrateSpeechMetrics evaluate_text_quality(const std::string& text) {
    SubstrateSpeechMetrics m;
    if (text.empty()) return m;
    std::map<char, size_t> counts;
    size_t whitespaces = 0;
    size_t alphabets = 0;

    for (char ch : text) {
        counts[ch]++;
        if (std::isspace(static_cast<unsigned char>(ch))) whitespaces++;
        if (std::isalpha(static_cast<unsigned char>(ch))) alphabets++;
    }

    m.unique_chars = counts.size();
    m.whitespace_ratio = static_cast<double>(whitespaces) / text.size();
    m.alphabet_ratio = static_cast<double>(alphabets) / text.size();

    double ent = 0.0;
    for (const auto& [ch, count] : counts) {
        double p = static_cast<double>(count) / text.size();
        if (p > 0.0) ent -= p * std::log2(p);
    }
    m.char_entropy = ent;
    return m;
}

// ============================================================================
// 1. WAVEFIELD CONTINUUM (512D Coupled SDE Phase Oscillators)
// ============================================================================
class SubstrateWavefield {
public:
    SubstrateWavefield(size_t dim = 512) : dim_(dim) {
        q_.assign(dim_, 0.0f);
        p_.assign(dim_, 0.0f);
        stiffness_.resize(dim_);
        damping_.resize(dim_);
        causal_credit_.assign(dim_, 0.0f);
        epigenetic_lock_.assign(dim_, 0.0f);
        for (size_t i = 0; i < dim_; ++i) {
            stiffness_[i] = 1.0f + 0.02f * static_cast<float>(i % 64);
            damping_[i] = 0.02f;
        }
        std::mt19937 rng(5781);
        std::normal_distribution<float> norm(0.0f, 0.04f);
        sensory_.resize(CHANNEL_IO_DIM * dim_);
        motor_.resize(CHANNEL_IO_DIM * dim_);
        for (auto& w : sensory_) w = norm(rng);
        for (auto& w : motor_) w = norm(rng);
        boundary_momentum_.fill(0.0f);
        latest_probs_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * dim_;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dim_; ++i) {
            float force = sensory_[in_offset + i] - stiffness_[i] * q_[i] - damping_[i] * p_[i];
            p_[i] += force * 0.05f;
            q_[i] = std::tanh(q_[i] + p_[i] * 0.05f);
        }

        std::array<float, CHANNEL_IO_DIM> logits{};
        float max_l = -1e9f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_[c * dim_], q_.data(), dim_);
            float norm_dot = dot / (std::sqrt(static_cast<float>(dim_)) + SAFETY_EPSILON);
            float eff_logit = norm_dot - boundary_momentum_[c];
            logits[c] = eff_logit;
            if (eff_logit > max_l) max_l = eff_logit;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((logits[c] - max_l) * 3.5f);
            latest_probs_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probs_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probs_[c];
            if (cum >= r) { chosen = c; break; }
        }
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) boundary_momentum_[c] *= 0.70f;
        boundary_momentum_[chosen] += 0.90f;

        return chosen;
    }

    void adapt(uint8_t target_byte, float lr = 0.04f) noexcept {
        float p_target = latest_probs_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probs_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t i = 0; i < dim_; ++i) {
                    motor_[c * dim_ + i] += alpha * q_[i];
                }
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }

private:
    size_t dim_;
    float free_energy_ = 1.0f;
    std::vector<float> q_, p_, stiffness_, damping_, causal_credit_, epigenetic_lock_;
    std::vector<float> sensory_, motor_;
    std::array<float, CHANNEL_IO_DIM> boundary_momentum_, latest_probs_;
};

// ============================================================================
// 2. SYMPLECTIC STRING CONTINUUM (256D Symplectic Rotation)
// ============================================================================
class SubstrateSymplecticString {
public:
    SubstrateSymplecticString(size_t num_modes = 256) : num_modes_(num_modes) {
        u_.assign(num_modes_, 0.0f);
        v_.assign(num_modes_, 0.0f);
        tensions_.resize(num_modes_);
        damping_.resize(num_modes_, 0.015f);
        for (size_t m = 0; m < num_modes_; ++m) tensions_[m] = 1.0f + 0.03f * static_cast<float>(m % 64);

        cos_w_.resize(num_modes_);
        sin_w_.resize(num_modes_);
        for (size_t m = 0; m < num_modes_; ++m) {
            float w = 0.12f * std::sqrt(tensions_[m]);
            cos_w_[m] = std::cos(w * 0.040f);
            sin_w_[m] = std::sin(w * 0.040f);
        }

        sensory_fourier_.resize(CHANNEL_IO_DIM * num_modes_);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t m = 0; m < num_modes_; ++m) {
                float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>((c + 1) * (m + 1)) / static_cast<float>(num_modes_);
                sensory_fourier_[c * num_modes_ + m] = std::sin(phase) * 0.15f;
            }
        }

        std::mt19937 rng(5782);
        std::normal_distribution<float> norm(0.0f, 0.03f);
        motor_readout_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : motor_readout_) w = norm(rng);

        anchor_momentum_.fill(0.0f);
        latest_probs_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * num_modes_;
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < num_modes_; ++m) {
            float u_rot = u_[m] * cos_w_[m] - v_[m] * sin_w_[m];
            float v_rot = u_[m] * sin_w_[m] + v_[m] * cos_w_[m];
            float pluck = sensory_fourier_[in_offset + m];
            u_[m] = (u_rot + pluck * 0.040f) * 0.985f;
            v_[m] = (v_rot + pluck * 0.040f) * 0.985f;
        }

        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&motor_readout_[c * num_modes_], u_.data(), num_modes_);
            float eff_res = dot / (std::sqrt(static_cast<float>(num_modes_)) + SAFETY_EPSILON) - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * 4.5f);
            latest_probs_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probs_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probs_[c];
            if (cum >= r) { chosen = c; break; }
        }
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f;

        return chosen;
    }

    void adapt(uint8_t target_byte, float lr = 0.04f) noexcept {
        float p_target = latest_probs_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probs_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t m = 0; m < num_modes_; ++m) {
                    motor_readout_[c * num_modes_ + m] += alpha * u_[m];
                }
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }

private:
    size_t num_modes_;
    float free_energy_ = 1.0f;
    std::vector<float> u_, v_, tensions_, damping_, cos_w_, sin_w_;
    std::vector<float> sensory_fourier_, motor_readout_;
    std::array<float, CHANNEL_IO_DIM> anchor_momentum_, latest_probs_;
};

// ============================================================================
// 3. BUDDING HYPER-OPERATOR CONTINUUM (EXP-577 Super-Primitives)
// ============================================================================
class SubstrateHyperOperator {
public:
    SubstrateHyperOperator(size_t num_ops = 32) : num_ops_(num_ops) {
        h_state_.assign(256, 0.0f);
        m_state_.assign(256, 0.0f);

        std::mt19937 rng(5783);
        std::normal_distribution<float> norm(0.0f, 0.03f);

        W_in_.resize(CHANNEL_IO_DIM * 256);
        W_out_.resize(CHANNEL_IO_DIM * 256);
        for (auto& w : W_in_) w = norm(rng);
        for (auto& w : W_out_) w = norm(rng);

        op_kernels_.resize(num_ops_ * 256);
        for (auto& w : op_kernels_) w = norm(rng);

        anchor_momentum_.fill(0.0f);
        latest_probs_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * 256;
        std::vector<float> hyper_accum(256, 0.0f);

        for (size_t k = 0; k < num_ops_; ++k) {
            const float* k_ptr = &op_kernels_[k * 256];
            for (size_t i = 0; i < 256; ++i) {
                float in_val = W_in_[in_offset + i];
                float lin = -0.81f * in_val;
                float gate = -0.48f * (h_state_[i] / (1.0f + std::exp(-k_ptr[i] * in_val)));
                float integ = -0.32f * m_state_[i];
                float attract = 0.06f * std::tanh(2.0f * h_state_[i]);
                hyper_accum[i] += (lin + gate + integ + attract) * 0.040f;
            }
        }

        float norm_factor = 1.0f / (std::sqrt(static_cast<float>(num_ops_)) + SAFETY_EPSILON);
        for (size_t i = 0; i < 256; ++i) {
            float drive = hyper_accum[i] * norm_factor;
            m_state_[i] = m_state_[i] * 0.95f + drive * 0.05f;
            h_state_[i] = std::tanh(h_state_[i] * 0.92f + drive + m_state_[i] * 0.1f);
        }

        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&W_out_[c * 256], h_state_.data(), 256);
            float eff_res = dot / (std::sqrt(256.0f) + SAFETY_EPSILON) - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * 4.0f);
            latest_probs_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probs_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probs_[c];
            if (cum >= r) { chosen = c; break; }
        }
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f;

        return chosen;
    }

    void adapt(uint8_t target_byte, float lr = 0.04f) noexcept {
        float p_target = latest_probs_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probs_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t i = 0; i < 256; ++i) {
                    W_out_[c * 256 + i] += alpha * h_state_[i];
                }
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }

private:
    size_t num_ops_;
    float free_energy_ = 1.0f;
    std::vector<float> h_state_, m_state_;
    std::vector<float> W_in_, W_out_, op_kernels_;
    std::array<float, CHANNEL_IO_DIM> anchor_momentum_, latest_probs_;
};

// ============================================================================
// 4. TOPO-GEOMETRIC FLUID LATTICE (NEW - EXP-578a)
// Hydrodynamic Euler-Vorticity Lattice with Continuous Circulation Integ
// du/dt = -(u . grad)u - grad(P) + nu * Laplacian(u) + Source(x)
// Circulation Gamma_k = \oint u . dl encodes long-range state memory
// ============================================================================
class SubstrateFluidLattice {
public:
    SubstrateFluidLattice(size_t grid_dim = 16) : grid_dim_(grid_dim), cells_(grid_dim * grid_dim) {
        u_vel_.assign(cells_, 0.0f);
        v_vel_.assign(cells_, 0.0f);
        vorticity_.assign(cells_, 0.0f);
        pressure_.assign(cells_, 0.0f);

        std::mt19937 rng(5784);
        std::normal_distribution<float> norm(0.0f, 0.04f);

        sensory_jets_.resize(CHANNEL_IO_DIM * cells_);
        for (auto& w : sensory_jets_) w = norm(rng);

        circulation_sensors_.resize(CHANNEL_IO_DIM * cells_);
        for (auto& w : circulation_sensors_) w = norm(rng);

        anchor_momentum_.fill(0.0f);
        latest_probs_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * cells_;

        // 1. Hydrodynamic Jet Impulse Injection
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells_; ++i) {
            float jet = sensory_jets_[in_offset + i];
            u_vel_[i] += jet * 0.05f;
            v_vel_[i] += jet * 0.05f * ((i % 2 == 0) ? 1.0f : -1.0f);
        }

        // 2. Navier-Stokes Vorticity & Pressure Advection step on 2D torus grid
        std::vector<float> next_u(cells_, 0.0f);
        std::vector<float> next_v(cells_, 0.0f);

        const int N = static_cast<int>(grid_dim_);
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int idx = y * N + x;
                int right = y * N + ((x + 1) % N);
                int left  = y * N + ((x - 1 + N) % N);
                int up    = ((y - 1 + N) % N) * N + x;
                int down  = ((y + 1) % N) * N + x;

                // Spatial derivatives via central differences
                float du_dx = (u_vel_[right] - u_vel_[left]) * 0.5f;
                float du_dy = (u_vel_[down] - u_vel_[up]) * 0.5f;
                float dv_dx = (v_vel_[right] - v_vel_[left]) * 0.5f;
                float dv_dy = (v_vel_[down] - v_vel_[up]) * 0.5f;

                // Vorticity w = dv/dx - du/dy
                float w = dv_dx - du_dy;
                vorticity_[idx] = w;

                // Non-linear advection -(u du/dx + v du/dy) + viscous diffusion
                float lap_u = u_vel_[right] + u_vel_[left] + u_vel_[up] + u_vel_[down] - 4.0f * u_vel_[idx];
                float lap_v = v_vel_[right] + v_vel_[left] + v_vel_[up] + v_vel_[down] - 4.0f * v_vel_[idx];

                float u_adv = -(u_vel_[idx] * du_dx + v_vel_[idx] * du_dy) + 0.01f * lap_u;
                float v_adv = -(u_vel_[idx] * dv_dx + v_vel_[idx] * dv_dy) + 0.01f * lap_v;

                next_u[idx] = std::tanh(u_vel_[idx] + u_adv * 0.040f);
                next_v[idx] = std::tanh(v_vel_[idx] + v_adv * 0.040f);
            }
        }
        u_vel_ = std::move(next_u);
        v_vel_ = std::move(next_v);

        // 3. Circulation Readout \oint u . dl
        std::array<float, CHANNEL_IO_DIM> circulation{};
        float max_circ = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float circ_sum = 0.0f;
            const float* w_ptr = &circulation_sensors_[c * cells_];
            for (size_t i = 0; i < cells_; ++i) {
                circ_sum += w_ptr[i] * (u_vel_[i] + vorticity_[i]);
            }
            float norm_circ = circ_sum / (std::sqrt(static_cast<float>(cells_)) + SAFETY_EPSILON) - anchor_momentum_[c];
            circulation[c] = norm_circ;
            if (norm_circ > max_circ) max_circ = norm_circ;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((circulation[c] - max_circ) * 4.0f);
            latest_probs_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probs_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probs_[c];
            if (cum >= r) { chosen = c; break; }
        }
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f;

        return chosen;
    }

    void adapt(uint8_t target_byte, float lr = 0.04f) noexcept {
        float p_target = latest_probs_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probs_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t i = 0; i < cells_; ++i) {
                    circulation_sensors_[c * cells_ + i] += alpha * (u_vel_[i] + vorticity_[i]);
                }
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }

private:
    size_t grid_dim_;
    size_t cells_;
    float free_energy_ = 1.0f;
    std::vector<float> u_vel_, v_vel_, vorticity_, pressure_;
    std::vector<float> sensory_jets_, circulation_sensors_;
    std::array<float, CHANNEL_IO_DIM> anchor_momentum_, latest_probs_;
};

// ============================================================================
// 5. NON-COMMUTATIVE QUANTUM PHASE SPACE (NEW - EXP-578b)
// Non-Commutative Quantum Phase Space with Weyl-Wigner Shift Operators
// [Q_i, P_j] = i hbar \theta_{ij}
// Unitary Phase Displacement D(\alpha) = exp(\alpha a^\dagger - \alpha^* a)
// ============================================================================
class SubstrateQuantumPhaseSpace {
public:
    SubstrateQuantumPhaseSpace(size_t num_modes = 256) : num_modes_(num_modes) {
        q_op_.assign(num_modes_, 0.0f);
        p_op_.assign(num_modes_, 0.0f);
        non_commutative_theta_.resize(num_modes_);
        for (size_t i = 0; i < num_modes_; ++i) {
            non_commutative_theta_[i] = 0.05f * static_cast<float>((i % 16) + 1);
        }

        std::mt19937 rng(5785);
        std::normal_distribution<float> norm(0.0f, 0.03f);

        weyl_disp_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : weyl_disp_) w = norm(rng);

        phase_readout_.resize(CHANNEL_IO_DIM * num_modes_);
        for (auto& w : phase_readout_) w = norm(rng);

        anchor_momentum_.fill(0.0f);
        latest_probs_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * num_modes_;

        // 1. Weyl-Wigner Non-Commutative Displacement Step
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < num_modes_; ++m) {
            float alpha = weyl_disp_[in_offset + m];
            float theta = non_commutative_theta_[m];

            // Non-commutative phase shift: dq = alpha * cos(theta), dp = alpha * sin(theta)
            // Plus non-commutative cross-coupling: [Q, P] -> theta * (q*p_prev - p*q_prev)
            float q_prev = q_op_[m];
            float p_prev = p_op_[m];

            float cross_comm = theta * (q_prev * p_prev);

            q_op_[m] = std::tanh(q_prev + alpha * std::cos(theta) + cross_comm * 0.02f);
            p_op_[m] = std::tanh(p_prev + alpha * std::sin(theta) - cross_comm * 0.02f);
        }

        // 2. Quantum Phase Overlap Readout \langle \psi | \hat{O} | \psi \rangle
        std::array<float, CHANNEL_IO_DIM> phase_overlap{};
        float max_overlap = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = avx2_dot_product(&phase_readout_[c * num_modes_], q_op_.data(), num_modes_);
            float norm_dot = dot / (std::sqrt(static_cast<float>(num_modes_)) + SAFETY_EPSILON) - anchor_momentum_[c];
            phase_overlap[c] = norm_dot;
            if (norm_dot > max_overlap) max_overlap = norm_dot;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((phase_overlap[c] - max_overlap) * 4.5f);
            latest_probs_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probs_[c] *= inv_sum;

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probs_[c];
            if (cum >= r) { chosen = c; break; }
        }
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f;

        return chosen;
    }

    void adapt(uint8_t target_byte, float lr = 0.04f) noexcept {
        float p_target = latest_probs_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probs_[c];
            if (std::abs(err) > 0.001f) {
                float alpha = lr * err;
                for (size_t m = 0; m < num_modes_; ++m) {
                    phase_readout_[c * num_modes_ + m] += alpha * q_op_[m];
                }
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }

private:
    size_t num_modes_;
    float free_energy_ = 1.0f;
    std::vector<float> q_op_, p_op_, non_commutative_theta_;
    std::vector<float> weyl_disp_, phase_readout_;
    std::array<float, CHANNEL_IO_DIM> anchor_momentum_, latest_probs_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-578: GRAND SOVEREIGN CONTINUUM BENCHMARK (5 SUBSTRATES)     ===\n";
    std::cout << "=== Standard KEP v30.0 (Principles 2, 3, 4, 5, 6, 29, 30)           ===\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) {
        text_path = "data/web_downloads/alpaca_data.json";
        file.open(text_path, std::ios::binary);
    }

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::cout << "[+] Corpus loaded: " << corpus.size() << " bytes.\n\n";
    } else {
        std::string fallback = "The capital of France is Paris. Language is physical resonance.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000;
    const std::string prompt = "The capital of France is ";

    // 1. Wavefield Continuum
    std::cout << ">>> [1/5] Running Wavefield Continuum (512D SDE Phase Oscillators)...\n";
    karyon::SubstrateWavefield wave_sub(512);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        wave_sub.step(in_b);
        wave_sub.adapt(tgt_b, 0.04f);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double wave_throughput = train_steps / std::chrono::duration<double>(t1 - t0).count();
    std::string wave_speech = prompt;
    uint8_t curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t act = wave_sub.step(curr);
        wave_speech += static_cast<char>(act);
        curr = static_cast<uint8_t>(act);
    }
    auto wave_m = karyon::evaluate_text_quality(wave_speech);

    // 2. Symplectic String Continuum
    std::cout << ">>> [2/5] Running Symplectic String Continuum (256D Phase Rotation)...\n";
    karyon::SubstrateSymplecticString string_sub(256);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        string_sub.step(in_b);
        string_sub.adapt(tgt_b, 0.04f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double string_throughput = train_steps / std::chrono::duration<double>(t1 - t0).count();
    std::string string_speech = prompt;
    curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t act = string_sub.step(curr);
        string_speech += static_cast<char>(act);
        curr = static_cast<uint8_t>(act);
    }
    auto string_m = karyon::evaluate_text_quality(string_speech);

    // 3. Budding Hyper-Operator Continuum
    std::cout << ">>> [3/5] Running Budding Hyper-Operator Continuum (Super-Primitives)...\n";
    karyon::SubstrateHyperOperator hyper_sub(32);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        hyper_sub.step(in_b);
        hyper_sub.adapt(tgt_b, 0.04f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double hyper_throughput = train_steps / std::chrono::duration<double>(t1 - t0).count();
    std::string hyper_speech = prompt;
    curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t act = hyper_sub.step(curr);
        hyper_speech += static_cast<char>(act);
        curr = static_cast<uint8_t>(act);
    }
    auto hyper_m = karyon::evaluate_text_quality(hyper_speech);

    // 4. Topo-Geometric Fluid Lattice
    std::cout << ">>> [4/5] Running Topo-Geometric Fluid Lattice (Vorticity Navier-Stokes)...\n";
    karyon::SubstrateFluidLattice fluid_sub(16);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        fluid_sub.step(in_b);
        fluid_sub.adapt(tgt_b, 0.04f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fluid_throughput = train_steps / std::chrono::duration<double>(t1 - t0).count();
    std::string fluid_speech = prompt;
    curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t act = fluid_sub.step(curr);
        fluid_speech += static_cast<char>(act);
        curr = static_cast<uint8_t>(act);
    }
    auto fluid_m = karyon::evaluate_text_quality(fluid_speech);

    // 5. Non-Commutative Quantum Phase Space
    std::cout << ">>> [5/5] Running Non-Commutative Quantum Phase Space (Weyl-Wigner)...\n";
    karyon::SubstrateQuantumPhaseSpace quantum_sub(256);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        quantum_sub.step(in_b);
        quantum_sub.adapt(tgt_b, 0.04f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double quantum_throughput = train_steps / std::chrono::duration<double>(t1 - t0).count();
    std::string quantum_speech = prompt;
    curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t act = quantum_sub.step(curr);
        quantum_speech += static_cast<char>(act);
        curr = static_cast<uint8_t>(act);
    }
    auto quantum_m = karyon::evaluate_text_quality(quantum_speech);

    // ========================================================================
    // FINAL GRAND TELEMETRY TABLE
    // ========================================================================
    std::cout << "\n=========================================================================================================\n";
    std::cout << "=== GRAND SOVEREIGN CONTINUUM BENCHMARK RESULTS (EXP-578)                                             ===\n";
    std::cout << "=========================================================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Free Energy (nats) | Alphabet % | Whitespace % | Entropy\n";
    std::cout << "--------------------------------|--------------------|--------------------|------------|--------------|--------\n";
    std::cout << "1. Wavefield (512D SDE)         | " << wave_throughput << "           | " << wave_sub.get_free_energy() << "             | " << (wave_m.alphabet_ratio * 100.0) << "%      | " << (wave_m.whitespace_ratio * 100.0) << "%        | " << wave_m.char_entropy << "\n";
    std::cout << "2. Symplectic String (256D)     | " << string_throughput << "          | " << string_sub.get_free_energy() << "             | " << (string_m.alphabet_ratio * 100.0) << "%      | " << (string_m.whitespace_ratio * 100.0) << "%        | " << string_m.char_entropy << "\n";
    std::cout << "3. Budding Hyper-Operator       | " << hyper_throughput << "           | " << hyper_sub.get_free_energy() << "             | " << (hyper_m.alphabet_ratio * 100.0) << "%      | " << (hyper_m.whitespace_ratio * 100.0) << "%        | " << hyper_m.char_entropy << "\n";
    std::cout << "4. Topo-Geometric Fluid Lattice | " << fluid_throughput << "           | " << fluid_sub.get_free_energy() << "             | " << (fluid_m.alphabet_ratio * 100.0) << "%      | " << (fluid_m.whitespace_ratio * 100.0) << "%        | " << fluid_m.char_entropy << "\n";
    std::cout << "5. Quantum Phase Space (Weyl)   | " << quantum_throughput << "          | " << quantum_sub.get_free_energy() << "             | " << (quantum_m.alphabet_ratio * 100.0) << "%      | " << (quantum_m.whitespace_ratio * 100.0) << "%        | " << quantum_m.char_entropy << "\n";
    std::cout << "=========================================================================================================\n\n";

    std::cout << ">>> SAMPLE GENERATED SPEECH COMPARISON:\n";
    std::cout << "1. Wavefield    : \"" << wave_speech.substr(0, 100) << "\"\n";
    std::cout << "2. String       : \"" << string_speech.substr(0, 100) << "\"\n";
    std::cout << "3. Hyper-Op     : \"" << hyper_speech.substr(0, 100) << "\"\n";
    std::cout << "4. Fluid        : \"" << fluid_speech.substr(0, 100) << "\"\n";
    std::cout << "5. Quantum Phase: \"" << quantum_speech.substr(0, 100) << "\"\n";

    return 0;
}
