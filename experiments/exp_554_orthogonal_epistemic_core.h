// ============================================================================
// EXP-554: Orthogonal Epistemic Field & True Categorical Surprise
// Standard: KEP v28.0 (Principles 2, 4, 5, 27, 28, 29)
//
// Root Fix:
// 1. Orthogonal Byte Substrate:
//    Characters are NOT scalar numbers on a 0..255 ruler!
//    Each byte c in {0..255} is an orthogonal unitary basis vector e_c in R^256.
//    No false Euclidean distance (|'a' - 'b'| = 1 vs |'a' - 'z'| = 25).
// 2. True Epistemic Surprise (Variational Free Energy):
//    F_t(c) = -ln(P(c)) = -logit_c + ln(sum_j exp(logit_j))
//    Surprise is exact information content: if model assigned probability P(c)=0.01,
//    surprise is -ln(0.01) = 4.60 nats. If P(c)=0.99, surprise is 0.01 nats.
// 3. Exact Analytic Causal Force into the Wavefield:
//    dF_t / dq_k = sum_c (P(c) - y_c) * W_{c, k}
//    This pulls the physical modes along the exact gradient of epistemic surprise,
//    breaking the scalar center-of-mass collapse (spacebar trap).
// ============================================================================

#ifndef EXP_554_ORTHOGONAL_EPISTEMIC_CORE_H
#define EXP_554_ORTHOGONAL_EPISTEMIC_CORE_H

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <iostream>
#include <immintrin.h>

namespace karyon {

constexpr size_t NUM_PHYSICAL_MODES = 128;
constexpr size_t ALPHABET_SIZE = 256;

struct alignas(64) WavefieldState {
    std::array<float, NUM_PHYSICAL_MODES> q;            // Physical mode displacements
    std::array<float, NUM_PHYSICAL_MODES> p;            // Conjugate momenta
    std::array<float, NUM_PHYSICAL_MODES> stiffness;    // Mode stiffness K_k
    std::array<float, NUM_PHYSICAL_MODES> damping;      // Mode damping Gamma_k
};

class OrthogonalEpistemicCore {
public:
    OrthogonalEpistemicCore() noexcept {
        reset();
    }

    void reset() noexcept {
        current_modes_ = NUM_PHYSICAL_MODES;
        std::mt19937 rng(554);
        std::normal_distribution<float> norm(0.0f, 0.05f);

        for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
            state_.q[k] = 0.0f;
            state_.p[k] = 0.0f;
            state_.stiffness[k] = 1.0f + 0.02f * static_cast<float>(k);
            state_.damping[k] = 0.4f + 0.01f * static_cast<float>(k);
        }

        // Projective embedding: map orthogonal byte e_c into continuous field drive
        // And readout: map continuous field mode q into byte logits
        // Random orthogonal initialization
        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            bias_[c] = 0.0f;
            for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
                // Symmetrical bidirectional coupling between field and orthogonal alphabet
                float val = norm(rng);
                proj_[c * NUM_PHYSICAL_MODES + k] = val;
            }
        }
    }

    // Step 1: Perceive current byte as an ORTHOGONAL impulse (e_in),
    // propagate physical wavefield by dt, compute exact categorical Free Energy,
    // and train field dynamics + projections on actual surprise.
    float process_byte_transition(uint8_t input_byte, uint8_t target_next_byte, float dt, float lr) noexcept {
        // --- 1. CONTINUOUS PHYSICAL DYNAMICS ---
        // Input byte exerts an orthogonal physical drive vector F_k = proj[input_byte, k]
        const size_t in_idx = static_cast<size_t>(input_byte) * NUM_PHYSICAL_MODES;

        #pragma GCC ivdep
        for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
            const float drive = proj_[in_idx + k] * 4.0f;
            const float k_val = static_cast<float>(k + 1);
            const float omega_sq = state_.stiffness[k] * (k_val * 0.1f + 1.0f);
            const float gamma = 0.5f * state_.damping[k];

            // Analytic harmonic step with external orthogonal drive
            const float q_star = drive / (omega_sq + 1e-6f);
            const float q_rel = state_.q[k] - q_star;
            const float p_curr = state_.p[k];

            const float disc = omega_sq - gamma * gamma;
            float q_next = 0.0f;
            float p_next = 0.0f;

            if (disc > 1e-5f) {
                const float omega_d = std::sqrt(disc);
                const float decay = std::exp(-gamma * dt);
                const float cos_wt = std::cos(omega_d * dt);
                const float sin_wt = std::sin(omega_d * dt);
                const float inv_wd = 1.0f / omega_d;

                q_next = decay * (q_rel * cos_wt + (p_curr + gamma * q_rel) * inv_wd * sin_wt) + q_star;
                p_next = decay * (p_curr * cos_wt - (omega_sq * q_rel + gamma * p_curr) * inv_wd * sin_wt);
            } else {
                const float decay = std::exp(-gamma * dt);
                q_next = decay * (q_rel + p_curr * dt) + q_star;
                p_next = decay * (p_curr - gamma * p_curr * dt);
            }

            state_.q[k] = std::clamp(q_next, -10.0f, 10.0f);
            state_.p[k] = std::clamp(p_next, -20.0f, 20.0f);
        }

        // --- 2. LOGIT COMPUTATION OVER 256 ORTHOGONAL CLASSES ---
        std::array<float, ALPHABET_SIZE> logits{};
        float max_logit = -1e9f;

        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            float sum = bias_[c];
            const size_t c_idx = c * NUM_PHYSICAL_MODES;
            for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
                sum += proj_[c_idx + k] * state_.q[k];
            }
            logits[c] = sum;
            if (sum > max_logit) max_logit = sum;
        }

        // --- 3. NUMERICALLY STABLE SOFTMAX ---
        float sum_exp = 0.0f;
        std::array<float, ALPHABET_SIZE> probs{};
        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            float p = std::exp(logits[c] - max_logit);
            probs[c] = p;
            sum_exp += p;
        }

        const float inv_sum_exp = 1.0f / (sum_exp + 1e-7f);
        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            probs[c] *= inv_sum_exp;
        }

        // --- 4. TRUE EPISTEMIC SURPRISE (VARIATIONAL FREE ENERGY) ---
        // S = -ln(P(target))
        const float target_p = std::max(probs[target_next_byte], 1e-7f);
        const float surprise_nats = -std::log(target_p);

        // --- 5. EXACT CAUSAL SURPRISE GRADIENT BACKPROPAGATION ---
        // dS / dlogit_c = P(c) - delta_{c, target}
        if (lr > 0.0f) {
            std::array<float, NUM_PHYSICAL_MODES> dF_dq{};
            dF_dq.fill(0.0f);

            for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
                const float grad_c = probs[c] - (c == target_next_byte ? 1.0f : 0.0f);
                bias_[c] -= lr * grad_c;

                const size_t c_idx = c * NUM_PHYSICAL_MODES;
                for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
                    const float W_ck = proj_[c_idx + k];
                    const float q_k = state_.q[k];

                    // Plasticity in projection weights
                    proj_[c_idx + k] -= lr * grad_c * q_k;

                    // Accumulate exact physical pull force into wavefield state
                    dF_dq[k] += grad_c * W_ck;
                }
            }

            // Direct Epistemic Field Reaction: pull wavefield coordinates directly
            // along the direction that annihilates surprise!
            for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
                state_.q[k] -= lr * 2.0f * dF_dq[k];
            }
        }

        return surprise_nats;
    }

    // Inference mode: sample or greedy pick next byte
    uint8_t sample_next_byte(uint8_t input_byte, float dt, float temperature = 0.7f) noexcept {
        // 1. Perceive input byte
        const size_t in_idx = static_cast<size_t>(input_byte) * NUM_PHYSICAL_MODES;
        for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
            const float drive = proj_[in_idx + k] * 4.0f;
            state_.q[k] += drive * dt;
        }

        // 2. Readout logits
        std::array<float, ALPHABET_SIZE> logits{};
        float max_logit = -1e9f;

        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            float sum = bias_[c];
            const size_t c_idx = c * NUM_PHYSICAL_MODES;
            for (size_t k = 0; k < NUM_PHYSICAL_MODES; ++k) {
                sum += proj_[c_idx + k] * state_.q[k];
            }
            logits[c] = sum;
            if (sum > max_logit) max_logit = sum;
        }

        // Greedy if temp <= 0
        if (temperature <= 0.01f) {
            uint8_t best = 0;
            float top = -1e9f;
            for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
                if (logits[c] > top) {
                    top = logits[c];
                    best = static_cast<uint8_t>(c);
                }
            }
            return best;
        }

        // Softmax with temperature
        float sum_exp = 0.0f;
        std::array<float, ALPHABET_SIZE> probs{};
        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            float p = std::exp((logits[c] - max_logit) / temperature);
            probs[c] = p;
            sum_exp += p;
        }

        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * sum_exp;
        float cum = 0.0f;
        for (size_t c = 0; c < ALPHABET_SIZE; ++c) {
            cum += probs[c];
            if (cum >= r) {
                return static_cast<uint8_t>(c);
            }
        }
        return static_cast<uint8_t>(ALPHABET_SIZE - 1);
    }

private:
    size_t current_modes_;
    WavefieldState state_;

    // Orthogonal coupling projection matrices: [256 * 128]
    alignas(64) std::array<float, ALPHABET_SIZE * NUM_PHYSICAL_MODES> proj_;
    alignas(64) std::array<float, ALPHABET_SIZE> bias_;
};

} // namespace karyon

#endif // EXP_554_ORTHOGONAL_EPISTEMIC_CORE_H
