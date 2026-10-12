// karyon_native_core.cpp — Pure C++20 Sovereign Unified Substrate Core Engine v7.3
// Standard: KEP v20.0 Sovereign Master (Principles 2 to 27 Compliant)
// Full AVX2-FMA Vectorized Plasticity & Laminar Compression Field

#include "karyon_native_core.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

namespace karyon {

// Fast AVX2 dot product of 8-float aligned vectors
static inline float avx2_dot_product(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float res = _mm_cvtss_f32(sums);

    for (; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

// ---------------------------------------------------------------------------
// 1. ContinuousSuperPrimitive Implementation v7.3
// ---------------------------------------------------------------------------
ContinuousSuperPrimitive::ContinuousSuperPrimitive(size_t p_id, size_t d, std::mt19937& rng, bool sprouted_zero_shock)
    : id(p_id), dim(d),
      alpha_lin(0.80f), alpha_bilin(0.20f), alpha_rec(0.85f), alpha_res(0.30f),
      epigenetic_lock(0.0f), cumulative_causal_utility(0.0f),
      birth_gain(sprouted_zero_shock ? 0.0f : 1.0f) {

    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
    W_lin.resize(dim * dim);
    W_bilin.resize(dim * dim);
    state.resize(dim, 0.0f);

    for (size_t i = 0; i < dim * dim; ++i) {
        W_lin[i] = dist(rng);
        W_bilin[i] = dist(rng) * 0.5f;
    }
}

void ContinuousSuperPrimitive::forward(std::span<const float> in,
                                       std::span<float> out,
                                       std::span<float> scratch_quad) {
    // 1. Precompute Hadamard quad product: scratch_quad[i] = in[i] * state[i] via AVX2
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vin = _mm256_loadu_ps(in.data() + i);
        __m256 vst = _mm256_loadu_ps(state.data() + i);
        __m256 vq = _mm256_mul_ps(vin, vst);
        _mm256_storeu_ps(scratch_quad.data() + i, vq);
    }
    for (; i < dim; ++i) {
        scratch_quad[i] = in[i] * state[i];
    }

    // 2. High-speed AVX2 dot products for linear and bilinear projections
    for (size_t d = 0; d < dim; ++d) {
        const float* w_lin_row = W_lin.data() + d * dim;
        const float* w_bi_row = W_bilin.data() + d * dim;

        float lin_val = avx2_dot_product(w_lin_row, in.data(), dim);
        float bilin_val = avx2_dot_product(w_bi_row, scratch_quad.data(), dim);

        float net_flow = alpha_lin * lin_val +
                         alpha_bilin * bilin_val +
                         alpha_rec * state[d] +
                         alpha_res * in[d];

        float h_new = std::tanh(net_flow);
        state[d] = 0.70f * state[d] + 0.30f * h_new;

        // Net2Net Identity Sprouting: out = (1 - g) * in + g * h_new
        out[d] = (1.0f - birth_gain) * in[d] + birth_gain * h_new;
    }
}

void ContinuousSuperPrimitive::mature_birth_gain(float rate) {
    if (birth_gain < 1.0f) {
        birth_gain = std::min(1.0f, birth_gain + rate);
    }
}

void ContinuousSuperPrimitive::update_plasticity_avx2(std::span<const float> in,
                                                     std::span<const float> target,
                                                     float plasticity_rate,
                                                     std::span<const float> scratch_quad) {
    float inv_plasticity = plasticity_rate * (1.0f - std::tanh(epigenetic_lock));
    if (inv_plasticity <= 1e-6f) return;

    for (size_t d = 0; d < dim; ++d) {
        float err = target[d] - state[d];
        float factor = inv_plasticity * err;
        __m256 v_factor = _mm256_set1_ps(factor);

        float* w_lin_row = W_lin.data() + d * dim;
        float* w_bi_row = W_bilin.data() + d * dim;

        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            // W_lin update
            __m256 w_l = _mm256_loadu_ps(w_lin_row + i);
            __m256 v_in = _mm256_loadu_ps(in.data() + i);
            w_l = _mm256_fmadd_ps(v_factor, v_in, w_l);
            _mm256_storeu_ps(w_lin_row + i, w_l);

            // W_bilin update
            __m256 w_b = _mm256_loadu_ps(w_bi_row + i);
            __m256 v_q = _mm256_loadu_ps(scratch_quad.data() + i);
            w_b = _mm256_fmadd_ps(v_factor, v_q, w_b);
            _mm256_storeu_ps(w_bi_row + i, w_b);
        }
        for (; i < dim; ++i) {
            w_lin_row[i] += factor * in[i];
            w_bi_row[i] += factor * scratch_quad[i];
        }
    }
}

void ContinuousSuperPrimitive::adapt_coordinates(float dF_dalpha_lin,
                                                 float dF_dalpha_bilin,
                                                 float plasticity_rate) {
    float eta = plasticity_rate * 0.05f;
    alpha_lin = std::clamp(alpha_lin - eta * dF_dalpha_lin, 0.1f, 2.5f);
    alpha_bilin = std::clamp(alpha_bilin - eta * dF_dalpha_bilin, 0.0f, 2.5f);
}

void ContinuousSuperPrimitive::update_epigenetic_lock(float baseline_fe_reduction) {
    cumulative_causal_utility += baseline_fe_reduction;
    epigenetic_lock = std::clamp(epigenetic_lock + baseline_fe_reduction * 0.10f, 0.0f, 3.0f);
}

// ---------------------------------------------------------------------------
// 2. KaryonSovereignCore Implementation v7.3
// ---------------------------------------------------------------------------
KaryonSovereignCore::KaryonSovereignCore(const CoreConfig& config)
    : config_(config), rng_(42),
      noradrenaline_arousal_(0.10f),
      dopamine_vigor_(config.da_baseline),
      somatic_energy_(1.0f),
      current_horizon_k_(config.min_horizon_k),
      ticks_since_last_sprout_(0),
      moving_fe_ema_(0.0f),
      scratchpad_(config.sensory_dim, config.latent_dim) {

    for (size_t p = 0; p < config_.initial_primitives; ++p) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(p, config_.latent_dim, rng_, false));
    }

    // Down-projection: [latent_dim x sensory_dim]
    W_sensory_down_.resize(config_.latent_dim * config_.sensory_dim, 0.0f);
    std::normal_distribution<float> dist_down(0.0f, 1.0f / std::sqrt(static_cast<float>(config_.sensory_dim)));
    for (size_t i = 0; i < W_sensory_down_.size(); ++i) {
        W_sensory_down_[i] = dist_down(rng_);
    }

    // Up-projection: [sensory_dim x latent_dim]
    W_sensory_up_.resize(config_.sensory_dim * config_.latent_dim, 0.0f);
    std::normal_distribution<float> dist_up(0.0f, 1.0f / std::sqrt(static_cast<float>(config_.latent_dim)));
    for (size_t i = 0; i < W_sensory_up_.size(); ++i) {
        W_sensory_up_[i] = dist_up(rng_);
    }

    // Motor field: [sensory_dim x action_dim]
    W_motor_.resize(config_.sensory_dim * config_.action_dim, 0.0f);
    std::normal_distribution<float> dist_motor(0.0f, 0.15f);
    for (size_t i = 0; i < W_motor_.size(); ++i) {
        W_motor_[i] = dist_motor(rng_);
    }
}

bool KaryonSovereignCore::trigger_morphogenetic_sprout() {
    if (primitives_.size() >= config_.max_primitives) {
        return false;
    }
    size_t new_id = primitives_.size();
    primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(new_id, config_.latent_dim, rng_, true));
    ticks_since_last_sprout_ = 0;
    return true;
}

bool KaryonSovereignCore::prune_least_causal_primitive() {
    if (primitives_.size() <= config_.initial_primitives) {
        return false;
    }

    size_t worst_idx = 0;
    float min_score = 1e9f;

    for (size_t i = 0; i < primitives_.size(); ++i) {
        float score = primitives_[i]->cumulative_causal_utility + 5.0f * primitives_[i]->epigenetic_lock;
        if (score < min_score) {
            min_score = score;
            worst_idx = i;
        }
    }

    primitives_.erase(primitives_.begin() + worst_idx);
    return true;
}

void KaryonSovereignCore::evaluate_autopoietic_morphogenesis(float fe_error) {
    ticks_since_last_sprout_++;

    if (moving_fe_ema_ == 0.0f) {
        moving_fe_ema_ = fe_error;
    } else {
        moving_fe_ema_ = 0.99f * moving_fe_ema_ + 0.01f * fe_error;
    }

    if (ticks_since_last_sprout_ > config_.sprouting_cooldown_ticks &&
        fe_error > config_.sprouting_fe_threshold &&
        noradrenaline_arousal_ > 0.40f) {
        trigger_morphogenetic_sprout();
    }
}

void KaryonSovereignCore::decide_continuous_action(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<float> action_out) {
    auto& e_adjoint = scratchpad_.buf_adjoint_sensory;
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        e_adjoint[d] = current_state[d] - target_goal[d];
    }

    float lr_u = 0.25f * (1.0f + dopamine_vigor_);

    for (size_t m = 0; m < config_.action_dim; ++m) {
        float grad_m = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            grad_m += W_motor_[d * config_.action_dim + m] * e_adjoint[d];
        }
        action_out[m] = std::clamp(action_out[m] - lr_u * grad_m, -1.0f, 1.0f);
    }
}

CoreTelemetry KaryonSovereignCore::step_environment(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<const float> next_state_observed,
                                                   std::span<const float> action_taken) {
    CoreTelemetry telem{};
    telem.action_vector.assign(action_taken.begin(), action_taken.end());
    telem.mental_horizon_k = current_horizon_k_;
    telem.noradrenaline_arousal = noradrenaline_arousal_;
    size_t count_before = primitives_.size();

    auto& s_in = scratchpad_.buf_sensory_in;
    auto& s_out = scratchpad_.buf_sensory_out;
    auto& l_cur = scratchpad_.buf_latent_a;
    auto& l_nxt = scratchpad_.buf_latent_b;
    auto& q_scratch = scratchpad_.buf_quad;

    // 1. Motor efference projection on sensory field
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float motor_flow = 0.0f;
        const float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            motor_flow += w_row[m] * action_taken[m];
        }
        s_in[d] = current_state[d] + motor_flow;
    }

    // 2. High-speed AVX2 Down-projection to Latent manifold: [latent_dim x sensory_dim]
    for (size_t l = 0; l < config_.latent_dim; ++l) {
        const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
        l_cur[l] = std::tanh(avx2_dot_product(w_row, s_in.data(), config_.sensory_dim));
    }

    // 3. Fast Latent Super-Primitives processing (D=32, tiny footprint, massive speed!)
    for (auto& p : primitives_) {
        p->forward(l_cur, l_nxt, q_scratch);
        p->mature_birth_gain(0.005f);
        std::copy(l_nxt.begin(), l_nxt.end(), l_cur.begin());
    }

    // 4. High-speed AVX2 Up-projection to Sensory field: [sensory_dim x latent_dim]
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        const float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
        s_out[s] = avx2_dot_product(w_row, l_cur.data(), config_.latent_dim);
    }

    // 5. Prediction error calculation on sensory manifold
    float fe_error = 0.0f;
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float diff = s_out[d] - next_state_observed[d];
        fe_error += diff * diff;
    }
    fe_error /= static_cast<float>(config_.sensory_dim);

    telem.free_energy = fe_error;
    telem.sensory_prediction_error = std::sqrt(fe_error);

    // 6. Somatic arousal & horizon modulation
    float delta_fe = fe_error - 0.02f;
    if (delta_fe > 0.0f) {
        noradrenaline_arousal_ = std::min(1.0f, noradrenaline_arousal_ + delta_fe * 1.5f);
    } else {
        noradrenaline_arousal_ = std::max(0.01f, noradrenaline_arousal_ - config_.noradrenaline_decay);
    }
    adapt_endogenous_horizon(fe_error);

    // 7. Evaluate Autopoietic Morphogenesis
    evaluate_autopoietic_morphogenesis(fe_error);
    telem.active_primitives_count = primitives_.size();
    telem.event_sprouted = (primitives_.size() > count_before);
    telem.event_pruned = (primitives_.size() < count_before);

    // 8. Fully Vectorized AVX2 Plasticity update
    float plasticity = config_.plasticity_rate * (1.0f + noradrenaline_arousal_);

    // Update Up-projection W_sensory_up_
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float err_s = next_state_observed[s] - s_out[s];
        float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
        for (size_t l = 0; l < config_.latent_dim; ++l) {
            w_row[l] += plasticity * err_s * l_cur[l];
        }
    }

    // Update Primitives via AVX2 FMA
    for (auto& p : primitives_) {
        p->update_plasticity_avx2(l_cur, l_cur, plasticity, q_scratch);
        p->adapt_coordinates(0.01f, 0.01f, plasticity);
        p->update_epigenetic_lock(0.05f - fe_error);
    }

    // Update Motor field
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float err_d = next_state_observed[d] - s_out[d];
        float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            w_row[m] += plasticity * err_d * action_taken[m];
        }
    }

    telem.lock_ratio = get_average_epigenetic_lock();
    return telem;
}

size_t KaryonSovereignCore::decide_action(const std::vector<float>& current_state,
                                         const std::vector<float>& target_goal) {
    std::vector<float> continuous_act(config_.action_dim, 0.0f);
    decide_continuous_action(current_state, target_goal, continuous_act);
    auto max_it = std::max_element(continuous_act.begin(), continuous_act.end());
    return std::distance(continuous_act.begin(), max_it);
}

CoreTelemetry KaryonSovereignCore::step_environment(const std::vector<float>& current_state,
                                                   const std::vector<float>& target_goal,
                                                   const std::vector<float>& next_state_observed,
                                                   size_t discrete_action) {
    std::vector<float> continuous_act(config_.action_dim, 0.0f);
    if (discrete_action < config_.action_dim) {
        continuous_act[discrete_action] = 1.0f;
    }
    return step_environment(current_state, target_goal, next_state_observed, continuous_act);
}

void KaryonSovereignCore::adapt_endogenous_horizon(float fe_error) {
    if (noradrenaline_arousal_ > 0.40f || fe_error > 0.08f) {
        current_horizon_k_ = std::min(config_.max_horizon_k, current_horizon_k_ + 1);
    } else if (noradrenaline_arousal_ < 0.15f && fe_error < 0.02f) {
        current_horizon_k_ = std::max(config_.min_horizon_k, (current_horizon_k_ > 1 ? current_horizon_k_ - 1 : 1));
    }
}

float KaryonSovereignCore::get_average_epigenetic_lock() const {
    if (primitives_.empty()) return 0.0f;
    float sum = 0.0f;
    for (const auto& p : primitives_) {
        sum += p->epigenetic_lock;
    }
    return sum / static_cast<float>(primitives_.size());
}

} // namespace karyon
