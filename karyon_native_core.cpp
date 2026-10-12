// karyon_native_core.cpp — Pure C++20 Sovereign Unified Substrate Core Engine v7.4
// Standard: KEP v20.0 Sovereign Master (Principles 2 to 27 Compliant)
// Full AVX2-FMA Vectorized Plasticity, Laminar Compression Field, and Mental Sandbox Rollout

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
// 1. ContinuousSuperPrimitive Implementation v7.4
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
            __m256 w_l = _mm256_loadu_ps(w_lin_row + i);
            __m256 v_in = _mm256_loadu_ps(in.data() + i);
            w_l = _mm256_fmadd_ps(v_factor, v_in, w_l);
            _mm256_storeu_ps(w_lin_row + i, w_l);

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
// 2. KaryonSovereignCore Implementation v7.4
// ---------------------------------------------------------------------------
KaryonSovereignCore::KaryonSovereignCore(const CoreConfig& config)
    : config_(config), rng_(42),
      noradrenaline_arousal_(0.10f),
      dopamine_vigor_(config.da_baseline),
      somatic_energy_(1.0f),
      current_horizon_k_(config.min_horizon_k),
      ticks_since_last_sprout_(0),
      moving_fe_ema_(0.0f) {

    size_t s_dim = config_.sensory_dim;
    size_t l_dim = config_.latent_dim;

    buf_sensory_in_.resize(s_dim, 0.0f);
    buf_sensory_out_.resize(s_dim, 0.0f);
    buf_latent_a_.resize(l_dim, 0.0f);
    buf_latent_b_.resize(l_dim, 0.0f);
    buf_quad_.resize(l_dim, 0.0f);
    buf_adjoint_sensory_.resize(s_dim, 0.0f);

    for (size_t p = 0; p < config_.initial_primitives; ++p) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(p, l_dim, rng_, false));
    }

    // Down-projection: [latent_dim x sensory_dim]
    W_sensory_down_.resize(l_dim * s_dim, 0.0f);
    std::normal_distribution<float> dist_down(0.0f, 1.0f / std::sqrt(static_cast<float>(s_dim)));
    for (size_t i = 0; i < W_sensory_down_.size(); ++i) {
        W_sensory_down_[i] = dist_down(rng_);
    }

    // Up-projection: [sensory_dim x latent_dim]
    W_sensory_up_.resize(s_dim * l_dim, 0.0f);
    std::normal_distribution<float> dist_up(0.0f, 1.0f / std::sqrt(static_cast<float>(l_dim)));
    for (size_t i = 0; i < W_sensory_up_.size(); ++i) {
        W_sensory_up_[i] = dist_up(rng_);
    }

    // Motor field: [sensory_dim x action_dim]
    W_motor_.resize(s_dim * config_.action_dim, 0.0f);
    std::normal_distribution<float> dist_motor(0.0f, 0.15f);
    for (size_t i = 0; i < W_motor_.size(); ++i) {
        W_motor_[i] = dist_motor(rng_);
    }

    // Action delta table for discrete mental rollout: [action_dim x sensory_dim]
    action_delta_table_.resize(config_.action_dim * s_dim, 0.0f);
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
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        buf_adjoint_sensory_[d] = current_state[d] - target_goal[d];
    }

    float lr_u = 0.25f * (1.0f + dopamine_vigor_);

    for (size_t m = 0; m < config_.action_dim; ++m) {
        float grad_m = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            grad_m += W_motor_[d * config_.action_dim + m] * buf_adjoint_sensory_[d];
        }
        action_out[m] = std::clamp(action_out[m] - lr_u * grad_m, -1.0f, 1.0f);
    }
}

float KaryonSovereignCore::evaluate_trajectory(size_t action_id,
                                              const std::vector<float>& s0,
                                              const std::vector<float>& target_goal,
                                              size_t horizon_k) {
    std::vector<float> sim = s0;
    const float* act_delta = action_delta_table_.data() + action_id * config_.sensory_dim;
    float total_cost = 0.0f;

    for (size_t tau = 1; tau <= horizon_k; ++tau) {
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            sim[d] += act_delta[d];
        }

        // Down-project
        for (size_t l = 0; l < config_.latent_dim; ++l) {
            const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
            buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, sim.data(), config_.sensory_dim));
        }

        // Primitives pass
        for (auto& p : primitives_) {
            p->forward(buf_latent_a_, buf_latent_b_, buf_quad_);
            std::copy(buf_latent_b_.begin(), buf_latent_b_.end(), buf_latent_a_.begin());
        }

        // Up-project
        for (size_t s = 0; s < config_.sensory_dim; ++s) {
            const float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
            sim[s] = avx2_dot_product(w_row, buf_latent_a_.data(), config_.latent_dim);
        }

        // Free Energy cost wrt target goal
        float dist_sq = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float diff = sim[d] - target_goal[d];
            dist_sq += diff * diff;
        }

        float discount = std::pow(config_.discount_gamma, static_cast<float>(tau - 1));
        total_cost += discount * dist_sq;
    }

    return total_cost;
}

size_t KaryonSovereignCore::decide_action(const std::vector<float>& current_state,
                                         const std::vector<float>& target_goal) {
    size_t best_a = 0;
    float min_cost = 1e9f;

    for (size_t a = 0; a < config_.action_dim; ++a) {
        float cost = evaluate_trajectory(a, current_state, target_goal, current_horizon_k_);
        if (cost < min_cost) {
            min_cost = cost;
            best_a = a;
        }
    }

    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    if (u(rng_) < 0.02f) {
        std::uniform_int_distribution<size_t> ad(0, config_.action_dim - 1);
        return ad(rng_);
    }

    return best_a;
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

    // 1. Motor efference projection
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float motor_flow = 0.0f;
        const float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            motor_flow += w_row[m] * action_taken[m];
        }
        buf_sensory_in_[d] = current_state[d] + motor_flow;
    }

    // 2. Down-projection: [latent_dim x sensory_dim]
    for (size_t l = 0; l < config_.latent_dim; ++l) {
        const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
        buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, buf_sensory_in_.data(), config_.sensory_dim));
    }

    // 3. Latent Super-Primitives processing
    for (auto& p : primitives_) {
        p->forward(buf_latent_a_, buf_latent_b_, buf_quad_);
        p->mature_birth_gain(0.005f);
        std::copy(buf_latent_b_.begin(), buf_latent_b_.end(), buf_latent_a_.begin());
    }

    // 4. Up-projection: [sensory_dim x latent_dim]
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        const float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
        buf_sensory_out_[s] = avx2_dot_product(w_row, buf_latent_a_.data(), config_.latent_dim);
    }

    // 5. Prediction error calculation
    float fe_error = 0.0f;
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float diff = buf_sensory_out_[d] - next_state_observed[d];
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

    evaluate_autopoietic_morphogenesis(fe_error);
    telem.active_primitives_count = primitives_.size();
    telem.event_sprouted = (primitives_.size() > count_before);
    telem.event_pruned = (primitives_.size() < count_before);

    // 7. Plasticity updates
    float plasticity = config_.plasticity_rate * (1.0f + noradrenaline_arousal_);

    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float err_s = next_state_observed[s] - buf_sensory_out_[s];
        float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
        for (size_t l = 0; l < config_.latent_dim; ++l) {
            w_row[l] += plasticity * err_s * buf_latent_a_[l];
        }
    }

    for (auto& p : primitives_) {
        p->update_plasticity_avx2(buf_latent_a_, buf_latent_a_, plasticity, buf_quad_);
        p->adapt_coordinates(0.01f, 0.01f, plasticity);
        p->update_epigenetic_lock(0.05f - fe_error);
    }

    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float err_d = next_state_observed[d] - buf_sensory_out_[d];
        float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            w_row[m] += plasticity * err_d * action_taken[m];
        }
    }

    telem.lock_ratio = get_average_epigenetic_lock();
    return telem;
}

CoreTelemetry KaryonSovereignCore::step_environment(const std::vector<float>& current_state,
                                                   const std::vector<float>& target_goal,
                                                   const std::vector<float>& next_state_observed,
                                                   size_t discrete_action) {
    std::vector<float> continuous_act(config_.action_dim, 0.0f);
    if (discrete_action < config_.action_dim) {
        continuous_act[discrete_action] = 1.0f;
    }

    // Update action delta table for future mental rollouts
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float delta = next_state_observed[d] - current_state[d];
        action_delta_table_[discrete_action * config_.sensory_dim + d] +=
            config_.plasticity_rate * 5.0f * (delta - action_delta_table_[discrete_action * config_.sensory_dim + d]);
    }

    auto telem = step_environment(current_state, target_goal, next_state_observed, continuous_act);
    telem.action_discrete_chosen = discrete_action;
    return telem;
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
