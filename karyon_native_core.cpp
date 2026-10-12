// karyon_native_core.cpp — Pure C++20 Sovereign Unified Substrate Core Engine v6.0
// Standard: KEP v20.0 Sovereign Master (Principle 2 to 27 Compliant)
// 100% Production Grade Implementation of EXP-470 to EXP-479 Core Mechanisms

#include "karyon_native_core.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace karyon {

// ---------------------------------------------------------------------------
// 1. ContinuousSuperPrimitive Implementation
// ---------------------------------------------------------------------------
ContinuousSuperPrimitive::ContinuousSuperPrimitive(size_t p_id, size_t d, std::mt19937& rng)
    : id(p_id), dim(d),
      alpha_lin(0.80f), alpha_bilin(0.20f), alpha_rec(0.85f), alpha_res(0.30f),
      epigenetic_lock(0.0f), cumulative_causal_utility(0.0f) {

    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
    W_lin.resize(dim * dim);
    for (auto& w : W_lin) w = dist(rng);

    W_bilin.resize(dim * dim);
    for (auto& w : W_bilin) w = dist(rng) * 0.1f;

    bias.assign(dim, 0.0f);
    state.assign(dim, 0.0f);
    last_x.assign(dim, 0.0f);
    last_out.assign(dim, 0.0f);
}

std::vector<float> ContinuousSuperPrimitive::forward(const std::vector<float>& x) {
    last_x = x;
    std::vector<float> out(dim, 0.0f);

    for (size_t d = 0; d < dim; ++d) {
        state[d] = alpha_rec * state[d] + (1.0f - alpha_rec) * x[d];

        float lin_val = 0.0f;
        float bilin_val = 0.0f;
        const float* w_lin_row = W_lin.data() + d * dim;
        const float* w_bi_row = W_bilin.data() + d * dim;

        #pragma omp simd reduction(+:lin_val, bilin_val)
        for (size_t i = 0; i < dim; ++i) {
            lin_val += w_lin_row[i] * x[i];
            bilin_val += w_bi_row[i] * (x[i] * state[i]);
        }

        float raw = alpha_lin * lin_val + alpha_bilin * bilin_val + alpha_res * x[d] + bias[d];
        out[d] = std::tanh(raw);
    }

    last_out = out;
    return out;
}

void ContinuousSuperPrimitive::causal_update(const std::vector<float>& error,
                                              float global_fe_reduction,
                                              float plasticity_rate) {
    if (epigenetic_lock >= 0.95f) return;

    float plasticity = 1.0f - epigenetic_lock;
    cumulative_causal_utility = 0.99f * cumulative_causal_utility + 0.01f * global_fe_reduction;

    // Epigenetic Lock: Proven structures are protected against destructive forgetting
    if (cumulative_causal_utility > 0.015f) {
        epigenetic_lock = std::min(1.0f, epigenetic_lock + 0.005f);
    }

    // Dynamic equation coordinate evolution (KEP Principle 4)
    if (global_fe_reduction > 0.0f) {
        alpha_bilin = std::clamp(alpha_bilin + 0.005f * plasticity, 0.0f, 2.5f);
    } else {
        alpha_lin = std::clamp(alpha_lin + 0.005f * plasticity, 0.1f, 2.5f);
        alpha_bilin = std::clamp(alpha_bilin - 0.005f * plasticity, 0.0f, 2.5f);
    }

    for (size_t o = 0; o < dim && o < error.size(); ++o) {
        float g_o = error[o] * (1.0f - last_out[o] * last_out[o]);
        bias[o] += plasticity_rate * plasticity * g_o;

        float* w_lin_row = W_lin.data() + o * dim;
        for (size_t i = 0; i < dim && i < last_x.size(); ++i) {
            w_lin_row[i] += plasticity_rate * plasticity * (g_o * last_x[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. KaryonSovereignCore Implementation
// ---------------------------------------------------------------------------
KaryonSovereignCore::KaryonSovereignCore(const CoreConfig& config)
    : config_(config) {
    reset();
}

void KaryonSovereignCore::reset() {
    primitives_.clear();
    for (size_t i = 0; i < config_.num_primitives; ++i) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(i, config_.state_dim, rng_));
    }

    action_delta_table_.assign(config_.num_actions * config_.state_dim, 0.0f);
    rolling_free_energy_ = 0.10f;
    noradrenaline_arousal_ = 0.05f;
    current_horizon_k_ = config_.min_horizon_k;
    total_goals_achieved_ = 0;
}

size_t KaryonSovereignCore::compute_endogenous_horizon() {
    // Somatic interoception: NA_t scales with Free Energy surprise
    noradrenaline_arousal_ = 0.90f * noradrenaline_arousal_ + 0.10f * std::clamp(5.0f * rolling_free_energy_, 0.0f, 1.0f);

    float float_k = static_cast<float>(config_.min_horizon_k)
                  + (config_.max_horizon_k - config_.min_horizon_k) * (0.7f * noradrenaline_arousal_ + 0.3f * rolling_free_energy_);

    size_t k = std::clamp(static_cast<size_t>(std::floor(float_k)),
                          config_.min_horizon_k,
                          config_.max_horizon_k);

    current_horizon_k_ = k;
    return k;
}

float KaryonSovereignCore::evaluate_trajectory(size_t action_id,
                                              const std::vector<float>& s0,
                                              const std::vector<float>& target_goal,
                                              size_t horizon_k) {
    std::vector<float> sim = s0;
    const float* act_delta = action_delta_table_.data() + action_id * config_.state_dim;

    float total_future_cost = 0.0f;

    for (size_t tau = 1; tau <= horizon_k; ++tau) {
        for (size_t d = 0; d < config_.state_dim; ++d) {
            sim[d] += act_delta[d];
        }

        for (auto& p : primitives_) {
            sim = p->forward(sim);
        }

        float dist_sq = 0.0f;
        for (size_t d = 0; d < config_.state_dim; ++d) {
            float diff = sim[d] - target_goal[d];
            dist_sq += diff * diff;
        }

        float discount = std::pow(config_.discount_gamma, static_cast<float>(tau - 1));
        total_future_cost += discount * dist_sq;
    }

    return total_future_cost;
}

size_t KaryonSovereignCore::decide_action(const std::vector<float>& current_state,
                                         const std::vector<float>& target_goal) {
    size_t horizon_k = compute_endogenous_horizon();

    size_t best_a = 0;
    float min_cost = 1e9f;

    for (size_t a = 0; a < config_.num_actions; ++a) {
        float cost = evaluate_trajectory(a, current_state, target_goal, horizon_k);
        if (cost < min_cost) {
            min_cost = cost;
            best_a = a;
        }
    }

    // Small exploratory impulse (continuous stochastic fluctuation)
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    if (u(rng_) < 0.02f) {
        std::uniform_int_distribution<size_t> ad(0, config_.num_actions - 1);
        return ad(rng_);
    }

    return best_a;
}

CoreTelemetry KaryonSovereignCore::step_environment(const std::vector<float>& current_state,
                                                   const std::vector<float>& target_goal,
                                                   const std::vector<float>& next_state_observed,
                                                   size_t action_taken) {
    CoreTelemetry telem{};
    telem.action_executed = action_taken;
    telem.mental_horizon_k = current_horizon_k_;
    telem.noradrenaline_arousal = noradrenaline_arousal_;

    // 1. Forward prediction of the physical world step
    const float* act_delta = action_delta_table_.data() + action_taken * config_.state_dim;
    std::vector<float> inp(config_.state_dim);
    for (size_t d = 0; d < config_.state_dim; ++d) {
        inp[d] = current_state[d] + act_delta[d];
    }

    std::vector<float> pred = inp;
    for (auto& p : primitives_) {
        pred = p->forward(pred);
    }

    // 2. Compute Prediction Error Free Energy \mathcal{F}_t
    float fe_err = 0.0f;
    std::vector<float> error(config_.state_dim);
    for (size_t d = 0; d < config_.state_dim; ++d) {
        float err = next_state_observed[d] - pred[d];
        error[d] = err;
        fe_err += err * err;
    }
    fe_err /= config_.state_dim;

    telem.free_energy = fe_err;

    float fe_delta = rolling_free_energy_ - fe_err;
    rolling_free_energy_ = 0.95f * rolling_free_energy_ + 0.05f * fe_err;
    telem.rolling_free_energy = rolling_free_energy_;

    // 3. Online Causal Credit Assignment & Plasticity
    for (auto& p : primitives_) {
        p->causal_update(error, fe_delta, config_.plasticity_rate);
    }

    // 4. Update physical action efference mapping
    for (size_t d = 0; d < config_.state_dim; ++d) {
        action_delta_table_[action_taken * config_.state_dim + d] +=
            config_.plasticity_rate * 5.0f * (next_state_observed[d] - current_state[d] - action_delta_table_[action_taken * config_.state_dim + d]);
    }

    // 5. Goal Distance Telemetry
    float dist_sq = 0.0f;
    for (size_t d = 0; d < config_.state_dim; ++d) {
        float diff = next_state_observed[d] - target_goal[d];
        dist_sq += diff * diff;
    }
    telem.goal_distance = std::sqrt(dist_sq);

    if (telem.goal_distance < 0.70f) {
        telem.goal_reached = true;
        total_goals_achieved_++;
    }
    telem.goals_achieved = total_goals_achieved_;

    if (!primitives_.empty()) {
        telem.primary_primitive_lock = primitives_[0]->epigenetic_lock;
        telem.primary_causal_utility = primitives_[0]->cumulative_causal_utility;
    }

    return telem;
}

} // namespace karyon
