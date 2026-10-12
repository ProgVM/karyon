// karyon_native_core.cpp — Pure C++20 Sovereign Unified Substrate Core Engine v7.0
// Standard: KEP v20.0 Sovereign Master (Principle 2 to 27 Compliant)
// Zero-Allocation, Continuous Efference-Afference Manifold, and Hamiltonian Action

#include "karyon_native_core.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace karyon {

// ---------------------------------------------------------------------------
// 1. ContinuousSuperPrimitive Implementation v7.0
// ---------------------------------------------------------------------------
ContinuousSuperPrimitive::ContinuousSuperPrimitive(size_t p_id, size_t d, std::mt19937& rng)
    : id(p_id), dim(d),
      alpha_lin(0.80f), alpha_bilin(0.20f), alpha_rec(0.85f), alpha_res(0.30f),
      epigenetic_lock(0.0f), cumulative_causal_utility(0.0f) {

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
    // 1. Precompute Hadamard quad product: scratch_quad[i] = in[i] * state[i]
    // Vectorized cache access for bilinear GEMV reduction
    for (size_t i = 0; i < dim; ++i) {
        scratch_quad[i] = in[i] * state[i];
    }

    // 2. Continuous operator projection (Linear + Bilinear + Recurrent state)
    for (size_t d = 0; d < dim; ++d) {
        const float* w_lin_row = W_lin.data() + d * dim;
        const float* w_bi_row = W_bilin.data() + d * dim;

        float lin_val = 0.0f;
        float bilin_val = 0.0f;

        #pragma omp simd reduction(+:lin_val, bilin_val)
        for (size_t i = 0; i < dim; ++i) {
            lin_val += w_lin_row[i] * in[i];
            bilin_val += w_bi_row[i] * scratch_quad[i];
        }

        // Stratonovich continuous integration step
        float net_flow = alpha_lin * lin_val +
                         alpha_bilin * bilin_val +
                         alpha_rec * state[d] +
                         alpha_res * in[d];

        float h_new = std::tanh(net_flow);
        state[d] = 0.70f * state[d] + 0.30f * h_new;
        out[d] = h_new;
    }
}

void ContinuousSuperPrimitive::update_plasticity(std::span<const float> in,
                                                 std::span<const float> target,
                                                 float plasticity_rate) {
    float inv_plasticity = plasticity_rate * (1.0f - std::tanh(epigenetic_lock));
    if (inv_plasticity <= 1e-6f) return;

    for (size_t d = 0; d < dim; ++d) {
        float err = target[d] - state[d];
        float* w_lin_row = W_lin.data() + d * dim;
        float* w_bi_row = W_bilin.data() + d * dim;

        for (size_t i = 0; i < dim; ++i) {
            w_lin_row[i] += inv_plasticity * err * in[i];
            w_bi_row[i] += inv_plasticity * err * (in[i] * state[i]);
        }
    }
}

void ContinuousSuperPrimitive::adapt_coordinates(float dF_dalpha_lin,
                                                 float dF_dalpha_bilin,
                                                 float plasticity_rate) {
    // Pure continuous gradient descent on equation coordinates: d alpha / dt = -grad_alpha F
    // Completely eradicating discrete heuristic if-else checks (KEP Principle 4)
    float eta = plasticity_rate * 0.05f;
    alpha_lin = std::clamp(alpha_lin - eta * dF_dalpha_lin, 0.1f, 2.5f);
    alpha_bilin = std::clamp(alpha_bilin - eta * dF_dalpha_bilin, 0.0f, 2.5f);
}

void ContinuousSuperPrimitive::update_epigenetic_lock(float baseline_fe_reduction) {
    cumulative_causal_utility += baseline_fe_reduction;
    epigenetic_lock = std::clamp(epigenetic_lock + baseline_fe_reduction * 0.10f, 0.0f, 3.0f);
}

// ---------------------------------------------------------------------------
// 2. KaryonSovereignCore Implementation v7.0
// ---------------------------------------------------------------------------
KaryonSovereignCore::KaryonSovereignCore(const CoreConfig& config)
    : config_(config), rng_(42),
      noradrenaline_arousal_(0.10f),
      dopamine_vigor_(config.da_baseline),
      somatic_energy_(1.0f),
      current_horizon_k_(config.min_horizon_k),
      scratchpad_(config.state_dim, config.action_dim) {

    for (size_t p = 0; p < config_.num_primitives; ++p) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(p, config_.state_dim, rng_));
    }

    // Initialize continuous motor efference projection matrix W_motor in R^{D x M}
    W_motor_.resize(config_.state_dim * config_.action_dim, 0.0f);
    std::normal_distribution<float> dist(0.0f, 0.15f);
    for (size_t i = 0; i < W_motor_.size(); ++i) {
        W_motor_[i] = dist(rng_);
    }
}

float KaryonSovereignCore::evaluate_continuous_trajectory(std::span<const float> u,
                                                          std::span<const float> s0,
                                                          std::span<const float> target,
                                                          size_t horizon_k) {
    // Zero-allocation trajectory simulation using CoreScratchpad
    auto& cur = scratchpad_.buf_a;
    auto& nxt = scratchpad_.buf_b;
    auto& q_scratch = scratchpad_.buf_quad;

    // 1. Motor efference projection: cur = s0 + W_motor * u
    for (size_t d = 0; d < config_.state_dim; ++d) {
        float motor_flow = 0.0f;
        const float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            motor_flow += w_row[m] * u[m];
        }
        cur[d] = s0[d] + motor_flow;
    }

    // 2. Latent rollout across super-primitives over depth K
    for (size_t tau = 0; tau < horizon_k; ++tau) {
        for (auto& p : primitives_) {
            p->forward(cur, nxt, q_scratch);
            std::copy(nxt.begin(), nxt.end(), cur.begin());
        }
    }

    // 3. Least-Action Hamiltonian metric: Distance to target goal
    float fe_dist = 0.0f;
    for (size_t d = 0; d < config_.state_dim; ++d) {
        float diff = cur[d] - target[d];
        fe_dist += diff * diff;
    }
    return fe_dist / static_cast<float>(config_.state_dim);
}

void KaryonSovereignCore::decide_continuous_action(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<float> action_out) {
    size_t horizon_k = current_horizon_k_;

    // Initialize action u around zero
    for (size_t m = 0; m < config_.action_dim; ++m) {
        action_out[m] = 0.0f;
    }

    // Continuous Hamiltonian Least-Action Motor Steering:
    // du/dt = -grad_u F (numeric gradient descent along internal Free Energy manifold)
    const float eps = 0.01f;
    const float lr_u = 0.20f * (1.0f + dopamine_vigor_);

    for (size_t step = 0; step < config_.hamiltonian_steps; ++step) {
        float base_cost = evaluate_continuous_trajectory(action_out, current_state, target_goal, horizon_k);

        for (size_t m = 0; m < config_.action_dim; ++m) {
            action_out[m] += eps;
            float pert_cost = evaluate_continuous_trajectory(action_out, current_state, target_goal, horizon_k);
            action_out[m] -= eps;

            float grad_m = (pert_cost - base_cost) / eps;
            scratchpad_.buf_grad[m] = grad_m;
        }

        for (size_t m = 0; m < config_.action_dim; ++m) {
            action_out[m] = std::clamp(action_out[m] - lr_u * scratchpad_.buf_grad[m], -1.0f, 1.0f);
        }
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

    // 1. Forward model prediction: inp = current_state + W_motor * action_taken
    auto& cur = scratchpad_.buf_a;
    auto& nxt = scratchpad_.buf_b;
    auto& q_scratch = scratchpad_.buf_quad;

    for (size_t d = 0; d < config_.state_dim; ++d) {
        float motor_flow = 0.0f;
        const float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            motor_flow += w_row[m] * action_taken[m];
        }
        cur[d] = current_state[d] + motor_flow;
    }

    for (auto& p : primitives_) {
        p->forward(cur, nxt, q_scratch);
        std::copy(nxt.begin(), nxt.end(), cur.begin());
    }

    // 2. Sensory prediction error & Free Energy
    float fe_error = 0.0f;
    for (size_t d = 0; d < config_.state_dim; ++d) {
        float diff = cur[d] - next_state_observed[d];
        fe_error += diff * diff;
    }
    fe_error /= static_cast<float>(config_.state_dim);

    telem.free_energy = fe_error;
    telem.sensory_prediction_error = std::sqrt(fe_error);

    // 3. Autonomous somatic arousal
    float delta_fe = fe_error - 0.02f;
    if (delta_fe > 0.0f) {
        noradrenaline_arousal_ = std::min(1.0f, noradrenaline_arousal_ + delta_fe * 1.5f);
    } else {
        noradrenaline_arousal_ = std::max(0.01f, noradrenaline_arousal_ - config_.noradrenaline_decay);
    }
    adapt_endogenous_horizon(fe_error);

    // 4. Update W_motor projection via continuous causal credit
    float plasticity = config_.plasticity_rate * (1.0f + noradrenaline_arousal_);
    for (size_t d = 0; d < config_.state_dim; ++d) {
        float err_d = next_state_observed[d] - cur[d];
        float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            w_row[m] += plasticity * err_d * action_taken[m];
        }
    }

    // 5. Plasticity update on super-primitives and dynamic coordinate evolution
    float dF_dlin = 0.0f;
    float dF_dbilin = 0.0f;
    for (size_t d = 0; d < config_.state_dim; ++d) {
        dF_dlin += (cur[d] - next_state_observed[d]);
        dF_dbilin += (cur[d] - next_state_observed[d]) * cur[d];
    }
    dF_dlin /= static_cast<float>(config_.state_dim);
    dF_dbilin /= static_cast<float>(config_.state_dim);

    for (auto& p : primitives_) {
        p->update_plasticity(cur, next_state_observed, plasticity);
        p->adapt_coordinates(dF_dlin, dF_dbilin, plasticity);
        p->update_epigenetic_lock(0.05f - fe_error);
    }

    telem.lock_ratio = get_average_epigenetic_lock();
    return telem;
}

// Discrete compatibility helpers
size_t KaryonSovereignCore::decide_action(const std::vector<float>& current_state,
                                         const std::vector<float>& target_goal) {
    std::vector<float> continuous_act(config_.action_dim, 0.0f);
    decide_continuous_action(current_state, target_goal, continuous_act);

    // Argmax mapping for legacy discrete environments
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
