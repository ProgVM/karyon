// karyon_native_core.cpp
// Karyon Native Core v7.6 - Zero-Allocation Autonomous C++20 Cybernetic Substrate
// Standard: KEP v20.0 Sovereign Master (Principles 2, 3, 4, 5, 6, 7, 27, 28)

#include "karyon_native_core.h"
#include <cstring>
#include <iostream>

namespace karyon {

// High-performance AVX2 dot product
static inline float avx2_dot_product(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

ContinuousSuperPrimitive::ContinuousSuperPrimitive(size_t pid, size_t d, std::mt19937& rng)
    : id(pid), dim(d), alpha_lin(0.6f), alpha_bilin(0.2f), alpha_rec(0.2f),
      tau_time(1.0f), epigenetic_lock(0.0f), birth_gain(1.0f) {
    W_lin.resize(dim * dim);
    W_bilin.resize(dim * dim);
    W_rec.resize(dim * dim);
    state_trace.resize(dim, 0.0f);

    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
    for (auto& w : W_lin) w = dist(rng);
    for (auto& w : W_bilin) w = dist(rng) * 0.1f;
    for (auto& w : W_rec) w = dist(rng) * 0.5f;
}

void ContinuousSuperPrimitive::forward(std::span<const float> x_in, std::span<float> h_out, std::span<float> quad_buf) {
    for (size_t i = 0; i < dim; ++i) {
        float x_val = x_in[i];
        quad_buf[i] = std::sin(x_val * 3.14159265f);
    }

    for (size_t i = 0; i < dim; ++i) {
        const float* row_lin = W_lin.data() + i * dim;
        const float* row_bilin = W_bilin.data() + i * dim;
        const float* row_rec = W_rec.data() + i * dim;

        float lin_term = avx2_dot_product(row_lin, x_in.data(), dim);
        float bilin_term = avx2_dot_product(row_bilin, quad_buf.data(), dim);
        float rec_term = avx2_dot_product(row_rec, state_trace.data(), dim);

        float dh = (alpha_lin * lin_term + alpha_bilin * bilin_term + alpha_rec * rec_term) / tau_time;
        float h_new = std::tanh(state_trace[i] + dh);

        h_out[i] = x_in[i] + birth_gain * (h_new - x_in[i]);
        state_trace[i] = h_new;
    }
}

void ContinuousSuperPrimitive::update_plasticity_avx2(std::span<const float> pre,
                                                      std::span<const float> post,
                                                      float lr,
                                                      std::span<float> quad_buf) {
    float effective_lr = lr * (1.0f - epigenetic_lock);
    if (effective_lr < 1e-7f) return;

    for (size_t i = 0; i < dim; ++i) {
        quad_buf[i] = std::sin(pre[i] * 3.14159265f);
    }

    for (size_t i = 0; i < dim; ++i) {
        float p_i = post[i];
        float* row_lin = W_lin.data() + i * dim;
        float* row_bilin = W_bilin.data() + i * dim;
        float* row_rec = W_rec.data() + i * dim;

        for (size_t j = 0; j < dim; ++j) {
            row_lin[j] += effective_lr * (p_i * pre[j] - 0.01f * row_lin[j]);
            row_bilin[j] += effective_lr * 0.5f * (p_i * quad_buf[j] - 0.01f * row_bilin[j]);
            row_rec[j] += effective_lr * (p_i * state_trace[j] - 0.01f * row_rec[j]);
        }
    }
}

void ContinuousSuperPrimitive::mature_birth_gain(float rate) {
    if (birth_gain < 1.0f) {
        birth_gain = std::min(1.0f, birth_gain + rate);
    }
}

void ContinuousSuperPrimitive::update_epigenetic_lock(float utility) {
    float delta = utility * 0.01f;
    epigenetic_lock = std::clamp(epigenetic_lock + delta, 0.0f, 0.95f);
}

void ContinuousSuperPrimitive::adapt_coordinates(float d_lin, float d_bilin, float d_rec) {
    alpha_lin = std::clamp(alpha_lin + d_lin, 0.05f, 0.90f);
    alpha_bilin = std::clamp(alpha_bilin + d_bilin, 0.05f, 0.90f);
    alpha_rec = std::clamp(alpha_rec + d_rec, 0.05f, 0.90f);
    float sum = alpha_lin + alpha_bilin + alpha_rec;
    alpha_lin /= sum;
    alpha_bilin /= sum;
    alpha_rec /= sum;
}

KaryonSovereignCore::KaryonSovereignCore(const CoreConfig& config)
    : config_(config), rng_(42),
      noradrenaline_arousal_(0.01f),
      dopamine_vigor_(config.da_baseline),
      somatic_energy_(1.0f),
      stagnation_potential_(0.0f),
      current_horizon_k_(config.min_horizon_k),
      ticks_since_last_sprout_(0),
      moving_fe_ema_(0.05f) {

    for (size_t i = 0; i < config_.initial_primitives; ++i) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(i, config_.latent_dim, rng_));
    }

    W_sensory_down_.resize(config_.latent_dim * config_.sensory_dim);
    W_sensory_up_.resize(config_.sensory_dim * config_.latent_dim);
    W_motor_.resize(config_.sensory_dim * config_.action_dim, 0.0f);
    action_delta_table_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    W_action_valence_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    continuous_sensory_ema_.resize(config_.sensory_dim, 0.0f);

    std::normal_distribution<float> dist_down(0.0f, 1.0f / std::sqrt(static_cast<float>(config_.sensory_dim)));
    for (auto& w : W_sensory_down_) w = dist_down(rng_);

    std::normal_distribution<float> dist_up(0.0f, 1.0f / std::sqrt(static_cast<float>(config_.latent_dim)));
    for (auto& w : W_sensory_up_) w = dist_up(rng_);

    buf_sensory_in_.resize(config_.sensory_dim, 0.0f);
    buf_sensory_out_.resize(config_.sensory_dim, 0.0f);
    buf_latent_a_.resize(config_.latent_dim, 0.0f);
    buf_latent_b_.resize(config_.latent_dim, 0.0f);
    buf_quad_.resize(config_.latent_dim, 0.0f);
    buf_adjoint_sensory_.resize(config_.sensory_dim, 0.0f);
}

bool KaryonSovereignCore::trigger_morphogenetic_sprout() {
    if (primitives_.size() >= config_.max_primitives) return false;
    size_t parent_idx = 0;
    float max_lock = -1.0f;
    for (size_t i = 0; i < primitives_.size(); ++i) {
        if (primitives_[i]->epigenetic_lock > max_lock) {
            max_lock = primitives_[i]->epigenetic_lock;
            parent_idx = i;
        }
    }
    const auto& parent = primitives_[parent_idx];
    size_t new_id = primitives_.size();
    auto child = std::make_unique<ContinuousSuperPrimitive>(new_id, config_.latent_dim, rng_);
    child->alpha_lin = parent->alpha_lin;
    child->alpha_bilin = parent->alpha_bilin;
    child->alpha_rec = parent->alpha_rec;
    child->tau_time = parent->tau_time * 1.2f;
    child->epigenetic_lock = 0.0f;
    child->birth_gain = 0.0f; // Smooth Net2Net identity birth
    child->W_lin = parent->W_lin;
    child->W_bilin = parent->W_bilin;
    child->W_rec = parent->W_rec;
    primitives_.push_back(std::move(child));
    ticks_since_last_sprout_ = 0;
    return true;
}

bool KaryonSovereignCore::prune_least_causal_primitive() {
    if (primitives_.size() <= config_.initial_primitives) return false;
    size_t victim_idx = 0;
    float min_lock = 1e9f;
    for (size_t i = 0; i < primitives_.size(); ++i) {
        if (primitives_[i]->epigenetic_lock < min_lock && primitives_[i]->birth_gain >= 0.8f) {
            min_lock = primitives_[i]->epigenetic_lock;
            victim_idx = i;
        }
    }
    primitives_.erase(primitives_.begin() + victim_idx);
    return true;
}

void KaryonSovereignCore::evaluate_autopoietic_morphogenesis(float fe_error) {
    ticks_since_last_sprout_++;
    moving_fe_ema_ = 0.95f * moving_fe_ema_ + 0.05f * fe_error;
    if (ticks_since_last_sprout_ > config_.sprouting_cooldown_ticks) {
        if (fe_error > config_.sprouting_fe_threshold && moving_fe_ema_ > config_.sprouting_fe_threshold) {
            trigger_morphogenetic_sprout();
        } else if (fe_error < 0.02f && moving_fe_ema_ < 0.02f) {
            prune_least_causal_primitive();
        }
    }
}

void KaryonSovereignCore::decide_continuous_action(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<float> action_out) {
    // Analytical Adjoint Motor Field:
    // grad_u = W_motor^T * (target_goal - current_state)
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        buf_adjoint_sensory_[d] = target_goal[d] - current_state[d];
    }

    for (size_t a = 0; a < config_.action_dim; ++a) {
        float motor_drive = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float w = W_motor_[d * config_.action_dim + a];
            motor_drive += w * buf_adjoint_sensory_[d];
        }

        // Subtract predicted pain/threat for this continuous action in current sensory context
        const float* v_row = W_action_valence_.data() + a * config_.sensory_dim;
        float expected_pain = avx2_dot_product(v_row, current_state.data(), config_.sensory_dim);
        motor_drive -= config_.aversive_weight * std::max(0.0f, -expected_pain);

        // Dopaminergic motor vigor scaling
        action_out[a] = std::tanh(motor_drive * (1.0f + 1.5f * dopamine_vigor_));
    }
}

float KaryonSovereignCore::evaluate_trajectory(size_t action_id,
                                               const std::vector<float>& s0,
                                               const std::vector<float>& target_goal,
                                               size_t horizon_k) {
    std::vector<float> sim = s0;
    float total_cost = 0.0f;

    // Direct Aversive Reflex Check at Step 0:
    // How much pain has this action produced from the immediate sensory state s0?
    const float* v_row = W_action_valence_.data() + action_id * config_.sensory_dim;
    float expected_valence = avx2_dot_product(v_row, s0.data(), config_.sensory_dim);
    // If expected valence is strongly negative (pain/collision), inject massive penalty!
    float immediate_pain_cost = config_.aversive_weight * 10.0f * std::max(0.0f, -expected_valence);

    for (size_t tau = 1; tau <= horizon_k; ++tau) {
        // Delta prediction rollout
        const float* delta_row = action_delta_table_.data() + action_id * config_.sensory_dim;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            sim[d] += delta_row[d];
        }

        // Down-project into latent manifold
        for (size_t l = 0; l < config_.latent_dim; ++l) {
            const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
            buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, sim.data(), config_.sensory_dim));
        }

        // Latent Primitives dynamic relaxation
        for (auto& p : primitives_) {
            p->forward(buf_latent_a_, buf_latent_b_, buf_quad_);
            std::copy(buf_latent_b_.begin(), buf_latent_b_.end(), buf_latent_a_.begin());
        }

        // Up-project
        for (size_t s = 0; s < config_.sensory_dim; ++s) {
            const float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
            sim[s] = avx2_dot_product(w_row, buf_latent_a_.data(), config_.latent_dim);
        }

        // 1. Pragmatic Cost: Distance to target goal
        float pragmatic_dist_sq = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float diff = sim[d] - target_goal[d];
            pragmatic_dist_sq += diff * diff;
        }

        // 2. Epistemic Novelty Drive (Anti-Stagnation):
        float novelty_sq = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float diff = sim[d] - continuous_sensory_ema_[d];
            novelty_sq += diff * diff;
        }

        // 3. Anticipated pain along the simulated trajectory
        float sim_val = avx2_dot_product(v_row, sim.data(), config_.sensory_dim);
        float sim_pain = config_.aversive_weight * std::max(0.0f, -sim_val);

        // 4. Expected Free Energy = Pragmatic Cost - Epistemic Drive + Anticipated Pain
        float step_cost = pragmatic_dist_sq - (stagnation_potential_ * 2.5f) * std::sqrt(novelty_sq + 1e-6f) + sim_pain;

        float discount = std::pow(config_.discount_gamma, static_cast<float>(tau - 1));
        total_cost += discount * step_cost;
    }

    return total_cost + immediate_pain_cost;
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

    // Dynamic Epistemic Exploration Probability
    float exploration_prob = 0.01f + 0.35f * std::min(1.0f, stagnation_potential_);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    if (u(rng_) < exploration_prob) {
        std::uniform_int_distribution<size_t> ad(0, config_.action_dim - 1);
        return ad(rng_);
    }

    return best_a;
}

CoreTelemetry KaryonSovereignCore::step_environment(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<const float> next_state_observed,
                                                   std::span<const float> action_taken,
                                                   float external_valence) {
    CoreTelemetry telem;
    size_t count_before = primitives_.size();

    // 0. Update continuous sensory EMA and non-static stagnation potential
    float sensory_delta_sq = 0.0f;
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float diff = next_state_observed[d] - current_state[d];
        sensory_delta_sq += diff * diff;
        continuous_sensory_ema_[d] = 0.90f * continuous_sensory_ema_[d] + 0.10f * next_state_observed[d];
    }
    float sensory_velocity = std::sqrt(sensory_delta_sq);

    // Continuous differential stagnation law (no static N)
    if (sensory_velocity < 0.02f) {
        stagnation_potential_ = std::min(2.0f, stagnation_potential_ + 0.08f);
    } else {
        stagnation_potential_ = std::max(0.0f, stagnation_potential_ - config_.stagnation_decay);
    }

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
    telem.stagnation_potential = stagnation_potential_;

    // 6. Direct Associative Valence Learning:
    // Update W_action_valence_ table so actions learn to predict external_valence from sensory context
    for (size_t a = 0; a < config_.action_dim; ++a) {
        float a_weight = action_taken[a];
        if (std::abs(a_weight) > 1e-4f) {
            float* v_row = W_action_valence_.data() + a * config_.sensory_dim;
            float pred_v = avx2_dot_product(v_row, current_state.data(), config_.sensory_dim);
            float v_error = external_valence - pred_v;
            for (size_t d = 0; d < config_.sensory_dim; ++d) {
                v_row[d] += config_.plasticity_rate * 5.0f * a_weight * v_error * current_state[d];
            }
        }
    }

    // 7. Endogenous Neuromodulator Coupling:
    if (external_valence > 0.0f) {
        dopamine_vigor_ = std::min(1.0f, dopamine_vigor_ + external_valence * 0.4f);
        noradrenaline_arousal_ = std::max(0.01f, noradrenaline_arousal_ - external_valence * 0.2f);
    } else if (external_valence < 0.0f) {
        noradrenaline_arousal_ = std::min(1.0f, noradrenaline_arousal_ + std::abs(external_valence) * 0.5f);
        dopamine_vigor_ = std::max(0.05f, dopamine_vigor_ - std::abs(external_valence) * 0.2f);
    }

    // Somatic arousal & horizon modulation
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

    // 8. Plasticity updates modulated by Neuromodulators & Valence
    float plasticity = config_.plasticity_rate * (1.0f + noradrenaline_arousal_) * (1.0f + dopamine_vigor_);

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
        float lock_utility = (external_valence * 0.10f) + (0.05f - fe_error);
        p->update_epigenetic_lock(lock_utility);
    }

    // Direct Motor Synaptic Rewiring:
    // When valence is negative (pain), directly DEPRESS the motor connections that led to this action!
    // When valence is positive (reward), FACILITATE the motor connections!
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float motor_gradient = (target_goal[d] - current_state[d]);
        float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            float delta = 0.0f;
            if (external_valence < 0.0f) {
                // Strong inhibitory depression on this action
                delta = -plasticity * 3.0f * std::abs(external_valence) * action_taken[m] * current_state[d];
            } else {
                delta = plasticity * (external_valence * 2.0f + 0.1f) * motor_gradient * action_taken[m];
            }
            w_row[m] += delta;
        }
    }

    telem.noradrenaline = noradrenaline_arousal_;
    telem.dopamine = dopamine_vigor_;
    telem.somatic_energy = somatic_energy_;
    telem.horizon_k = current_horizon_k_;
    telem.lock_ratio = get_average_epigenetic_lock();
    return telem;
}

CoreTelemetry KaryonSovereignCore::step_environment(const std::vector<float>& current_state,
                                                   const std::vector<float>& target_goal,
                                                   const std::vector<float>& next_state_observed,
                                                   size_t discrete_action,
                                                   float external_valence) {
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

    auto telem = step_environment(current_state, target_goal, next_state_observed, continuous_act, external_valence);
    telem.action_discrete_chosen = discrete_action;
    return telem;
}

void KaryonSovereignCore::adapt_endogenous_horizon(float fe_error) {
    if (noradrenaline_arousal_ > 0.40f || fe_error > 0.08f || stagnation_potential_ > 0.50f) {
        current_horizon_k_ = std::min(config_.max_horizon_k, current_horizon_k_ + 1);
    } else if (noradrenaline_arousal_ < 0.15f && fe_error < 0.02f && stagnation_potential_ < 0.10f) {
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

// C-ABI Export Bindings
extern "C" {

void* karyon_core_create(const CoreConfig* cfg) {
    CoreConfig config;
    if (cfg) config = *cfg;
    return new KaryonSovereignCore(config);
}

void karyon_core_destroy(void* handle) {
    delete static_cast<KaryonSovereignCore*>(handle);
}

void karyon_core_decide_action(void* handle, const float* current_state, const float* target_goal, float* action_out) {
    auto* core = static_cast<KaryonSovereignCore*>(handle);
    core->decide_continuous_action(
        std::span<const float>(current_state, 256),
        std::span<const float>(target_goal, 256),
        std::span<float>(action_out, 4)
    );
}

size_t karyon_core_decide_discrete(void* handle, const float* current_state, const float* target_goal) {
    auto* core = static_cast<KaryonSovereignCore*>(handle);
    std::vector<float> s_curr(current_state, current_state + 256);
    std::vector<float> s_goal(target_goal, target_goal + 256);
    return core->decide_action(s_curr, s_goal);
}

void karyon_core_step(void* handle,
                      const float* current_state,
                      const float* target_goal,
                      const float* next_state_observed,
                      const float* action_taken,
                      float external_valence,
                      CoreTelemetry* telem_out) {
    auto* core = static_cast<KaryonSovereignCore*>(handle);
    auto telem = core->step_environment(
        std::span<const float>(current_state, 256),
        std::span<const float>(target_goal, 256),
        std::span<const float>(next_state_observed, 256),
        std::span<const float>(action_taken, 4),
        external_valence
    );
    if (telem_out) *telem_out = telem;
}

} // extern "C"

} // namespace karyon
