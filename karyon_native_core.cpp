// karyon_native_core.cpp
// Karyon Native Core v8.0 - Autonomous C++20 Cybernetic Substrate with Sovereign Endogenous Hormones
// Standard: KEP v20.0 Sovereign Master (Principles 2, 3, 4, 5, 6, 7, 27, 28)

#include "karyon_native_core.h"
#include <cstring>
#include <iostream>

namespace karyon {

// Helper: Vectorized AVX2-FMA Dot Product
inline float avx2_dot_product(const float* a, const float* b, size_t size) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    float sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                buffer[4] + buffer[5] + buffer[6] + buffer[7];
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// =========================================================================
// ContinuousSuperPrimitive Implementation
// =========================================================================
ContinuousSuperPrimitive::ContinuousSuperPrimitive(size_t pid, size_t d, std::mt19937& rng)
    : id(pid), dim(d),
      alpha_lin(0.34f), alpha_bilin(0.33f), alpha_rec(0.33f),
      tau_time(0.1f), epigenetic_lock(0.0f), birth_gain(1.0f) {
    W_lin.resize(d * d);
    W_bilin.resize(d * d);
    W_rec.resize(d * d);
    state_trace.resize(d, 0.0f);

    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(d)));
    for (auto& w : W_lin) w = dist(rng);
    for (auto& w : W_bilin) w = dist(rng);
    for (auto& w : W_rec) w = dist(rng);
}

void ContinuousSuperPrimitive::forward(std::span<const float> x_in,
                                      std::span<float> h_out,
                                      std::span<float> quad_buf) {
    for (size_t i = 0; i < dim; ++i) {
        const float* row_lin = W_lin.data() + i * dim;
        const float* row_bilin = W_bilin.data() + i * dim;
        const float* row_rec = W_rec.data() + i * dim;

        float y_lin = avx2_dot_product(row_lin, x_in.data(), dim);
        float y_bilin = avx2_dot_product(row_bilin, x_in.data(), dim);
        float y_rec = avx2_dot_product(row_rec, state_trace.data(), dim);

        float fused = alpha_lin * y_lin +
                      alpha_bilin * (y_bilin * y_lin) +
                      alpha_rec * std::tanh(y_rec);

        float dh = (fused - state_trace[i]) * tau_time;
        state_trace[i] += dh;
        h_out[i] = state_trace[i] * birth_gain;
    }
}

void ContinuousSuperPrimitive::update_plasticity_avx2(std::span<const float> pre,
                                                      std::span<const float> post,
                                                      float lr,
                                                      std::span<float> quad_buf) {
    float effective_lr = lr * (1.0f - epigenetic_lock) * birth_gain;
    if (effective_lr < 1e-6f) return;

    for (size_t i = 0; i < dim; ++i) {
        float post_i = post[i];
        float* row_lin = W_lin.data() + i * dim;
        float* row_bilin = W_bilin.data() + i * dim;
        float* row_rec = W_rec.data() + i * dim;
        float trace_i = state_trace[i];

        size_t j = 0;
        __m256 v_lr = _mm256_set1_ps(effective_lr);
        __m256 v_post = _mm256_set1_ps(post_i);
        __m256 v_trace_i = _mm256_set1_ps(trace_i);

        for (; j + 8 <= dim; j += 8) {
            __m256 v_pre = _mm256_loadu_ps(pre.data() + j);
            __m256 v_wlin = _mm256_loadu_ps(row_lin + j);
            __m256 v_wrec = _mm256_loadu_ps(row_rec + j);

            __m256 d_wlin = _mm256_mul_ps(v_lr, _mm256_mul_ps(v_post, v_pre));
            _mm256_storeu_ps(row_lin + j, _mm256_add_ps(v_wlin, d_wlin));

            __m256 d_wrec = _mm256_mul_ps(v_lr, _mm256_mul_ps(v_trace_i, v_pre));
            _mm256_storeu_ps(row_rec + j, _mm256_add_ps(v_wrec, d_wrec));
        }

        for (; j < dim; ++j) {
            row_lin[j] += effective_lr * post_i * pre[j];
            row_bilin[j] += effective_lr * post_i * (pre[j] * pre[j]);
            row_rec[j] += effective_lr * trace_i * state_trace[j];
        }
    }
}

void ContinuousSuperPrimitive::mature_birth_gain(float rate) {
    if (birth_gain < 1.0f) {
        birth_gain = std::min(1.0f, birth_gain + rate);
    }
}

void ContinuousSuperPrimitive::update_epigenetic_lock(float utility) {
    if (utility > 0.0f) {
        epigenetic_lock = std::min(0.99f, epigenetic_lock + 0.01f * utility);
    } else {
        epigenetic_lock = std::max(0.0f, epigenetic_lock - 0.005f * std::abs(utility));
    }
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

// =========================================================================
// SovereignEndogenousHormone Implementation
// =========================================================================
SovereignHormone::SovereignHormone(size_t hid, size_t sensory_dim, std::mt19937& rng)
    : id(hid), concentration(0.5f), basal_level(0.5f), decay_rate(0.04f),
      valence_sensitivity(0.2f), surprise_sensitivity(0.2f),
      causal_utility(1.0f), birth_gain(1.0f),
      w_plasticity(0.0f), w_horizon(0.0f), w_motor_vigor(0.0f),
      w_threat_aversive(0.0f), w_exploration(0.0f) {
    W_sensory_emission.resize(sensory_dim, 0.0f);

    std::uniform_real_distribution<float> dist_decay(0.02f, 0.08f);
    decay_rate = dist_decay(rng);

    std::normal_distribution<float> dist_w(0.0f, 0.2f);
    for (auto& w : W_sensory_emission) w = dist_w(rng);

    // Initial distinct differentiation across initial hormones
    if (hid == 0) {
        // Axis 0: Emergency / Surprise / Rapid Arousal
        surprise_sensitivity = 0.6f;
        valence_sensitivity = -0.4f;
        w_plasticity = 0.5f;
        w_horizon = 0.8f;
        w_threat_aversive = 0.7f;
        decay_rate = 0.06f;
    } else if (hid == 1) {
        // Axis 1: Goal Appetence / Motor Vigor
        surprise_sensitivity = -0.1f;
        valence_sensitivity = 0.7f;
        w_motor_vigor = 0.9f;
        w_plasticity = 0.3f;
        w_horizon = -0.2f;
        decay_rate = 0.03f;
    } else if (hid == 2) {
        // Axis 2: Environmental Threat Sensitivity & Caution
        surprise_sensitivity = 0.3f;
        valence_sensitivity = -0.8f;
        w_threat_aversive = 1.0f;
        w_motor_vigor = -0.4f;
        w_horizon = 0.5f;
        decay_rate = 0.04f;
    } else {
        // De novo sprouted hormone
        std::uniform_real_distribution<float> dist_sens(-0.5f, 0.5f);
        valence_sensitivity = dist_sens(rng);
        surprise_sensitivity = dist_sens(rng);
        w_plasticity = dist_sens(rng);
        w_horizon = dist_sens(rng);
        w_motor_vigor = dist_sens(rng);
        w_threat_aversive = dist_sens(rng);
        w_exploration = dist_sens(rng);
        birth_gain = 0.05f; // Smooth identity birth
    }
}

// =========================================================================
// KaryonSovereignCore Implementation
// =========================================================================
KaryonSovereignCore::KaryonSovereignCore(const CoreConfig& config)
    : config_(config), rng_(42),
      effective_plasticity_(config.plasticity_rate),
      effective_motor_vigor_(1.0f),
      effective_threat_weight_(config.aversive_weight),
      effective_exploration_(0.0f),
      somatic_energy_(1.0f),
      stagnation_potential_(0.0f),
      appetitive_hunger_(0.0f),
      last_goal_dist_sq_(10.0f),
      current_horizon_k_(config.min_horizon_k),
      ticks_since_last_sprout_(0),
      ticks_since_last_hormone_sprout_(0),
      moving_fe_ema_(0.05f) {

    for (size_t i = 0; i < config_.initial_primitives; ++i) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(i, config_.latent_dim, rng_));
    }

    for (size_t h = 0; h < config_.initial_hormones; ++h) {
        hormones_.emplace_back(h, config_.sensory_dim, rng_);
    }

    W_sensory_down_.resize(config_.latent_dim * config_.sensory_dim);
    W_sensory_up_.resize(config_.sensory_dim * config_.latent_dim);
    W_motor_.resize(config_.sensory_dim * config_.action_dim, 0.0f);
    action_delta_table_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    W_action_valence_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    W_peril_latent_.resize(config_.action_dim * config_.latent_dim, 0.0f);
    action_eligibility_trace_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
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

    compute_effector_modulations();
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

    size_t new_id = primitives_.size();
    auto child = std::make_unique<ContinuousSuperPrimitive>(new_id, config_.latent_dim, rng_);
    const auto& parent = primitives_[parent_idx];

    child->W_lin = parent->W_lin;
    child->W_bilin = parent->W_bilin;
    child->W_rec = parent->W_rec;
    child->alpha_lin = parent->alpha_lin;
    child->alpha_bilin = parent->alpha_bilin;
    child->alpha_rec = parent->alpha_rec;
    child->epigenetic_lock = 0.0f;
    child->birth_gain = 0.05f; // Smooth identity sprouting

    std::normal_distribution<float> mut_dist(0.0f, 0.02f);
    for (auto& w : child->W_lin) w += mut_dist(rng_);
    for (auto& w : child->W_bilin) w += mut_dist(rng_);
    for (auto& w : child->W_rec) w += mut_dist(rng_);

    primitives_.push_back(std::move(child));
    ticks_since_last_sprout_ = 0;
    return true;
}

bool KaryonSovereignCore::prune_least_causal_primitive() {
    if (primitives_.size() <= config_.initial_primitives) return false;
    size_t victim_idx = 0;
    float min_lock = 1e9f;
    for (size_t i = 0; i < primitives_.size(); ++i) {
        if (primitives_[i]->epigenetic_lock < min_lock && primitives_[i]->birth_gain >= 0.95f) {
            min_lock = primitives_[i]->epigenetic_lock;
            victim_idx = i;
        }
    }
    primitives_.erase(primitives_.begin() + victim_idx);
    return true;
}

bool KaryonSovereignCore::trigger_hormone_sprout() {
    if (hormones_.size() >= config_.max_hormones) return false;
    size_t new_id = hormones_.size();
    
    // Principle 5 (Causal Credit Assignment): Sprout along the gradient of current need
    // If agent is experiencing appetitive hunger/frustration, sprout an aggressive goal-seeking hormone
    // If agent is suffering high negative valence, sprout an evasive/peril-reduction hormone
    SovereignHormone new_h(new_id, config_.sensory_dim, rng_);
    
    // Data-driven endogenous initialization
    if (appetitive_hunger_ > 0.4f || stagnation_potential_ > 0.3f) {
        // Frustration / Stagnation: boost motor vigor, appetence and goal pursuit
        new_h.w_motor_vigor = 1.0f;
        new_h.w_threat_aversive = -0.3f; // Disinhibit timidity
        new_h.w_exploration = 0.4f;
        new_h.valence_sensitivity = 0.8f;
        new_h.surprise_sensitivity = 0.2f;
    } else {
        // Balanced adaptive axis
        new_h.w_motor_vigor = 0.4f;
        new_h.w_plasticity = 0.3f;
        new_h.w_threat_aversive = 0.3f;
        new_h.valence_sensitivity = 0.4f;
    }
    new_h.birth_gain = 0.15f; // Net2Net smooth birth

    hormones_.push_back(std::move(new_h));
    ticks_since_last_hormone_sprout_ = 0;
    return true;
}

bool KaryonSovereignCore::prune_least_causal_hormone() {
    if (hormones_.size() <= config_.initial_hormones) return false;
    size_t victim_idx = 0;
    float min_utility = 1e9f;
    for (size_t i = 0; i < hormones_.size(); ++i) {
        if (hormones_[i].causal_utility < min_utility && hormones_[i].birth_gain >= 0.90f) {
            min_utility = hormones_[i].causal_utility;
            victim_idx = i;
        }
    }
    hormones_.erase(hormones_.begin() + victim_idx);
    return true;
}

void KaryonSovereignCore::update_sovereign_hormones(std::span<const float> sensory_state,
                                                   float surprise,
                                                   float valence) {
    for (auto& h : hormones_) {
        // 1. Sensory emission projection
        float sensory_drive = 0.0f;
        if (!h.W_sensory_emission.empty() && sensory_state.size() >= config_.sensory_dim) {
            sensory_drive = avx2_dot_product(h.W_sensory_emission.data(), sensory_state.data(), config_.sensory_dim);
        }

        // 2. Continuous Differential Equation of Hormone Kinetics:
        // dh/dt = -gamma * (h - h0) + surprise_gain * surprise + valence_gain * valence + sensory_drive
        float dh = -h.decay_rate * (h.concentration - h.basal_level)
                   + h.surprise_sensitivity * surprise
                   + h.valence_sensitivity * valence
                   + 0.1f * std::tanh(sensory_drive);

        h.concentration = std::clamp(h.concentration + dh, 0.01f, 2.0f);

        // Mature newly sprouted hormones
        if (h.birth_gain < 1.0f) {
            h.birth_gain = std::min(1.0f, h.birth_gain + 0.005f);
        }

        // Causal utility tracking: rewarded if concentration aligns with valence, penalized on collision
        float alignment = (h.concentration - h.basal_level) * valence;
        h.causal_utility = 0.995f * h.causal_utility + 0.005f * std::max(0.0f, 1.0f + alignment);
    }

    compute_effector_modulations();
}

void KaryonSovereignCore::compute_effector_modulations() {
    float mod_plast = 0.0f;
    float mod_horizon = 0.0f;
    float mod_vigor = 0.0f;
    float mod_threat = 0.0f;
    float mod_expl = 0.0f;

    for (const auto& h : hormones_) {
        float dev = (h.concentration - h.basal_level) * h.birth_gain;
        mod_plast += h.w_plasticity * dev;
        mod_horizon += h.w_horizon * dev;
        mod_vigor += h.w_motor_vigor * dev;
        mod_threat += h.w_threat_aversive * dev;
        mod_expl += h.w_exploration * dev;
    }

    effective_plasticity_ = std::clamp(config_.plasticity_rate * std::exp(std::clamp(mod_plast, -1.5f, 1.5f)),
                                       0.002f, 0.08f);
    effective_motor_vigor_ = std::clamp(1.0f + 0.8f * std::tanh(mod_vigor), 0.4f, 2.0f);
    // Preserving strong aversive threat weighting when peril hormone spikes:
    effective_threat_weight_ = std::clamp(config_.aversive_weight * (1.0f + 2.0f * std::tanh(std::max(0.0f, mod_threat))),
                                         4.0f, 60.0f);
    effective_exploration_ = std::clamp(std::tanh(mod_expl), 0.0f, 0.5f);

    // Endogenous mental trajectory horizon
    int h_offset = static_cast<int>(std::round(mod_horizon * 2.0f));
    int target_k = static_cast<int>(config_.min_horizon_k) + h_offset;
    if (stagnation_potential_ > 0.4f) target_k += 1;
    current_horizon_k_ = static_cast<size_t>(std::clamp(target_k,
                                                        static_cast<int>(config_.min_horizon_k),
                                                        static_cast<int>(config_.max_horizon_k)));
}

float KaryonSovereignCore::compute_latent_peril(size_t action_id, const std::vector<float>& h_latent) {
    if (action_id >= config_.action_dim) return 0.0f;
    const float* p_row = W_peril_latent_.data() + action_id * config_.latent_dim;
    float linear_hazard = avx2_dot_product(p_row, h_latent.data(), config_.latent_dim);
    return std::tanh(linear_hazard);
}

void KaryonSovereignCore::decide_continuous_action(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<float> action_out) {
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        buf_adjoint_sensory_[d] = target_goal[d] - current_state[d];
    }

    for (size_t l = 0; l < config_.latent_dim; ++l) {
        const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
        buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, current_state.data(), config_.sensory_dim));
    }

    for (size_t a = 0; a < config_.action_dim; ++a) {
        float motor_drive = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float w = W_motor_[d * config_.action_dim + a];
            motor_drive += w * buf_adjoint_sensory_[d];
        }

        float peril = compute_latent_peril(a, buf_latent_a_);
        if (peril > config_.peril_threshold) {
            motor_drive -= effective_threat_weight_ * (peril - config_.peril_threshold) * 5.0f;
        }

        // Sovereign Hormonal motor vigor scaling
        action_out[a] = std::tanh(motor_drive * effective_motor_vigor_);
    }
}

float KaryonSovereignCore::evaluate_trajectory(size_t action_id,
                                               const std::vector<float>& s0,
                                               const std::vector<float>& target_goal,
                                               size_t horizon_k) {
    std::vector<float> sim = s0;
    float total_cost = 0.0f;
    float max_trajectory_peril = 0.0f;

    const float* v_row = W_action_valence_.data() + action_id * config_.sensory_dim;
    float expected_valence = avx2_dot_product(v_row, s0.data(), config_.sensory_dim);
    float immediate_pain_cost = effective_threat_weight_ * 5.0f * std::max(0.0f, -expected_valence);

    for (size_t tau = 1; tau <= horizon_k; ++tau) {
        const float* delta_row = action_delta_table_.data() + action_id * config_.sensory_dim;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            sim[d] += delta_row[d];
        }

        for (size_t l = 0; l < config_.latent_dim; ++l) {
            const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
            buf_latent_b_[l] = std::tanh(avx2_dot_product(w_row, sim.data(), config_.sensory_dim));
        }

        float step_peril = compute_latent_peril(action_id, buf_latent_b_);
        max_trajectory_peril = std::max(max_trajectory_peril, step_peril);

        float goal_dist_sq = 0.0f;
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float diff = target_goal[d] - sim[d];
            goal_dist_sq += diff * diff;
        }

        // Active Teleonomic Drive: Appetitive hunger amplifies weight on goal approach
        float teleonomic_weight = 1.0f + 2.5f * appetitive_hunger_;
        float discount = std::pow(config_.discount_gamma, static_cast<float>(tau - 1));
        total_cost += discount * goal_dist_sq * teleonomic_weight;
    }

    // Peril barrier threshold is dynamic: when starving/hungry, agent takes calculated risks
    float dynamic_peril_thresh = config_.peril_threshold + 0.15f * std::min(1.0f, appetitive_hunger_);
    if (max_trajectory_peril > dynamic_peril_thresh) {
        total_cost += 1e5f * (max_trajectory_peril - dynamic_peril_thresh);
    }
    total_cost += immediate_pain_cost;
    return total_cost;
}

size_t KaryonSovereignCore::decide_action(const std::vector<float>& current_state,
                                         const std::vector<float>& target_goal) {
    size_t best_action = 0;
    float min_cost = 1e9f;

    for (size_t a = 0; a < config_.action_dim; ++a) {
        float cost = evaluate_trajectory(a, current_state, target_goal, current_horizon_k_);
        if (cost < min_cost) {
            min_cost = cost;
            best_action = a;
        }
    }
    return best_action;
}

CoreTelemetry KaryonSovereignCore::step_environment(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<const float> next_state_observed,
                                                   std::span<const float> action_taken,
                                                   float external_valence) {
    CoreTelemetry telem;
    size_t count_before = primitives_.size();
    size_t hormone_count_before = hormones_.size();

    // 1. Sensory Compression (Observation -> Latent Manifold)
    for (size_t l = 0; l < config_.latent_dim; ++l) {
        const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
        buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, current_state.data(), config_.sensory_dim));
    }

    // 2. Super-Primitive Layer Feedforward
    for (size_t l = 0; l < config_.latent_dim; ++l) buf_latent_b_[l] = 0.0f;
    for (auto& p : primitives_) {
        p->forward(buf_latent_a_, buf_latent_b_, buf_quad_);
    }

    // 3. Sensory Expansion (Latent -> Predicted Next Observation)
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        const float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
        buf_sensory_out_[s] = avx2_dot_product(w_row, buf_latent_b_.data(), config_.latent_dim);
    }

    // 4. Free Energy / Surprise Calculation
    float fe_error = 0.0f;
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float err = next_state_observed[s] - buf_sensory_out_[s];
        fe_error += err * err;
    }
    fe_error /= static_cast<float>(config_.sensory_dim);
    moving_fe_ema_ = 0.95f * moving_fe_ema_ + 0.05f * fe_error;
    telem.free_energy = fe_error;
    telem.sensory_prediction_error = fe_error;

    // 5. Stagnation & Goal Progress Detection
    float sensory_motion = 0.0f;
    float current_goal_dist_sq = 0.0f;
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float diff = current_state[s] - continuous_sensory_ema_[s];
        sensory_motion += diff * diff;
        continuous_sensory_ema_[s] = 0.90f * continuous_sensory_ema_[s] + 0.10f * current_state[s];

        float g_diff = target_goal[s] - current_state[s];
        current_goal_dist_sq += g_diff * g_diff;
    }
    sensory_motion = std::sqrt(sensory_motion / static_cast<float>(config_.sensory_dim));
    if (sensory_motion < 0.005f) {
        stagnation_potential_ = std::min(1.0f, stagnation_potential_ + 0.05f);
    } else {
        stagnation_potential_ = std::max(0.0f, stagnation_potential_ - config_.stagnation_decay);
    }
    telem.stagnation_potential = stagnation_potential_;

    // Teleonomic Progress & Appetitive Hunger:
    // If distance to goal did not decrease, increase appetitive drive
    float goal_delta = last_goal_dist_sq_ - current_goal_dist_sq;
    last_goal_dist_sq_ = current_goal_dist_sq;

    if (external_valence > 0.5f) {
        // Goal achieved! Satiation resets hunger
        appetitive_hunger_ = 0.0f;
    } else if (goal_delta <= 0.001f) {
        appetitive_hunger_ = std::min(1.5f, appetitive_hunger_ + 0.02f);
    } else {
        appetitive_hunger_ = std::max(0.0f, appetitive_hunger_ - 0.04f);
    }

    // 6. Update Sovereign Endogenous Hormones
    update_sovereign_hormones(current_state, fe_error, external_valence);

    // 7. Non-Linear Peril Field & Temporal Eligibility Credit
    for (size_t a = 0; a < config_.action_dim; ++a) {
        float a_act = action_taken[a];
        if (a_act > 0.05f) {
            float* p_row = W_peril_latent_.data() + a * config_.latent_dim;
            if (external_valence < 0.0f) {
                float pain_mag = std::abs(external_valence);
                for (size_t l = 0; l < config_.latent_dim; ++l) {
                    p_row[l] += effective_plasticity_ * 4.0f * pain_mag * buf_latent_a_[l];
                    p_row[l] = std::clamp(p_row[l], -4.0f, 4.0f);
                }
            } else if (external_valence > 0.0f) {
                for (size_t l = 0; l < config_.latent_dim; ++l) {
                    p_row[l] -= effective_plasticity_ * 1.5f * external_valence * buf_latent_a_[l];
                    p_row[l] = std::clamp(p_row[l], -4.0f, 4.0f);
                }
            }

            float* v_row = W_action_valence_.data() + a * config_.sensory_dim;
            float pred_v = avx2_dot_product(v_row, current_state.data(), config_.sensory_dim);
            float v_error = external_valence - pred_v;
            for (size_t d = 0; d < config_.sensory_dim; ++d) {
                v_row[d] += effective_plasticity_ * 5.0f * a_act * v_error * current_state[d];
                v_row[d] = std::clamp(v_row[d], -4.0f, 4.0f);
            }
        }
    }

    // 8. Autopoietic Genesis of Primitives & Hormones
    evaluate_autopoietic_morphogenesis(fe_error);
    telem.active_primitives_count = primitives_.size();
    telem.active_hormones_count = hormones_.size();
    telem.event_sprouted = (primitives_.size() > count_before);
    telem.event_pruned = (primitives_.size() < count_before);
    telem.event_hormone_sprouted = (hormones_.size() > hormone_count_before);
    telem.event_hormone_pruned = (hormones_.size() < hormone_count_before);

    // 9. Numerically Bounded Plasticity Updates
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float err_s = next_state_observed[s] - buf_sensory_out_[s];
        err_s = std::clamp(err_s, -2.0f, 2.0f);
        float* w_row = W_sensory_up_.data() + s * config_.latent_dim;
        for (size_t l = 0; l < config_.latent_dim; ++l) {
            w_row[l] += effective_plasticity_ * err_s * buf_latent_a_[l];
            w_row[l] = std::clamp(w_row[l], -3.0f, 3.0f);
        }
    }

    for (auto& p : primitives_) {
        p->update_plasticity_avx2(buf_latent_a_, buf_latent_a_, effective_plasticity_, buf_quad_);
        p->adapt_coordinates(0.01f, 0.01f, effective_plasticity_);
        float lock_utility = (external_valence * 0.10f) + (0.05f - fe_error);
        p->update_epigenetic_lock(lock_utility);
    }

    // Direct Motor Synaptic Rewiring
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float motor_gradient = (target_goal[d] - current_state[d]);
        float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            float delta = 0.0f;
            if (external_valence < 0.0f) {
                delta = -effective_plasticity_ * 3.0f * std::abs(external_valence) * action_taken[m] * current_state[d];
            } else {
                delta = effective_plasticity_ * (external_valence * 2.0f + 0.1f) * motor_gradient * action_taken[m];
            }
            w_row[m] += delta;
            w_row[m] = std::clamp(w_row[m], -4.0f, 4.0f);
        }
    }

    telem.primary_hormone = get_hormone(0);
    telem.secondary_hormone = get_hormone(1);
    telem.effective_plasticity = effective_plasticity_;
    telem.effective_motor_vigor = effective_motor_vigor_;
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

    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float delta = next_state_observed[d] - current_state[d];
        action_delta_table_[discrete_action * config_.sensory_dim + d] +=
            effective_plasticity_ * 5.0f * (delta - action_delta_table_[discrete_action * config_.sensory_dim + d]);
    }

    auto telem = step_environment(current_state, target_goal, next_state_observed, continuous_act, external_valence);
    telem.action_discrete_chosen = discrete_action;
    return telem;
}

void KaryonSovereignCore::evaluate_autopoietic_morphogenesis(float fe_error) {
    ticks_since_last_sprout_++;
    ticks_since_last_hormone_sprout_++;

    // 1. Primitive Morphogenesis
    if (ticks_since_last_sprout_ > config_.sprouting_cooldown_ticks) {
        if (fe_error > config_.sprouting_fe_threshold && primitives_.size() < config_.max_primitives) {
            trigger_morphogenetic_sprout();
        } else if (fe_error < 0.015f && primitives_.size() > config_.initial_primitives) {
            prune_least_causal_primitive();
        }
    }

    // 2. Sovereign Hormone Morphogenesis
    if (ticks_since_last_hormone_sprout_ > (config_.sprouting_cooldown_ticks * 2)) {
        if (moving_fe_ema_ > (config_.sprouting_fe_threshold * 0.8f) && hormones_.size() < config_.max_hormones) {
            trigger_hormone_sprout();
        } else if (moving_fe_ema_ < 0.02f && hormones_.size() > 2) {
            prune_least_causal_hormone();
        }
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
    if (!handle) return;
    auto* core = static_cast<KaryonSovereignCore*>(handle);
    core->decide_continuous_action(std::span<const float>(current_state, 256),
                                   std::span<const float>(target_goal, 256),
                                   std::span<float>(action_out, 4));
}

size_t karyon_core_decide_discrete(void* handle, const float* current_state, const float* target_goal) {
    if (!handle) return 0;
    auto* core = static_cast<KaryonSovereignCore*>(handle);
    std::vector<float> s_curr(current_state, current_state + 256);
    std::vector<float> s_tgt(target_goal, target_goal + 256);
    return core->decide_action(s_curr, s_tgt);
}

void karyon_core_step(void* handle,
                      const float* current_state,
                      const float* target_goal,
                      const float* next_state_observed,
                      const float* action_taken,
                      float external_valence,
                      CoreTelemetry* telem_out) {
    if (!handle) return;
    auto* core = static_cast<KaryonSovereignCore*>(handle);
    auto telem = core->step_environment(std::span<const float>(current_state, 256),
                                        std::span<const float>(target_goal, 256),
                                        std::span<const float>(next_state_observed, 256),
                                        std::span<const float>(action_taken, 4),
                                        external_valence);
    if (telem_out) *telem_out = telem;
}

} // extern "C"

} // namespace karyon
