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
// Continuous Dynamic Episodic Memory Bank Implementation
// =========================================================================
DynamicEpisodicMemory::QueryResult DynamicEpisodicMemory::query(const std::vector<float>& query_key, float* scratch_read) const {
    QueryResult res;
    if (slots.empty()) return res;

    float best_sim = -1.0f;
    int best_i = -1;

    float q_norm_sq = avx2_dot_product(query_key.data(), query_key.data(), key_dim);
    float q_norm = std::sqrt(std::max(1e-8f, q_norm_sq));

    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& sl = slots[i];
        float k_norm_sq = avx2_dot_product(sl.key.data(), sl.key.data(), key_dim);
        float k_norm = std::sqrt(std::max(1e-8f, k_norm_sq));

        float dot = avx2_dot_product(query_key.data(), sl.key.data(), key_dim);
        float cosine_sim = dot / (q_norm * k_norm + 1e-8f);

        if (cosine_sim > best_sim) {
            best_sim = cosine_sim;
            best_i = static_cast<int>(i);
        }
    }

    res.max_sim = best_sim;
    res.best_idx = best_i;
    if (best_i >= 0) {
        res.familiarity = slots[best_i].visit_count;
        res.retrieved_value = slots[best_i].value;
        if (scratch_read) {
            std::memcpy(scratch_read, slots[best_i].value.data(), sizeof(float) * value_dim);
        }
    }
    return res;
}

void DynamicEpisodicMemory::record(const std::vector<float>& key,
                                   const std::vector<float>& value,
                                   float utility,
                                   uint64_t current_step) {
    auto q = query(key);
    if (q.best_idx >= 0 && q.max_sim >= similarity_threshold) {
        // Soft update existing memory trace (Bayesian fusion of consequences)
        auto& sl = slots[q.best_idx];
        sl.visit_count += 1.0f;
        sl.last_accessed_step = current_step;
        sl.utility = std::max(sl.utility, utility);

        float blend = 0.25f / std::sqrt(sl.visit_count);
        for (size_t d = 0; d < value_dim; ++d) {
            sl.value[d] = (1.0f - blend) * sl.value[d] + blend * value[d];
        }
    } else {
        // Sprout a new episodic topological memory node (Unbounded Lifelong Expansion)
        EpisodicMemorySlot new_slot;
        new_slot.key = key;
        new_slot.value = value;
        new_slot.utility = utility;
        new_slot.visit_count = 1.0f;
        new_slot.last_accessed_step = current_step;
        slots.push_back(std::move(new_slot));
    }
}

float DynamicEpisodicMemory::compute_inhibition_of_return(const std::vector<float>& candidate_key) const {
    if (slots.empty()) return 0.0f;

    float q_norm_sq = avx2_dot_product(candidate_key.data(), candidate_key.data(), key_dim);
    float q_norm = std::sqrt(std::max(1e-8f, q_norm_sq));

    float max_penalty = 0.0f;
    for (const auto& sl : slots) {
        float k_norm_sq = avx2_dot_product(sl.key.data(), sl.key.data(), key_dim);
        float k_norm = std::sqrt(std::max(1e-8f, k_norm_sq));
        float dot = avx2_dot_product(candidate_key.data(), sl.key.data(), key_dim);
        float sim = dot / (q_norm * k_norm + 1e-8f);

        if (sim > 0.85f) {
            // Repulsion scales with how often this state has been visited
            float recency_boost = std::min(5.0f, sl.visit_count * 0.35f);
            float penalty = sim * recency_boost;
            if (penalty > max_penalty) {
                max_penalty = penalty;
            }
        }
    }
    return max_penalty;
}

// =========================================================================
// Continuous Sovereign Tensor-Field Super-Primitive Implementation (KEP Principle 3)
// =========================================================================
ContinuousSuperPrimitive::ContinuousSuperPrimitive(size_t pid, size_t d, std::mt19937& rng)
    : id(pid), dim(d),
      gate_bias(0.5f), decay_rate(0.1f), tau_time(0.2f),
      epigenetic_lock(0.0f), birth_gain(1.0f) {
    
    // W_field is [dim x (2 * dim)] projecting [x; h_prev]
    W_field.resize(dim * (2 * dim), 0.0f);
    // W_gate is [dim x dim] projecting gating flow
    W_gate.resize(dim * dim, 0.0f);
    state_trace.resize(dim, 0.0f);

    std::normal_distribution<float> dist_init(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
    for (auto& w : W_field) w = dist_init(rng);
    for (auto& w : W_gate) w = dist_init(rng);

    // Initial diversity of continuous operator coordinates
    if (pid % 3 == 0) {
        // Degenerates towards fast temporal integrator / attractor basin
        gate_bias = 0.1f;
        decay_rate = 0.05f;
        tau_time = 0.4f;
    } else if (pid % 3 == 1) {
        // Degenerates towards non-linear multiplicative gate (SwiGLU-like)
        gate_bias = 0.85f;
        decay_rate = 0.3f;
        tau_time = 0.25f;
    } else {
        // Balanced continuous dynamic field
        gate_bias = 0.5f;
        decay_rate = 0.15f;
        tau_time = 0.2f;
    }
}

void ContinuousSuperPrimitive::forward(std::span<const float> x_in,
                                       std::span<float> h_out,
                                       std::span<float> quad_buf) {
    size_t in_dim = 2 * dim;

    for (size_t i = 0; i < dim; ++i) {
        const float* row_field = W_field.data() + i * in_dim;
        const float* row_gate = W_gate.data() + i * dim;

        // Split field dot product: x part and h_prev part
        float f_x = avx2_dot_product(row_field, x_in.data(), dim);
        float f_h = avx2_dot_product(row_field + dim, state_trace.data(), dim);
        float f_total = f_x + f_h;

        // Gate activation sigma(W_gate * x)
        float g_lin = avx2_dot_product(row_gate, x_in.data(), dim);
        float g_act = 1.0f / (1.0f + std::exp(-g_lin));

        // Unified continuous operator degeneration
        float y_i = (1.0f - gate_bias) * std::tanh(f_total) + gate_bias * (g_act * f_x);

        // Continuous state integration with dissipation decay
        float dh = tau_time * (y_i - decay_rate * state_trace[i]);
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

    size_t in_dim = 2 * dim;
    for (size_t i = 0; i < dim; ++i) {
        float post_i = post[i];
        float* row_field = W_field.data() + i * in_dim;
        float* row_gate = W_gate.data() + i * dim;
        float trace_i = state_trace[i];

        // 1. Plasticity for W_field (pre and trace)
        size_t j = 0;
        __m256 v_lr = _mm256_set1_ps(effective_lr);
        __m256 v_post = _mm256_set1_ps(post_i);

        for (; j + 8 <= dim; j += 8) {
            __m256 v_pre = _mm256_loadu_ps(pre.data() + j);
            __m256 v_wfield_x = _mm256_loadu_ps(row_field + j);
            __m256 v_wgate = _mm256_loadu_ps(row_gate + j);

            __m256 d_wfield = _mm256_mul_ps(v_lr, _mm256_mul_ps(v_post, v_pre));
            _mm256_storeu_ps(row_field + j, _mm256_add_ps(v_wfield_x, d_wfield));
            _mm256_storeu_ps(row_gate + j, _mm256_add_ps(v_wgate, d_wfield));
        }
        for (; j < dim; ++j) {
            row_field[j] += effective_lr * post_i * pre[j];
            row_gate[j] += effective_lr * post_i * pre[j];
        }

        // 2. Plasticity for W_field recurrent block (state_trace)
        float* row_field_h = row_field + dim;
        j = 0;
        for (; j + 8 <= dim; j += 8) {
            __m256 v_trace = _mm256_loadu_ps(state_trace.data() + j);
            __m256 v_wfield_h = _mm256_loadu_ps(row_field_h + j);
            __m256 d_wfield_h = _mm256_mul_ps(v_lr, _mm256_mul_ps(v_post, v_trace));
            _mm256_storeu_ps(row_field_h + j, _mm256_add_ps(v_wfield_h, d_wfield_h));
        }
        for (; j < dim; ++j) {
            row_field_h[j] += effective_lr * post_i * state_trace[j];
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

void ContinuousSuperPrimitive::adapt_manifold_coordinates(float d_gate, float d_decay, float lr) {
    gate_bias = std::clamp(gate_bias + d_gate * lr, 0.01f, 0.99f);
    decay_rate = std::clamp(decay_rate + d_decay * lr, 0.01f, 0.80f);
}

void ContinuousSuperPrimitive::sprout_latent_dimension() {
    size_t old_d = dim;
    size_t new_d = old_d + 1;
    size_t old_in_dim = 2 * old_d;
    size_t new_in_dim = 2 * new_d;

    std::vector<float> new_field(new_d * new_in_dim, 0.0f);
    std::vector<float> new_gate(new_d * new_d, 0.0f);

    for (size_t r = 0; r < old_d; ++r) {
        for (size_t c = 0; c < old_d; ++c) {
            new_field[r * new_in_dim + c] = W_field[r * old_in_dim + c];
        }
        for (size_t c = 0; c < old_d; ++c) {
            new_field[r * new_in_dim + new_d + c] = W_field[r * old_in_dim + old_d + c];
        }
        for (size_t c = 0; c < old_d; ++c) {
            new_gate[r * new_d + c] = W_gate[r * old_d + c];
        }
    }

    new_field[old_d * new_in_dim + old_d] = 1.0f;
    new_gate[old_d * new_d + old_d] = 1.0f;

    W_field = std::move(new_field);
    W_gate = std::move(new_gate);
    state_trace.resize(new_d, 0.0f);
    dim = new_d;
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
      current_latent_dim_(config.initial_latent_dim),
      current_horizon_k_(config.min_horizon_k),
      ticks_since_last_sprout_(0),
      ticks_since_last_hormone_sprout_(0),
      ticks_since_last_latent_sprout_(0),
      moving_fe_ema_(0.05f),
      episodic_memory_(config.initial_latent_dim, config.action_dim + 2, 0.90f),
      global_step_counter_(0) {

    for (size_t i = 0; i < config_.initial_primitives; ++i) {
        primitives_.push_back(std::make_unique<ContinuousSuperPrimitive>(i, current_latent_dim_, rng_));
    }

    for (size_t h = 0; h < config_.initial_hormones; ++h) {
        hormones_.emplace_back(h, config_.sensory_dim, rng_);
    }

    W_sensory_down_.resize(current_latent_dim_ * config_.sensory_dim);
    W_sensory_up_.resize(config_.sensory_dim * current_latent_dim_);
    W_motor_.resize(config_.sensory_dim * config_.action_dim, 0.0f);
    W_motor_latent_.resize(config_.action_dim * current_latent_dim_, 0.0f);
    action_delta_table_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    W_action_valence_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    W_peril_latent_.resize(config_.action_dim * current_latent_dim_, 0.0f);
    action_eligibility_trace_.resize(config_.action_dim * config_.sensory_dim, 0.0f);
    continuous_sensory_ema_.resize(config_.sensory_dim, 0.0f);

    std::normal_distribution<float> dist_m(0.0f, 0.15f);
    for (auto& w : W_motor_) w = dist_m(rng_);

    std::normal_distribution<float> dist_ml(0.0f, 0.15f);
    for (auto& w : W_motor_latent_) w = dist_ml(rng_);

    M_successor_.resize(current_latent_dim_ * current_latent_dim_, 0.0f);
    for (size_t l = 0; l < current_latent_dim_; ++l) {
        M_successor_[l * current_latent_dim_ + l] = 1.0f; // Diagonal identity
    }
    goal_successor_map_.resize(current_latent_dim_, 0.0f);
    macro_subgoal_latent_.resize(current_latent_dim_, 0.0f);
    macro_subgoal_timer_ = 0.0f;

    std::normal_distribution<float> dist_down(0.0f, 1.0f / std::sqrt(static_cast<float>(config_.sensory_dim)));
    for (auto& w : W_sensory_down_) w = dist_down(rng_);

    std::normal_distribution<float> dist_up(0.0f, 1.0f / std::sqrt(static_cast<float>(current_latent_dim_)));
    for (auto& w : W_sensory_up_) w = dist_up(rng_);

    buf_sensory_in_.resize(config_.sensory_dim, 0.0f);
    buf_sensory_out_.resize(config_.sensory_dim, 0.0f);
    buf_latent_a_.resize(current_latent_dim_, 0.0f);
    buf_latent_b_.resize(current_latent_dim_, 0.0f);
    buf_quad_.resize(current_latent_dim_, 0.0f);
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
    auto child = std::make_unique<ContinuousSuperPrimitive>(new_id, current_latent_dim_, rng_);
    const auto& parent = primitives_[parent_idx];

    child->W_field = parent->W_field;
    child->W_gate = parent->W_gate;
    child->gate_bias = parent->gate_bias;
    child->decay_rate = parent->decay_rate;
    child->tau_time = parent->tau_time;
    child->epigenetic_lock = 0.0f;
    child->birth_gain = 0.05f; // Smooth identity sprouting

    std::normal_distribution<float> mut_dist(0.0f, 0.02f);
    for (auto& w : child->W_field) w += mut_dist(rng_);
    for (auto& w : child->W_gate) w += mut_dist(rng_);

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

bool KaryonSovereignCore::trigger_latent_dimension_sprout() {
    if (current_latent_dim_ >= config_.max_latent_dim) return false;

    size_t old_d = current_latent_dim_;
    size_t new_d = old_d + 1;
    size_t S = config_.sensory_dim;
    size_t A = config_.action_dim;

    // 1. Sprout all primitives in stack
    for (auto& p : primitives_) {
        p->sprout_latent_dimension();
    }

    // 2. Expand W_sensory_down_ [new_d x S]
    std::vector<float> new_down(new_d * S, 0.0f);
    std::copy(W_sensory_down_.begin(), W_sensory_down_.end(), new_down.begin());
    std::normal_distribution<float> dist_down(0.0f, 0.01f); // Gentle initial projection
    for (size_t s = 0; s < S; ++s) {
        new_down[old_d * S + s] = dist_down(rng_);
    }
    W_sensory_down_ = std::move(new_down);

    // 3. Expand W_sensory_up_ [S x new_d]
    std::vector<float> new_up(S * new_d, 0.0f);
    for (size_t s = 0; s < S; ++s) {
        for (size_t l = 0; l < old_d; ++l) {
            new_up[s * new_d + l] = W_sensory_up_[s * old_d + l];
        }
        new_up[s * new_d + old_d] = 0.0f; // Zero-shock Net2Net identity
    }
    W_sensory_up_ = std::move(new_up);

    // 4. Expand W_peril_latent_ [A x new_d]
    std::vector<float> new_peril(A * new_d, 0.0f);
    for (size_t a = 0; a < A; ++a) {
        for (size_t l = 0; l < old_d; ++l) {
            new_peril[a * new_d + l] = W_peril_latent_[a * old_d + l];
        }
        new_peril[a * new_d + old_d] = 0.0f;
    }
    W_peril_latent_ = std::move(new_peril);

    // 4a. Expand W_motor_latent_ [A x new_d]
    std::vector<float> new_motor_latent(A * new_d, 0.0f);
    for (size_t a = 0; a < A; ++a) {
        for (size_t l = 0; l < old_d; ++l) {
            new_motor_latent[a * new_d + l] = W_motor_latent_[a * old_d + l];
        }
        new_motor_latent[a * new_d + old_d] = 0.0f; // Zero-shock Net2Net identity
    }
    W_motor_latent_ = std::move(new_motor_latent);

    // 4b. Expand M_successor_ [new_d x new_d]
    std::vector<float> new_m(new_d * new_d, 0.0f);
    for (size_t r = 0; r < old_d; ++r) {
        for (size_t c = 0; c < old_d; ++c) {
            new_m[r * new_d + c] = M_successor_[r * old_d + c];
        }
    }
    new_m[old_d * new_d + old_d] = 1.0f;
    M_successor_ = std::move(new_m);
    goal_successor_map_.resize(new_d, 0.0f);
    macro_subgoal_latent_.resize(new_d, 0.0f);

    // 4c. Expand Dynamic Episodic Memory Key Dimension
    episodic_memory_.sprout_dimension();

    // 5. Expand internal scratchpads
    buf_latent_a_.resize(new_d, 0.0f);
    buf_latent_b_.resize(new_d, 0.0f);
    buf_quad_.resize(new_d, 0.0f);

    current_latent_dim_ = new_d;
    ticks_since_last_latent_sprout_ = 0;
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
    const float* p_row = W_peril_latent_.data() + action_id * current_latent_dim_;
    float linear_hazard = avx2_dot_product(p_row, h_latent.data(), current_latent_dim_);
    return std::tanh(linear_hazard);
}

void KaryonSovereignCore::decide_continuous_action(std::span<const float> current_state,
                                                   std::span<const float> target_goal,
                                                   std::span<float> action_out) {
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        buf_adjoint_sensory_[d] = target_goal[d] - current_state[d];
    }

    for (size_t l = 0; l < current_latent_dim_; ++l) {
        const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
        buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, current_state.data(), config_.sensory_dim));
    }

    for (size_t a = 0; a < config_.action_dim; ++a) {
        float motor_drive = 0.0f;
        // Direct sensory adjoint error drive
        for (size_t d = 0; d < config_.sensory_dim; ++d) {
            float w = W_motor_[d * config_.action_dim + a];
            motor_drive += w * buf_adjoint_sensory_[d];
        }

        // Deep Latent Manifold Coupling (Cortex to Motor Field)
        const float* ml_row = W_motor_latent_.data() + a * current_latent_dim_;
        motor_drive += avx2_dot_product(ml_row, buf_latent_a_.data(), current_latent_dim_);

        // Riemannian metric curvature barrier:
        // Threat is an endogenous geometric curvature in latent manifold rather than hardcoded threshold.
        // g_a = 1 + exp(5.0 * peril)
        // High peril curves space exponentially, naturally deflecting geodesic motor trajectories.
        float peril = compute_latent_peril(a, buf_latent_a_);
        if (peril > 0.0f) {
            float curvature_drag = std::expm1(4.0f * peril) * effective_threat_weight_ * 0.25f;
            motor_drive -= curvature_drag;
        }

        // Sovereign Hormonal motor vigor scaling
        action_out[a] = std::tanh(motor_drive * effective_motor_vigor_);
    }

    // Refine action using continuous variational trajectory optimization
    optimize_continuous_action_trajectory(current_state, target_goal, action_out, 3);
}

void KaryonSovereignCore::optimize_continuous_action_trajectory(std::span<const float> current_state,
                                                                std::span<const float> target_goal,
                                                                std::span<float> action_inout,
                                                                size_t variational_steps) {
    if (variational_steps == 0) return;

    size_t S = config_.sensory_dim;
    size_t A = config_.action_dim;
    float step_size = 0.15f * effective_motor_vigor_;

    for (size_t step = 0; step < variational_steps; ++step) {
        // Continuous trajectory forward simulation along the continuous action vector:
        // Efference projection through W_motor:
        for (size_t d = 0; d < S; ++d) {
            float delta_d = 0.0f;
            for (size_t a = 0; a < A; ++a) {
                delta_d += W_motor_[d * A + a] * action_inout[a];
            }
            buf_sensory_out_[d] = current_state[d] + delta_d;
        }

        // Compute error gradient towards target goal
        for (size_t d = 0; d < S; ++d) {
            buf_adjoint_sensory_[d] = target_goal[d] - buf_sensory_out_[d];
        }

        // Variational adjoint step in action space: dF/du_a = sum_d W_motor[d, a] * (target[d] - pred[d])
        for (size_t a = 0; a < A; ++a) {
            float grad_a = 0.0f;
            for (size_t d = 0; d < S; ++d) {
                grad_a += W_motor_[d * A + a] * buf_adjoint_sensory_[d];
            }
            // Continuous gradient update along Free Energy manifold
            action_inout[a] = std::clamp(action_inout[a] + step_size * grad_a, -1.0f, 1.0f);
        }
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

    size_t S = config_.sensory_dim;
    size_t L = current_latent_dim_;

    for (size_t tau = 1; tau <= horizon_k; ++tau) {
        const float* delta_row = action_delta_table_.data() + action_id * S;
        for (size_t d = 0; d < S; ++d) {
            sim[d] += delta_row[d];
        }

        for (size_t l = 0; l < L; ++l) {
            const float* w_row = W_sensory_down_.data() + l * S;
            buf_latent_b_[l] = std::tanh(avx2_dot_product(w_row, sim.data(), S));
        }

        float step_peril = compute_latent_peril(action_id, buf_latent_b_);
        max_trajectory_peril = std::max(max_trajectory_peril, step_peril);

        // 1. Direct Euclidean goal distance
        float goal_dist_sq = 0.0f;
        for (size_t d = 0; d < S; ++d) {
            float diff = target_goal[d] - sim[d];
            goal_dist_sq += diff * diff;
        }

        // 2. Successor Topological Geodesic:
        // Project predicted latent state through successor diffusion map M:
        // m_sim = M * buf_latent_b_
        float topo_geodesic_dist = 0.0f;
        for (size_t l = 0; l < L; ++l) {
            const float* m_row = M_successor_.data() + l * L;
            float m_val = avx2_dot_product(m_row, buf_latent_b_.data(), L);
            float diff = goal_successor_map_[l] - m_val;
            topo_geodesic_dist += diff * diff;
        }

        // 3. Hierarchical Macro-Attractor Alignment (Level 2 Sub-Goal):
        float macro_cost = 0.0f;
        if (stagnation_potential_ > 0.05f || appetitive_hunger_ > 0.1f) {
            for (size_t l = 0; l < L; ++l) {
                float diff = macro_subgoal_latent_[l] - buf_latent_b_[l];
                macro_cost += diff * diff;
            }
        }

        // 4. Non-Parametric Episodic Memory Query & Inhibition-of-Return
        // Prevent cyclic traps and attract towards historically fruitful paths
        float memory_ior = episodic_memory_.compute_inhibition_of_return(buf_latent_b_);
        auto mem_q = episodic_memory_.query(buf_latent_b_);
        float memory_val_bonus = 0.0f;
        if (mem_q.best_idx >= 0 && mem_q.max_sim > 0.88f) {
            // value[0] is historical valence of this state
            if (!mem_q.retrieved_value.empty()) {
                memory_val_bonus = mem_q.retrieved_value[0];
            }
        }

        // Blend Euclidean and Topological Geodesic by hunger and stagnation
        float topo_weight = std::clamp(stagnation_potential_ * 2.5f + appetitive_hunger_ * 1.5f, 0.15f, 0.90f);
        float effective_spatial_cost = (1.0f - topo_weight) * goal_dist_sq +
                                       topo_weight * topo_geodesic_dist +
                                       0.35f * macro_cost +
                                       0.15f * memory_ior -
                                       0.30f * memory_val_bonus;

        // Active Teleonomic Drive: Appetitive hunger amplifies weight on goal approach
        float teleonomic_weight = 1.0f + 2.5f * appetitive_hunger_;
        float discount = std::pow(config_.discount_gamma, static_cast<float>(tau - 1));
        total_cost += discount * effective_spatial_cost * teleonomic_weight;
    }

    // Endogenous Riemannian Geodesic Curvature Barrier:
    // Replaces hardcoded peril_threshold and artificial 1e5f infinite penalty.
    // The metric tensor deformation along the trajectory is:
    // ds^2 = (1 + expm1(6.0 * max_trajectory_peril)) * ||v||^2
    // Starvation/appetitive hunger flattens the perceived metric curvature, allowing bold risk-taking.
    if (max_trajectory_peril > 0.0f) {
        float hunger_mitigation = 1.0f / (1.0f + 2.0f * appetitive_hunger_);
        float metric_curvature = std::expm1(5.0f * max_trajectory_peril) * hunger_mitigation;
        total_cost += effective_threat_weight_ * config_.metric_curvature_scale * metric_curvature;
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
    for (size_t l = 0; l < current_latent_dim_; ++l) {
        const float* w_row = W_sensory_down_.data() + l * config_.sensory_dim;
        buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, current_state.data(), config_.sensory_dim));
    }

    // 2. Super-Primitive Layer Feedforward
    for (size_t l = 0; l < current_latent_dim_; ++l) buf_latent_b_[l] = 0.0f;
    for (auto& p : primitives_) {
        p->forward(buf_latent_a_, buf_latent_b_, buf_quad_);
    }

    // 3. Sensory Expansion (Latent -> Predicted Next Observation)
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        const float* w_row = W_sensory_up_.data() + s * current_latent_dim_;
        buf_sensory_out_[s] = avx2_dot_product(w_row, buf_latent_b_.data(), current_latent_dim_);
    }

    // 4. Free Energy / Surprise Calculation
    // Free Energy F = Epistemic Surprise (sensory prediction error) + Teleonomic Divergence (pragmatic goal distance error)
    float sensory_pred_err = 0.0f;
    float pragmatic_goal_err = 0.0f;

    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float s_err = next_state_observed[s] - buf_sensory_out_[s];
        sensory_pred_err += s_err * s_err;

        float g_err = next_state_observed[s] - target_goal[s];
        pragmatic_goal_err += g_err * g_err;
    }
    sensory_pred_err /= static_cast<float>(config_.sensory_dim);
    pragmatic_goal_err /= static_cast<float>(config_.sensory_dim);

    // Unified Variational Free Energy: F = F_epistemic + F_pragmatic
    float fe_error = sensory_pred_err + 0.5f * pragmatic_goal_err;

    moving_fe_ema_ = 0.95f * moving_fe_ema_ + 0.05f * fe_error;
    telem.free_energy = fe_error;
    telem.sensory_prediction_error = sensory_pred_err;

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
            float* p_row = W_peril_latent_.data() + a * current_latent_dim_;
            if (external_valence < 0.0f) {
                float pain_mag = std::abs(external_valence);
                for (size_t l = 0; l < current_latent_dim_; ++l) {
                    p_row[l] += effective_plasticity_ * 4.0f * pain_mag * buf_latent_a_[l];
                    p_row[l] = std::clamp(p_row[l], -4.0f, 4.0f);
                }
            } else if (external_valence > 0.0f) {
                for (size_t l = 0; l < current_latent_dim_; ++l) {
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
    // A. Update Sensory Expansion (W_sensory_up_): Latent -> Observed
    for (size_t s = 0; s < config_.sensory_dim; ++s) {
        float err_s = next_state_observed[s] - buf_sensory_out_[s];
        err_s = std::clamp(err_s, -2.0f, 2.0f);
        float* w_row = W_sensory_up_.data() + s * current_latent_dim_;
        for (size_t l = 0; l < current_latent_dim_; ++l) {
            w_row[l] += effective_plasticity_ * err_s * buf_latent_a_[l];
            w_row[l] = std::clamp(w_row[l], -3.0f, 3.0f);
        }
    }

    // B. Update Sensory Compression Encoder (W_sensory_down_): Sensory -> Latent
    for (size_t l = 0; l < current_latent_dim_; ++l) {
        float grad_z = 0.0f;
        for (size_t s = 0; s < config_.sensory_dim; ++s) {
            float err_s = next_state_observed[s] - buf_sensory_out_[s];
            grad_z += err_s * W_sensory_up_[s * current_latent_dim_ + l];
        }
        float z_val = buf_latent_a_[l];
        float dtanh = 1.0f - z_val * z_val;
        float delta_l = std::clamp(grad_z * dtanh, -2.0f, 2.0f);

        float* down_row = W_sensory_down_.data() + l * config_.sensory_dim;
        for (size_t s = 0; s < config_.sensory_dim; ++s) {
            down_row[s] += effective_plasticity_ * 0.5f * delta_l * current_state[s];
            down_row[s] = std::clamp(down_row[s], -3.0f, 3.0f);
        }
    }

    for (auto& p : primitives_) {
        p->update_plasticity_avx2(buf_latent_a_, buf_latent_a_, effective_plasticity_, buf_quad_);
        p->adapt_manifold_coordinates(0.01f, 0.01f, effective_plasticity_);
        float lock_utility = (external_valence * 0.10f) + (0.05f - fe_error);
        p->update_epigenetic_lock(lock_utility);
    }

    // Direct Motor Synaptic Rewiring via Causal Afference Credit
    // delta_obs = next_state_observed - current_state
    // If motor action m moved state towards target_goal, reward W_motor:
    for (size_t d = 0; d < config_.sensory_dim; ++d) {
        float observed_motion = next_state_observed[d] - current_state[d];
        float goal_dir = target_goal[d] - current_state[d];
        float* w_row = W_motor_.data() + d * config_.action_dim;
        for (size_t m = 0; m < config_.action_dim; ++m) {
            float act = action_taken[m];
            // Track causal correlation between motor action and physical observed movement
            float motor_causality = observed_motion * act;
            float alignment_with_goal = observed_motion * goal_dir;
            
            float dW = 0.0f;
            if (external_valence < 0.0f) {
                dW = -effective_plasticity_ * 4.0f * std::abs(external_valence) * act * current_state[d];
            } else {
                // Natural Hebbian causal reinforcement: actions that move state towards goal are strengthened
                dW = effective_plasticity_ * (motor_causality * 2.0f + alignment_with_goal * 1.5f + (external_valence * 2.0f + 0.1f) * goal_dir * act);
            }
            w_row[m] += dW;
            w_row[m] = std::clamp(w_row[m], -4.0f, 4.0f);
        }
    }

    // Direct Cortical-Motor Rewiring (W_motor_latent_):
    // Actions aligned with goal motion reinforce the latent state that produced them
    for (size_t a = 0; a < config_.action_dim; ++a) {
        float act = action_taken[a];
        if (std::abs(act) > 0.05f) {
            float* ml_row = W_motor_latent_.data() + a * current_latent_dim_;
            for (size_t l = 0; l < current_latent_dim_; ++l) {
                float d_ml = 0.0f;
                if (external_valence < 0.0f) {
                    d_ml = -effective_plasticity_ * 2.0f * std::abs(external_valence) * act * buf_latent_a_[l];
                } else {
                    float goal_alignment = (last_goal_dist_sq_ - current_goal_dist_sq);
                    d_ml = effective_plasticity_ * (goal_alignment * 2.0f + external_valence * 1.5f + 0.05f) * act * buf_latent_a_[l];
                }
                ml_row[l] += d_ml;
                ml_row[l] = std::clamp(ml_row[l], -4.0f, 4.0f);
            }
        }
    }

    telem.primary_hormone = get_hormone(0);
    telem.secondary_hormone = get_hormone(1);
    telem.effective_plasticity = effective_plasticity_;
    telem.effective_motor_vigor = effective_motor_vigor_;
    telem.somatic_energy = somatic_energy_;
    telem.horizon_k = current_horizon_k_;
    telem.current_latent_dim = current_latent_dim_;
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

    // Successor Representation & Hierarchical Macro-Attractor Dynamics:
    size_t L = current_latent_dim_;
    size_t S = config_.sensory_dim;

    for (size_t l = 0; l < L; ++l) {
        const float* w_row = W_sensory_down_.data() + l * S;
        buf_latent_a_[l] = std::tanh(avx2_dot_product(w_row, current_state.data(), S));
        buf_latent_b_[l] = std::tanh(avx2_dot_product(w_row, next_state_observed.data(), S));
    }

    // M_{t+1} = M_t + eta * (phi_t + gamma * M * phi_{t+1} - M * phi_t)
    float gamma_succ = 0.90f;
    float lr_succ = effective_plasticity_ * 2.0f;
    for (size_t r = 0; r < L; ++r) {
        float* m_row = M_successor_.data() + r * L;
        float m_next = avx2_dot_product(m_row, buf_latent_b_.data(), L);
        float m_curr = avx2_dot_product(m_row, buf_latent_a_.data(), L);
        float td_error = buf_latent_a_[r] + gamma_succ * m_next - m_curr;
        for (size_t c = 0; c < L; ++c) {
            m_row[c] += lr_succ * td_error * buf_latent_a_[c];
            m_row[c] = std::clamp(m_row[c], -3.0f, 3.0f);
        }
    }

    // Goal successor map imprinting:
    if (external_valence > 0.5f) {
        for (size_t l = 0; l < L; ++l) {
            const float* m_row = M_successor_.data() + l * L;
            goal_successor_map_[l] = 0.75f * goal_successor_map_[l] + 0.25f * avx2_dot_product(m_row, buf_latent_b_.data(), L);
        }
    } else {
        for (size_t l = 0; l < L; ++l) {
            const float* w_down = W_sensory_down_.data() + l * S;
            float z_goal = std::tanh(avx2_dot_product(w_down, target_goal.data(), S));
            goal_successor_map_[l] = 0.995f * goal_successor_map_[l] + 0.005f * z_goal;
        }
    }

    // Level 2 Slow Macro-Attractor Evolution:
    macro_subgoal_timer_ += 1.0f;
    if (macro_subgoal_timer_ > 40.0f || stagnation_potential_ > 0.35f) {
        macro_subgoal_timer_ = 0.0f;
        if (stagnation_potential_ > 0.25f) {
            std::normal_distribution<float> macro_dist(0.0f, 0.7f);
            for (size_t l = 0; l < L; ++l) {
                macro_subgoal_latent_[l] = std::tanh(-buf_latent_a_[l] + macro_dist(rng_));
            }
        } else {
            for (size_t l = 0; l < L; ++l) {
                macro_subgoal_latent_[l] = 0.65f * macro_subgoal_latent_[l] + 0.35f * goal_successor_map_[l];
            }
        }
    }

    // Unbounded Episodic Memory Recording:
    // Encode current transition outcome into non-parametric memory
    global_step_counter_++;
    std::vector<float> mem_val(config_.action_dim + 2, 0.0f);
    mem_val[0] = external_valence;
    if (discrete_action < config_.action_dim) {
        mem_val[1 + discrete_action] = 1.0f;
    }
    // High utility if state yielded non-zero valence or high surprise
    float mem_util = std::abs(external_valence) * 2.0f + 0.1f;
    episodic_memory_.record(buf_latent_a_, mem_val, mem_util, global_step_counter_);

    auto telem = step_environment(current_state, target_goal, next_state_observed, continuous_act, external_valence);
    telem.action_discrete_chosen = discrete_action;
    telem.episodic_memory_slots = episodic_memory_.size();
    auto cur_q = episodic_memory_.query(buf_latent_a_);
    telem.episodic_familiarity = cur_q.familiarity;
    return telem;
}

void KaryonSovereignCore::evaluate_autopoietic_morphogenesis(float fe_error) {
    ticks_since_last_sprout_++;
    ticks_since_last_hormone_sprout_++;
    ticks_since_last_latent_sprout_++;

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

    // 3. Sovereign Latent Manifold Morphogenesis (Dimensional Sprouting)
    // If prediction error remains high and persistent despite primitive and hormonal adaptations,
    // the compression bottleneck is too tight: sprout latent dimension!
    if (ticks_since_last_latent_sprout_ > (config_.sprouting_cooldown_ticks * 3)) {
        if (moving_fe_ema_ > 0.04f && current_latent_dim_ < config_.max_latent_dim) {
            trigger_latent_dimension_sprout();
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
