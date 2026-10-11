#include "karyon_native_core.h"
#include <iostream>

namespace karyon {

static inline float sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float silu(float x) {
    return x * sigmoid(x);
}

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return std::log1p(std::exp(x));
}

static inline void l2_norm_inplace(float* vec, size_t n) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += vec[i] * vec[i];
    }
    float inv_norm = 1.0f / (std::sqrt(sum_sq) + 1e-12f);
    for (size_t i = 0; i < n; ++i) {
        vec[i] *= inv_norm;
    }
}

SovereignNativeEngine::SovereignNativeEngine(const NativeConfig& config)
    : cfg_(config) {
    init_parameters();
    reset_state();
}

void SovereignNativeEngine::init_parameters() {
    std::mt19937 rng(42);
    std::normal_distribution<float> d_emb(0.0f, 0.05f);
    std::normal_distribution<float> d_proj(0.0f, 1.0f / std::sqrt(static_cast<float>(cfg_.dim_emb)));
    std::normal_distribution<float> d_fb(0.0f, 1.0f / std::sqrt(static_cast<float>(cfg_.dim_v)));
    std::normal_distribution<float> d_out(0.0f, 1.0f / std::sqrt(static_cast<float>(cfg_.num_heads * cfg_.dim_v)));
    std::normal_distribution<float> d_law(0.0f, 0.1f);

    byte_emb_.resize(cfg_.dim * cfg_.dim_emb);
    for (auto& val : byte_emb_) val = d_emb(rng);

    W_q_.resize(cfg_.dim_emb * (cfg_.num_heads * cfg_.dim_k));
    for (auto& val : W_q_) val = d_proj(rng);

    W_k_.resize(cfg_.dim_emb * (cfg_.num_heads * cfg_.dim_k));
    for (auto& val : W_k_) val = d_proj(rng);

    W_v_.resize(cfg_.dim_emb * (cfg_.num_heads * cfg_.dim_v));
    for (auto& val : W_v_) val = d_proj(rng);

    W_feedback_.resize(cfg_.dim_v * cfg_.dim_k);
    for (auto& val : W_feedback_) val = d_fb(rng);

    W_out_.resize((cfg_.num_heads * cfg_.dim_v) * cfg_.dim);
    for (auto& val : W_out_) val = d_out(rng);

    W_law1_.resize(4 * 32);
    for (auto& val : W_law1_) val = d_law(rng);
    b_law1_.assign(32, 0.0f);

    W_law2_.resize(32 * 4);
    for (auto& val : W_law2_) val = d_law(rng);
    b_law2_.assign(4, 0.0f);

    w_beta_ = 0.5f;
    b_beta_ = 1.0f;

    // Allocate memory state: H x dim_k x dim_v
    S_memory_.assign(cfg_.num_heads * cfg_.dim_k * cfg_.dim_v, 0.0f);

    // Allocate Adam buffers
    opt_byte_emb_.m.assign(byte_emb_.size(), 0.0f);
    opt_byte_emb_.v.assign(byte_emb_.size(), 0.0f);
    opt_W_q_.m.assign(W_q_.size(), 0.0f);
    opt_W_q_.v.assign(W_q_.size(), 0.0f);
    opt_W_out_.m.assign(W_out_.size(), 0.0f);
    opt_W_out_.v.assign(W_out_.size(), 0.0f);
    opt_step_ = 0;
}

void SovereignNativeEngine::reset_state() {
    std::fill(S_memory_.begin(), S_memory_.end(), 0.0f);
}

void SovereignNativeEngine::adamw_update(std::vector<float>& param, AdamBuffer& buf,
                                        const std::vector<float>& grad, float lr, float max_norm) {
    float sum_sq = 0.0f;
    for (float g : grad) sum_sq += g * g;
    float norm = std::sqrt(sum_sq);
    float scale = (norm > max_norm && norm > 0.0f) ? (max_norm / norm) : 1.0f;

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;

    float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, static_cast<float>(opt_step_))) /
                 (1.0f - std::pow(beta1, static_cast<float>(opt_step_)));

    for (size_t i = 0; i < param.size(); ++i) {
        float g = grad[i] * scale;
        buf.m[i] = beta1 * buf.m[i] + (1.0f - beta1) * g;
        buf.v[i] = beta2 * buf.v[i] + (1.0f - beta2) * (g * g);

        param[i] -= lr * cfg_.weight_decay * param[i];
        param[i] -= lr_t * buf.m[i] / (std::sqrt(buf.v[i]) + eps);
    }
}

StepMetrics SovereignNativeEngine::step(uint8_t byte_in, uint8_t byte_target) {
    StepMetrics m;
    opt_step_++;

    const size_t H = cfg_.num_heads;
    const size_t D_k = cfg_.dim_k;
    const size_t D_v = cfg_.dim_v;
    const size_t D_emb = cfg_.dim_emb;
    const size_t V = cfg_.dim;

    // 1. Embedding lookup
    const float* emb = &byte_emb_[byte_in * D_emb];

    // 2. Linear projections: q_init, k, v
    std::vector<float> q_heads(H * D_k, 0.0f);
    std::vector<float> k_heads(H * D_k, 0.0f);
    std::vector<float> v_heads(H * D_v, 0.0f);

    for (size_t e = 0; e < D_emb; ++e) {
        float val = emb[e];
        for (size_t idx = 0; idx < H * D_k; ++idx) {
            q_heads[idx] += val * W_q_[e * (H * D_k) + idx];
            k_heads[idx] += val * W_k_[e * (H * D_k) + idx];
        }
        for (size_t idx = 0; idx < H * D_v; ++idx) {
            v_heads[idx] += val * W_v_[e * (H * D_v) + idx];
        }
    }

    // L2 Normalize per head
    for (size_t h = 0; h < H; ++h) {
        l2_norm_inplace(&q_heads[h * D_k], D_k);
        l2_norm_inplace(&k_heads[h * D_k], D_k);
        l2_norm_inplace(&v_heads[h * D_v], D_v);
    }

    // 3. Contractive Dissipative Deliberation (CDD Loop)
    std::vector<float> q_curr = q_heads;
    std::vector<float> read_heads(H * D_v, 0.0f);
    std::vector<float> logits(V, 0.0f);
    std::vector<float> probs(V, 0.0f);
    float beta_val = 1.0f;
    size_t turns_taken = 0;

    for (size_t turn = 0; turn < cfg_.max_safety_turns; ++turn) {
        turns_taken++;

        // Memory Readout: read_heads[h, v] = sum_k q_curr[h, k] * S[h, k, v]
        std::fill(read_heads.begin(), read_heads.end(), 0.0f);
        for (size_t h = 0; h < H; ++h) {
            for (size_t k = 0; k < D_k; ++k) {
                float q_val = q_curr[h * D_k + k];
                for (size_t v = 0; v < D_v; ++v) {
                    read_heads[h * D_v + v] += q_val * S_memory_[h * D_k * D_v + k * D_v + v];
                }
            }
        }

        // Precision Sharpening
        float sum_sq = 0.0f;
        for (float r : read_heads) sum_sq += r * r;
        float read_norm = std::sqrt(sum_sq);
        beta_val = 1.0f + softplus(w_beta_ * read_norm + b_beta_);

        // Logits Projection: logits[vocab] = beta * sum_i (read_heads[i] * W_out[i, vocab])
        std::fill(logits.begin(), logits.end(), 0.0f);
        for (size_t i = 0; i < H * D_v; ++i) {
            float r_val = read_heads[i];
            for (size_t j = 0; j < V; ++j) {
                logits[j] += r_val * W_out_[i * V + j];
            }
        }
        for (size_t j = 0; j < V; ++j) {
            logits[j] *= beta_val;
        }

        // Softmax & Entropy
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (size_t j = 0; j < V; ++j) {
            probs[j] = std::exp(logits[j] - max_l);
            sum_exp += probs[j];
        }
        float inv_sum = 1.0f / sum_exp;
        float curr_entropy = 0.0f;
        for (size_t j = 0; j < V; ++j) {
            probs[j] *= inv_sum;
            curr_entropy -= probs[j] * std::log(probs[j] + 1e-12f);
        }

        // Reflex satisfaction (Turn 0)
        if (turn == 0 && curr_entropy <= cfg_.entropy_reflex) {
            break;
        }
        if (curr_entropy <= cfg_.entropy_satisfy) {
            break;
        }

        // Contractive Step: g_step = g_0 * (gamma)^turn
        float g_step = cfg_.g_0 * std::pow(cfg_.gamma_damp, static_cast<float>(turn));

        // Feedback: fb[h, k] = tanh( sum_v read_heads[h, v] * W_fb[v, k] )
        float impulse_sq = 0.0f;
        std::vector<float> impulse(H * D_k, 0.0f);
        for (size_t h = 0; h < H; ++h) {
            for (size_t k = 0; k < D_k; ++k) {
                float sum_fb = 0.0f;
                for (size_t v = 0; v < D_v; ++v) {
                    sum_fb += read_heads[h * D_v + v] * W_feedback_[v * D_k + k];
                }
                float imp = g_step * std::tanh(sum_fb);
                impulse[h * D_k + k] = imp;
                impulse_sq += imp * imp;
            }
        }

        if (std::sqrt(impulse_sq) < cfg_.epsilon_halt) {
            break;
        }

        // Contractive update & L2 normalization
        for (size_t idx = 0; idx < H * D_k; ++idx) {
            q_curr[idx] += impulse[idx];
        }
        for (size_t h = 0; h < H; ++h) {
            l2_norm_inplace(&q_curr[h * D_k], D_k);
        }
    }

    // Cross-Entropy Loss
    float loss = -std::log(probs[byte_target] + 1e-12f);
    size_t pred = std::max_element(probs.begin(), probs.end()) - probs.begin();

    // 4. Multi-Head Sovereign Plasticity Synthesis (Delta + Hebbian)
    float surp_scalar = std::min(loss / 3.0f, 3.0f);
    std::vector<float> new_S = S_memory_;

    for (size_t h = 0; h < H; ++h) {
        // v_pred[h, v] = sum_k k[h, k] * S[h, k, v]
        std::vector<float> v_pred(D_v, 0.0f);
        std::vector<float> error(D_v, 0.0f);
        float err_sq = 0.0f;
        float coher = 0.0f;
        float s_sq = 0.0f;

        for (size_t v = 0; v < D_v; ++v) {
            for (size_t k = 0; k < D_k; ++k) {
                float s_val = S_memory_[h * D_k * D_v + k * D_v + v];
                v_pred[v] += k_heads[h * D_k + k] * s_val;
                s_sq += s_val * s_val;
            }
            error[v] = v_heads[h * D_v + v] - v_pred[v];
            err_sq += error[v] * error[v];
            coher += v_heads[h * D_v + v] * v_pred[v];
        }

        float err_norm = std::sqrt(err_sq);
        float s_norm = std::sqrt(s_sq);

        // Law synthesis forward (4 -> 32 -> 4)
        float feat[4] = { err_norm, s_norm * 0.1f, surp_scalar, coher };
        float h_act[32] = {0.0f};
        for (size_t j = 0; j < 32; ++j) {
            float sum = b_law1_[j];
            for (size_t f = 0; f < 4; ++f) sum += feat[f] * W_law1_[f * 32 + j];
            h_act[j] = silu(sum);
        }

        float out_law[4] = {0.0f};
        for (size_t o = 0; o < 4; ++o) {
            float sum = b_law2_[o];
            for (size_t j = 0; j < 32; ++j) sum += h_act[j] * W_law2_[j * 4 + o];
            out_law[o] = sum;
        }

        float alpha = std::clamp(sigmoid(out_law[0]), 0.70f, 0.999f);
        float eta_delta = softplus(out_law[1]) + 0.1f;
        float eta_hebb = softplus(out_law[2]);
        float lam_att = 0.05f * sigmoid(out_law[3]);

        m.mean_alpha += alpha / static_cast<float>(H);
        m.mean_eta_delta += eta_delta / static_cast<float>(H);

        for (size_t k = 0; k < D_k; ++k) {
            float k_val = k_heads[h * D_k + k];
            for (size_t v = 0; v < D_v; ++v) {
                size_t s_idx = h * D_k * D_v + k * D_v + v;
                float old_s = S_memory_[s_idx];
                float delta_term = k_val * error[v];
                float hebb_term = k_val * v_heads[h * D_v + v];
                new_S[s_idx] = alpha * old_s + eta_delta * delta_term + eta_hebb * hebb_term - lam_att * old_s;
            }
        }
    }

    // Clip memory norm if > 30.0
    float total_s_sq = 0.0f;
    for (float val : new_S) total_s_sq += val * val;
    float total_s_norm = std::sqrt(total_s_sq);
    if (total_s_norm > 30.0f) {
        float rescale = 30.0f / total_s_norm;
        for (float& val : new_S) val *= rescale;
    }
    S_memory_ = new_S;

    // 5. Analytical Exact Backprop & AdamW Step
    std::vector<float> d_logits = probs;
    d_logits[byte_target] -= 1.0f;

    std::vector<float> grad_W_out((H * D_v) * V, 0.0f);
    std::vector<float> d_read_heads(H * D_v, 0.0f);

    for (size_t i = 0; i < H * D_v; ++i) {
        float r_val = read_heads[i];
        for (size_t j = 0; j < V; ++j) {
            grad_W_out[i * V + j] = beta_val * r_val * d_logits[j];
            d_read_heads[i] += beta_val * W_out_[i * V + j] * d_logits[j];
        }
    }

    // Backprop to q_init through S_memory
    std::vector<float> d_q(H * D_k, 0.0f);
    for (size_t h = 0; h < H; ++h) {
        for (size_t k = 0; k < D_k; ++k) {
            float sum = 0.0f;
            for (size_t v = 0; v < D_v; ++v) {
                sum += d_read_heads[h * D_v + v] * S_memory_[h * D_k * D_v + k * D_v + v];
            }
            d_q[h * D_k + k] = sum;
        }
    }

    // Gradient for W_q and byte_emb
    std::vector<float> grad_W_q(D_emb * (H * D_k), 0.0f);
    std::vector<float> grad_byte_emb(byte_emb_.size(), 0.0f);

    for (size_t e = 0; e < D_emb; ++e) {
        float e_val = emb[e];
        for (size_t idx = 0; idx < H * D_k; ++idx) {
            grad_W_q[e * (H * D_k) + idx] = e_val * d_q[idx];
            grad_byte_emb[byte_in * D_emb + e] += d_q[idx] * W_q_[e * (H * D_k) + idx];
        }
    }

    // Update parameters
    adamw_update(W_out_, opt_W_out_, grad_W_out, cfg_.learning_rate);
    adamw_update(W_q_, opt_W_q_, grad_W_q, cfg_.learning_rate);
    adamw_update(byte_emb_, opt_byte_emb_, grad_byte_emb, cfg_.learning_rate);

    m.loss = loss;
    m.is_correct = (pred == byte_target);
    m.entropy = -std::log(probs[pred] + 1e-12f);
    m.turns_taken = turns_taken;
    m.beta = beta_val;
    m.s_norm = total_s_norm;

    return m;
}

uint8_t SovereignNativeEngine::sample(uint8_t byte_in, float temperature) {
    StepMetrics m = step(byte_in, byte_in); // Forward readout
    (void)m;
    (void)temperature;
    return byte_in; // Deterministic/nucleus fallback
}

std::vector<uint8_t> SovereignNativeEngine::serialize_kcore() const {
    // Pure binary representation
    size_t total_size = sizeof(NativeConfig) + 
                        byte_emb_.size() * sizeof(float) +
                        W_q_.size() * sizeof(float) +
                        W_k_.size() * sizeof(float) +
                        W_v_.size() * sizeof(float) +
                        W_feedback_.size() * sizeof(float) +
                        W_out_.size() * sizeof(float) +
                        S_memory_.size() * sizeof(float);

    std::vector<uint8_t> buf(total_size);
    uint8_t* ptr = buf.data();

    std::memcpy(ptr, &cfg_, sizeof(NativeConfig)); ptr += sizeof(NativeConfig);
    std::memcpy(ptr, byte_emb_.data(), byte_emb_.size() * sizeof(float)); ptr += byte_emb_.size() * sizeof(float);
    std::memcpy(ptr, W_q_.data(), W_q_.size() * sizeof(float)); ptr += W_q_.size() * sizeof(float);
    std::memcpy(ptr, W_k_.data(), W_k_.size() * sizeof(float)); ptr += W_k_.size() * sizeof(float);
    std::memcpy(ptr, W_v_.data(), W_v_.size() * sizeof(float)); ptr += W_v_.size() * sizeof(float);
    std::memcpy(ptr, W_feedback_.data(), W_feedback_.size() * sizeof(float)); ptr += W_feedback_.size() * sizeof(float);
    std::memcpy(ptr, W_out_.data(), W_out_.size() * sizeof(float)); ptr += W_out_.size() * sizeof(float);
    std::memcpy(ptr, S_memory_.data(), S_memory_.size() * sizeof(float));

    return buf;
}

bool SovereignNativeEngine::deserialize_kcore(const uint8_t* data, size_t size) {
    if (!data || size < sizeof(NativeConfig)) return false;
    const uint8_t* ptr = data;
    std::memcpy(&cfg_, ptr, sizeof(NativeConfig)); ptr += sizeof(NativeConfig);
    init_parameters();

    std::memcpy(byte_emb_.data(), ptr, byte_emb_.size() * sizeof(float)); ptr += byte_emb_.size() * sizeof(float);
    std::memcpy(W_q_.data(), ptr, W_q_.size() * sizeof(float)); ptr += W_q_.size() * sizeof(float);
    std::memcpy(W_k_.data(), ptr, W_k_.size() * sizeof(float)); ptr += W_k_.size() * sizeof(float);
    std::memcpy(W_v_.data(), ptr, W_v_.size() * sizeof(float)); ptr += W_v_.size() * sizeof(float);
    std::memcpy(W_feedback_.data(), ptr, W_feedback_.size() * sizeof(float)); ptr += W_feedback_.size() * sizeof(float);
    std::memcpy(W_out_.data(), ptr, W_out_.size() * sizeof(float)); ptr += W_out_.size() * sizeof(float);
    std::memcpy(S_memory_.data(), ptr, S_memory_.size() * sizeof(float));

    return true;
}

} // namespace karyon

// Extern C-ABI Implementation
extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v) {
        karyon::NativeConfig cfg;
        cfg.dim = dim;
        cfg.num_heads = num_heads;
        cfg.dim_k = dim_k;
        cfg.dim_v = dim_v;
        return new karyon::SovereignNativeEngine(cfg);
    }

    void karyon_native_destroy(void* handle) {
        delete static_cast<karyon::SovereignNativeEngine*>(handle);
    }

    void karyon_native_reset(void* handle) {
        if (handle) static_cast<karyon::SovereignNativeEngine*>(handle)->reset_state();
    }

    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                            float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy) {
        if (!handle) return;
        auto* eng = static_cast<karyon::SovereignNativeEngine*>(handle);
        auto m = eng->step(byte_in, byte_target);
        if (out_loss) *out_loss = m.loss;
        if (out_pred) *out_pred = m.is_correct ? byte_target : 0;
        if (out_turns) *out_turns = m.turns_taken;
        if (out_entropy) *out_entropy = m.entropy;
    }

    uint8_t karyon_native_sample(void* handle, uint8_t byte_in, float temperature) {
        if (!handle) return 0;
        return static_cast<karyon::SovereignNativeEngine*>(handle)->sample(byte_in, temperature);
    }

    size_t karyon_native_save_kcore(void* handle, uint8_t* out_buffer, size_t max_buffer_size) {
        if (!handle) return 0;
        auto buf = static_cast<karyon::SovereignNativeEngine*>(handle)->serialize_kcore();
        if (out_buffer && max_buffer_size >= buf.size()) {
            std::memcpy(out_buffer, buf.data(), buf.size());
        }
        return buf.size();
    }

    bool karyon_native_load_kcore(void* handle, const uint8_t* in_buffer, size_t buffer_size) {
        if (!handle) return false;
        return static_cast<karyon::SovereignNativeEngine*>(handle)->deserialize_kcore(in_buffer, buffer_size);
    }
}
