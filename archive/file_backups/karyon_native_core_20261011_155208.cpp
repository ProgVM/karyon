// karyon_native_core.cpp — Pure C++20 Sovereign Substrate Core Engine Implementation
// Standard: KEP v16.0 Sovereign Master & EXP-456 Master Autopoiesis
// Zero-Python / Zero-Torch / Pure Native C++20 Substrate Architecture

#include "karyon_native_core.h"
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>

// Optional Rust FFI declarations with weak linkage or dynamic dlopen stubs
extern "C" {
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((weak)) void* karyon_rust_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v, size_t dim_emb) { return nullptr; }
    __attribute__((weak)) void karyon_rust_destroy(void* handle) {}
    __attribute__((weak)) void karyon_rust_reset(void* handle) {}
    __attribute__((weak)) void karyon_rust_associative_read(const void* handle, const float* q_heads, float* read_out) {}
    __attribute__((weak)) void karyon_rust_apply_laws(void* handle, const float* k_heads, const float* v_heads, 
                                const float* v_pred_heads, const float* alphas, 
                                const float* etas_delta, const float* etas_hebb, const float* lambdas) {}
#else
    void* karyon_rust_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v, size_t dim_emb);
    void karyon_rust_destroy(void* handle);
    void karyon_rust_reset(void* handle);
    void karyon_rust_associative_read(const void* handle, const float* q_heads, float* read_out);
    void karyon_rust_apply_laws(void* handle, const float* k_heads, const float* v_heads, 
                                const float* v_pred_heads, const float* alphas, 
                                const float* etas_delta, const float* etas_hebb, const float* lambdas);
#endif
}

namespace karyon {

SovereignNativeEngine::SovereignNativeEngine(const NativeConfig& config)
    : config_(config), adam_t_(0) {
    init_parameters();
    if (config_.use_rust_backend) {
        rust_handle_ = karyon_rust_create(config_.dim, config_.num_heads, config_.dim_k, config_.dim_v, config_.dim_emb);
    }
}

SovereignNativeEngine::~SovereignNativeEngine() {
    if (rust_handle_) {
        karyon_rust_destroy(rust_handle_);
        rust_handle_ = nullptr;
    }
}

void SovereignNativeEngine::reset_state() {
    std::fill(S_memory_.begin(), S_memory_.end(), 0.0f);
    if (rust_handle_) {
        karyon_rust_reset(rust_handle_);
    }
}

void SovereignNativeEngine::init_parameters() {
    std::mt19937 rng(42);
    std::normal_distribution<float> d_emb(0.0f, 0.05f);
    
    byte_emb_.resize(config_.dim * config_.dim_emb);
    for (auto& val : byte_emb_) val = d_emb(rng);

    auto init_proj = [&](std::vector<float>& W, size_t r, size_t c) {
        W.resize(r * c);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(c)));
        for (auto& val : W) val = dist(rng);
    };

    init_proj(W_q_, config_.num_heads * config_.dim_k, config_.dim_emb);
    init_proj(W_k_, config_.num_heads * config_.dim_k, config_.dim_emb);
    init_proj(W_v_, config_.num_heads * config_.dim_v, config_.dim_emb);
    init_proj(W_feedback_, config_.dim_k, config_.dim_v);
    init_proj(W_out_, config_.dim, config_.num_heads * config_.dim_v);

    m_out_.assign(W_out_.size(), 0.0f);
    v_out_.assign(W_out_.size(), 0.0f);

    // Law Synthesizer MLP: 4 -> 32 -> 4
    W_law1_.resize(32 * 4);
    b_law1_.assign(32, 0.0f);
    std::normal_distribution<float> d_law(0.0f, 0.1f);
    for (auto& val : W_law1_) val = d_law(rng);

    W_law2_.resize(4 * 32);
    b_law2_.assign(4, 0.0f);
    for (auto& val : W_law2_) val = d_law(rng);

    S_memory_.assign(config_.num_heads * config_.dim_k * config_.dim_v, 0.0f);
}

StepMetrics SovereignNativeEngine::step(uint8_t byte_in, uint8_t byte_target) {
    adam_t_++;
    StepMetrics metrics;

    // 1. Forward Embedding
    const float* emb_ptr = byte_emb_.data() + byte_in * config_.dim_emb;
    std::vector<float> emb(config_.dim_emb);
    for (size_t e = 0; e < config_.dim_emb; ++e) emb[e] = emb_ptr[e];

    // Multi-head projections
    std::vector<float> q_proj(config_.num_heads * config_.dim_k);
    std::vector<float> k_proj(config_.num_heads * config_.dim_k);
    std::vector<float> v_proj(config_.num_heads * config_.dim_v);
    matvec_mul(W_q_.data(), emb.data(), q_proj.data(), q_proj.size(), config_.dim_emb);
    matvec_mul(W_k_.data(), emb.data(), k_proj.data(), k_proj.size(), config_.dim_emb);
    matvec_mul(W_v_.data(), emb.data(), v_proj.data(), v_proj.size(), config_.dim_emb);

    std::vector<float> q_heads = q_proj;
    normalize_heads(q_heads.data(), config_.num_heads, config_.dim_k);

    // 2. Associative Memory Read
    std::vector<float> read_heads(config_.num_heads * config_.dim_v, 0.0f);
    if (rust_handle_) {
        karyon_rust_associative_read(rust_handle_, q_heads.data(), read_heads.data());
    } else {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* qh = q_heads.data() + h * config_.dim_k;
            float* rh = read_heads.data() + h * config_.dim_v;
            const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            for (size_t k = 0; k < config_.dim_k; ++k) {
                float qv = qh[k];
                const float* srow = sh + k * config_.dim_v;
                for (size_t v = 0; v < config_.dim_v; ++v) rh[v] += qv * srow[v];
            }
        }
    }

    // 3. Deliberative Dissipative Recurrent Thinking
    size_t turns_taken = 1;
    float g_0 = 0.85f;
    std::vector<float> q_curr = q_heads;
    std::vector<float> read_flat = read_heads;
    float read_norm = 0.0f;
    float beta_val = 1.0f;
    std::vector<float> logits(config_.dim, 0.0f);

    for (size_t turn = 0; turn < config_.max_thinking_turns; ++turn) {
        read_flat = read_heads;
        read_norm = compute_l2_norm(read_flat.data(), read_flat.size());

        float u_beta = std::clamp(w_beta_ * read_norm + b_beta_, -20.0f, 20.0f);
        beta_val = 1.0f + std::log1p(std::exp(u_beta));

        matvec_mul(W_out_.data(), read_flat.data(), logits.data(), config_.dim, read_flat.size());
        for (size_t d = 0; d < config_.dim; ++d) logits[d] *= beta_val;

        if (turn == config_.max_thinking_turns - 1) break;

        float curr_entropy = compute_entropy_from_logits(logits.data(), config_.dim);
        if (curr_entropy <= config_.entropy_satisfy) break;

        // Feedback impulse
        float g_step = g_0 * std::pow(config_.gamma_dissipation, static_cast<float>(turn));
        std::vector<float> fb(config_.num_heads * config_.dim_k, 0.0f);
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* rh = read_heads.data() + h * config_.dim_v;
            float* fbh = fb.data() + h * config_.dim_k;
            matvec_mul(W_feedback_.data(), rh, fbh, config_.dim_k, config_.dim_v);
            for (size_t k = 0; k < config_.dim_k; ++k) fbh[k] = std::tanh(fbh[k]);
        }

        float impulse_sq = 0.0f;
        for (size_t i = 0; i < fb.size(); ++i) {
            float imp = g_step * fb[i];
            q_curr[i] += imp;
            impulse_sq += imp * imp;
        }
        if (std::sqrt(impulse_sq) < config_.eps_halt) break;

        normalize_heads(q_curr.data(), config_.num_heads, config_.dim_k);
        if (rust_handle_) {
            karyon_rust_associative_read(rust_handle_, q_curr.data(), read_heads.data());
        } else {
            std::fill(read_heads.begin(), read_heads.end(), 0.0f);
            for (size_t h = 0; h < config_.num_heads; ++h) {
                const float* qh = q_curr.data() + h * config_.dim_k;
                float* rh = read_heads.data() + h * config_.dim_v;
                const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
                for (size_t k = 0; k < config_.dim_k; ++k) {
                    float qv = qh[k];
                    const float* srow = sh + k * config_.dim_v;
                    for (size_t v = 0; v < config_.dim_v; ++v) rh[v] += qv * srow[v];
                }
            }
        }
        turns_taken++;
    }

    // Softmax & Surprise
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(config_.dim);
    for (size_t d = 0; d < config_.dim; ++d) {
        probs[d] = std::exp(logits[d] - max_l);
        sum_exp += probs[d];
    }
    float entropy = 0.0f;
    for (size_t d = 0; d < config_.dim; ++d) {
        probs[d] /= sum_exp;
        if (probs[d] > 1e-12f) entropy -= probs[d] * std::log(probs[d]);
    }

    float p_target = std::max(probs[byte_target], 1e-7f);
    float loss = -std::log(p_target);
    uint8_t pred = static_cast<uint8_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));

    // 4. Multi-Head Law Synthesis
    std::vector<float> v_pred(config_.num_heads * config_.dim_v, 0.0f);
    for (size_t h = 0; h < config_.num_heads; ++h) {
        const float* kh = k_proj.data() + h * config_.dim_k;
        float* vph = v_pred.data() + h * config_.dim_v;
        const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
        for (size_t k = 0; k < config_.dim_k; ++k) {
            float kv = kh[k];
            const float* srow = sh + k * config_.dim_v;
            for (size_t v = 0; v < config_.dim_v; ++v) vph[v] += kv * srow[v];
        }
    }

    std::vector<float> features(config_.num_heads * 4);
    for (size_t h = 0; h < config_.num_heads; ++h) {
        const float* vh = v_proj.data() + h * config_.dim_v;
        const float* vph = v_pred.data() + h * config_.dim_v;
        const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;

        float err_sq = 0.0f, s_sq = 0.0f, coher = 0.0f;
        for (size_t v = 0; v < config_.dim_v; ++v) {
            float diff = vh[v] - vph[v];
            err_sq += diff * diff;
            coher += vh[v] * vph[v];
        }
        for (size_t i = 0; i < config_.dim_k * config_.dim_v; ++i) s_sq += sh[i] * sh[i];

        features[h * 4 + 0] = std::sqrt(err_sq);
        features[h * 4 + 1] = std::sqrt(s_sq) / 10.0f;
        features[h * 4 + 2] = std::min(loss / 3.0f, 3.0f);
        features[h * 4 + 3] = coher;
    }

    std::vector<float> alphas(config_.num_heads);
    std::vector<float> etas_delta(config_.num_heads);
    std::vector<float> etas_hebb(config_.num_heads);
    std::vector<float> lambdas(config_.num_heads);

    for (size_t h = 0; h < config_.num_heads; ++h) {
        const float* feat = features.data() + h * 4;
        std::vector<float> h_law(32, 0.0f);
        matvec_mul(W_law1_.data(), feat, h_law.data(), 32, 4);
        for (size_t i = 0; i < 32; ++i) {
            h_law[i] += b_law1_[i];
            float sig = 1.0f / (1.0f + std::exp(-std::clamp(h_law[i], -20.0f, 20.0f)));
            h_law[i] *= sig;
        }

        std::vector<float> ol(4, 0.0f);
        matvec_mul(W_law2_.data(), h_law.data(), ol.data(), 4, 32);
        for (size_t i = 0; i < 4; ++i) ol[i] += b_law2_[i];

        alphas[h] = std::clamp(1.0f / (1.0f + std::exp(-ol[0])), 0.70f, 0.999f);
        etas_delta[h] = std::log1p(std::exp(std::clamp(ol[1], -20.0f, 20.0f))) + 0.1f;
        etas_hebb[h] = std::log1p(std::exp(std::clamp(ol[2], -20.0f, 20.0f)));
        lambdas[h] = 0.05f / (1.0f + std::exp(-ol[3]));
    }

    // Apply into Memory (C++ or Rust)
    if (rust_handle_) {
        karyon_rust_apply_laws(rust_handle_, k_proj.data(), v_proj.data(), v_pred.data(),
                               alphas.data(), etas_delta.data(), etas_hebb.data(), lambdas.data());
    } else {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            float alpha = alphas[h];
            float eta_d = etas_delta[h];
            float eta_h = etas_hebb[h];
            float lam = lambdas[h];

            const float* kh = k_proj.data() + h * config_.dim_k;
            const float* vh = v_proj.data() + h * config_.dim_v;
            const float* vph = v_pred.data() + h * config_.dim_v;
            float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;

            for (size_t k = 0; k < config_.dim_k; ++k) {
                float kval = kh[k];
                float* srow = sh + k * config_.dim_v;
                for (size_t v = 0; v < config_.dim_v; ++v) {
                    float err_val = vh[v] - vph[v];
                    srow[v] = alpha * srow[v] + eta_d * (kval * err_val) + eta_h * (kval * vh[v]) - lam * srow[v];
                }
            }
        }

        // Norm clamp
        float s_total_sq = 0.0f;
        for (size_t i = 0; i < S_memory_.size(); ++i) s_total_sq += S_memory_[i] * S_memory_[i];
        float norm_s = std::sqrt(s_total_sq);
        if (norm_s > 30.0f) {
            float scale = 30.0f / norm_s;
            for (size_t i = 0; i < S_memory_.size(); ++i) S_memory_[i] *= scale;
        }
    }

    // 5. Output Weights Optimization
    std::vector<float> d_logits = probs;
    d_logits[byte_target] -= 1.0f;

    std::vector<float> grad_out(W_out_.size(), 0.0f);
    for (size_t r = 0; r < config_.dim; ++r) {
        float dl = d_logits[r];
        for (size_t c = 0; c < read_flat.size(); ++c) {
            grad_out[r * read_flat.size() + c] = beta_val * dl * read_flat[c];
        }
    }
    adamw_step(W_out_, grad_out, m_out_, v_out_, config_.lr);

    float d_beta = 0.0f;
    for (size_t d = 0; d < config_.dim; ++d) d_beta += (logits[d] / beta_val) * d_logits[d];
    float sig_u = 1.0f / (1.0f + std::exp(-std::clamp(w_beta_ * read_norm + b_beta_, -20.0f, 20.0f)));
    float d_u = d_beta * sig_u;

    m_w_beta_ = 0.9f * m_w_beta_ + 0.1f * (d_u * read_norm);
    v_w_beta_ = 0.999f * v_w_beta_ + 0.001f * (d_u * read_norm) * (d_u * read_norm);
    w_beta_ -= 0.015f * m_w_beta_ / (std::sqrt(v_w_beta_) + 1e-8f);

    m_b_beta_ = 0.9f * m_b_beta_ + 0.1f * d_u;
    v_b_beta_ = 0.999f * v_b_beta_ + 0.001f * d_u * d_u;
    b_beta_ -= 0.015f * m_b_beta_ / (std::sqrt(v_b_beta_) + 1e-8f);

    metrics.loss = loss;
    metrics.pred_byte = pred;
    metrics.is_correct = (pred == byte_target);
    metrics.turns_taken = turns_taken;
    metrics.entropy = entropy;
    metrics.beta = beta_val;
    metrics.mean_alpha = std::accumulate(alphas.begin(), alphas.end(), 0.0f) / config_.num_heads;
    metrics.mean_eta_delta = std::accumulate(etas_delta.begin(), etas_delta.end(), 0.0f) / config_.num_heads;
    metrics.mean_eta_hebb = std::accumulate(etas_hebb.begin(), etas_hebb.end(), 0.0f) / config_.num_heads;
    metrics.s_norm = read_norm;

    return metrics;
}

uint8_t SovereignNativeEngine::sample_next(uint8_t byte_in, float temperature) {
    const float* emb_ptr = byte_emb_.data() + byte_in * config_.dim_emb;
    std::vector<float> emb(config_.dim_emb);
    for (size_t e = 0; e < config_.dim_emb; ++e) emb[e] = emb_ptr[e];

    std::vector<float> q_proj(config_.num_heads * config_.dim_k);
    matvec_mul(W_q_.data(), emb.data(), q_proj.data(), q_proj.size(), config_.dim_emb);
    normalize_heads(q_proj.data(), config_.num_heads, config_.dim_k);

    std::vector<float> read_heads(config_.num_heads * config_.dim_v, 0.0f);
    if (rust_handle_) {
        karyon_rust_associative_read(rust_handle_, q_proj.data(), read_heads.data());
    } else {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* qh = q_proj.data() + h * config_.dim_k;
            float* rh = read_heads.data() + h * config_.dim_v;
            const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            for (size_t k = 0; k < config_.dim_k; ++k) {
                float qv = qh[k];
                const float* srow = sh + k * config_.dim_v;
                for (size_t v = 0; v < config_.dim_v; ++v) rh[v] += qv * srow[v];
            }
        }
    }

    std::vector<float> logits(config_.dim, 0.0f);
    matvec_mul(W_out_.data(), read_heads.data(), logits.data(), config_.dim, read_heads.size());

    // Temperature scale
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(config_.dim);
    for (size_t d = 0; d < config_.dim; ++d) {
        probs[d] = std::exp((logits[d] - max_l) / std::max(temperature, 0.01f));
        sum_exp += probs[d];
    }
    for (size_t d = 0; d < config_.dim; ++d) probs[d] /= sum_exp;

    return static_cast<uint8_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

void SovereignNativeEngine::matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const float* row = W + r * cols;
        for (size_t c = 0; c < cols; ++c) sum += row[c] * x[c];
        y[r] = sum;
    }
}

void SovereignNativeEngine::normalize_heads(float* ptr, size_t num_heads, size_t head_dim) {
    for (size_t h = 0; h < num_heads; ++h) {
        float* head_ptr = ptr + h * head_dim;
        float sq = 0.0f;
        for (size_t i = 0; i < head_dim; ++i) sq += head_ptr[i] * head_ptr[i];
        float inv = 1.0f / (std::sqrt(sq) + 1e-8f);
        for (size_t i = 0; i < head_dim; ++i) head_ptr[i] *= inv;
    }
}

float SovereignNativeEngine::compute_l2_norm(const float* ptr, size_t n) {
    float sq = 0.0f;
    for (size_t i = 0; i < n; ++i) sq += ptr[i] * ptr[i];
    return std::sqrt(sq);
}

float SovereignNativeEngine::compute_entropy_from_logits(const float* logits, size_t n) {
    float max_l = *std::max_element(logits, logits + n);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) sum_exp += std::exp(logits[i] - max_l);
    float ent = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float p = std::exp(logits[i] - max_l) / sum_exp;
        if (p > 1e-12f) ent -= p * std::log(p);
    }
    return ent;
}

void SovereignNativeEngine::adamw_step(std::vector<float>& W, const std::vector<float>& grad, 
                                      std::vector<float>& m, std::vector<float>& v, 
                                      float lr, float weight_decay) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - std::pow(b1, static_cast<float>(adam_t_));
    float bc2 = 1.0f - std::pow(b2, static_cast<float>(adam_t_));

    for (size_t i = 0; i < W.size(); ++i) {
        float g = std::clamp(grad[i], -1.0f, 1.0f);
        W[i] -= lr * weight_decay * W[i];
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        W[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

std::vector<uint8_t> SovereignNativeEngine::serialize_kcore() const {
    std::vector<uint8_t> buf;
    const char magic[] = "KCORE\x06\x00\x00";
    buf.insert(buf.end(), magic, magic + 8);
    
    auto append_vec = [&](const std::vector<float>& v) {
        size_t n = v.size();
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&n);
        buf.insert(buf.end(), p, p + sizeof(n));
        const uint8_t* vp = reinterpret_cast<const uint8_t*>(v.data());
        buf.insert(buf.end(), vp, vp + n * sizeof(float));
    };

    append_vec(byte_emb_);
    append_vec(W_q_);
    append_vec(W_k_);
    append_vec(W_v_);
    append_vec(W_feedback_);
    append_vec(W_out_);
    append_vec(W_law1_);
    append_vec(b_law1_);
    append_vec(W_law2_);
    append_vec(b_law2_);
    append_vec(S_memory_);

    return buf;
}

bool SovereignNativeEngine::deserialize_kcore(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 8) return false;
    if (std::memcmp(buffer.data(), "KCORE\x06\x00\x00", 8) != 0) return false;

    size_t offset = 8;
    auto read_vec = [&](std::vector<float>& v) -> bool {
        if (offset + sizeof(size_t) > buffer.size()) return false;
        size_t n = *reinterpret_cast<const size_t*>(buffer.data() + offset);
        offset += sizeof(size_t);
        if (offset + n * sizeof(float) > buffer.size()) return false;
        v.resize(n);
        std::memcpy(v.data(), buffer.data() + offset, n * sizeof(float));
        offset += n * sizeof(float);
        return true;
    };

    if (!read_vec(byte_emb_)) return false;
    if (!read_vec(W_q_)) return false;
    if (!read_vec(W_k_)) return false;
    if (!read_vec(W_v_)) return false;
    if (!read_vec(W_feedback_)) return false;
    if (!read_vec(W_out_)) return false;
    if (!read_vec(W_law1_)) return false;
    if (!read_vec(b_law1_)) return false;
    if (!read_vec(W_law2_)) return false;
    if (!read_vec(b_law2_)) return false;
    if (!read_vec(S_memory_)) return false;

    return true;
}

} // namespace karyon

// C-ABI Safe Exports

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
        if (handle) delete static_cast<karyon::SovereignNativeEngine*>(handle);
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
        if (out_pred) *out_pred = m.pred_byte;
        if (out_turns) *out_turns = m.turns_taken;
        if (out_entropy) *out_entropy = m.entropy;
    }
}
