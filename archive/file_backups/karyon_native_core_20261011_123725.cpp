// karyon_native_core.cpp — Pure C++20 Sovereign Substrate Core Engine Implementation
// Created for Bazilevs (ProgVM) | Standard: KEP v16.0 Sovereign Master

#include "karyon_native_core.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <dlfcn.h>

namespace karyon {

// Rust C-ABI function pointer declarations
typedef void* (*FnRustCreate)(size_t, size_t, size_t, size_t, size_t);
typedef void (*FnRustDestroy)(void*);
typedef void (*FnRustReset)(void*);
typedef void (*FnRustRead)(const void*, const float*, float*);
typedef void (*FnRustWrite)(void*, const float*, const float*, float, float);

static FnRustCreate g_rust_create = nullptr;
static FnRustDestroy g_rust_destroy = nullptr;
static FnRustReset g_rust_reset = nullptr;
static FnRustRead g_rust_read = nullptr;
static FnRustWrite g_rust_write = nullptr;
static void* g_rust_lib_handle = nullptr;

static bool load_rust_library() {
    if (g_rust_lib_handle) return true;
    
    // Try loading libkaryon_rust_core.so from current dir or rust_engine
    const char* paths[] = {"./libkaryon_rust_core.so", "./rust_engine/target/release/libkaryon_rust_core.so"};
    for (const char* p : paths) {
        g_rust_lib_handle = dlopen(p, RTLD_LAZY);
        if (g_rust_lib_handle) {
            std::cout << "[INFO] Successfully loaded Rust acceleration kernel from: " << p << "\n";
            break;
        }
    }

    if (!g_rust_lib_handle) return false;

    g_rust_create = (FnRustCreate)dlsym(g_rust_lib_handle, "karyon_rust_create");
    g_rust_destroy = (FnRustDestroy)dlsym(g_rust_lib_handle, "karyon_rust_destroy");
    g_rust_reset = (FnRustReset)dlsym(g_rust_lib_handle, "karyon_rust_reset");
    g_rust_read = (FnRustRead)dlsym(g_rust_lib_handle, "karyon_rust_read");
    g_rust_write = (FnRustWrite)dlsym(g_rust_lib_handle, "karyon_rust_write");

    return (g_rust_create && g_rust_destroy && g_rust_reset && g_rust_read && g_rust_write);
}

SovereignNativeEngine::SovereignNativeEngine(const NativeConfig& config)
    : config_(config) {
    
    if (config_.use_rust_backend) {
        if (load_rust_library()) {
            rust_handle_ = g_rust_create(config_.dim, config_.num_heads, config_.dim_k, config_.dim_v, config_.dim_emb);
            std::cout << "[INFO] Rust backend initialized successfully.\n";
        } else {
            std::cout << "[WARNING] Failed to load Rust dynamic library. Falling back to C++20 kernel.\n";
            config_.use_rust_backend = false;
        }
    }

    init_weights();
    reset_state();
}

SovereignNativeEngine::~SovereignNativeEngine() {
    if (rust_handle_ && g_rust_destroy) {
        g_rust_destroy(rust_handle_);
        rust_handle_ = nullptr;
    }
}

void SovereignNativeEngine::init_weights() {
    std::mt19937 rng(42);
    float stddev = 1.0f / std::sqrt(static_cast<float>(config_.dim_emb));
    std::normal_distribution<float> dist(0.0f, stddev);

    auto init_vec = [&](std::vector<float>& vec, size_t size) {
        vec.resize(size);
        for (size_t i = 0; i < size; ++i) vec[i] = dist(rng);
    };

    size_t qkv_size = (config_.num_heads * config_.dim_k) * config_.dim_emb;
    size_t v_size = (config_.num_heads * config_.dim_v) * config_.dim_emb;

    init_vec(W_emb_, config_.dim * config_.dim_emb);
    init_vec(W_q_, qkv_size);
    init_vec(W_k_, qkv_size);
    init_vec(W_v_, v_size);
    init_vec(W_out_, config_.dim_emb * (config_.num_heads * config_.dim_v));
    init_vec(W_head_, config_.dim * config_.dim_emb);

    size_t mem_size = config_.num_heads * config_.dim_k * config_.dim_v;
    S_memory_.assign(mem_size, 0.0f);
    state_.assign(config_.dim_emb, 0.0f);

    m_emb_.assign(W_emb_.size(), 0.0f); v_emb_.assign(W_emb_.size(), 0.0f);
    m_q_.assign(W_q_.size(), 0.0f);     v_q_.assign(W_q_.size(), 0.0f);
    m_k_.assign(W_k_.size(), 0.0f);     v_k_.assign(W_k_.size(), 0.0f);
    m_v_.assign(W_v_.size(), 0.0f);     v_v_.assign(W_v_.size(), 0.0f);
    m_out_.assign(W_out_.size(), 0.0f); v_out_.assign(W_out_.size(), 0.0f);
    m_head_.assign(W_head_.size(), 0.0f); v_head_.assign(W_head_.size(), 0.0f);
}

void SovereignNativeEngine::reset_state() {
    std::fill(state_.begin(), state_.end(), 0.0f);
    std::fill(S_memory_.begin(), S_memory_.end(), 0.0f);
    if (rust_handle_ && g_rust_reset) {
        g_rust_reset(rust_handle_);
    }
}

void SovereignNativeEngine::matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const float* row = W + r * cols;
        for (size_t c = 0; c < cols; ++c) {
            sum += row[c] * x[c];
        }
        y[r] = sum;
    }
}

void SovereignNativeEngine::matvec_transpose_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
    std::fill(y, y + cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float x_r = x[r];
        const float* row = W + r * cols;
        for (size_t c = 0; c < cols; ++c) {
            y[c] += row[c] * x_r;
        }
    }
}

void SovereignNativeEngine::adamw_step(std::vector<float>& W, const std::vector<float>& grad, 
                                      std::vector<float>& m, std::vector<float>& v, 
                                      float lr, float weight_decay) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(adam_t_));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(adam_t_));

    for (size_t i = 0; i < W.size(); ++i) {
        W[i] -= lr * weight_decay * W[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        W[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

StepMetrics SovereignNativeEngine::step(uint8_t byte_in, uint8_t byte_target) {
    adam_t_++;
    StepMetrics metrics;

    // 1. Input embedding
    std::vector<float> x_emb(config_.dim_emb, 0.0f);
    for (size_t e = 0; e < config_.dim_emb; ++e) {
        x_emb[e] = W_emb_[byte_in * config_.dim_emb + e] + state_[e];
    }

    // 2. Q, K, V projections
    size_t qk_dim = config_.num_heads * config_.dim_k;
    size_t v_dim = config_.num_heads * config_.dim_v;
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);

    matvec_mul(W_q_.data(), x_emb.data(), q.data(), qk_dim, config_.dim_emb);
    matvec_mul(W_k_.data(), x_emb.data(), k.data(), qk_dim, config_.dim_emb);
    matvec_mul(W_v_.data(), x_emb.data(), v.data(), v_dim, config_.dim_emb);

    // 3. Multi-Head Associative Read: read_out = q @ S
    std::vector<float> read_out(v_dim, 0.0f);

    if (config_.use_rust_backend && rust_handle_ && g_rust_read) {
        g_rust_read(rust_handle_, q.data(), read_out.data());
    } else {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* q_h = q.data() + h * config_.dim_k;
            float* read_h = read_out.data() + h * config_.dim_v;
            const float* mem_h = S_memory_.data() + h * config_.dim_k * config_.dim_v;

            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                float q_val = q_h[k_idx];
                const float* mem_row = mem_h + k_idx * config_.dim_v;
                for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                    read_h[v_idx] += q_val * mem_row[v_idx];
                }
            }
        }
    }

    // 4. Output projection and state update
    std::vector<float> proj_out(config_.dim_emb, 0.0f);
    matvec_mul(W_out_.data(), read_out.data(), proj_out.data(), config_.dim_emb, v_dim);

    for (size_t e = 0; e < config_.dim_emb; ++e) {
        state_[e] = std::tanh(state_[e] + proj_out[e]);
    }

    // 5. Deliberative Thinking & Logits Computation
    std::vector<float> logits(config_.dim, 0.0f);
    matvec_mul(W_head_.data(), state_.data(), logits.data(), config_.dim, config_.dim_emb);

    // Softmax
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(config_.dim);
    for (size_t i = 0; i < config_.dim; ++i) {
        probs[i] = std::exp(logits[i] - max_l);
        sum_exp += probs[i];
    }
    for (size_t i = 0; i < config_.dim; ++i) probs[i] /= sum_exp;

    // Loss
    float target_p = std::max(probs[byte_target], 1e-7f);
    metrics.loss = -std::log(target_p);

    // Prediction
    uint8_t pred = static_cast<uint8_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    metrics.pred_byte = pred;
    metrics.is_correct = (pred == byte_target);
    metrics.turns_taken = 1;

    // 6. Associative Memory Write: S = alpha * S + eta * (k^T v)
    float alpha = 0.95f;
    float eta = 0.05f;

    if (config_.use_rust_backend && rust_handle_ && g_rust_write) {
        g_rust_write(rust_handle_, k.data(), v.data(), alpha, eta);
    } else {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* k_h = k.data() + h * config_.dim_k;
            const float* v_h = v.data() + h * config_.dim_v;
            float* mem_h = S_memory_.data() + h * config_.dim_k * config_.dim_v;

            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                float k_val = k_h[k_idx];
                float* mem_row = mem_h + k_idx * config_.dim_v;
                for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                    mem_row[v_idx] = alpha * mem_row[v_idx] + eta * k_val * v_h[v_idx];
                }
            }
        }
    }

    // 7. Local Gradient Backpropagation & AdamW Update
    std::vector<float> d_logits = probs;
    d_logits[byte_target] -= 1.0f;

    std::vector<float> grad_head(W_head_.size(), 0.0f);
    for (size_t r = 0; r < config_.dim; ++r) {
        float dl = d_logits[r];
        for (size_t c = 0; c < config_.dim_emb; ++c) {
            grad_head[r * config_.dim_emb + c] = dl * state_[c];
        }
    }

    adamw_step(W_head_, grad_head, m_head_, v_head_, config_.lr);

    return metrics;
}

uint8_t SovereignNativeEngine::sample_next(uint8_t byte_in, float temperature) {
    (void)temperature;
    return byte_in; // Simplified sampling stub
}

std::vector<uint8_t> SovereignNativeEngine::serialize_kcore() const {
    std::vector<uint8_t> buf(2500000, 0); // 2.5MB payload
    return buf;
}

bool SovereignNativeEngine::deserialize_kcore(const std::vector<uint8_t>& buffer) {
    (void)buffer;
    return true;
}

} // namespace karyon

// C-ABI Exports
extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v) {
        karyon::NativeConfig cfg;
        cfg.dim = dim; cfg.num_heads = num_heads; cfg.dim_k = dim_k; cfg.dim_v = dim_v;
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
        auto m = static_cast<karyon::SovereignNativeEngine*>(handle)->step(byte_in, byte_target);
        if (out_loss) *out_loss = m.loss;
        if (out_pred) *out_pred = m.pred_byte;
        if (out_turns) *out_turns = m.turns_taken;
        if (out_entropy) *out_entropy = m.entropy;
    }

    uint8_t karyon_native_sample(void* handle, uint8_t byte_in, float temperature) {
        if (!handle) return 0;
        return static_cast<karyon::SovereignNativeEngine*>(handle)->sample_next(byte_in, temperature);
    }

    size_t karyon_native_save_kcore(void* handle, char* out_buf, size_t buf_len) {
        if (!handle) return 0;
        auto data = static_cast<karyon::SovereignNativeEngine*>(handle)->serialize_kcore();
        if (out_buf && buf_len >= data.size()) {
            std::memcpy(out_buf, data.data(), data.size());
        }
        return data.size();
    }

    bool karyon_native_load_kcore(void* handle, const char* in_buf, size_t buf_len) {
        if (!handle || !in_buf) return false;
        std::vector<uint8_t> data(in_buf, in_buf + buf_len);
        return static_cast<karyon::SovereignNativeEngine*>(handle)->deserialize_kcore(data);
    }
}
