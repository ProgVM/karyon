// karyon_native_core.cpp — Pure C++20 Sovereign Substrate Core Engine Implementation
// Standard: KEP v16.0 Sovereign Master & EXP-456 Master Autopoiesis
// Zero-Python / Zero-Torch / Pure Native C++20 Substrate Architecture

#include "karyon_native_core.h"
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <dlfcn.h>

// Dynamic Rust Engine Loader Struct
namespace karyon {

struct RustBackendDispatch {
    void* lib_handle = nullptr;
    void* (*create_fn)(size_t, size_t, size_t, size_t, size_t) = nullptr;
    void (*destroy_fn)(void*) = nullptr;
    void (*reset_fn)(void*) = nullptr;
    void (*read_fn)(const void*, const float*, float*) = nullptr;
    void (*apply_laws_fn)(void*, const float*, const float*, const float*, const float*, const float*, const float*, const float*) = nullptr;

    bool is_available() const {
        return lib_handle && create_fn && destroy_fn && reset_fn && read_fn && apply_laws_fn;
    }

    void load() {
        const char* lib_paths[] = {
            "./libkaryon_rust_core.so",
            "libkaryon_rust_core.so",
            "../libkaryon_rust_core.so",
            "rust_engine/target/release/libkaryon_rust_core.so"
        };

        for (const char* path : lib_paths) {
            lib_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (lib_handle) {
                break;
            }
        }

        if (!lib_handle) {
            return;
        }

        create_fn = reinterpret_cast<decltype(create_fn)>(dlsym(lib_handle, "karyon_rust_create"));
        destroy_fn = reinterpret_cast<decltype(destroy_fn)>(dlsym(lib_handle, "karyon_rust_destroy"));
        reset_fn = reinterpret_cast<decltype(reset_fn)>(dlsym(lib_handle, "karyon_rust_reset"));
        read_fn = reinterpret_cast<decltype(read_fn)>(dlsym(lib_handle, "karyon_rust_associative_read"));
        apply_laws_fn = reinterpret_cast<decltype(apply_laws_fn)>(dlsym(lib_handle, "karyon_rust_apply_laws"));

        if (!is_available()) {
            dlclose(lib_handle);
            lib_handle = nullptr;
        }
    }

    ~RustBackendDispatch() {
        if (lib_handle) {
            dlclose(lib_handle);
            lib_handle = nullptr;
        }
    }
};

static RustBackendDispatch& get_rust_dispatch() {
    static RustBackendDispatch dispatch;
    static bool initialized = false;
    if (!initialized) {
        dispatch.load();
        initialized = true;
    }
    return dispatch;
}

SovereignNativeEngine::SovereignNativeEngine(const NativeConfig& config)
    : config_(config), adam_t_(0), rust_handle_(nullptr) {
    init_parameters();
    
    if (config_.use_rust_backend) {
        auto& dispatch = get_rust_dispatch();
        if (dispatch.is_available()) {
            rust_handle_ = dispatch.create_fn(config_.dim, config_.num_heads, config_.dim_k, config_.dim_v, config_.dim_emb);
            if (rust_handle_) {
                std::cout << "[INFO] SovereignNativeEngine: Rust SIMD acceleration backend dynamically engaged.\n";
            } else {
                std::cout << "[WARN] SovereignNativeEngine: Failed to instantiate Rust engine. Falling back to native C++20 AVX2.\n";
                config_.use_rust_backend = false;
            }
        } else {
            std::cout << "[INFO] SovereignNativeEngine: libkaryon_rust_core.so not found or symbols missing. Falling back to native C++20 AVX2.\n";
            config_.use_rust_backend = false;
        }
    }
}

SovereignNativeEngine::~SovereignNativeEngine() {
    if (rust_handle_) {
        auto& dispatch = get_rust_dispatch();
        if (dispatch.is_available()) {
            dispatch.destroy_fn(rust_handle_);
        }
        rust_handle_ = nullptr;
    }
}

void SovereignNativeEngine::reset_state() {
    std::fill(psi_.begin(), psi_.end(), 0.0f);
    std::fill(p_momentum_.begin(), p_momentum_.end(), 0.0f);
    std::fill(S_memory_.begin(), S_memory_.end(), 0.0f);
    beta_eff_ = 1.0f;
    if (rust_handle_) {
        auto& dispatch = get_rust_dispatch();
        if (dispatch.is_available()) {
            dispatch.reset_fn(rust_handle_);
        }
    }
}

void SovereignNativeEngine::init_parameters() {
    std::mt19937 gen(42);
    float scale = 0.02f;
    std::normal_distribution<float> d(0.0f, scale);

    byte_emb_.resize(config_.dim * config_.dim_emb);
    for (auto& w : byte_emb_) w = d(gen);

    W_q_.resize(config_.num_heads * config_.dim_k * config_.dim_emb);
    for (auto& w : W_q_) w = d(gen);

    W_k_.resize(config_.num_heads * config_.dim_k * config_.dim_emb);
    for (auto& w : W_k_) w = d(gen);

    W_v_.resize(config_.num_heads * config_.dim_v * config_.dim_emb);
    for (auto& w : W_v_) w = d(gen);

    W_feedback_.resize(config_.dim_emb * (config_.num_heads * config_.dim_v));
    for (auto& w : W_feedback_) w = d(gen);

    W_out_.resize(config_.dim * config_.dim_emb);
    for (auto& w : W_out_) w = d(gen);

    // Sovereign Law Synth MLPs
    size_t dim_laws_in = config_.dim_emb + 4; // h + [entropy, delta_mag, hebb_mag, S_norm]
    size_t dim_laws_hidden = 64;
    size_t dim_laws_out = 4 * config_.num_heads; // alpha, eta_delta, eta_hebb, lambda

    W_law1_.resize(dim_laws_hidden * dim_laws_in);
    b_law1_.assign(dim_laws_hidden, 0.0f);
    W_law2_.resize(dim_laws_out * dim_laws_hidden);
    b_law2_.assign(dim_laws_out, 0.0f);

    for (auto& w : W_law1_) w = d(gen);
    for (auto& w : W_law2_) w = d(gen);

    // Allocate persistent state
    psi_.assign(config_.dim_emb, 0.0f);
    p_momentum_.assign(config_.dim_emb, 0.0f);
    S_memory_.assign(config_.num_heads * config_.dim_k * config_.dim_v, 0.0f);
    beta_eff_ = 1.0f;

    // Allocate AdamW state
    auto alloc_adam = [](const std::vector<float>& p, std::vector<float>& m, std::vector<float>& v) {
        m.assign(p.size(), 0.0f);
        v.assign(p.size(), 0.0f);
    };

    alloc_adam(byte_emb_, m_byte_emb_, v_byte_emb_);
    alloc_adam(W_q_, m_W_q_, v_W_q_);
    alloc_adam(W_k_, m_W_k_, v_W_k_);
    alloc_adam(W_v_, m_W_v_, v_W_v_);
    alloc_adam(W_feedback_, m_W_feedback_, v_W_feedback_);
    alloc_adam(W_out_, m_W_out_, v_W_out_);
    alloc_adam(W_law1_, m_W_law1_, v_W_law1_);
    alloc_adam(b_law1_, m_b_law1_, v_b_law1_);
    alloc_adam(W_law2_, m_W_law2_, v_W_law2_);
    alloc_adam(b_law2_, m_b_law2_, v_b_law2_);
}

StepMetrics SovereignNativeEngine::step(uint8_t byte_in, uint8_t byte_target) {
    StepMetrics metrics;

    // 1. Sensory Embedding
    const float* emb_in = &byte_emb_[byte_in * config_.dim_emb];
    for (size_t d = 0; d < config_.dim_emb; ++d) {
        p_momentum_[d] += emb_in[d];
    }

    size_t turns = 0;
    std::vector<float> logits(config_.dim, 0.0f);
    std::vector<float> probs(config_.dim, 0.0f);
    float entropy = 0.0f;

    std::vector<float> q_heads(config_.num_heads * config_.dim_k, 0.0f);
    std::vector<float> k_heads(config_.num_heads * config_.dim_k, 0.0f);
    std::vector<float> v_heads(config_.num_heads * config_.dim_v, 0.0f);
    std::vector<float> read_heads(config_.num_heads * config_.dim_v, 0.0f);
    std::vector<float> feedback(config_.dim_emb, 0.0f);

    auto& dispatch = get_rust_dispatch();

    // 2. Dissipative Recurrent Thinking Loop
    while (turns < config_.max_thinking_turns) {
        turns++;

        // Kinetic Update: psi += p_momentum
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            psi_[d] += p_momentum_[d];
        }

        // Multi-Head Projections: Q, K, V
        for (size_t h = 0; h < config_.num_heads; ++h) {
            for (size_t k = 0; k < config_.dim_k; ++k) {
                float sum_q = 0.0f, sum_k = 0.0f;
                const float* wq = &W_q_[(h * config_.dim_k + k) * config_.dim_emb];
                const float* wk = &W_k_[(h * config_.dim_k + k) * config_.dim_emb];
                for (size_t d = 0; d < config_.dim_emb; ++d) {
                    sum_q += wq[d] * psi_[d];
                    sum_k += wk[d] * psi_[d];
                }
                q_heads[h * config_.dim_k + k] = sum_q;
                k_heads[h * config_.dim_k + k] = sum_k;
            }

            for (size_t v = 0; v < config_.dim_v; ++v) {
                float sum_v = 0.0f;
                const float* wv = &W_v_[(h * config_.dim_v + v) * config_.dim_emb];
                for (size_t d = 0; d < config_.dim_emb; ++d) {
                    sum_v += wv[d] * psi_[d];
                }
                v_heads[h * config_.dim_v + v] = sum_v;
            }
        }

        // Associative Memory Read: Read = Q * S
        if (rust_handle_ && dispatch.is_available()) {
            dispatch.read_fn(rust_handle_, q_heads.data(), read_heads.data());
        } else {
            for (size_t h = 0; h < config_.num_heads; ++h) {
                const float* q = &q_heads[h * config_.dim_k];
                const float* S = &S_memory_[h * config_.dim_k * config_.dim_v];
                float* out = &read_heads[h * config_.dim_v];
                for (size_t v = 0; v < config_.dim_v; ++v) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < config_.dim_k; ++k) {
                        sum += q[k] * S[k * config_.dim_v + v];
                    }
                    out[v] = sum;
                }
            }
        }

        // Feedback Projection: feedback = W_feedback * read_heads
        size_t total_v = config_.num_heads * config_.dim_v;
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            float sum = 0.0f;
            const float* wf = &W_feedback_[d * total_v];
            for (size_t j = 0; j < total_v; ++j) {
                sum += wf[j] * read_heads[j];
            }
            feedback[d] = sum;
        }

        // Dissipative Momentum Drag: p = gamma * p + feedback
        float p_norm_sq = 0.0f;
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            p_momentum_[d] = config_.gamma_dissipation * p_momentum_[d] + feedback[d];
            p_norm_sq += p_momentum_[d] * p_momentum_[d];
        }

        // Output Logits & Prediction: logits = beta_eff * (W_out * psi)
        float max_l = -1e9f;
        for (size_t c = 0; c < config_.dim; ++c) {
            float sum = 0.0f;
            const float* wo = &W_out_[c * config_.dim_emb];
            for (size_t d = 0; d < config_.dim_emb; ++d) {
                sum += wo[d] * psi_[d];
            }
            logits[c] = beta_eff_ * sum;
            if (logits[c] > max_l) max_l = logits[c];
        }

        // Softmax & Entropy
        float sum_exp = 0.0f;
        for (size_t c = 0; c < config_.dim; ++c) {
            probs[c] = std::exp(logits[c] - max_l);
            sum_exp += probs[c];
        }
        entropy = 0.0f;
        float inv_sum = 1.0f / sum_exp;
        for (size_t c = 0; c < config_.dim; ++c) {
            probs[c] *= inv_sum;
            if (probs[c] > 1e-12f) {
                entropy -= probs[c] * std::log(probs[c]);
            }
        }

        // Halting condition
        if (entropy < config_.entropy_satisfy || std::sqrt(p_norm_sq) < config_.eps_halt) {
            break;
        }
    }

    // 3. Evaluation & Metrics
    uint8_t pred_byte = 0;
    float max_p = -1.0f;
    for (size_t c = 0; c < config_.dim; ++c) {
        if (probs[c] > max_p) {
            max_p = probs[c];
            pred_byte = static_cast<uint8_t>(c);
        }
    }

    float target_prob = std::max(probs[byte_target], 1e-12f);
    float loss = -std::log(target_prob);

    // 4. Memory Law Synthesis
    float s_norm_sq = 0.0f;
    for (float s : S_memory_) s_norm_sq += s * s;
    float s_norm = std::sqrt(s_norm_sq);

    // Feature Vector for Law Synth MLP
    std::vector<float> law_in(config_.dim_emb + 4, 0.0f);
    std::copy(psi_.begin(), psi_.end(), law_in.begin());
    law_in[config_.dim_emb] = entropy;
    law_in[config_.dim_emb + 1] = loss;
    law_in[config_.dim_emb + 2] = s_norm;
    law_in[config_.dim_emb + 3] = static_cast<float>(turns);

    size_t dim_laws_in = config_.dim_emb + 4;
    size_t dim_laws_hidden = 64;
    std::vector<float> law_h(dim_laws_hidden, 0.0f);

    for (size_t i = 0; i < dim_laws_hidden; ++i) {
        float sum = b_law1_[i];
        const float* w = &W_law1_[i * dim_laws_in];
        for (size_t j = 0; j < dim_laws_in; ++j) {
            sum += w[j] * law_in[j];
        }
        law_h[i] = std::tanh(sum);
    }

    size_t dim_laws_out = 4 * config_.num_heads;
    std::vector<float> law_out(dim_laws_out, 0.0f);
    for (size_t i = 0; i < dim_laws_out; ++i) {
        float sum = b_law2_[i];
        const float* w = &W_law2_[i * dim_laws_hidden];
        for (size_t j = 0; j < dim_laws_hidden; ++j) {
            sum += w[j] * law_h[j];
        }
        law_out[i] = sum;
    }

    // Dynamic Laws extraction per head
    std::vector<float> alphas(config_.num_heads);
    std::vector<float> etas_delta(config_.num_heads);
    std::vector<float> etas_hebb(config_.num_heads);
    std::vector<float> lambdas(config_.num_heads);

    float sum_alpha = 0.0f, sum_eta_delta = 0.0f, sum_eta_hebb = 0.0f;

    for (size_t h = 0; h < config_.num_heads; ++h) {
        alphas[h] = 0.50f + 0.499f / (1.0f + std::exp(-law_out[h * 4]));
        etas_delta[h] = 1.0f / (1.0f + std::exp(-law_out[h * 4 + 1]));
        etas_hebb[h] = 1.0f / (1.0f + std::exp(-law_out[h * 4 + 2]));
        lambdas[h] = 0.01f / (1.0f + std::exp(-law_out[h * 4 + 3]));

        sum_alpha += alphas[h];
        sum_eta_delta += etas_delta[h];
        sum_eta_hebb += etas_hebb[h];
    }

    // 5. Associative Memory Law Execution: Delta + Hebbian Plasticity
    std::vector<float> v_pred_heads(config_.num_heads * config_.dim_v, 0.0f);
    for (size_t h = 0; h < config_.num_heads; ++h) {
        const float* k = &k_heads[h * config_.dim_k];
        const float* S = &S_memory_[h * config_.dim_k * config_.dim_v];
        float* vp = &v_pred_heads[h * config_.dim_v];
        for (size_t v = 0; v < config_.dim_v; ++v) {
            float sum = 0.0f;
            for (size_t k_i = 0; k_i < config_.dim_k; ++k_i) {
                sum += k[k_i] * S[k_i * config_.dim_v + v];
            }
            vp[v] = sum;
        }
    }

    if (rust_handle_ && dispatch.is_available()) {
        dispatch.apply_laws_fn(rust_handle_, k_heads.data(), v_heads.data(), 
                               v_pred_heads.data(), alphas.data(), 
                               etas_delta.data(), etas_hebb.data(), lambdas.data());
    } else {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            float alpha = alphas[h];
            float e_delta = etas_delta[h];
            float e_hebb = etas_hebb[h];
            float lambda = lambdas[h];

            const float* k = &k_heads[h * config_.dim_k];
            const float* v = &v_heads[h * config_.dim_v];
            const float* vp = &v_pred_heads[h * config_.dim_v];
            float* S = &S_memory_[h * config_.dim_k * config_.dim_v];

            for (size_t k_i = 0; k_i < config_.dim_k; ++k_i) {
                for (size_t v_j = 0; v_j < config_.dim_v; ++v_j) {
                    size_t idx = k_i * config_.dim_v + v_j;
                    float delta_v = v[v_j] - vp[v_j];
                    float s_old = S[idx];
                    float s_new = alpha * s_old + e_delta * k[k_i] * delta_v + e_hebb * k[k_i] * v[v_j] - lambda * s_old;
                    S[idx] = s_new;
                }
            }
        }
    }

    // Dynamic Precision Beta Adaptation
    if (pred_byte == byte_target) {
        beta_eff_ = std::min(beta_eff_ + 0.05f, 21.0f);
    } else {
        beta_eff_ = std::max(beta_eff_ - 0.02f, 1.0f);
    }

    // 6. Online Micro-AdamW Plasticity Step
    adam_step(byte_in, byte_target, probs);

    // Return Step Metrics
    metrics.loss = loss;
    metrics.pred_byte = pred_byte;
    metrics.is_correct = (pred_byte == byte_target);
    metrics.turns_taken = turns;
    metrics.entropy = entropy;
    metrics.beta = beta_eff_;
    metrics.mean_alpha = sum_alpha / config_.num_heads;
    metrics.mean_eta_delta = sum_eta_delta / config_.num_heads;
    metrics.mean_eta_hebb = sum_eta_hebb / config_.num_heads;
    metrics.s_norm = s_norm;

    return metrics;
}

uint8_t SovereignNativeEngine::sample_next(uint8_t byte_in, float temperature) {
    const float* emb_in = &byte_emb_[byte_in * config_.dim_emb];
    for (size_t d = 0; d < config_.dim_emb; ++d) {
        p_momentum_[d] += emb_in[d];
    }

    size_t turns = 0;
    std::vector<float> logits(config_.dim, 0.0f);
    std::vector<float> probs(config_.dim, 0.0f);
    std::vector<float> q_heads(config_.num_heads * config_.dim_k, 0.0f);
    std::vector<float> read_heads(config_.num_heads * config_.dim_v, 0.0f);
    std::vector<float> feedback(config_.dim_emb, 0.0f);

    auto& dispatch = get_rust_dispatch();

    while (turns < config_.max_thinking_turns) {
        turns++;
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            psi_[d] += p_momentum_[d];
        }

        for (size_t h = 0; h < config_.num_heads; ++h) {
            for (size_t k = 0; k < config_.dim_k; ++k) {
                float sum_q = 0.0f;
                const float* wq = &W_q_[(h * config_.dim_k + k) * config_.dim_emb];
                for (size_t d = 0; d < config_.dim_emb; ++d) {
                    sum_q += wq[d] * psi_[d];
                }
                q_heads[h * config_.dim_k + k] = sum_q;
            }
        }

        if (rust_handle_ && dispatch.is_available()) {
            dispatch.read_fn(rust_handle_, q_heads.data(), read_heads.data());
        } else {
            for (size_t h = 0; h < config_.num_heads; ++h) {
                const float* q = &q_heads[h * config_.dim_k];
                const float* S = &S_memory_[h * config_.dim_k * config_.dim_v];
                float* out = &read_heads[h * config_.dim_v];
                for (size_t v = 0; v < config_.dim_v; ++v) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < config_.dim_k; ++k) {
                        sum += q[k] * S[k * config_.dim_v + v];
                    }
                    out[v] = sum;
                }
            }
        }

        size_t total_v = config_.num_heads * config_.dim_v;
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            float sum = 0.0f;
            const float* wf = &W_feedback_[d * total_v];
            for (size_t j = 0; j < total_v; ++j) {
                sum += wf[j] * read_heads[j];
            }
            feedback[d] = sum;
        }

        float p_norm_sq = 0.0f;
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            p_momentum_[d] = config_.gamma_dissipation * p_momentum_[d] + feedback[d];
            p_norm_sq += p_momentum_[d] * p_momentum_[d];
        }

        float max_l = -1e9f;
        for (size_t c = 0; c < config_.dim; ++c) {
            float sum = 0.0f;
            const float* wo = &W_out_[c * config_.dim_emb];
            for (size_t d = 0; d < config_.dim_emb; ++d) {
                sum += wo[d] * psi_[d];
            }
            logits[c] = (beta_eff_ * sum) / std::max(temperature, 0.01f);
            if (logits[c] > max_l) max_l = logits[c];
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < config_.dim; ++c) {
            probs[c] = std::exp(logits[c] - max_l);
            sum_exp += probs[c];
        }
        float inv_sum = 1.0f / sum_exp;
        float entropy = 0.0f;
        for (size_t c = 0; c < config_.dim; ++c) {
            probs[c] *= inv_sum;
            if (probs[c] > 1e-12f) entropy -= probs[c] * std::log(probs[c]);
        }

        if (entropy < config_.entropy_satisfy || std::sqrt(p_norm_sq) < config_.eps_halt) {
            break;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<uint8_t> dist(probs.begin(), probs.end());
    return dist(gen);
}

void SovereignNativeEngine::adam_step(uint8_t byte_in, uint8_t byte_target, const std::vector<float>& probs) {
    adam_t_++;

    // Gradient of Cross-Entropy Loss w.r.t logits: grad = probs - one_hot(target)
    std::vector<float> grad_logits = probs;
    grad_logits[byte_target] -= 1.0f;

    // Grad W_out: grad_logits * psi^T
    std::vector<float> grad_W_out(W_out_.size(), 0.0f);
    std::vector<float> grad_psi(config_.dim_emb, 0.0f);

    for (size_t c = 0; c < config_.dim; ++c) {
        float g_c = grad_logits[c] * beta_eff_;
        const float* wo = &W_out_[c * config_.dim_emb];
        float* g_wo = &grad_W_out[c * config_.dim_emb];
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            g_wo[d] = g_c * psi_[d];
            grad_psi[d] += g_c * wo[d];
        }
    }

    // Grad byte_emb of current byte
    std::vector<float> grad_emb(config_.dim_emb, 0.0f);
    for (size_t d = 0; d < config_.dim_emb; ++d) {
        grad_emb[d] = grad_psi[d];
    }

    // Apply AdamW to W_out and byte_emb
    update_adam_param(W_out_, grad_W_out, m_W_out_, v_W_out_);

    // Slice Adam update for single byte embedding
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, lr = config_.lr;
    float bc1 = 1.0f - std::pow(b1, static_cast<float>(adam_t_));
    float bc2 = 1.0f - std::pow(b2, static_cast<float>(adam_t_));

    float* emb_w = &byte_emb_[byte_in * config_.dim_emb];
    float* m_emb = &m_byte_emb_[byte_in * config_.dim_emb];
    float* v_emb = &v_byte_emb_[byte_in * config_.dim_emb];

    for (size_t d = 0; d < config_.dim_emb; ++d) {
        float g = grad_emb[d];
        m_emb[d] = b1 * m_emb[d] + (1.0f - b1) * g;
        v_emb[d] = b2 * v_emb[d] + (1.0f - b2) * g * g;
        float m_hat = m_emb[d] / bc1;
        float v_hat = v_emb[d] / bc2;
        emb_w[d] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void SovereignNativeEngine::update_adam_param(std::vector<float>& W, const std::vector<float>& grad,
                                             std::vector<float>& m, std::vector<float>& v) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, lr = config_.lr;
    float bc1 = 1.0f - std::pow(b1, static_cast<float>(adam_t_));
    float bc2 = 1.0f - std::pow(b2, static_cast<float>(adam_t_));

    size_t n = W.size();
    for (size_t i = 0; i < n; ++i) {
        float g = grad[i];
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
        if (handle) {
            delete static_cast<karyon::SovereignNativeEngine*>(handle);
        }
    }

    void karyon_native_reset(void* handle) {
        if (handle) {
            static_cast<karyon::SovereignNativeEngine*>(handle)->reset_state();
        }
    }

    float karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target, uint8_t* pred_out) {
        if (!handle) return 0.0f;
        auto* eng = static_cast<karyon::SovereignNativeEngine*>(handle);
        auto m = eng->step(byte_in, byte_target);
        if (pred_out) *pred_out = m.pred_byte;
        return m.loss;
    }
}
