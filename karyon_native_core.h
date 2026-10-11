// karyon_native_core.h — Pure C++20 Sovereign Substrate Core Engine Interface
// Standard: KEP v16.0 Sovereign Master & EXP-456 Master Autopoiesis
// Zero-Python / Zero-Torch / Pure Native C++20 Substrate Architecture

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace karyon {

struct NativeConfig {
    size_t dim = 258;          // Byte vocabulary size (0..255 + pad + eos)
    size_t num_heads = 4;      // Sovereign memory heads
    size_t dim_k = 16;         // Key dimension per head
    size_t dim_v = 32;         // Value dimension per head
    size_t dim_emb = 128;      // Continuous state embedding dimension
    float lr = 0.015f;         // Sovereign plastic learning rate
    float gamma_dissipation = 0.70f; // Deliberation impulse decay
    size_t max_thinking_turns = 8;   // Maximum deliberative thinking turns
    float entropy_satisfy = 0.35f;   // Target surprise entropy cutoff
    float eps_halt = 0.01f;          // Momentum equilibrium cutoff
    bool use_rust_backend = false;   // Flag whether Rust kernel is active
};

struct StepMetrics {
    float loss = 0.0f;
    uint8_t pred_byte = 0;
    bool is_correct = false;
    size_t turns_taken = 0;
    float entropy = 0.0f;
    float beta = 0.0f;
    float mean_alpha = 0.0f;
    float mean_eta_delta = 0.0f;
    float mean_eta_hebb = 0.0f;
    float s_norm = 0.0f;
};

class SovereignNativeEngine {
public:
    SovereignNativeEngine(const NativeConfig& config);
    virtual ~SovereignNativeEngine();

    void reset_state();
    StepMetrics step(uint8_t byte_in, uint8_t byte_target);
    uint8_t sample_next(uint8_t byte_in, float temperature = 0.5f);

    std::vector<uint8_t> serialize_kcore() const;
    bool deserialize_kcore(const std::vector<uint8_t>& buffer);

    bool is_rust_active() const { return config_.use_rust_backend; }

protected:
    NativeConfig config_;
    
    // Core parameters (std::vector<float>)
    std::vector<float> byte_emb_;    // [dim * dim_emb]
    std::vector<float> W_q_;         // [(num_heads * dim_k) * dim_emb]
    std::vector<float> W_k_;         // [(num_heads * dim_k) * dim_emb]
    std::vector<float> W_v_;         // [(num_heads * dim_v) * dim_emb]
    std::vector<float> W_feedback_;  // [dim_k * dim_v]
    std::vector<float> W_out_;       // [dim * (num_heads * dim_v)]
    
    // Sovereign Multi-Head Law Synthesizer: 4 -> 32 -> 4
    std::vector<float> W_law1_, b_law1_;
    std::vector<float> W_law2_, b_law2_;
    
    // Beta precision gain parameters
    float w_beta_ = 0.5f, b_beta_ = 1.0f;
    float m_w_beta_ = 0.0f, v_w_beta_ = 0.0f;
    float m_b_beta_ = 0.0f, v_b_beta_ = 0.0f;

    // Multi-head Associative Memory Matrix: [num_heads * dim_k * dim_v]
    std::vector<float> S_memory_;

    // AdamW Momentum Buffers
    std::vector<float> m_out_, v_out_;
    size_t adam_t_ = 0;

    // Pointer to Rust engine instance if Rust backend is active
    void* rust_handle_ = nullptr;

    // Internal helper math
    void init_parameters();
    void matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols);
    void normalize_heads(float* ptr, size_t num_heads, size_t head_dim);
    float compute_l2_norm(const float* ptr, size_t n);
    float compute_entropy_from_logits(const float* logits, size_t n);
    void adamw_step(std::vector<float>& W, const std::vector<float>& grad, 
                    std::vector<float>& m, std::vector<float>& v, 
                    float lr, float weight_decay = 0.0001f);
};

} // namespace karyon

// C-ABI Export Interface for ctypes / Python / Rust FFI
extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v);
    void karyon_native_destroy(void* handle);
    void karyon_native_reset(void* handle);
    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                           float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy);
}
