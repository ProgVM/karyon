// karyon_native_core.h — Pure C++20 Sovereign Substrate Core Engine Header
// Created for Bazilevs (ProgVM) | Standard: KEP v16.0 Sovereign Master
// Zero-Python / Zero-Torch / Pure Native C++20 Substrate Architecture

#ifndef KARYON_NATIVE_CORE_H
#define KARYON_NATIVE_CORE_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>

namespace karyon {

struct NativeConfig {
    size_t dim = 258;          // Raw byte vocabulary V=258 (256 bytes + pad + eos)
    size_t num_heads = 8;      // Multi-head associative memory
    size_t dim_k = 64;         // Key dimension per head
    size_t dim_v = 64;         // Value dimension per head
    size_t dim_emb = 128;      // Continuous state embedding dimension
    float lr = 0.015f;         // Sovereign plastic learning rate
    float gamma_dissipation = 0.85f; // Exponential contractive dissipation (EXP-453)
    size_t max_thinking_turns = 12;  // Maximum deliberative thinking turns
    float entropy_threshold = 0.10f; // Target surprise entropy cutoff
    bool use_rust_backend = false;  // Flag whether Rust kernel is active
};

struct StepMetrics {
    float loss = 0.0f;
    uint8_t pred_byte = 0;
    bool is_correct = false;
    size_t turns_taken = 0;
    float entropy = 0.0f;
    float beta = 0.0f;
    float s_norm = 0.0f;
};

class SovereignNativeEngine {
public:
    SovereignNativeEngine(const NativeConfig& config = NativeConfig());
    ~SovereignNativeEngine();

    void reset_state();
    StepMetrics step(uint8_t byte_in, uint8_t byte_target);
    uint8_t sample_next(uint8_t byte_in, float temperature = 0.5f);

    // .kcore v6.0 Container Serialization & Deserialization
    std::vector<uint8_t> serialize_kcore() const;
    bool deserialize_kcore(const std::vector<uint8_t>& buffer);

    bool is_rust_active() const { return config_.use_rust_backend; }

private:
    NativeConfig config_;
    
    // Core parameters (std::vector<float>)
    std::vector<float> W_emb_;     // [dim * dim_emb]
    std::vector<float> W_q_;       // [(num_heads * dim_k) * dim_emb]
    std::vector<float> W_k_;       // [(num_heads * dim_k) * dim_emb]
    std::vector<float> W_v_;       // [(num_heads * dim_v) * dim_emb]
    std::vector<float> W_out_;     // [dim_emb * (num_heads * dim_v)]
    std::vector<float> W_head_;    // [dim * dim_emb]

    // Multi-head Associative Memory Matrix: [num_heads * dim_k * dim_v]
    std::vector<float> S_memory_;

    // Continuous Hidden State Vector: [dim_emb]
    std::vector<float> state_;

    // AdamW Momentum Buffers
    std::vector<float> m_emb_, v_emb_;
    std::vector<float> m_q_, v_q_;
    std::vector<float> m_k_, v_k_;
    std::vector<float> m_v_, v_v_;
    std::vector<float> m_out_, v_out_;
    std::vector<float> m_head_, v_head_;
    size_t adam_t_ = 0;

    // Pointer to Rust engine instance if Rust backend is active
    void* rust_handle_ = nullptr;

    // Internal helper math
    void init_weights();
    void matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols);
    void matvec_transpose_mul(const float* W, const float* x, float* y, size_t rows, size_t cols);
    void adamw_step(std::vector<float>& W, const std::vector<float>& grad, 
                    std::vector<float>& m, std::vector<float>& v, 
                    float lr, float weight_decay = 0.01f);
};

} // namespace karyon

// C-ABI Export Interface for ctypes / Rust FFI
extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v);
    void karyon_native_destroy(void* handle);
    void karyon_native_reset(void* handle);
    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                           float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy);
    uint8_t karyon_native_sample(void* handle, uint8_t byte_in, float temperature);
    size_t karyon_native_save_kcore(void* handle, char* out_buf, size_t buf_len);
    bool karyon_native_load_kcore(void* handle, const char* in_buf, size_t buf_len);
}

#endif // KARYON_NATIVE_CORE_H
