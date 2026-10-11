#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <random>
#include <algorithm>
#include <cstring>

namespace karyon {

struct NativeConfig {
    size_t dim = 258;
    size_t num_heads = 8;
    size_t dim_k = 64;
    size_t dim_v = 64;
    size_t dim_emb = 256;
    float g_0 = 0.35f;
    float gamma_damp = 0.55f;
    float epsilon_halt = 0.012f;
    float entropy_reflex = 1.20f;
    float entropy_satisfy = 0.35f;
    size_t max_safety_turns = 16;
    float learning_rate = 0.015f;
    float weight_decay = 1e-4f;
};

struct StepMetrics {
    float loss = 0.0f;
    bool is_correct = false;
    float entropy = 0.0f;
    size_t turns_taken = 0;
    float beta = 0.0f;
    float s_norm = 0.0f;
    float mean_alpha = 0.0f;
    float mean_eta_delta = 0.0f;
};

class SovereignNativeEngine {
public:
    explicit SovereignNativeEngine(const NativeConfig& config = NativeConfig());
    ~SovereignNativeEngine() = default;

    void reset_state();
    StepMetrics step(uint8_t byte_in, uint8_t byte_target);
    uint8_t sample(uint8_t byte_in, float temperature = 0.45f);

    // Serialization
    std::vector<uint8_t> serialize_kcore() const;
    bool deserialize_kcore(const uint8_t* data, size_t size);

private:
    NativeConfig cfg_;

    // Parameters
    std::vector<float> byte_emb_;       // [dim x dim_emb]
    std::vector<float> W_q_;            // [dim_emb x (H * dim_k)]
    std::vector<float> W_k_;            // [dim_emb x (H * dim_k)]
    std::vector<float> W_v_;            // [dim_emb x (H * dim_v)]
    std::vector<float> W_feedback_;     // [dim_v x dim_k]
    std::vector<float> W_out_;          // [(H * dim_v) x dim]
    
    // Multi-head Law Synthesizer (4 -> 32 -> 4)
    std::vector<float> W_law1_;         // [4 x 32]
    std::vector<float> b_law1_;         // [32]
    std::vector<float> W_law2_;         // [32 x 4]
    std::vector<float> b_law2_;         // [4]

    // Precision scaling
    float w_beta_ = 0.5f;
    float b_beta_ = 1.0f;

    // Persistent Memory State S: [H x dim_k x dim_v]
    std::vector<float> S_memory_;

    // Persistent AdamW Optimizer State
    struct AdamBuffer {
        std::vector<float> m;
        std::vector<float> v;
    };
    AdamBuffer opt_byte_emb_;
    AdamBuffer opt_W_q_;
    AdamBuffer opt_W_out_;
    float opt_w_beta_m_ = 0.0f;
    float opt_w_beta_v_ = 0.0f;
    float opt_b_beta_m_ = 0.0f;
    float opt_b_beta_v_ = 0.0f;
    size_t opt_step_ = 0;

    void init_parameters();
    void adamw_update(std::vector<float>& param, AdamBuffer& buf, const std::vector<float>& grad, float lr, float max_norm = 1.0f);
};

} // namespace karyon

// Pure C-ABI Export (Zero Dependencies)
extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v);
    void karyon_native_destroy(void* handle);
    void karyon_native_reset(void* handle);
    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                            float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy);
    uint8_t karyon_native_sample(void* handle, uint8_t byte_in, float temperature);
    size_t karyon_native_save_kcore(void* handle, uint8_t* out_buffer, size_t max_buffer_size);
    bool karyon_native_load_kcore(void* handle, const uint8_t* in_buffer, size_t buffer_size);
}
