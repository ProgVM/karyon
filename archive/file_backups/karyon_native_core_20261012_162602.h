// karyon_native_core.h — Pure C++20 Sovereign Substrate Core Engine Interface v5.0
// Standard: KEP v16.0 Sovereign Master & Principle 27 (Anti-Hardcode Sovereignty / Absolute Substrate Freedom)
// Zero-AdamW / Zero-Backprop / Pure Continuous Physics-Driven State Dynamics
// EXP-472 Integrated: Unbounded Sovereign Attractor Manifold & Self-Organizing Head Dynamics

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <omp.h>

namespace karyon {

struct NativeConfig {
    size_t dim = 258;                // Byte vocabulary size (0..255 + pad + eos)
    size_t num_heads = 4;            // Sovereign memory heads
    size_t dim_k = 16;               // Key dimension per head
    size_t dim_v = 16;               // Value dimension per head (num_heads * dim_v = dim_emb)
    size_t dim_emb = 64;             // Continuous state embedding & dual-tied manifold dimension

    float beta_scale = 12.0f;        // Base attractor sharpness / inverse temperature
    float eta_mem = 0.45f;           // Local physical learning rate for memory trace
    float mu_emb = 0.05f;            // Local embedding manifold plasticity
    float gamma_dissipation = 0.70f; // Deliberation impulse decay
    size_t max_thinking_turns = 8;   // Maximum deliberative thinking turns
    float entropy_satisfy = 0.35f;   // Target surprise entropy cutoff
    float eps_halt = 0.01f;          // Momentum equilibrium cutoff

    // EXP-470: Sovereign Self-Organizing Head Dynamics
    float eta_theta = 0.01f;         // Plasticity rate for head timescale coordinates

    // EXP-472: Unbounded Sovereign Attractor Manifold
    bool use_unbounded_memory = true;
    float surprise_threshold = 2.0f;
    float beta_attractor = 16.0f;

    // EXP-458 & EXP-462: Poincare Hyperbolic Memory Graph & Geodesic Retrieval
    size_t poincare_dim = 16;
    float max_poincare_curvature = -1.0f;
    float geodesic_gamma = 0.5f;
    size_t geodesic_k = 3;

    // EXP-461 & EXP-463: Synaptic Mesh P2P & Multi-Node Swarm
    uint32_t node_id = 101;
    int p2p_port = 19001;
    int target_peer_port = 19002;
    bool p2p_enabled = true;
};

struct StepMetrics {
    float loss = 0.0f;
    uint8_t pred_byte = 0;
    bool is_correct = false;
    size_t turns_taken = 0;
    float entropy = 0.0f;
    float beta = 1.0f;
    float mean_alpha = 0.0f;
    float mean_eta_delta = 0.0f;
    float mean_eta_hebb = 0.0f;
    float error_norm = 0.0f;
    float h_norm = 0.0f;
    size_t memory_nodes = 0;
    size_t poincare_nodes = 0;
    size_t geodesic_recalls = 0;
    bool p2p_spike_sent = false;
    bool p2p_spike_received = false;
    size_t swarm_delivered = 0;
    bool jit_patch_triggered = false;
};

// EXP-472: Unbounded Sovereign Attractor Node
struct AttractorNode {
    std::vector<float> key;    // Trigger context pattern in R^dim_emb
    uint8_t target_byte = 0;   // Target transition byte
    float usage_count = 1.0f;  // Synaptic consolidation strength
};

struct PoincareMemoryNode {
    std::vector<float> center; // Inside unit disk |x| < 1
    std::vector<float> value;
    float access_count = 1.0f;
    float curvature = -1.0f;
};

typedef void (*JitKernelFn)(float* S, const float* q, const float* v, float lr, size_t num_heads, size_t dim_k, size_t dim_v);

class SovereignNativeEngine {
public:
    explicit SovereignNativeEngine(const NativeConfig& config = NativeConfig());
    ~SovereignNativeEngine();

    void reset();
    StepMetrics step(uint8_t byte_in, uint8_t byte_target);

    // High-Throughput Stream Ingestion
    size_t stream_from_file(const std::string& filepath, size_t max_bytes);

    // EXP-461 & EXP-463: Synaptic Mesh P2P UDP Broadcast & Swarm Multicast
    bool broadcast_mesh_spike(float error, const std::vector<float>& key, const std::vector<float>& value);
    size_t broadcast_to_swarm(float error, const std::vector<float>& key, const std::vector<float>& value);
    bool receive_mesh_spike();

    // EXP-458 & EXP-462: Poincare Geodesic Retrieval
    size_t retrieve_geodesic_memories(const std::vector<float>& query_key, size_t top_k, float gamma);

    // EXP-459: JIT Hot-Patching Engine
    bool trigger_jit_hotpatch();

    const NativeConfig& config() const { return config_; }
    size_t poincare_node_count() const { return poincare_graph_.size(); }
    size_t attractor_node_count() const { return unbounded_memory_.size(); }

private:
    NativeConfig config_;

    // Dual-Tied Continuous Coordinate Manifold (Input Embedding + Output Resonance Basis)
    std::vector<float> byte_emb_; // [dim x dim_emb]
    std::vector<float> S_memory_; // [num_heads x dim_k x dim_v] Associative field
    std::vector<float> theta_heads_; // [num_heads] Endogenous timescale coordinates (EXP-470)

    // EXP-472: Unbounded Sovereign Attractor Manifold (Zero artificial limits)
    std::vector<AttractorNode> unbounded_memory_;

    // EXP-458 & EXP-462: Poincare Hyperbolic Memory Graph
    std::vector<PoincareMemoryNode> poincare_graph_;

    // JIT Hot-Patching
    void* jit_dl_handle_ = nullptr;
    JitKernelFn active_jit_kernel_ = nullptr;
    bool is_jit_patched_ = false;

    // Synaptic UDP Mesh Socket
    int p2p_sock_fd_ = -1;

    void init_parameters();
    void init_p2p_socket();

    void normalize_vector(float* ptr, size_t n);
    float compute_l2_norm(const float* ptr, size_t n);
    float dot_product(const float* a, const float* b, size_t n);
    float compute_entropy_from_logits(const float* logits, size_t n);

    float poincare_distance(const std::vector<float>& u, const std::vector<float>& v_vec);
    void sprout_poincare_node(const std::vector<float>& key, const std::vector<float>& value);
};

} // namespace karyon

extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v);
    void karyon_native_destroy(void* handle);
    void karyon_native_reset(void* handle);
    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                           float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy);
    size_t karyon_native_get_memory_nodes(void* handle);
}
