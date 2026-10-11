// karyon_native_core.h — Pure C++20 Sovereign Substrate Core Engine Interface v3.0 Integrated
// Standard: KEP v16.0 Sovereign Master & Integrated Sovereign Organs (EXP-457 to EXP-465)
// Zero-Python / Zero-Torch / Pure Native C++20 Substrate Architecture

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
#include <liburing.h>
#include <immintrin.h>
#include <omp.h>

namespace karyon {

struct NativeConfig {
    size_t dim = 258;                // Byte vocabulary size (0..255 + pad + eos)
    size_t num_heads = 4;            // Sovereign memory heads
    size_t dim_k = 16;               // Key dimension per head
    size_t dim_v = 32;               // Value dimension per head
    size_t dim_emb = 128;            // Continuous state embedding dimension
    float lr = 0.015f;               // Sovereign plastic learning rate
    float gamma_dissipation = 0.70f; // Deliberation impulse decay
    size_t max_thinking_turns = 8;   // Maximum deliberative thinking turns
    float entropy_satisfy = 0.35f;   // Target surprise entropy cutoff
    float eps_halt = 0.01f;          // Momentum equilibrium cutoff
    bool use_rust_backend = false;   // Flag whether Rust kernel is active

    // EXP-458 & EXP-462: Poincare Hyperbolic Memory Graph & Geodesic Retrieval
    size_t poincare_dim = 16;
    float max_poincare_curvature = -1.0f;
    float geodesic_gamma = 0.5f;     // EXP-462 Geodesic retrieval scaling factor
    size_t geodesic_k = 3;           // Top-k nearest hyperbolic attractors

    // EXP-460 & EXP-465: io_uring Gateway
    size_t uring_queue_depth = 128;

    // EXP-461 & EXP-463: Synaptic Mesh P2P & Multi-Node Swarm
    uint32_t node_id = 101;
    int p2p_port = 19001;
    int target_peer_port = 19002;
    std::vector<int> swarm_peer_ports; // EXP-463: Multi-node peer ports
    bool p2p_enabled = true;

    // EXP-464: SIMD & Parallelism
    bool enable_avx2_fma = true;
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

    // Integrated Organ Telemetry
    size_t poincare_nodes = 0;
    size_t geodesic_recalls = 0;
    bool jit_hotpatched = false;
    bool p2p_spike_broadcast = false;
    bool p2p_spike_received = false;
    size_t swarm_delivered_spikes = 0;
};

// EXP-458 & EXP-462 Poincare Hyperbolic Memory Node
struct PoincareMemoryNode {
    std::vector<float> coords;
    std::vector<float> payload;
    float curvature = -1.0f;
    size_t access_count = 0;
};

// UDP Packet payload for EXP-461 & EXP-463 Synaptic Mesh
struct SynapticSpikePacket {
    uint32_t node_id;
    float free_energy;
    float phase_q[16];
    float phase_v[32];
};

typedef void (*JitKernelFn)(float* S, const float* q, const float* v, float lr, size_t num_heads, size_t dim_k, size_t dim_v);

class SovereignNativeEngine {
public:
    SovereignNativeEngine(const NativeConfig& config);
    virtual ~SovereignNativeEngine();

    void reset_state();
    StepMetrics step(uint8_t byte_in, uint8_t byte_target);
    uint8_t sample_next(uint8_t byte_in, float temperature = 0.5f);

    // EXP-460 & EXP-465: io_uring Async Byte Gateway Stream
    bool async_stream_ingest(const std::string& filepath, std::vector<uint8_t>& out_buffer, size_t chunk_size = 65536);

    // EXP-461 & EXP-463: Synaptic Mesh P2P & Multi-Node Swarm Broadcast
    bool broadcast_mesh_spike(float free_energy, const std::vector<float>& q, const std::vector<float>& v);
    size_t broadcast_to_swarm(float free_energy, const std::vector<float>& q, const std::vector<float>& v);
    bool receive_mesh_spike();

    // EXP-459: JIT Hot-Patching Engine
    bool trigger_jit_hotpatch();

    // EXP-462: Riemannian Geodesic Memory Retrieval
    size_t retrieve_geodesic_memories(const std::vector<float>& query_key, size_t k, float gamma, float mix_weight = 0.20f);

    std::vector<uint8_t> serialize_kcore() const;
    bool deserialize_kcore(const std::vector<uint8_t>& buffer);

    bool is_rust_active() const { return config_.use_rust_backend; }

protected:
    NativeConfig config_;

    // Core parameters
    std::vector<float> byte_emb_;
    std::vector<float> W_q_;
    std::vector<float> W_k_;
    std::vector<float> W_v_;
    std::vector<float> W_feedback_;
    std::vector<float> W_out_;

    // Sovereign Multi-Head Law Synthesizer
    std::vector<float> W_law1_, b_law1_;
    std::vector<float> W_law2_, b_law2_;

    // Beta precision gain parameters
    float w_beta_ = 0.5f, b_beta_ = 1.0f;
    float m_w_beta_ = 0.0f, v_w_beta_ = 0.0f;
    float m_b_beta_ = 0.0f, v_b_beta_ = 0.0f;

    // Multi-head Associative Memory Matrix
    std::vector<float> S_memory_;

    // EXP-458 & EXP-462: Poincare Hyperbolic Memory Graph
    std::vector<PoincareMemoryNode> poincare_graph_;

    // EXP-459: JIT Hot-Patching function pointer and dlopen handle
    void* jit_dl_handle_ = nullptr;
    JitKernelFn active_jit_kernel_ = nullptr;
    bool is_jit_patched_ = false;

    // EXP-460: io_uring Struct
    struct io_uring uring_ring_;
    bool uring_initialized_ = false;

    // EXP-461 & EXP-463: Synaptic UDP Mesh Socket
    int p2p_sock_fd_ = -1;

    // AdamW Momentum Buffers
    std::vector<float> m_out_, v_out_;
    size_t adam_t_ = 0;

    void* rust_handle_ = nullptr;

    // Internal helper math
    void init_parameters();
    void init_io_uring();
    void init_p2p_socket();

    // EXP-464: AVX2 SIMD Accelerated MatVec Multiplication
    void matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols);
    void simdfma_matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols);

    void normalize_heads(float* ptr, size_t num_heads, size_t head_dim);
    float compute_l2_norm(const float* ptr, size_t n);
    float compute_entropy_from_logits(const float* logits, size_t n);
    void adamw_step(std::vector<float>& W, const std::vector<float>& grad,
                    std::vector<float>& m, std::vector<float>& v,
                    float lr, float weight_decay = 0.0001f);

    // EXP-458 & EXP-462: Poincare Distance & Geodesics
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
}
