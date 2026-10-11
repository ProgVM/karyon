// karyon_native_core.cpp — Integrated C++20 Sovereign Substrate Core Engine v4.0 (Dual-Tied Resonance Engine)
// Standard: KEP v16.0 Sovereign Master & Principle 27 (Anti-Hardcode Sovereignty / Absolute Substrate Freedom)
// Zero-AdamW / Zero-Backprop / Pure Continuous Physics-Driven State Dynamics

#include "karyon_native_core.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <immintrin.h>
#include <omp.h>

namespace karyon {

SovereignNativeEngine::SovereignNativeEngine(const NativeConfig& config)
    : config_(config) {
    init_parameters();
    init_io_uring();
    init_p2p_socket();
}

SovereignNativeEngine::~SovereignNativeEngine() {
    if (jit_dl_handle_) {
        dlclose(jit_dl_handle_);
        jit_dl_handle_ = nullptr;
    }
    if (uring_initialized_) {
        io_uring_queue_exit(&uring_ring_);
        uring_initialized_ = false;
    }
    if (p2p_sock_fd_ >= 0) {
        close(p2p_sock_fd_);
        p2p_sock_fd_ = -1;
    }
}

void SovereignNativeEngine::init_io_uring() {
    int ret = io_uring_queue_init(config_.uring_queue_depth, &uring_ring_, 0);
    if (ret == 0) {
        uring_initialized_ = true;
    }
}

void SovereignNativeEngine::init_p2p_socket() {
    if (!config_.p2p_enabled) return;
    p2p_sock_fd_ = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (p2p_sock_fd_ < 0) return;

    struct sockaddr_in local_addr{};
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = INADDR_ANY;
    local_addr.sin_port = htons(config_.p2p_port);

    bind(p2p_sock_fd_, (struct sockaddr*)&local_addr, sizeof(local_addr));
}

void SovereignNativeEngine::init_parameters() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    byte_emb_.resize(config_.dim * config_.dim_emb);
    for (size_t b = 0; b < config_.dim; ++b) {
        float* eb = byte_emb_.data() + b * config_.dim_emb;
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            eb[d] = dist(gen);
        }
        normalize_vector(eb, config_.dim_emb);
    }

    S_memory_.resize(config_.num_heads * config_.dim_k * config_.dim_v, 0.0f);
}

void SovereignNativeEngine::normalize_vector(float* ptr, size_t n) {
    float norm = compute_l2_norm(ptr, n);
    if (norm > 1e-6f) {
        float inv = 1.0f / norm;
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) ptr[i] *= inv;
    }
}

float SovereignNativeEngine::compute_l2_norm(const float* ptr, size_t n) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; ++i) sum += ptr[i] * ptr[i];
    return std::sqrt(sum);
}

float SovereignNativeEngine::compute_entropy_from_logits(const float* logits, size_t n) {
    float max_l = logits[0];
    for (size_t i = 1; i < n; ++i) if (logits[i] > max_l) max_l = logits[i];

    float sum_exp = 0.0f;
    std::vector<float> p(n);
    for (size_t i = 0; i < n; ++i) {
        p[i] = std::exp(logits[i] - max_l);
        sum_exp += p[i];
    }
    float inv_sum = 1.0f / sum_exp;
    float entropy = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        p[i] *= inv_sum;
        if (p[i] > 1e-7f) {
            entropy -= p[i] * std::log(p[i]);
        }
    }
    return entropy;
}

void SovereignNativeEngine::reset() {
    std::fill(S_memory_.begin(), S_memory_.end(), 0.0f);
    poincare_graph_.clear();
}

float SovereignNativeEngine::poincare_distance(const std::vector<float>& u, const std::vector<float>& v_vec) {
    float norm_u_sq = 0.0f, norm_v_sq = 0.0f, diff_sq = 0.0f;
    for (size_t i = 0; i < u.size(); ++i) {
        norm_u_sq += u[i] * u[i];
        norm_v_sq += v_vec[i] * v_vec[i];
        float d = u[i] - v_vec[i];
        diff_sq += d * d;
    }
    norm_u_sq = std::min(norm_u_sq, 0.9999f);
    norm_v_sq = std::min(norm_v_sq, 0.9999f);
    float alpha = 1.0f - norm_u_sq;
    float beta_den = 1.0f - norm_v_sq;
    float delta = 2.0f * diff_sq / (alpha * beta_den + 1e-7f);
    return std::acosh(1.0f + delta);
}

void SovereignNativeEngine::sprout_poincare_node(const std::vector<float>& key, const std::vector<float>& value) {
    if (poincare_graph_.size() >= 512) return;
    PoincareMemoryNode node;
    node.center = key;
    float norm = compute_l2_norm(node.center.data(), node.center.size());
    if (norm >= 0.95f) {
        float scale = 0.90f / (norm + 1e-6f);
        for (auto& x : node.center) x *= scale;
    }
    node.value = value;
    node.access_count = 1.0f;
    node.curvature = config_.max_poincare_curvature;
    poincare_graph_.push_back(std::move(node));
}

size_t SovereignNativeEngine::retrieve_geodesic_memories(const std::vector<float>& query_key, size_t top_k, float gamma) {
    if (poincare_graph_.empty()) return 0;
    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(poincare_graph_.size());
    for (size_t i = 0; i < poincare_graph_.size(); ++i) {
        float d = poincare_distance(query_key, poincare_graph_[i].center);
        dists.push_back({d, i});
    }
    std::sort(dists.begin(), dists.end());
    size_t k = std::min(top_k, dists.size());
    for (size_t i = 0; i < k; ++i) {
        size_t idx = dists[i].second;
        poincare_graph_[idx].access_count += gamma / (1.0f + dists[i].first);
    }
    return k;
}

bool SovereignNativeEngine::broadcast_mesh_spike(float error, const std::vector<float>& key, const std::vector<float>& value) {
    if (p2p_sock_fd_ < 0) return false;
    struct sockaddr_in peer_addr{};
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_port = htons(config_.target_peer_port);
    inet_pton(AF_INET, "127.0.0.1", &peer_addr.sin_addr);

    std::vector<float> packet;
    packet.push_back(static_cast<float>(config_.node_id));
    packet.push_back(error);
    packet.insert(packet.end(), key.begin(), key.end());
    packet.insert(packet.end(), value.begin(), value.end());

    ssize_t sent = sendto(p2p_sock_fd_, packet.data(), packet.size() * sizeof(float), 0,
                          (struct sockaddr*)&peer_addr, sizeof(peer_addr));
    return sent > 0;
}

size_t SovereignNativeEngine::broadcast_to_swarm(float error, const std::vector<float>& key, const std::vector<float>& value) {
    return broadcast_mesh_spike(error, key, value) ? 1 : 0;
}

bool SovereignNativeEngine::receive_mesh_spike() {
    if (p2p_sock_fd_ < 0) return false;
    std::vector<float> buf(256);
    ssize_t rec = recv(p2p_sock_fd_, buf.data(), buf.size() * sizeof(float), 0);
    return rec > 0;
}

bool SovereignNativeEngine::trigger_jit_hotpatch() {
    return false;
}

size_t SovereignNativeEngine::stream_from_file_uring(const std::string& filepath, size_t max_bytes) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    std::vector<uint8_t> buffer(max_bytes);
    ssize_t r = read(fd, buffer.data(), max_bytes);
    close(fd);
    if (r <= 1) return 0;

    size_t steps = r - 1;
    for (size_t i = 0; i < steps; ++i) {
        step(buffer[i], buffer[i + 1]);
    }
    return steps;
}

StepMetrics SovereignNativeEngine::step(uint8_t byte_in, uint8_t byte_target) {
    StepMetrics metrics;

    // 1. Dual-Tied Readout: Look up input byte directly on unified coordinate manifold
    float* e_in = byte_emb_.data() + byte_in * config_.dim_emb;

    // 2. Head Keys Partitioning
    std::vector<float> k(config_.num_heads * config_.dim_k);
    for (size_t h = 0; h < config_.num_heads; ++h) {
        for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
            k[h * config_.dim_k + k_idx] = e_in[h * config_.dim_k + k_idx];
        }
        float* kh = k.data() + h * config_.dim_k;
        normalize_vector(kh, config_.dim_k);
    }

    // 3. Associative Continuous Field Retrieval: h_field = S @ k
    std::vector<float> h_field(config_.dim_emb, 0.0f);
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < config_.num_heads; ++h) {
        const float* s_head = S_memory_.data() + h * config_.dim_k * config_.dim_v;
        const float* kh = k.data() + h * config_.dim_k;
        float* hh = h_field.data() + h * config_.dim_v;

        for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                sum += kh[k_idx] * s_head[k_idx * config_.dim_v + v_idx];
            }
            hh[v_idx] = sum;
        }
    }

    // Direct wave transmission leak along substrate
    #pragma omp simd
    for (size_t i = 0; i < config_.dim_emb; ++i) {
        h_field[i] = 0.70f * h_field[i] + 0.30f * e_in[i];
    }

    float h_norm = compute_l2_norm(h_field.data(), config_.dim_emb);
    metrics.h_norm = h_norm;

    // 4. Dual-Tied Resonance Projection (Cosine Physical Overlap onto Byte Manifold)
    std::vector<float> logits(config_.dim, 0.0f);
    float max_logit = -1e9f;
    float inv_h_norm = 1.0f / (h_norm + 1e-4f);

    #pragma omp parallel for schedule(static) reduction(max:max_logit)
    for (size_t b = 0; b < config_.dim; ++b) {
        const float* eb = byte_emb_.data() + b * config_.dim_emb;
        float dot = 0.0f;
        #pragma omp simd reduction(+:dot)
        for (size_t d = 0; d < config_.dim_emb; ++d) {
            dot += h_field[d] * eb[d];
        }
        float l = config_.beta_scale * (dot * inv_h_norm);
        logits[b] = l;
        if (l > max_logit) max_logit = l;
    }

    // 5. Gibbs-Boltzmann Probability Distribution
    float sum_exp = 0.0f;
    std::vector<float> probs(config_.dim);
    for (size_t b = 0; b < config_.dim; ++b) {
        probs[b] = std::exp(logits[b] - max_logit);
        sum_exp += probs[b];
    }
    float inv_sum = 1.0f / sum_exp;
    uint8_t best_byte = 0;
    float max_prob = 0.0f;
    for (size_t b = 0; b < config_.dim; ++b) {
        probs[b] *= inv_sum;
        if (probs[b] > max_prob) {
            max_prob = probs[b];
            best_byte = static_cast<uint8_t>(b);
        }
    }

    float loss = -std::log(std::max(probs[byte_target], 1e-7f));
    metrics.loss = loss;
    metrics.pred_byte = best_byte;
    metrics.is_correct = (best_byte == byte_target);
    metrics.entropy = compute_entropy_from_logits(logits.data(), config_.dim);
    metrics.beta = config_.beta_scale;
    metrics.mean_alpha = config_.alpha_decay;

    // 6. Prediction Error Wave: e_t = E_{target} - h_{field}
    float* e_target = byte_emb_.data() + byte_target * config_.dim_emb;
    std::vector<float> error_wave(config_.dim_emb);
    float err_sq = 0.0f;
    for (size_t d = 0; d < config_.dim_emb; ++d) {
        error_wave[d] = e_target[d] - h_field[d];
        err_sq += error_wave[d] * error_wave[d];
    }
    metrics.error_norm = std::sqrt(err_sq);

    // 7. Pure Endogenous Delta-Hebbian Dissipation (Zero AdamW / Zero Backprop)
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < config_.num_heads; ++h) {
        float* s_head = S_memory_.data() + h * config_.dim_k * config_.dim_v;
        const float* kh = k.data() + h * config_.dim_k;
        const float* err_h = error_wave.data() + h * config_.dim_v;

        for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
            float kval = kh[k_idx];
            float* s_row = s_head + k_idx * config_.dim_v;
            #pragma omp simd
            for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                s_row[v_idx] = config_.alpha_decay * s_row[v_idx] + config_.eta_mem * (kval * err_h[v_idx]);
            }
        }
    }

    // 8. Endogenous Manifold Self-Organization
    for (size_t d = 0; d < config_.dim_emb; ++d) {
        e_target[d] += config_.mu_emb * h_field[d];
    }
    normalize_vector(e_target, config_.dim_emb);

    // 9. Poincare Geodesic Retrieval & Dynamic Sprouting on High Surprise
    if (loss > 2.5f) {
        metrics.geodesic_recalls = retrieve_geodesic_memories(k, config_.geodesic_k, config_.geodesic_gamma);
    }
    if (loss > 3.0f) {
        std::vector<float> val_v(error_wave.begin(), error_wave.begin() + config_.poincare_dim);
        std::vector<float> key_k(k.begin(), k.begin() + config_.poincare_dim);
        sprout_poincare_node(key_k, val_v);
    }
    metrics.poincare_nodes = poincare_graph_.size();

    // 10. Synaptic P2P & Multi-Node Swarm Broadcast
    metrics.p2p_spike_sent = broadcast_mesh_spike(loss, k, error_wave);
    metrics.swarm_delivered = broadcast_to_swarm(loss, k, error_wave);
    metrics.p2p_spike_received = receive_mesh_spike();

    return metrics;
}

} // namespace karyon

extern "C" {
    void* karyon_native_create(size_t dim, size_t num_heads, size_t dim_k, size_t dim_v) {
        karyon::NativeConfig cfg;
        cfg.dim = dim;
        cfg.num_heads = num_heads;
        cfg.dim_k = dim_k;
        cfg.dim_v = dim_v;
        cfg.dim_emb = num_heads * dim_v;
        return new karyon::SovereignNativeEngine(cfg);
    }

    void karyon_native_destroy(void* handle) {
        if (handle) {
            delete static_cast<karyon::SovereignNativeEngine*>(handle);
        }
    }

    void karyon_native_reset(void* handle) {
        if (handle) {
            static_cast<karyon::SovereignNativeEngine*>(handle)->reset();
        }
    }

    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                           float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy) {
        if (!handle) return;
        auto* engine = static_cast<karyon::SovereignNativeEngine*>(handle);
        auto metrics = engine->step(byte_in, byte_target);
        if (out_loss) *out_loss = metrics.loss;
        if (out_pred) *out_pred = metrics.pred_byte;
        if (out_turns) *out_turns = metrics.turns_taken;
        if (out_entropy) *out_entropy = metrics.entropy;
    }
}
