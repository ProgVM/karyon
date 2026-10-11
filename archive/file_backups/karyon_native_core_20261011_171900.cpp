// karyon_native_core.cpp — Integrated C++20 Sovereign Substrate Core Engine v3.0
// Standard: KEP v16.0 Sovereign Master & Integrated Sovereign Organs (EXP-457 to EXP-465)
// Zero-Python / Zero-Torch / Pure Native C++20 Substrate Architecture

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
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dim_emb));
    std::uniform_real_distribution<float> dist(-scale, scale);

    byte_emb_.resize(config_.dim * config_.dim_emb);
    for (auto& w : byte_emb_) w = dist(gen);

    W_q_.resize((config_.num_heads * config_.dim_k) * config_.dim_emb);
    for (auto& w : W_q_) w = dist(gen);

    W_k_.resize((config_.num_heads * config_.dim_k) * config_.dim_emb);
    for (auto& w : W_k_) w = dist(gen);

    W_v_.resize((config_.num_heads * config_.dim_v) * config_.dim_emb);
    for (auto& w : W_v_) w = dist(gen);

    W_feedback_.resize(config_.dim_k * config_.dim_v);
    for (auto& w : W_feedback_) w = dist(gen);

    W_out_.resize(config_.dim * (config_.num_heads * config_.dim_v));
    for (auto& w : W_out_) w = dist(gen);

    W_law1_.resize(32 * 4);
    b_law1_.resize(32, 0.0f);
    for (auto& w : W_law1_) w = dist(gen);

    W_law2_.resize(4 * 32);
    b_law2_.resize(4, 0.0f);
    for (auto& w : W_law2_) w = dist(gen);

    S_memory_.resize(config_.num_heads * config_.dim_k * config_.dim_v, 0.0f);
    m_out_.resize(W_out_.size(), 0.0f);
    v_out_.resize(W_out_.size(), 0.0f);
}

// EXP-464: High-Performance AVX2 SIMD MatVec Implementation
void SovereignNativeEngine::simdfma_matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
    #pragma omp parallel for schedule(static) if(rows >= 32)
    for (size_t i = 0; i < rows; ++i) {
        const float* w_row = W + i * cols;
        __m256 acc = _mm256_setzero_ps();
        size_t j = 0;

        for (; j + 8 <= cols; j += 8) {
            __m256 w_vec = _mm256_loadu_ps(w_row + j);
            __m256 x_vec = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w_vec, x_vec, acc);
        }

        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

        for (; j < cols; ++j) {
            sum += w_row[j] * x[j];
        }
        y[i] = sum;
    }
}

void SovereignNativeEngine::matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
    if (config_.enable_avx2_fma) {
        simdfma_matvec_mul(W, x, y, rows, cols);
        return;
    }
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        const float* w_row = W + i * cols;
        for (size_t j = 0; j < cols; ++j) {
            sum += w_row[j] * x[j];
        }
        y[i] = sum;
    }
}

void SovereignNativeEngine::normalize_heads(float* ptr, size_t num_heads, size_t head_dim) {
    for (size_t h = 0; h < num_heads; ++h) {
        float* h_ptr = ptr + h * head_dim;
        float norm = compute_l2_norm(h_ptr, head_dim);
        if (norm > 1e-6f) {
            float inv_norm = 1.0f / norm;
            for (size_t i = 0; i < head_dim; ++i) {
                h_ptr[i] *= inv_norm;
            }
        }
    }
}

float SovereignNativeEngine::compute_l2_norm(const float* ptr, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += ptr[i] * ptr[i];
    return std::sqrt(sum);
}

float SovereignNativeEngine::compute_entropy_from_logits(const float* logits, size_t n) {
    float max_l = logits[0];
    for (size_t i = 1; i < n; ++i) if (logits[i] > max_l) max_l = logits[i];

    float sum_exp = 0.0f;
    std::vector<float> probs(n);
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - max_l);
        sum_exp += probs[i];
    }

    float inv_sum = 1.0f / sum_exp;
    float entropy = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        probs[i] *= inv_sum;
        if (probs[i] > 1e-9f) {
            entropy -= probs[i] * std::log(probs[i]);
        }
    }
    return entropy;
}

float SovereignNativeEngine::poincare_distance(const std::vector<float>& u, const std::vector<float>& v_vec) {
    float u_sq = 0.0f, v_sq = 0.0f, diff_sq = 0.0f;
    size_t dim = std::min(u.size(), v_vec.size());
    for (size_t i = 0; i < dim; ++i) {
        u_sq += u[i] * u[i];
        v_sq += v_vec[i] * v_vec[i];
        float d = u[i] - v_vec[i];
        diff_sq += d * d;
    }
    u_sq = std::min(u_sq, 0.99f);
    v_sq = std::min(v_sq, 0.99f);

    float num = 2.0f * diff_sq;
    float den = (1.0f - u_sq) * (1.0f - v_sq);
    float arg = 1.0f + num / (den + 1e-6f);
    return std::acosh(std::max(1.0f, arg));
}

void SovereignNativeEngine::sprout_poincare_node(const std::vector<float>& key, const std::vector<float>& value) {
    PoincareMemoryNode node;
    node.coords.resize(config_.poincare_dim);
    for (size_t i = 0; i < config_.poincare_dim; ++i) {
        node.coords[i] = (i < key.size()) ? std::tanh(key[i]) * 0.5f : 0.0f;
    }
    node.payload = value;
    node.curvature = config_.max_poincare_curvature;
    node.access_count = 1;
    poincare_graph_.push_back(node);
}

// EXP-462: Riemannian Geodesic Retrieval Implementation
size_t SovereignNativeEngine::retrieve_geodesic_memories(const std::vector<float>& query_key, size_t k, float gamma, float mix_weight) {
    if (poincare_graph_.empty()) return 0;

    std::vector<float> q_poincare(config_.poincare_dim);
    for (size_t i = 0; i < config_.poincare_dim; ++i) {
        q_poincare[i] = (i < query_key.size()) ? std::tanh(query_key[i]) * 0.5f : 0.0f;
    }

    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(poincare_graph_.size());
    for (size_t idx = 0; idx < poincare_graph_.size(); ++idx) {
        float d_H = poincare_distance(q_poincare, poincare_graph_[idx].coords);
        dists.push_back({d_H, idx});
    }

    std::sort(dists.begin(), dists.end());
    size_t top_k = std::min(k, dists.size());

    for (size_t rank = 0; rank < top_k; ++rank) {
        float d_H = dists[rank].first;
        size_t node_idx = dists[rank].second;
        float w_i = std::exp(-gamma * d_H);

        poincare_graph_[node_idx].access_count++;
        const auto& payload = poincare_graph_[node_idx].payload;

        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < config_.num_heads; ++h) {
            float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                float q_val = (k_idx < query_key.size()) ? query_key[k_idx] : 0.0f;
                float* srow = sh + k_idx * config_.dim_v;
                #pragma omp simd
                for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                    float p_val = (v_idx < payload.size()) ? payload[v_idx] : 0.0f;
                    srow[v_idx] += mix_weight * w_i * (q_val * p_val);
                }
            }
        }
    }
    return top_k;
}

bool SovereignNativeEngine::trigger_jit_hotpatch() {
    std::string cpp_code = R"(
#include <cstddef>
extern "C" void active_kernel(float* S, const float* q, const float* v, float lr, size_t num_heads, size_t dim_k, size_t dim_v) {
    for (size_t h = 0; h < num_heads; ++h) {
        float* sh = S + h * dim_k * dim_v;
        const float* qh = q + h * dim_k;
        const float* vh = v + h * dim_v;
        for (size_t k = 0; k < dim_k; ++k) {
            float q_val = qh[k];
            float* srow = sh + k * dim_v;
            #pragma omp simd
            for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                srow[v_idx] += lr * (q_val * vh[v_idx]);
            }
        }
    }
}
)";
    std::ofstream out("/tmp/karyon_integrated_jit.cpp");
    out << cpp_code;
    out.close();

    int res = system("g++ -O3 -shared -fPIC -mavx2 -mfma -o /tmp/karyon_integrated_jit.so /tmp/karyon_integrated_jit.cpp");
    if (res != 0) return false;

    if (jit_dl_handle_) dlclose(jit_dl_handle_);
    jit_dl_handle_ = dlopen("/tmp/karyon_integrated_jit.so", RTLD_NOW);
    if (!jit_dl_handle_) return false;

    active_jit_kernel_ = (JitKernelFn)dlsym(jit_dl_handle_, "active_kernel");
    if (!active_jit_kernel_) return false;

    is_jit_patched_ = true;
    return true;
}

bool SovereignNativeEngine::async_stream_ingest(const std::string& filepath, std::vector<uint8_t>& out_buffer, size_t chunk_size) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return false; }

    size_t file_size = st.st_size;
    out_buffer.resize(file_size);

    if (!uring_initialized_) {
        ssize_t bytes_read = read(fd, out_buffer.data(), file_size);
        close(fd);
        return (bytes_read == static_cast<ssize_t>(file_size));
    }

    size_t offset = 0;
    size_t bytes_remaining = file_size;

    while (bytes_remaining > 0) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&uring_ring_);
        if (!sqe) {
            io_uring_submit(&uring_ring_);
            sqe = io_uring_get_sqe(&uring_ring_);
        }
        size_t bytes_to_read = std::min(bytes_remaining, chunk_size);
        io_uring_prep_read(sqe, fd, out_buffer.data() + offset, bytes_to_read, offset);
        sqe->user_data = offset;
        offset += bytes_to_read;
        bytes_remaining -= bytes_to_read;
    }

    io_uring_submit(&uring_ring_);

    size_t completed_chunks = (file_size + chunk_size - 1) / chunk_size;
    for (size_t c = 0; c < completed_chunks; ++c) {
        struct io_uring_cqe* cqe = nullptr;
        int ret = io_uring_wait_cqe(&uring_ring_, &cqe);
        if (ret < 0) { close(fd); return false; }
        io_uring_cqe_seen(&uring_ring_, cqe);
    }
    close(fd);
    return true;
}

bool SovereignNativeEngine::broadcast_mesh_spike(float free_energy, const std::vector<float>& q, const std::vector<float>& v) {
    if (p2p_sock_fd_ < 0) return false;
    SynapticSpikePacket pkt;
    pkt.node_id = config_.node_id;
    pkt.free_energy = free_energy;

    size_t k_len = std::min(q.size(), static_cast<size_t>(16));
    size_t v_len = std::min(v.size(), static_cast<size_t>(32));

    for (size_t i = 0; i < k_len; ++i) pkt.phase_q[i] = q[i];
    for (size_t i = 0; i < v_len; ++i) pkt.phase_v[i] = v[i];

    struct sockaddr_in dest_addr{};
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(config_.target_peer_port);
    inet_pton(AF_INET, "127.0.0.1", &dest_addr.sin_addr);

    ssize_t sent = sendto(p2p_sock_fd_, &pkt, sizeof(pkt), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    return sent == static_cast<ssize_t>(sizeof(pkt));
}

// EXP-463: Multi-Node Swarm Broadcast Implementation
size_t SovereignNativeEngine::broadcast_to_swarm(float free_energy, const std::vector<float>& q, const std::vector<float>& v) {
    if (p2p_sock_fd_ < 0 || config_.swarm_peer_ports.empty()) return 0;

    SynapticSpikePacket pkt;
    pkt.node_id = config_.node_id;
    pkt.free_energy = free_energy;

    size_t k_len = std::min(q.size(), static_cast<size_t>(16));
    size_t v_len = std::min(v.size(), static_cast<size_t>(32));

    for (size_t i = 0; i < k_len; ++i) pkt.phase_q[i] = q[i];
    for (size_t i = 0; i < v_len; ++i) pkt.phase_v[i] = v[i];

    size_t sent_count = 0;
    for (int p_port : config_.swarm_peer_ports) {
        struct sockaddr_in dest_addr{};
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(p_port);
        inet_pton(AF_INET, "127.0.0.1", &dest_addr.sin_addr);

        ssize_t sent = sendto(p2p_sock_fd_, &pkt, sizeof(pkt), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
        if (sent == static_cast<ssize_t>(sizeof(pkt))) sent_count++;
    }
    return sent_count;
}

bool SovereignNativeEngine::receive_mesh_spike() {
    if (p2p_sock_fd_ < 0) return false;
    SynapticSpikePacket pkt;
    struct sockaddr_in src_addr{};
    socklen_t addr_len = sizeof(src_addr);

    ssize_t bytes_recv = recvfrom(p2p_sock_fd_, &pkt, sizeof(pkt), 0, (struct sockaddr*)&src_addr, &addr_len);
    if (bytes_recv != static_cast<ssize_t>(sizeof(pkt))) return false;

    float coupling_gamma = 0.15f;
    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < config_.num_heads; ++h) {
        float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
        for (size_t k = 0; k < config_.dim_k; ++k) {
            float q_val = pkt.phase_q[k % 16];
            float* srow = sh + k * config_.dim_v;
            #pragma omp simd
            for (size_t v = 0; v < config_.dim_v; ++v) {
                float v_val = pkt.phase_v[v % 32];
                srow[v] += coupling_gamma * (q_val * v_val);
            }
        }
    }
    return true;
}

void SovereignNativeEngine::reset_state() {
    std::fill(S_memory_.begin(), S_memory_.end(), 0.0f);
    poincare_graph_.clear();
}

StepMetrics SovereignNativeEngine::step(uint8_t byte_in, uint8_t byte_target) {
    StepMetrics metrics;
    const float* emb_in = byte_emb_.data() + byte_in * config_.dim_emb;

    std::vector<float> q(config_.num_heads * config_.dim_k);
    std::vector<float> k(config_.num_heads * config_.dim_k);
    std::vector<float> v(config_.num_heads * config_.dim_v);

    matvec_mul(W_q_.data(), emb_in, q.data(), config_.num_heads * config_.dim_k, config_.dim_emb);
    matvec_mul(W_k_.data(), emb_in, k.data(), config_.num_heads * config_.dim_k, config_.dim_emb);
    matvec_mul(W_v_.data(), emb_in, v.data(), config_.num_heads * config_.dim_v, config_.dim_emb);

    normalize_heads(q.data(), config_.num_heads, config_.dim_k);
    normalize_heads(k.data(), config_.num_heads, config_.dim_k);

    // EXP-456 Multi-Head Plastic Law Synthesizer
    std::vector<float> head_norms(4, 0.0f);
    size_t heads_to_norm = std::min(config_.num_heads, static_cast<size_t>(4));
    for (size_t h = 0; h < heads_to_norm; ++h) {
        head_norms[h] = compute_l2_norm(q.data() + h * config_.dim_k, config_.dim_k);
    }

    std::vector<float> law1_out(32);
    matvec_mul(W_law1_.data(), head_norms.data(), law1_out.data(), 32, 4);
    for (size_t i = 0; i < 32; ++i) law1_out[i] = std::max(0.0f, law1_out[i] + b_law1_[i]);

    std::vector<float> law2_out(4);
    matvec_mul(W_law2_.data(), law1_out.data(), law2_out.data(), 4, 32);
    for (size_t i = 0; i < 4; ++i) law2_out[i] = 1.0f / (1.0f + std::exp(-(law2_out[i] + b_law2_[i])));

    float alpha = law2_out[0];
    float eta_delta = law2_out[1];
    float eta_hebb = law2_out[2];
    float beta = 1.0f + 2.0f * law2_out[3];

    // Deliberative Thinking Recurrent Horizon
    std::vector<float> y_read(config_.num_heads * config_.dim_v, 0.0f);
    std::vector<float> q_curr = q;
    float prev_entropy = 100.0f;
    size_t turn = 0;

    for (turn = 0; turn < config_.max_thinking_turns; ++turn) {
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            const float* qh = q_curr.data() + h * config_.dim_k;
            float* yh = y_read.data() + h * config_.dim_v;

            for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                    sum += qh[k_idx] * sh[k_idx * config_.dim_v + v_idx];
                }
                yh[v_idx] = sum;
            }
        }

        std::vector<float> logits(config_.dim);
        matvec_mul(W_out_.data(), y_read.data(), logits.data(), config_.dim, config_.num_heads * config_.dim_v);

        float entropy = compute_entropy_from_logits(logits.data(), config_.dim);
        if (entropy < config_.entropy_satisfy || std::abs(prev_entropy - entropy) < config_.eps_halt) {
            break;
        }
        prev_entropy = entropy;

        std::vector<float> fb(config_.dim_k, 0.0f);
        matvec_mul(W_feedback_.data(), y_read.data(), fb.data(), config_.dim_k, config_.dim_v);

        for (size_t h = 0; h < config_.num_heads; ++h) {
            float* qh = q_curr.data() + h * config_.dim_k;
            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                qh[k_idx] = config_.gamma_dissipation * qh[k_idx] + (1.0f - config_.gamma_dissipation) * fb[k_idx];
            }
        }
        normalize_heads(q_curr.data(), config_.num_heads, config_.dim_k);
    }

    std::vector<float> final_logits(config_.dim);
    matvec_mul(W_out_.data(), y_read.data(), final_logits.data(), config_.dim, config_.num_heads * config_.dim_v);

    for (size_t i = 0; i < config_.dim; ++i) final_logits[i] *= beta;

    float max_l = final_logits[0];
    uint8_t best_byte = 0;
    for (size_t i = 1; i < config_.dim; ++i) {
        if (final_logits[i] > max_l) {
            max_l = final_logits[i];
            best_byte = static_cast<uint8_t>(i);
        }
    }

    float sum_exp = 0.0f;
    std::vector<float> probs(config_.dim);
    for (size_t i = 0; i < config_.dim; ++i) {
        probs[i] = std::exp(final_logits[i] - max_l);
        sum_exp += probs[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < config_.dim; ++i) probs[i] *= inv_sum;

    float loss = -std::log(std::max(probs[byte_target], 1e-7f));

    // Update Associative Memory Matrix (JIT vs Builtin OpenMP SIMD)
    if (is_jit_patched_ && active_jit_kernel_) {
        active_jit_kernel_(S_memory_.data(), q.data(), v.data(), config_.lr * eta_hebb, config_.num_heads, config_.dim_k, config_.dim_v);
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < config_.num_heads; ++h) {
            float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            const float* kh = k.data() + h * config_.dim_k;
            const float* vh = v.data() + h * config_.dim_v;

            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                float q_val = kh[k_idx];
                float* srow = sh + k_idx * config_.dim_v;
                #pragma omp simd
                for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                    float delta_rule = -eta_delta * srow[v_idx];
                    float hebb_rule = eta_hebb * (q_val * vh[v_idx]);
                    srow[v_idx] = alpha * srow[v_idx] + config_.lr * (delta_rule + hebb_rule);
                }
            }
        }
    }

    // EXP-458 & EXP-462: Poincare Geodesic Retrieval and Dynamic Sprouting
    size_t geodesic_recalls = 0;
    if (loss > 2.5f) {
        geodesic_recalls = retrieve_geodesic_memories(q, config_.geodesic_k, config_.geodesic_gamma);
    }
    if (loss > 3.0f) {
        sprout_poincare_node(q, v);
    }

    // EXP-461 & EXP-463: Synaptic P2P & Multi-Node Swarm Broadcast
    bool broadcast_ok = broadcast_mesh_spike(loss, q, v);
    size_t swarm_delivered = broadcast_to_swarm(loss, q, v);
    bool recv_ok = receive_mesh_spike();

    // EXP-459: Auto JIT Hot-Patch Trigger if loss remains high
    if (loss > 4.5f && !is_jit_patched_) {
        trigger_jit_hotpatch();
    }

    metrics.loss = loss;
    metrics.pred_byte = best_byte;
    metrics.is_correct = (best_byte == byte_target);
    metrics.turns_taken = turn + 1;
    metrics.entropy = compute_entropy_from_logits(final_logits.data(), config_.dim);
    metrics.beta = beta;
    metrics.mean_alpha = alpha;
    metrics.mean_eta_delta = eta_delta;
    metrics.mean_eta_hebb = eta_hebb;
    metrics.s_norm = compute_l2_norm(S_memory_.data(), S_memory_.size());
    metrics.poincare_nodes = poincare_graph_.size();
    metrics.geodesic_recalls = geodesic_recalls;
    metrics.jit_hotpatched = is_jit_patched_;
    metrics.p2p_spike_broadcast = broadcast_ok;
    metrics.p2p_spike_received = recv_ok;
    metrics.swarm_delivered_spikes = swarm_delivered;

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

    void karyon_native_step(void* handle, uint8_t byte_in, uint8_t byte_target,
                           float* out_loss, uint8_t* out_pred, size_t* out_turns, float* out_entropy) {
        if (handle) {
            auto m = static_cast<karyon::SovereignNativeEngine*>(handle)->step(byte_in, byte_target);
            if (out_loss) *out_loss = m.loss;
            if (out_pred) *out_pred = m.pred_byte;
            if (out_turns) *out_turns = m.turns_taken;
            if (out_entropy) *out_entropy = m.entropy;
        }
    }
}
