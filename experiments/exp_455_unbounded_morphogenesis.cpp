// exp_455_unbounded_morphogenesis.cpp
// EXP-455: Unbounded Morphogenetic Potential & Dynamic Operator Genesis Substrate
// Standard: KEP v16.0 Sovereign Master & Principle 28 (C++20/Rust High-Performance Imperative)
// Author: Bazilevs (ProgVM) & Autonomous Lead AI Cyberneticist

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <string>
#include <numeric>

namespace karyon {

// Mathematical operator types synthesized spontaneously in the unbounded field
enum class DynamicOpType {
    LINEAR_PROJECTION,       // Standard spatial projection W @ x
    QUADRATIC_CONJUNCTION,   // Bilinear / Multiplicative interaction (x * (W @ x))
    HOLOGRAPHIC_PHASE_SHIFT, // Rotational phase shift (x * cos(th) - x_rot * sin(th))
    SATURATED_ATTRACTOR,     // Nonlinear Hopfield basin snap tanh(2.0 * (W @ x))
    CONTINUOUS_LEAKY_INTEG   // Dynamic temporal state integration
};

struct DynamicSynthesizedNode {
    size_t node_id;
    DynamicOpType op_type;
    size_t dim;
    std::vector<float> W;
    float alpha_epigenetic;   // Net2Net epigenetic gate tanh(alpha_epi)
    float m_alpha, v_alpha;   // Adam momentum for epigenetic gate
    float viability_score;    // Thermodynamic vitality (FEP surprise reduction)
    size_t age_steps;
};

struct MorphogenesisConfig {
    size_t dim = 258;
    size_t num_heads = 8;
    size_t dim_k = 32;
    size_t dim_v = 32;
    size_t dim_emb = 128;
    float lr = 0.015f;
    float lr_epi = 0.02f;
    float pruning_threshold = 0.005f;
    size_t genesis_interval = 250;     // Steps between spontaneous genesis events
    size_t max_unbounded_nodes = 32;   // Upper cap for safety
    size_t total_steps = 4000;
};

class SovereignMorphogenesisEngine {
public:
    SovereignMorphogenesisEngine(const MorphogenesisConfig& cfg)
        : cfg_(cfg), rng_(1337), adam_t_(0) {
        init_base_substrate();
    }

    struct StepResult {
        float loss;
        uint8_t pred_byte;
        bool is_correct;
        float entropy;
        size_t turns_taken;
        size_t active_nodes_count;
        float mean_alpha_epi;
    };

    StepResult step(uint8_t byte_in, uint8_t byte_target) {
        adam_t_++;
        StepResult res;

        // 1. Spontaneous Morphogenesis Check:
        // Under persistent thermodynamic stress or periodic intervals, spontaneously synthesize new operators!
        if (cfg_.genesis_interval < 100000 && adam_t_ % cfg_.genesis_interval == 0 && dynamic_nodes_.size() < cfg_.max_unbounded_nodes) {
            synthesize_spontaneous_node();
        }

        // 2. Base Input Embedding
        const float* emb_ptr = W_emb_.data() + byte_in * cfg_.dim_emb;
        std::vector<float> emb(cfg_.dim_emb);
        for (size_t e = 0; e < cfg_.dim_emb; ++e) emb[e] = emb_ptr[e];

        // 3. Multi-Head Projections: Q, K, V
        std::vector<float> q(cfg_.num_heads * cfg_.dim_k);
        std::vector<float> k(cfg_.num_heads * cfg_.dim_k);
        std::vector<float> v(cfg_.num_heads * cfg_.dim_v);

        matvec_mul(W_q_.data(), emb.data(), q.data(), q.size(), cfg_.dim_emb);
        matvec_mul(W_k_.data(), emb.data(), k.data(), k.size(), cfg_.dim_emb);
        matvec_mul(W_v_.data(), emb.data(), v.data(), v.size(), cfg_.dim_emb);

        // Normalize Q per head
        for (size_t h = 0; h < cfg_.num_heads; ++h) {
            float* q_h = q.data() + h * cfg_.dim_k;
            float sq = 0.0f;
            for (size_t i = 0; i < cfg_.dim_k; ++i) sq += q_h[i] * q_h[i];
            float inv_norm = 1.0f / (std::sqrt(sq) + 1e-8f);
            for (size_t i = 0; i < cfg_.dim_k; ++i) q_h[i] *= inv_norm;
        }

        // Multi-head associative read: read_flat = Q @ S
        std::vector<float> read_flat(cfg_.num_heads * cfg_.dim_v, 0.0f);
        for (size_t h = 0; h < cfg_.num_heads; ++h) {
            const float* q_h = q.data() + h * cfg_.dim_k;
            float* r_h = read_flat.data() + h * cfg_.dim_v;
            const float* s_h = S_mem_.data() + h * cfg_.dim_k * cfg_.dim_v;

            for (size_t ki = 0; ki < cfg_.dim_k; ++ki) {
                float qv = q_h[ki];
                const float* s_row = s_h + ki * cfg_.dim_v;
                for (size_t vi = 0; vi < cfg_.dim_v; ++vi) {
                    r_h[vi] += qv * s_row[vi];
                }
            }
        }

        // Base spatial projection back to embedding space: h = W_proj @ read_flat
        std::vector<float> h(cfg_.dim_emb, 0.0f);
        matvec_mul(W_proj_.data(), read_flat.data(), h.data(), cfg_.dim_emb, read_flat.size());

        // 4. Pass through Spontaneously Synthesized Dynamic Operators
        // Net2Net smooth grafting: tanh(alpha_epi) * Op(h)
        std::vector<std::vector<float>> node_outputs(dynamic_nodes_.size());
        for (size_t i = 0; i < dynamic_nodes_.size(); ++i) {
            auto& node = dynamic_nodes_[i];
            node.age_steps++;
            float gate = std::tanh(node.alpha_epigenetic);

            std::vector<float> op_val(cfg_.dim_emb, 0.0f);
            switch (node.op_type) {
                case DynamicOpType::LINEAR_PROJECTION:
                    matvec_mul(node.W.data(), h.data(), op_val.data(), cfg_.dim_emb, cfg_.dim_emb);
                    break;
                case DynamicOpType::QUADRATIC_CONJUNCTION: {
                    std::vector<float> tmp(cfg_.dim_emb, 0.0f);
                    matvec_mul(node.W.data(), h.data(), tmp.data(), cfg_.dim_emb, cfg_.dim_emb);
                    for (size_t e = 0; e < cfg_.dim_emb; ++e) op_val[e] = h[e] * tmp[e];
                    break;
                }
                case DynamicOpType::SATURATED_ATTRACTOR: {
                    std::vector<float> tmp(cfg_.dim_emb, 0.0f);
                    matvec_mul(node.W.data(), h.data(), tmp.data(), cfg_.dim_emb, cfg_.dim_emb);
                    for (size_t e = 0; e < cfg_.dim_emb; ++e) op_val[e] = std::tanh(2.0f * tmp[e]);
                    break;
                }
                case DynamicOpType::HOLOGRAPHIC_PHASE_SHIFT: {
                    for (size_t e = 0; e < cfg_.dim_emb; ++e) {
                        float th = node.W[e];
                        op_val[e] = h[e] * std::cos(th) - h[(e + 1) % cfg_.dim_emb] * std::sin(th);
                    }
                    break;
                }
                case DynamicOpType::CONTINUOUS_LEAKY_INTEG: {
                    for (size_t e = 0; e < cfg_.dim_emb; ++e) {
                        float decay = 1.0f / (1.0f + std::exp(-node.W[e]));
                        op_val[e] = decay * state_[e] + (1.0f - decay) * h[e];
                    }
                    break;
                }
            }

            node_outputs[i] = op_val;

            // Smooth additive contribution to h
            for (size_t e = 0; e < cfg_.dim_emb; ++e) {
                h[e] += gate * op_val[e];
            }
        }

        // Continuous State update
        for (size_t e = 0; e < cfg_.dim_emb; ++e) {
            state_[e] = std::tanh(0.85f * state_[e] + 0.15f * h[e]);
        }

        // 5. Output Logits: logits = W_out @ state
        std::vector<float> logits(cfg_.dim, 0.0f);
        matvec_mul(W_out_.data(), state_.data(), logits.data(), cfg_.dim, cfg_.dim_emb);

        // Softmax & Entropy
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        std::vector<float> probs(cfg_.dim);
        for (size_t i = 0; i < cfg_.dim; ++i) {
            probs[i] = std::exp(logits[i] - max_l);
            sum_exp += probs[i];
        }
        float entropy = 0.0f;
        for (size_t i = 0; i < cfg_.dim; ++i) {
            probs[i] /= sum_exp;
            if (probs[i] > 1e-12f) entropy -= probs[i] * std::log(probs[i]);
        }

        float p_target = std::max(probs[byte_target], 1e-7f);
        res.loss = -std::log(p_target);
        uint8_t pred = static_cast<uint8_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
        res.pred_byte = pred;
        res.is_correct = (pred == byte_target);
        res.entropy = entropy;
        res.turns_taken = 1;
        res.active_nodes_count = dynamic_nodes_.size();

        float sum_alpha = 0.0f;
        for (const auto& nd : dynamic_nodes_) sum_alpha += std::abs(nd.alpha_epigenetic);
        res.mean_alpha_epi = dynamic_nodes_.empty() ? 0.0f : (sum_alpha / dynamic_nodes_.size());

        // 6. Associative Memory Online Plasticity (Exact Multi-Head Hebbian/Delta Law)
        // Error computation: error = V - (K @ S)
        std::vector<float> v_pred(cfg_.num_heads * cfg_.dim_v, 0.0f);
        for (size_t head = 0; head < cfg_.num_heads; ++head) {
            const float* k_h = k.data() + head * cfg_.dim_k;
            float* vp_h = v_pred.data() + head * cfg_.dim_v;
            const float* s_h = S_mem_.data() + head * cfg_.dim_k * cfg_.dim_v;

            for (size_t ki = 0; ki < cfg_.dim_k; ++ki) {
                float kv = k_h[ki];
                const float* s_row = s_h + ki * cfg_.dim_v;
                for (size_t vi = 0; vi < cfg_.dim_v; ++vi) {
                    vp_h[vi] += kv * s_row[vi];
                }
            }
        }

        // Memory update S = alpha * S + eta_delta * (k^T error) + eta_hebb * (k^T v)
        float alpha_mem = 0.94f;
        float eta_delta = 0.10f;
        float eta_hebb = 0.04f;
        for (size_t head = 0; head < cfg_.num_heads; ++head) {
            const float* k_h = k.data() + head * cfg_.dim_k;
            const float* v_h = v.data() + head * cfg_.dim_v;
            const float* vp_h = v_pred.data() + head * cfg_.dim_v;
            float* s_h = S_mem_.data() + head * cfg_.dim_k * cfg_.dim_v;

            for (size_t ki = 0; ki < cfg_.dim_k; ++ki) {
                float kv = k_h[ki];
                float* s_row = s_h + ki * cfg_.dim_v;
                for (size_t vi = 0; vi < cfg_.dim_v; ++vi) {
                    float err_val = v_h[vi] - vp_h[vi];
                    s_row[vi] = alpha_mem * s_row[vi] + eta_delta * kv * err_val + eta_hebb * kv * v_h[vi];
                }
            }
        }

        // 7. Gradients & Persistent AdamW Updates
        // dL / dLogits = probs; probs[target] -= 1.0
        std::vector<float> d_logits = probs;
        d_logits[byte_target] -= 1.0f;

        // Gradient for W_out
        std::vector<float> grad_out(W_out_.size(), 0.0f);
        for (size_t r = 0; r < cfg_.dim; ++r) {
            float dl = d_logits[r];
            for (size_t c = 0; c < cfg_.dim_emb; ++c) {
                grad_out[r * cfg_.dim_emb + c] = dl * state_[c];
            }
        }
        adamw_step(W_out_, grad_out, m_out_, v_out_, cfg_.lr);

        // Gradient backpropagating to state
        std::vector<float> d_state(cfg_.dim_emb, 0.0f);
        matvec_transpose_mul(W_out_.data(), d_logits.data(), d_state.data(), cfg_.dim, cfg_.dim_emb);

        // Update epigenetic gating for dynamic nodes
        for (size_t i = 0; i < dynamic_nodes_.size(); ++i) {
            auto& node = dynamic_nodes_[i];
            float dtanh = 1.0f - std::pow(std::tanh(node.alpha_epigenetic), 2.0f);

            float d_alpha = 0.0f;
            for (size_t e = 0; e < cfg_.dim_emb; ++e) {
                d_alpha += d_state[e] * node_outputs[i][e] * dtanh;
            }

            // Clip gradient to prevent explosion
            d_alpha = std::clamp(d_alpha, -1.0f, 1.0f);

            // Epigenetic learning
            node.m_alpha = 0.9f * node.m_alpha + 0.1f * d_alpha;
            node.v_alpha = 0.99f * node.v_alpha + 0.01f * d_alpha * d_alpha;
            float bc1 = 1.0f - std::pow(0.9f, static_cast<float>(node.age_steps));
            float bc2 = 1.0f - std::pow(0.99f, static_cast<float>(node.age_steps));

            node.alpha_epigenetic -= cfg_.lr_epi * (node.m_alpha / bc1) / (std::sqrt(node.v_alpha / bc2) + 1e-8f);

            // Thermodynamic vitality: lower loss increases survival score
            if (res.loss < 2.5f) {
                node.viability_score += 0.02f;
            } else {
                node.viability_score -= 0.01f;
            }
        }

        // 8. Edelman Neural Darwinism: Prune dead / unviable nodes
        prune_unviable_nodes();

        return res;
    }

private:
    MorphogenesisConfig cfg_;
    std::mt19937 rng_;
    size_t adam_t_;

    // Core weights
    std::vector<float> W_emb_;
    std::vector<float> W_q_, W_k_, W_v_, W_proj_;
    std::vector<float> W_out_;
    std::vector<float> state_;
    std::vector<float> S_mem_;

    // AdamW momentum buffers
    std::vector<float> m_out_, v_out_;

    // Dynamic self-synthesized operator graph
    std::vector<DynamicSynthesizedNode> dynamic_nodes_;
    size_t next_node_id_ = 0;

    void init_base_substrate() {
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(cfg_.dim_emb));
        auto init_v = [&](std::vector<float>& v, size_t n) {
            v.resize(n);
            for (size_t i = 0; i < n; ++i) v[i] = dist(rng_);
        };

        init_v(W_emb_, cfg_.dim * cfg_.dim_emb);
        init_v(W_q_, (cfg_.num_heads * cfg_.dim_k) * cfg_.dim_emb);
        init_v(W_k_, (cfg_.num_heads * cfg_.dim_k) * cfg_.dim_emb);
        init_v(W_v_, (cfg_.num_heads * cfg_.dim_v) * cfg_.dim_emb);
        init_v(W_proj_, cfg_.dim_emb * (cfg_.num_heads * cfg_.dim_v));
        init_v(W_out_, cfg_.dim * cfg_.dim_emb);

        m_out_.assign(W_out_.size(), 0.0f);
        v_out_.assign(W_out_.size(), 0.0f);

        state_.assign(cfg_.dim_emb, 0.0f);
        S_mem_.assign(cfg_.num_heads * cfg_.dim_k * cfg_.dim_v, 0.0f);
    }

    void synthesize_spontaneous_node() {
        DynamicSynthesizedNode node;
        node.node_id = next_node_id_++;
        node.dim = cfg_.dim_emb;
        
        std::uniform_int_distribution<int> type_dist(0, 4);
        node.op_type = static_cast<DynamicOpType>(type_dist(rng_));

        std::normal_distribution<float> w_dist(0.0f, 0.05f);
        size_t w_size = (node.op_type == DynamicOpType::HOLOGRAPHIC_PHASE_SHIFT ||
                         node.op_type == DynamicOpType::CONTINUOUS_LEAKY_INTEG) 
                        ? cfg_.dim_emb : cfg_.dim_emb * cfg_.dim_emb;
        
        node.W.resize(w_size);
        for (size_t i = 0; i < w_size; ++i) node.W[i] = w_dist(rng_);

        // Net2Net Identity Principle: alpha_epi = 0.0 -> tanh(0.0) = 0.0 (Strict Zero Shock at birth!)
        node.alpha_epigenetic = 0.0f;
        node.m_alpha = 0.0f;
        node.v_alpha = 0.0f;
        node.viability_score = 1.0f;
        node.age_steps = 0;

        dynamic_nodes_.push_back(node);
    }

    void prune_unviable_nodes() {
        if (dynamic_nodes_.empty()) return;

        dynamic_nodes_.erase(
            std::remove_if(dynamic_nodes_.begin(), dynamic_nodes_.end(),
                [&](const DynamicSynthesizedNode& nd) {
                    return (nd.age_steps > 350 && std::abs(std::tanh(nd.alpha_epigenetic)) < cfg_.pruning_threshold && nd.viability_score < 0.2f);
                }),
            dynamic_nodes_.end());
    }

    void matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
        for (size_t r = 0; r < rows; ++r) {
            float sum = 0.0f;
            const float* row = W + r * cols;
            for (size_t c = 0; c < cols; ++c) {
                sum += row[c] * x[c];
            }
            y[r] = sum;
        }
    }

    void matvec_transpose_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
        std::fill(y, y + cols, 0.0f);
        for (size_t r = 0; r < rows; ++r) {
            float xr = x[r];
            const float* row = W + r * cols;
            for (size_t c = 0; c < cols; ++c) {
                y[c] += row[c] * xr;
            }
        }
    }

    void adamw_step(std::vector<float>& W, const std::vector<float>& grad,
                    std::vector<float>& m, std::vector<float>& v, float lr) {
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - std::pow(b1, static_cast<float>(adam_t_));
        float bc2 = 1.0f - std::pow(b2, static_cast<float>(adam_t_));

        for (size_t i = 0; i < W.size(); ++i) {
            float g = std::clamp(grad[i], -1.0f, 1.0f); // Gradient clip
            W[i] -= lr * 0.001f * W[i];                 // Soft weight decay
            m[i] = b1 * m[i] + (1.0f - b1) * g;
            v[i] = b2 * v[i] + (1.0f - b2) * g * g;
            float m_hat = m[i] / bc1;
            float v_hat = v[i] / bc2;
            W[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
};

} // namespace karyon

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  EXP-455: UNBOUNDED MORPHOGENETIC POTENTIAL & OPERATOR GENESIS   \n";
    std::cout << "  Zero-Limit Sovereign Open-Ended Evolution Benchmark (C++20 Native)\n";
    std::cout << "==================================================================\n";

    // 1. Prepare benchmark corpus
    std::string text = 
        "Karyon-CoRE autopoietic continuous field dynamics. "
        "Mind emerges unconditionally from substrate physics without biological mimicry. "
        "Unbounded morphogenetic potential synthesizes novel operators on the fly. "
        "Zero-shock Net2Net epigenetic gating guarantees smooth functional integration. "
        "Neural Darwinism prunes degenerate configurations under thermodynamic pressure. ";
    
    std::vector<uint8_t> stream;
    while (stream.size() < 4000) {
        stream.insert(stream.end(), text.begin(), text.end());
    }
    stream.resize(4000);

    karyon::MorphogenesisConfig cfg;
    cfg.total_steps = stream.size() - 1;

    // ------------------------------------------------------------------------
    // Baseline A: Static Fixed Substrate (Zero Morphogenesis, No Genesis)
    // ------------------------------------------------------------------------
    std::cout << "\n>>> Phase 1: Benchmarking Static Baseline (Genesis Disabled)...\n";
    cfg.genesis_interval = 999999; // Never sprout
    karyon::SovereignMorphogenesisEngine engine_static(cfg);

    float sum_loss_static = 0.0f;
    size_t correct_static = 0;
    std::vector<float> losses_static;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < cfg.total_steps; ++i) {
        auto res = engine_static.step(stream[i], stream[i + 1]);
        sum_loss_static += res.loss;
        if (res.is_correct) correct_static++;
        losses_static.push_back(res.loss);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_static = std::chrono::duration<double>(t1 - t0).count();

    // ------------------------------------------------------------------------
    // Experimental B: Unbounded Morphogenetic Genesis (Spontaneous Sprouting & Pruning)
    // ------------------------------------------------------------------------
    std::cout << "\n>>> Phase 2: Benchmarking Unbounded Morphogenesis Engine (Zero Limits)...\n";
    cfg.genesis_interval = 200; // Sprout every 200 steps
    karyon::SovereignMorphogenesisEngine engine_unbounded(cfg);

    float sum_loss_unbounded = 0.0f;
    size_t correct_unbounded = 0;
    std::vector<float> losses_unbounded;
    size_t final_nodes_count = 0;
    float final_alpha_epi = 0.0f;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < cfg.total_steps; ++i) {
        auto res = engine_unbounded.step(stream[i], stream[i + 1]);
        sum_loss_unbounded += res.loss;
        if (res.is_correct) correct_unbounded++;
        losses_unbounded.push_back(res.loss);
        final_nodes_count = res.active_nodes_count;
        final_alpha_epi = res.mean_alpha_epi;

        if ((i + 1) % 1000 == 0) {
            std::cout << "    [Step " << (i + 1) << "/" << cfg.total_steps << "] "
                      << "Loss: " << std::fixed << std::setprecision(4) << res.loss
                      << " | Acc: " << (res.is_correct ? "1" : "0")
                      << " | Active Nodes: " << res.active_nodes_count
                      << " | Mean |alpha_epi|: " << std::setprecision(4) << res.mean_alpha_epi << "\n";
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double time_unbounded = std::chrono::duration<double>(t3 - t2).count();

    // Calculate tail statistics (second half of stream)
    size_t half = cfg.total_steps / 2;
    float tail_loss_static = 0.0f, tail_loss_unbounded = 0.0f;
    size_t tail_correct_static = 0, tail_correct_unbounded = 0;
    for (size_t i = half; i < cfg.total_steps; ++i) {
        tail_loss_static += losses_static[i];
        tail_loss_unbounded += losses_unbounded[i];
        if (losses_static[i] < 2.0f) tail_correct_static++;
        if (losses_unbounded[i] < 2.0f) tail_correct_unbounded++;
    }
    tail_loss_static /= (cfg.total_steps - half);
    tail_loss_unbounded /= (cfg.total_steps - half);

    std::cout << "\n==================================================================\n";
    std::cout << "                 EXP-455 COMPARATIVE RESULTS                      \n";
    std::cout << "==================================================================\n";
    std::cout << "Metric                            | Static Baseline  | Unbounded Morphogenesis\n";
    std::cout << "----------------------------------+------------------+------------------------\n";
    std::cout << "Overall Mean Loss (nats)          | " << std::setw(16) << (sum_loss_static / cfg.total_steps) 
              << " | " << std::setw(22) << (sum_loss_unbounded / cfg.total_steps) << "\n";
    std::cout << "Tail 2nd-Half Loss (nats)         | " << std::setw(16) << tail_loss_static 
              << " | " << std::setw(22) << tail_loss_unbounded << "\n";
    std::cout << "Overall Accuracy (%)              | " << std::setw(15) << (static_cast<double>(correct_static) / cfg.total_steps * 100.0) << "%"
              << " | " << std::setw(21) << (static_cast<double>(correct_unbounded) / cfg.total_steps * 100.0) << "%\n";
    std::cout << "2nd-Half Low Loss Mass (<2.0) (%) | " << std::setw(15) << (static_cast<double>(tail_correct_static) / (cfg.total_steps - half) * 100.0) << "%"
              << " | " << std::setw(21) << (static_cast<double>(tail_correct_unbounded) / (cfg.total_steps - half) * 100.0) << "%\n";
    std::cout << "Throughput (steps/sec)            | " << std::setw(16) << (cfg.total_steps / time_static)
              << " | " << std::setw(22) << (cfg.total_steps / time_unbounded) << "\n";
    std::cout << "Final Active Sprouted Nodes       | " << std::setw(16) << 0 
              << " | " << std::setw(22) << final_nodes_count << "\n";
    std::cout << "Final Mean Epigenetic Expression  | " << std::setw(16) << 0.0 
              << " | " << std::setw(22) << final_alpha_epi << "\n";
    std::cout << "==================================================================\n";

    // Write JSON results
    std::ofstream out("experiments/exp_455_results.json");
    out << "{\n"
        << "  \"exp_id\": \"EXP-455\",\n"
        << "  \"static_final_loss\": " << (sum_loss_static / cfg.total_steps) << ",\n"
        << "  \"unbounded_final_loss\": " << (sum_loss_unbounded / cfg.total_steps) << ",\n"
        << "  \"static_tail_loss\": " << tail_loss_static << ",\n"
        << "  \"unbounded_tail_loss\": " << tail_loss_unbounded << ",\n"
        << "  \"unbounded_accuracy\": " << (static_cast<double>(correct_unbounded) / cfg.total_steps * 100.0) << ",\n"
        << "  \"unbounded_throughput\": " << (cfg.total_steps / time_unbounded) << ",\n"
        << "  \"final_active_nodes\": " << final_nodes_count << ",\n"
        << "  \"final_mean_alpha_epi\": " << final_alpha_epi << "\n"
        << "}\n";
    out.close();

    return 0;
}
