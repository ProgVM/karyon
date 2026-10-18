// ============================================================================
// EXP-596: SOVEREIGN TOTIPOTENT FIELD WITH RECIPROCAL TOP-DOWN FEEDBACK,
//          SOVEREIGN CONSTRAINT CONTROL, AND SYSTEM 2 LATENT ROLLOUT (STF-RTDF-S2)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 8, 27, 28, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
//
// Scientific Objectives of EXP-596:
// 1. SOLVING SEMANTIC DRIFT & BRIDGING MIN FE TO FINAL FE:
//    In EXP-595, transient FE hit 0.058 nats, but final FE was 2.11 nats and output
//    lacked high-level semantic coherence.
//
// 2. UNRESTRICTED CONTINUOUS TENSOR-GENERATOR:
//    Node interaction is governed by an unconstrained continuous tensor field:
//    F(u_a, u_b) = W_a u_a + W_b u_b + SiLU(W_gate u_a) * u_b + u_a * (M_bilinear u_b) + W_feedback e_topdown
//
// 3. RECIPROCAL TOP-DOWN PREDICTIVE FEEDBACK:
//    Higher semantic cognitive nodes send top-down contextual expectations e_topdown back
//    to lower sensory/cognitive nodes, actively constraining byte generation to semantic concepts.
//
// 4. SOVEREIGN CONSTRAINT CONTROL (DIMENSION & LEARNING RATE):
//    Organism dynamically controls its active dimension D_act in [64, 256] and local learning rate
//    eta_k based on local Free Energy gradients dF_k/dt.
//
// 5. SYSTEM 2 LATENT ROLLOUT (EXPECTED FREE ENERGY MINIMIZATION):
//    Before emitting a motor byte, the motor head conducts K=3 internal mental simulation
//    steps in latent space, selecting the candidate action that minimizes Expected Free Energy G.
// ============================================================================

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>
#include <numbers>
#include <omp.h>
#include <map>
#include <memory>
#include <string>

#include "karyon_core.h"

namespace karyon {

enum class NodeType : uint8_t {
    SENSORY = 0,
    COGNITIVE = 1,
    MOTOR_HEAD = 2
};

struct SovereignSuperPrimitive {
    size_t node_id;
    NodeType type;
    size_t input_node_a;
    size_t input_node_b;

    float coefficient;      // Output injection strength
    float causal_credit;    // Running vector causal efficacy
    float epigenetic_lock;  // Epigenetic protection level

    // Dynamic Memory & Physical Bandwidth
    float alpha_decay;            // Memory timescale
    std::vector<float> state_s;   // Internal memory state
    std::vector<float> latest_out;// Output vector Phi_k

    // Dynamic Sovereign Constraint Control
    size_t active_dim;            // Sovereign active dimension D_act in [64, max_dim]
    float local_learning_rate;    // Sovereign self-tuned learning rate
    float local_free_energy;      // Local predictive error trace

    // Unconstrained Continuous Tensor-Generator Substrates
    std::vector<float> W_a;       // Projection matrix A (dim x dim)
    std::vector<float> W_b;       // Projection matrix B (dim x dim)
    std::vector<float> W_gate;    // Non-linear SiLU gate matrix (dim x dim)
    std::vector<float> M_bilinear;// Bilinear tensor coupling matrix (dim x dim)
    std::vector<float> W_feedback;// Top-down feedback matrix (dim x dim)

    // Motor Readout Projections
    std::vector<float> motor_proj;

    SovereignSuperPrimitive(size_t id, NodeType t, size_t max_dim, size_t in_a, size_t in_b,
                            float init_alpha, std::mt19937& rng)
        : node_id(id), type(t), input_node_a(in_a), input_node_b(in_b),
          coefficient(0.40f), causal_credit(0.0f), epigenetic_lock(0.0f),
          alpha_decay(init_alpha), active_dim(max_dim), local_learning_rate(0.04f),
          local_free_energy(1.0f) {

        state_s.assign(max_dim, 0.0f);
        latest_out.assign(max_dim, 0.0f);

        std::normal_distribution<float> dist(0.0f, 0.02f);

        if (type == NodeType::SENSORY) {
            W_a.assign(CHANNEL_IO_DIM * max_dim, 0.0f);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>(c + 1) / static_cast<float>(CHANNEL_IO_DIM);
                for (size_t i = 0; i < max_dim; ++i) {
                    W_a[c * max_dim + i] = std::sin(phase * static_cast<float>(i + 1)) * 0.15f + dist(rng);
                }
            }
            W_b.clear();
        } else {
            W_a.assign(max_dim * max_dim, 0.0f);
            W_b.assign(max_dim * max_dim, 0.0f);
            W_gate.assign(max_dim * max_dim, 0.0f);
            M_bilinear.assign(max_dim * max_dim, 0.0f);
            W_feedback.assign(max_dim * max_dim, 0.0f);

            for (auto& w : W_a) w = dist(rng);
            for (auto& w : W_b) w = dist(rng);
            for (auto& w : W_gate) w = dist(rng);
            for (auto& w : M_bilinear) w = dist(rng);
            for (auto& w : W_feedback) w = dist(rng);
        }

        if (type == NodeType::MOTOR_HEAD) {
            motor_proj.assign(CHANNEL_IO_DIM * max_dim, 0.0f);
            for (auto& w : motor_proj) w = dist(rng);
        }
    }
};

struct SpeechMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
};

SpeechMetrics evaluate_speech(const std::string& text) {
    SpeechMetrics m;
    if (text.empty()) return m;
    std::map<char, size_t> counts;
    size_t whitespaces = 0;
    size_t alphabets = 0;

    for (char ch : text) {
        counts[ch]++;
        if (std::isspace(static_cast<unsigned char>(ch))) whitespaces++;
        if (std::isalpha(static_cast<unsigned char>(ch))) alphabets++;
    }

    m.unique_chars = counts.size();
    m.whitespace_ratio = static_cast<double>(whitespaces) / text.size();
    m.alphabet_ratio = static_cast<double>(alphabets) / text.size();

    double ent = 0.0;
    for (const auto& [ch, count] : counts) {
        double p = static_cast<double>(count) / text.size();
        if (p > 0.0) ent -= p * std::log2(p);
    }
    m.char_entropy = ent;
    return m;
}

class SovereignTotipotentOrganism {
public:
    SovereignTotipotentOrganism(size_t max_dim = 256) noexcept
        : max_dim_(max_dim), rng_(5960) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        min_free_energy_ = 999.0f;
        prev_free_energy_ = 1.0f;
        dt_eff_ = 0.040f;
        endogenous_beta_ = 4.0f;
        total_sub_ticks_ = 0;
        total_sprouts_ = 0;
        total_prunes_ = 0;

        unified_field_.assign(max_dim_, 0.0f);
        dfield_.assign(max_dim_, 0.0f);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        // Node 0: Sensory Inflow (Alpha = 0.15, Nyquist bounded)
        nodes_.emplace_back(0, NodeType::SENSORY, max_dim_, 0, 0, 0.15f, rng_);
        nodes_[0].coefficient = 1.0f;
        nodes_[0].epigenetic_lock = 0.95f;

        // Node 1: Fast Cognitive Recurrence
        nodes_.emplace_back(1, NodeType::COGNITIVE, max_dim_, 0, 0, 0.85f, rng_);
        nodes_[1].coefficient = 0.6f;
        nodes_[1].epigenetic_lock = 0.6f;

        // Node 2: Deep Semantic Context Reservoir
        nodes_.emplace_back(2, NodeType::COGNITIVE, max_dim_, 1, 0, 0.995f, rng_);
        nodes_[2].coefficient = 0.5f;
        nodes_[2].epigenetic_lock = 0.6f;

        // Node 3: Sovereign Volitional Motor Readout Head
        nodes_.emplace_back(3, NodeType::MOTOR_HEAD, max_dim_, 2, 1, 0.35f, rng_);
        nodes_[3].coefficient = 1.0f;
        nodes_[3].epigenetic_lock = 0.8f;
    }

    void sprout_new_operator(size_t parent_a, size_t parent_b) {
        if (nodes_.size() >= 16) return;
        size_t new_id = nodes_.size();

        NodeType t = NodeType::COGNITIVE;
        size_t motor_count = 0;
        for (const auto& n : nodes_) if (n.type == NodeType::MOTOR_HEAD) motor_count++;
        if (motor_count < 3 && (rng_() % 4 == 0)) {
            t = NodeType::MOTOR_HEAD;
        }

        float init_alpha = 0.70f;
        if (t == NodeType::MOTOR_HEAD) {
            std::uniform_real_distribution<float> m_dist(0.10f, 0.45f);
            init_alpha = m_dist(rng_);
        } else {
            std::uniform_real_distribution<float> c_dist(0.25f, 0.995f);
            init_alpha = c_dist(rng_);
        }

        nodes_.emplace_back(new_id, t, max_dim_, parent_a, parent_b, init_alpha, rng_);
        auto& new_node = nodes_.back();

        new_node.coefficient = 0.12f;
        new_node.causal_credit = 0.0f;
        new_node.epigenetic_lock = 0.0f;
        total_sprouts_++;

        std::cout << "\n[+] SOVEREIGN GENESIS: Synthesized "
                  << (t == NodeType::MOTOR_HEAD ? "MotorHead [" : "CognitiveNode [")
                  << new_id << "] parents=(" << parent_a << ", " << parent_b << ") | Tau(alpha): "
                  << init_alpha << " | FE: " << free_energy_ << "\n";
    }

    void prune_harmful_operators() {
        if (nodes_.size() <= 4) return;

        for (auto it = nodes_.begin() + 4; it != nodes_.end(); ) {
            if (it->causal_credit < -0.04f && it->epigenetic_lock < 0.15f) {
                std::cout << "\n[-] SOVEREIGN PRUNING: Dissolved harmful Node [" << it->node_id
                          << "] (Type: " << (it->type == NodeType::MOTOR_HEAD ? "Motor" : "Cognitive")
                          << ", Credit: " << it->causal_credit
                          << ", Lock: " << (it->epigenetic_lock * 100.0f) << "%)\n";
                size_t erased_id = it->node_id;
                it = nodes_.erase(it);
                total_prunes_++;

                for (auto& n : nodes_) {
                    if (n.input_node_a == erased_id) n.input_node_a = 0;
                    if (n.input_node_b == erased_id) n.input_node_b = 1;
                }
            } else {
                ++it;
            }
        }
    }

    // Fast Vectorized Forward Pass with OpenMP + SIMD
    void forward_pass(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * max_dim_;

        // 1. Top-Down Context Expectations from Higher Cognitive Nodes
        std::vector<std::vector<float>> topdown_signals(nodes_.size(), std::vector<float>(max_dim_, 0.0f));

        for (size_t k = nodes_.size(); k-- > 0;) {
            const auto& node = nodes_[k];
            if (node.type == NodeType::COGNITIVE || node.type == NodeType::MOTOR_HEAD) {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                size_t p_b = std::min(node.input_node_b, nodes_.size() - 1);
                size_t d_act = node.active_dim;

                if (p_a < k && !node.W_feedback.empty()) {
                    for (size_t i = 0; i < d_act; ++i) {
                        float s = 0.0f;
                        const float* w_row = &node.W_feedback[i * max_dim_];
                        for (size_t j = 0; j < d_act; ++j) s += w_row[j] * node.latest_out[j];
                        topdown_signals[p_a][i] += 0.15f * s;
                    }
                }
                if (p_b < k && !node.W_feedback.empty()) {
                    for (size_t i = 0; i < d_act; ++i) {
                        float s = 0.0f;
                        const float* w_row = &node.W_feedback[i * max_dim_];
                        for (size_t j = 0; j < d_act; ++j) s += w_row[j] * node.latest_out[j];
                        topdown_signals[p_b][i] += 0.15f * s;
                    }
                }
            }
        }

        // 2. Fast Forward Propagation across DAG
        std::vector<float> field_accel(max_dim_, 0.0f);

        for (size_t k = 0; k < nodes_.size(); ++k) {
            auto& node = nodes_[k];
            if (std::abs(node.coefficient) < 1e-5f) continue;

            size_t d_act = node.active_dim;

            if (node.type == NodeType::SENSORY) {
                for (size_t i = 0; i < d_act; ++i) {
                    float in_val = node.W_a[in_offset + i] + topdown_signals[k][i];
                    node.state_s[i] = node.alpha_decay * node.state_s[i] + (1.0f - node.alpha_decay) * in_val;
                    node.latest_out[i] = std::tanh(node.state_s[i]);
                    field_accel[i] += node.coefficient * node.latest_out[i];
                }
            } else {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                size_t p_b = std::min(node.input_node_b, nodes_.size() - 1);
                const float* u_a = nodes_[p_a].latest_out.data();
                const float* u_b = nodes_[p_b].latest_out.data();

                // RMS Energy Normalization
                float sum_sq_a = 0.0f, sum_sq_b = 0.0f;
                for (size_t i = 0; i < d_act; ++i) {
                    sum_sq_a += u_a[i] * u_a[i];
                    sum_sq_b += u_b[i] * u_b[i];
                }
                float inv_rms_a = 1.0f / (std::sqrt(sum_sq_a / static_cast<float>(d_act)) + SAFETY_EPSILON);
                float inv_rms_b = 1.0f / (std::sqrt(sum_sq_b / static_cast<float>(d_act)) + SAFETY_EPSILON);

                std::vector<float> hat_u_a(d_act);
                std::vector<float> hat_u_b(d_act);
                for (size_t i = 0; i < d_act; ++i) {
                    hat_u_a[i] = u_a[i] * inv_rms_a;
                    hat_u_b[i] = u_b[i] * inv_rms_b;
                }

                #pragma omp parallel for schedule(static) if(d_act >= 128)
                for (size_t i = 0; i < d_act; ++i) {
                    float s_a = 0.0f, s_b = 0.0f, s_gate = 0.0f, s_bilinear = 0.0f;
                    const float* wa_row = &node.W_a[i * max_dim_];
                    const float* wb_row = &node.W_b[i * max_dim_];
                    const float* wg_row = &node.W_gate[i * max_dim_];
                    const float* mb_row = &node.M_bilinear[i * max_dim_];

                    for (size_t j = 0; j < d_act; ++j) {
                        s_a += wa_row[j] * hat_u_a[j];
                        s_b += wb_row[j] * hat_u_b[j];
                        s_gate += wg_row[j] * hat_u_a[j];
                        s_bilinear += mb_row[j] * hat_u_b[j];
                    }

                    // SiLU Gating & Bilinear Concept Coupling
                    float sig_a = 1.0f / (1.0f + std::exp(-s_gate));
                    float silu_gate = (s_gate * sig_a) * hat_u_b[i];

                    float tensor_val = s_a + s_b + silu_gate + (hat_u_a[i] * s_bilinear) + topdown_signals[k][i];

                    // Attractor Snapping
                    float snapped = tensor_val / std::sqrt(1.0f + 0.5f * tensor_val * tensor_val);

                    // Dynamic Memory Integration
                    node.state_s[i] = node.alpha_decay * node.state_s[i] + (1.0f - node.alpha_decay) * snapped;
                    node.latest_out[i] = std::tanh(node.state_s[i]);

                    if (node.type != NodeType::MOTOR_HEAD) {
                        #pragma omp atomic
                        field_accel[i] += node.coefficient * node.latest_out[i];
                    }
                }
            }
        }

        // Field Integration
        for (size_t i = 0; i < max_dim_; ++i) {
            dfield_[i] = (dfield_[i] + field_accel[i] * dt_eff_) * 0.980f;
            unified_field_[i] = std::tanh(unified_field_[i] + dfield_[i] * dt_eff_);
        }
    }

    // System 2 Latent Attractor Rollout (Expected Free Energy Minimization)
    size_t system2_mental_rollout(const std::array<float, CHANNEL_IO_DIM>& base_logits) noexcept {
        // Select top-4 candidate motor bytes from base logits
        std::vector<std::pair<float, size_t>> candidates;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            candidates.push_back({base_logits[c], c});
        }
        std::partial_sort(candidates.begin(), candidates.begin() + 4, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float best_efe = 1e9f;
        size_t best_action = candidates[0].second;

        // Save organism state snapshot for non-destructive mental simulation
        auto nodes_snapshot = nodes_;
        auto field_snapshot = unified_field_;

        for (size_t k = 0; k < 4; ++k) {
            size_t cand_byte = candidates[k].second;

            // Conduct K=2 mental simulation steps into latent future
            float mental_fe = 0.0f;
            for (size_t t = 0; t < 2; ++t) {
                forward_pass(static_cast<uint8_t>(cand_byte));

                // Predict expected target byte probability
                float sum_exp = 0.0f;
                std::array<float, CHANNEL_IO_DIM> cand_logits{};
                cand_logits.fill(0.0f);

                for (const auto& node : nodes_) {
                    if (node.type == NodeType::MOTOR_HEAD) {
                        const float* m_proj = node.motor_proj.data();
                        const float* head_out = node.latest_out.data();
                        size_t d_act = node.active_dim;

                        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                            float dot = 0.0f;
                            const float* w_ptr = &m_proj[c * max_dim_];
                            for (size_t i = 0; i < d_act; ++i) dot += w_ptr[i] * head_out[i];
                            cand_logits[c] += node.coefficient * (dot / std::sqrt(static_cast<float>(d_act)));
                        }
                    }
                }

                float max_res = *std::max_element(cand_logits.begin(), cand_logits.end());
                for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) sum_exp += std::exp(cand_logits[c] - max_res);

                float p_cand = std::exp(cand_logits[cand_byte] - max_res) / (sum_exp + SAFETY_EPSILON);
                mental_fe += -std::log(p_cand + SAFETY_EPSILON);
            }

            if (mental_fe < best_efe) {
                best_efe = mental_fe;
                best_action = cand_byte;
            }

            // Restore state snapshot
            nodes_ = nodes_snapshot;
            unified_field_ = field_snapshot;
        }

        return best_action;
    }

    size_t step(uint8_t input_byte) noexcept {
        // Forward Pass with Sub-Stepping
        size_t sub_ticks = (free_energy_ > 2.5f) ? 2 : 1;
        total_sub_ticks_ += sub_ticks;

        for (size_t tick = 0; tick < sub_ticks; ++tick) {
            forward_pass(input_byte);
        }

        // Motor Readout Aggregation
        std::array<float, CHANNEL_IO_DIM> aggregated_logits{};
        aggregated_logits.fill(0.0f);

        for (const auto& node : nodes_) {
            if (node.type == NodeType::MOTOR_HEAD && std::abs(node.coefficient) > 1e-4f) {
                const float* m_proj = node.motor_proj.data();
                const float* head_out = node.latest_out.data();
                size_t d_act = node.active_dim;

                for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                    float dot = 0.0f;
                    const float* w_ptr = &m_proj[c * max_dim_];
                    for (size_t i = 0; i < d_act; ++i) dot += w_ptr[i] * head_out[i];
                    aggregated_logits[c] += node.coefficient * (dot / std::sqrt(static_cast<float>(d_act)));
                }
            }
        }

        // Resonances with Refractory Momentum
        float max_res = -1e9f;
        std::array<float, CHANNEL_IO_DIM> resonance{};
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float r_c = aggregated_logits[c] - anchor_momentum_[c];
            resonance[c] = r_c;
            if (r_c > max_res) max_res = r_c;
        }

        // Softmax & Probabilities
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        // System 2 Latent Rollout Selection if Free Energy is elevated
        size_t chosen = 0;
        if (free_energy_ > 1.8f) {
            chosen = system2_mental_rollout(resonance);
        } else {
            float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            float cum = 0.0f;
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                cum += latest_probabilities_[c];
                if (cum >= r) { chosen = c; break; }
            }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f;

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float base_lr = 0.04f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;
        if (free_energy_ < min_free_energy_) min_free_energy_ = free_energy_;

        float fe_delta = free_energy_ - prev_free_energy_;

        float f_sq = 0.0f;
        for (size_t i = 0; i < max_dim_; ++i) f_sq += unified_field_[i] * unified_field_[i];
        float mean_f_sq = f_sq / static_cast<float>(max_dim_);

        // 1. Motor Error Gradient Backpropagation
        std::vector<float> G_motor(max_dim_, 0.0f);

        for (auto& node : nodes_) {
            if (node.type == NodeType::MOTOR_HEAD) {
                float plastic = (1.0f - node.epigenetic_lock);
                const float* head_out = node.latest_out.data();
                size_t d_act = node.active_dim;

                for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                    float target_val = (c == target_byte) ? 1.0f : 0.0f;
                    float err = target_val - latest_probabilities_[c];

                    if (std::abs(err) > 0.0005f) {
                        float alpha = node.local_learning_rate * err * node.coefficient;
                        float* m_ptr = &node.motor_proj[c * max_dim_];

                        for (size_t i = 0; i < d_act; ++i) {
                            G_motor[i] += err * node.coefficient * m_ptr[i];
                            float oja_decay = 0.02f * latest_probabilities_[c] * mean_f_sq * m_ptr[i];
                            m_ptr[i] += alpha * head_out[i] - plastic * oja_decay;
                        }
                    }
                }
            }
        }

        float g_norm_sq = 0.0f;
        for (size_t i = 0; i < max_dim_; ++i) g_norm_sq += G_motor[i] * G_motor[i];
        float g_inv_norm = 1.0f / (std::sqrt(g_norm_sq) + SAFETY_EPSILON);

        // 2. Continuous Causal Credit Assignment, Sovereign Dimension Control & Substrate Plasticity
        for (auto& node : nodes_) {
            size_t d_act = node.active_dim;

            float alignment = 0.0f, out_sq = 0.0f;
            for (size_t i = 0; i < d_act; ++i) {
                alignment += node.latest_out[i] * G_motor[i];
                out_sq += node.latest_out[i] * node.latest_out[i];
            }
            float out_norm = std::sqrt(out_sq) + SAFETY_EPSILON;
            float local_causal = (alignment / out_norm) * g_inv_norm * 5.0f;

            node.causal_credit = 0.985f * node.causal_credit + 0.015f * local_causal;
            node.local_free_energy = 0.95f * node.local_free_energy + 0.05f * ce;

            // Epigenetic Protection
            if (node.causal_credit > 0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock + 0.002f, 0.0f, 0.98f);
            } else if (node.causal_credit < -0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock - 0.005f, 0.0f, 0.98f);
            }

            float plastic = (1.0f - node.epigenetic_lock);

            // Sovereign Self-Tuning Learning Rate
            if (fe_delta > 0.05f) {
                node.local_learning_rate = std::clamp(node.local_learning_rate * 0.95f, 0.005f, 0.10f);
            } else {
                node.local_learning_rate = std::clamp(node.local_learning_rate * 1.02f, 0.005f, 0.10f);
            }

            // Sovereign Dimension Scaling: If credit is high and FE is elevated, expand active dimension!
            if (node.causal_credit > 0.04f && ce > 1.2f && node.active_dim < max_dim_) {
                node.active_dim = std::min(node.active_dim + 16, max_dim_);
            } else if (node.causal_credit < -0.02f && node.active_dim > 64) {
                node.active_dim = std::max(node.active_dim - 16, static_cast<size_t>(64));
            }

            node.coefficient += node.local_learning_rate * 0.05f * plastic * node.causal_credit;
            node.coefficient = std::clamp(node.coefficient, 0.05f, 2.0f);

            // Nyquist Bandwidth Law Enforcement
            if (node.type == NodeType::SENSORY) {
                node.alpha_decay = std::clamp(node.alpha_decay + node.local_learning_rate * 0.01f * plastic * local_causal, 0.01f, 0.30f);
            } else if (node.type == NodeType::MOTOR_HEAD) {
                node.alpha_decay = std::clamp(node.alpha_decay + node.local_learning_rate * 0.01f * plastic * local_causal, 0.05f, 0.50f);
            } else {
                node.alpha_decay = std::clamp(node.alpha_decay + node.local_learning_rate * 0.01f * plastic * local_causal, 0.15f, 0.9995f);
            }

            // Substrate Weight Updates with Oja Normalization
            if (node.type == NodeType::SENSORY) {
                const size_t in_offset = static_cast<size_t>(input_byte) * max_dim_;
                for (size_t i = 0; i < d_act; ++i) {
                    size_t s_idx = in_offset + i;
                    float oja_decay = 0.02f * mean_f_sq * node.W_a[s_idx];
                    node.W_a[s_idx] += node.local_learning_rate * 0.05f * plastic * (G_motor[i] - oja_decay);
                }
            } else {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                size_t p_b = std::min(node.input_node_b, nodes_.size() - 1);
                const float* u_a = nodes_[p_a].latest_out.data();
                const float* u_b = nodes_[p_b].latest_out.data();

                for (size_t i = 0; i < d_act; ++i) {
                    for (size_t j = 0; j < d_act; ++j) {
                        size_t idx = i * max_dim_ + j;
                        float oja = mean_f_sq * node.W_a[idx];

                        node.W_a[idx] += node.local_learning_rate * 0.02f * plastic * (G_motor[i] * u_a[j] - 0.01f * oja);
                        node.W_b[idx] += node.local_learning_rate * 0.02f * plastic * (G_motor[i] * u_b[j] - 0.01f * oja);
                        node.W_gate[idx] += node.local_learning_rate * 0.02f * plastic * (G_motor[i] * u_a[j] - 0.01f * oja);
                        node.M_bilinear[idx] += node.local_learning_rate * 0.02f * plastic * (G_motor[i] * u_b[j] - 0.01f * oja);

                        // Top-Down Reciprocal Feedback Learning
                        node.W_feedback[idx] += node.local_learning_rate * 0.015f * plastic * (u_a[i] * G_motor[j]);
                    }
                }
            }
        }

        // 3. Periodic Morphogenetic Audit & Structure Evolution
        if (total_steps_ % 20000 == 0) {
            evaluate_and_evolve();
        }
    }

    void evaluate_and_evolve() {
        std::cout << "\n--- [SOVEREIGN TOTIPOTENT AUDIT @ Step " << total_steps_ << "] ---\n";
        std::cout << "  Free Energy: " << free_energy_
                  << " | Min FE: " << min_free_energy_
                  << " | Active Nodes: " << nodes_.size()
                  << " | Sprouts: " << total_sprouts_
                  << " | Prunes: " << total_prunes_ << "\n";

        for (const auto& node : nodes_) {
            std::string t_str = "Cognitive";
            if (node.type == NodeType::SENSORY) t_str = "Sensory";
            else if (node.type == NodeType::MOTOR_HEAD) t_str = "MotorHead";

            std::cout << "  Node [" << node.node_id << "] " << t_str
                      << " (" << node.input_node_a << ", " << node.input_node_b << ")"
                      << " | Coeff: " << node.coefficient
                      << " | Tau(Alpha): " << node.alpha_decay
                      << " | Credit: " << node.causal_credit
                      << " | Lock: " << (node.epigenetic_lock * 100.0f) << "%"
                      << " | Dim: " << node.active_dim
                      << " | LR: " << node.local_learning_rate << "\n";
        }

        prune_harmful_operators();

        if (free_energy_ > 1.6f && nodes_.size() < 14) {
            size_t p_a = rng_() % nodes_.size();
            size_t p_b = rng_() % nodes_.size();
            sprout_new_operator(p_a, p_b);
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_min_free_energy() const noexcept { return min_free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }
    size_t get_active_nodes() const noexcept { return nodes_.size(); }
    size_t get_max_dim() const noexcept { return max_dim_; }
    size_t get_sprouts() const noexcept { return total_sprouts_; }
    size_t get_prunes() const noexcept { return total_prunes_; }

private:
    size_t max_dim_;
    size_t total_steps_;
    size_t total_sub_ticks_;
    size_t total_sprouts_;
    size_t total_prunes_;
    float free_energy_;
    float min_free_energy_;
    float prev_free_energy_;
    float dt_eff_;
    float endogenous_beta_;
    std::mt19937 rng_;

    std::vector<float> unified_field_;
    std::vector<float> dfield_;

    std::vector<SovereignSuperPrimitive> nodes_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-596: SOVEREIGN TOTIPOTENT FIELD WITH RECIPROCAL FEEDBACK     ===\n";
    std::cout << "=== Continuous Tensor Generator + Top-Down Feedback + Sovereign Dim  ===\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) {
        text_path = "data/web_downloads/alpaca_data.json";
        file.open(text_path, std::ios::binary);
    }

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::cout << "[+] Corpus loaded: " << corpus.size() << " bytes.\n\n";
    } else {
        std::string fallback = "The capital of France is Paris. Language reaches zero free energy when context never decays.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 200000;

    std::cout << ">>> Initializing Sovereign Totipotent Organism (Max Dim: 256)...\n";
    karyon::SovereignTotipotentOrganism sovereign_org(256);

    std::cout << ">>> Launching Sovereign Autopoietic Learning (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        sovereign_org.step(in_b);
        sovereign_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 40000 == 0) {
            std::cout << "  >>> Step " << (step + 1)
                      << " | Free Energy: " << sovereign_org.get_free_energy()
                      << " | Min FE: " << sovereign_org.get_min_free_energy()
                      << " | Beta: " << sovereign_org.get_beta()
                      << " | Nodes: " << sovereign_org.get_active_nodes()
                      << " | Sprouts: " << sovereign_org.get_sprouts()
                      << " | Prunes: " << sovereign_org.get_prunes() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(train_steps) / duration;

    // Directed Speech Generation Test
    std::string prompt = "The capital of France is ";
    std::string speech_output = prompt;
    uint8_t curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t next_act = sovereign_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-596 vs EXP-595 vs EXP-594 vs EXP-593) ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | Min FE (nats) | Active Nodes | Sprouts | Prunes | Alphabet %\n";
    std::cout << "--------------------------------|--------------------|-----------------|---------------|--------------|---------|--------|-----------\n";
    std::cout << "STF-RTDF-S2 Sovereign (EXP-596) | " << throughput << "          | " << sovereign_org.get_free_energy() << "          | " << sovereign_org.get_min_free_energy() << "       | " << sovereign_org.get_active_nodes() << "            | " << sovereign_org.get_sprouts() << "       | " << sovereign_org.get_prunes() << "      | " << (metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "NS-OL-TF Nyquist-Omni (EXP-595) | 632.02             | 2.1109          | 0.0586        | 11           | 13      | 6      | 82.67%\n";
    std::cout << "OL-TSPF Omni-Layer (EXP-594)    | 560.96             | 3.4011          | 0.3279        | 7            | 12      | 9      | 80.89%\n";
    std::cout << "EC-SP-DAG Conserved (EXP-593)   | 394.04             | 3.1598          | 3.0623        | 15           | 15      | 3      | 79.56%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
