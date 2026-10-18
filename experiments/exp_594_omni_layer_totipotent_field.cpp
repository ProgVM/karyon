// ============================================================================
// EXP-594: OMNI-LAYER TOTIPOTENT SUPER-PRIMITIVE FIELD (OL-TSPF)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 8, 27, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
//
// Scientific Objectives of EXP-594:
// 1. OMNI-LAYER SOVEREIGNTY:
//    Eliminate the artificial divide where sensory and motor layers are rigid,
//    top-down hardcoded matrices while only hidden nodes evolve.
//    Now, the entire perception-cognition-action continuum is composed of
//    totipotent super-primitives:
//    - Sensory Inflow Nodes: map byte stream into continuous field coordinates;
//    - Cognitive DAG Nodes: combine, associate, differentiate, filter, integrate;
//    - Motor Outflow Nodes: volitional readout heads are THEMSELVES super-primitives
//      with dynamic routing, selecting which cognitive attractors to express into
//      action logits, rather than a single static flat matrix!
//
// 2. EXPANDED ALGEBRAIC CONTINUUM OPERATORS (THE TOTIPOTENT BASIS):
//    Beyond simple linear addition and elementwise multiplication, the node
//    possesses full continuum coordinates over 6 fundamental operations:
//    (a) Linear Projection: W_a * hat{u}_a + W_b * hat{u}_b
//    (b) Associative Inner-Product Resonant Routing (Attention):
//        dot_ab = <hat{u}_a, hat{u}_b> / sqrt(D); attn_out = dot_ab * hat{u}_b
//    (c) Multiplicative Outer/Bilinear Concept Binding:
//        bilin = hat{u}_a \odot hat{u}_b
//    (d) Differential / Contrastive Rate of Change (Temporal Gradient):
//        diff = hat{u}_a - hat{u}_b
//    (e) Non-linear Gating (SwiGLU / Selective Filter):
//        gate = silu(hat{u}_a) \odot hat{u}_b
//    (f) Harmonic Phase Wave Packet:
//        wave = sin(Omega * hat{u}_a + Phi)
//    (g) Bounded Attractor Snapping:
//        snap = composed / sqrt(1 + composed^2)
//
// 3. PHYSICAL ENERGY CONSERVATION (First Law):
//    Strict RMS normalization preserves unit energy:
//    hat{u} = u / sqrt(mean(u^2) + eps)
//    guaranteeing that even recursive self-interactions (u_k * u_k)
//    cannot trigger numerical divergence!
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

struct OmniSuperPrimitive {
    size_t node_id;
    NodeType type;
    size_t input_node_a; // Index of parent node A in DAG
    size_t input_node_b; // Index of parent node B in DAG

    float coefficient;      // Output injection strength
    float causal_credit;    // Running vector causal efficacy <Phi_k, G_motor>
    float epigenetic_lock;  // 0.0 (plastic) -> 1.0 (shielded invariant)

    // Memory State & Dynamic Timescale
    float alpha_decay;            // Memory persistence: 0.05 -> 0.9995
    std::vector<float> state_s;   // Internal persistent memory state s_k in R^D
    std::vector<float> latest_out;// Output vector Phi_k in R^D

    // Extended Functional Continuum Coordinates (The Totipotent Basis)
    float mix_linear;     // Weight of linear projection
    float mix_attention;  // Weight of inner-product resonant routing
    float mix_bilinear;   // Weight of multiplicative concept binding
    float mix_diff;       // Weight of differential contrast (u_a - u_b)
    float mix_gating;     // Weight of selective gating (silu(u_a) * u_b)
    float mix_phase;      // Weight of harmonic wave packets
    float lambda_snap;    // Attractor snapping coordinate

    // Parametric Substrates
    std::vector<float> W_a;       // Projection / sensory weights
    std::vector<float> W_b;       // Projection weights
    std::vector<float> phase_freq;// Harmonic frequencies (dim)
    std::vector<float> phase_bias;// Harmonic offsets (dim)

    // Specific to MOTOR_HEAD: maps internal field to 258 byte logits
    std::vector<float> motor_proj;// (258 * dim) if MOTOR_HEAD

    OmniSuperPrimitive(size_t id, NodeType t, size_t dim, size_t in_a, size_t in_b,
                       float init_alpha, std::mt19937& rng)
        : node_id(id), type(t), input_node_a(in_a), input_node_b(in_b),
          coefficient(0.30f), causal_credit(0.0f), epigenetic_lock(0.0f),
          alpha_decay(init_alpha) {

        state_s.assign(dim, 0.0f);
        latest_out.assign(dim, 0.0f);

        std::normal_distribution<float> dist(0.0f, 0.02f);
        std::uniform_real_distribution<float> udist(0.1f, 0.9f);

        // Normalized initial operator coordinates
        mix_linear = 0.35f;
        mix_attention = 0.25f;
        mix_bilinear = 0.25f;
        mix_diff = 0.20f;
        mix_gating = 0.25f;
        mix_phase = 0.20f;
        lambda_snap = 0.50f;

        if (type == NodeType::SENSORY) {
            W_a.assign(CHANNEL_IO_DIM * dim, 0.0f);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>(c + 1) / static_cast<float>(CHANNEL_IO_DIM);
                for (size_t i = 0; i < dim; ++i) {
                    W_a[c * dim + i] = std::sin(phase * static_cast<float>(i + 1)) * 0.15f + dist(rng);
                }
            }
            W_b.clear();
        } else {
            W_a.assign(dim * dim, 0.0f);
            W_b.assign(dim * dim, 0.0f);
            for (auto& w : W_a) w = dist(rng);
            for (auto& w : W_b) w = dist(rng);
        }

        phase_freq.assign(dim, 0.0f);
        phase_bias.assign(dim, 0.0f);
        for (size_t i = 0; i < dim; ++i) {
            phase_freq[i] = udist(rng) * 3.14159f;
            phase_bias[i] = udist(rng) * 6.28318f;
        }

        if (type == NodeType::MOTOR_HEAD) {
            motor_proj.assign(CHANNEL_IO_DIM * dim, 0.0f);
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

class OmniOrganism {
public:
    OmniOrganism(size_t dim = 256) noexcept
        : dim_(dim), rng_(5940) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        dt_eff_ = 0.040f;
        endogenous_beta_ = 4.0f;
        total_sub_ticks_ = 0;
        total_sprouts_ = 0;
        total_prunes_ = 0;

        unified_field_.assign(dim_, 0.0f);
        dfield_.assign(dim_, 0.0f);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        // Node 0: Sensory Inflow (Alpha = 0.10, fast sensor)
        nodes_.emplace_back(0, NodeType::SENSORY, dim_, 0, 0, 0.10f, rng_);
        nodes_[0].coefficient = 1.0f;
        nodes_[0].epigenetic_lock = 0.95f;

        // Node 1: Fast Cognitive Recurrence (Alpha = 0.85)
        nodes_.emplace_back(1, NodeType::COGNITIVE, dim_, 0, 0, 0.85f, rng_);
        nodes_[1].coefficient = 0.6f;
        nodes_[1].epigenetic_lock = 0.6f;

        // Node 2: Ultra-Slow Persistent Context Reservoir (Alpha = 0.995)
        nodes_.emplace_back(2, NodeType::COGNITIVE, dim_, 1, 0, 0.995f, rng_);
        nodes_[2].coefficient = 0.5f;
        nodes_[2].epigenetic_lock = 0.6f;

        // Node 3: Sovereign Volitional Motor Readout Head (Motor Super-Primitive!)
        // It directly selects and decodes cognitive attractors from Node 2 and Node 1
        nodes_.emplace_back(3, NodeType::MOTOR_HEAD, dim_, 2, 1, 0.40f, rng_);
        nodes_[3].coefficient = 1.0f;
        nodes_[3].epigenetic_lock = 0.8f;
    }

    void sprout_new_operator(size_t parent_a, size_t parent_b) {
        if (nodes_.size() >= 24) return;
        size_t new_id = nodes_.size();

        // 80% cognitive, 20% second motor head
        NodeType t = NodeType::COGNITIVE;
        size_t motor_count = 0;
        for (const auto& n : nodes_) if (n.type == NodeType::MOTOR_HEAD) motor_count++;
        if (motor_count < 3 && (rng_() % 5 == 0)) {
            t = NodeType::MOTOR_HEAD;
        }

        std::uniform_real_distribution<float> alpha_dist(0.15f, 0.995f);
        float init_alpha = alpha_dist(rng_);

        nodes_.emplace_back(new_id, t, dim_, parent_a, parent_b, init_alpha, rng_);
        auto& new_node = nodes_.back();

        new_node.coefficient = 0.08f;
        new_node.causal_credit = 0.0f;
        new_node.epigenetic_lock = 0.0f;
        total_sprouts_++;

        std::cout << "\n[+] OMNI-LAYER GENESIS: Synthesized "
                  << (t == NodeType::MOTOR_HEAD ? "MotorHead [" : "CognitiveNode [")
                  << new_id << "] parents=(" << parent_a << ", " << parent_b << ") | Tau(alpha): "
                  << init_alpha << " | FE: " << free_energy_ << "\n";
    }

    void prune_harmful_operators() {
        if (nodes_.size() <= 4) return; // Protect primordial foundation (0, 1, 2, 3)

        for (auto it = nodes_.begin() + 4; it != nodes_.end(); ) {
            if (it->causal_credit < -0.04f && it->epigenetic_lock < 0.15f) {
                std::cout << "\n[-] OMNI-LAYER PRUNING: Dissolved harmful Node [" << it->node_id
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

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * dim_;

        size_t sub_ticks = 1;
        if (free_energy_ > 3.8f) sub_ticks = 4;
        else if (free_energy_ > 3.0f) sub_ticks = 3;
        else if (free_energy_ > 2.0f) sub_ticks = 2;

        total_sub_ticks_ += sub_ticks;

        for (size_t tick = 0; tick < sub_ticks; ++tick) {
            std::vector<float> field_accel(dim_, 0.0f);

            // Forward Pass across Omni-Layer DAG
            for (size_t k = 0; k < nodes_.size(); ++k) {
                auto& node = nodes_[k];
                if (std::abs(node.coefficient) < 1e-5f) continue;

                if (node.type == NodeType::SENSORY) {
                    for (size_t i = 0; i < dim_; ++i) {
                        float in_val = node.W_a[in_offset + i];
                        node.state_s[i] = node.alpha_decay * node.state_s[i] + (1.0f - node.alpha_decay) * in_val;
                        node.latest_out[i] = std::tanh(node.state_s[i]);
                        field_accel[i] += node.coefficient * node.latest_out[i];
                    }
                } else {
                    size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                    size_t p_b = std::min(node.input_node_b, nodes_.size() - 1);
                    const float* u_a = nodes_[p_a].latest_out.data();
                    const float* u_b = nodes_[p_b].latest_out.data();

                    // === ENERGY CONSERVATION: RMS Normalization ===
                    float sum_sq_a = 0.0f, sum_sq_b = 0.0f;
                    for (size_t i = 0; i < dim_; ++i) {
                        sum_sq_a += u_a[i] * u_a[i];
                        sum_sq_b += u_b[i] * u_b[i];
                    }
                    float inv_rms_a = 1.0f / (std::sqrt(sum_sq_a / static_cast<float>(dim_)) + SAFETY_EPSILON);
                    float inv_rms_b = 1.0f / (std::sqrt(sum_sq_b / static_cast<float>(dim_)) + SAFETY_EPSILON);

                    std::vector<float> hat_u_a(dim_);
                    std::vector<float> hat_u_b(dim_);
                    for (size_t i = 0; i < dim_; ++i) {
                        hat_u_a[i] = u_a[i] * inv_rms_a;
                        hat_u_b[i] = u_b[i] * inv_rms_b;
                    }

                    // 1. Linear Projection
                    std::vector<float> lin(dim_, 0.0f);
                    for (size_t i = 0; i < dim_; ++i) {
                        float s_a = 0.0f, s_b = 0.0f;
                        for (size_t j = 0; j < dim_; ++j) {
                            s_a += node.W_a[i * dim_ + j] * hat_u_a[j];
                            s_b += node.W_b[i * dim_ + j] * hat_u_b[j];
                        }
                        lin[i] = s_a + s_b;
                    }

                    // 2. Resonant Inner-Product Attention Routing (<hat_u_a, hat_u_b> / sqrt(D))
                    float dot_ab = 0.0f;
                    for (size_t i = 0; i < dim_; ++i) dot_ab += hat_u_a[i] * hat_u_b[i];
                    float attn_scale = std::tanh(dot_ab / std::sqrt(static_cast<float>(dim_)));

                    // 3. Full Totipotent Basis Evaluation
                    for (size_t i = 0; i < dim_; ++i) {
                        float attn_out = attn_scale * hat_u_b[i];
                        float binding = hat_u_a[i] * hat_u_b[i];
                        float diff = hat_u_a[i] - hat_u_b[i];
                        
                        // SiLU Gating: silu(x) = x * sigmoid(x)
                        float sig = 1.0f / (1.0f + std::exp(-hat_u_a[i]));
                        float gating = (hat_u_a[i] * sig) * hat_u_b[i];

                        float phase_wave = std::sin(node.phase_freq[i] * hat_u_a[i] + node.phase_bias[i]);

                        float composed = node.mix_linear * lin[i] +
                                         node.mix_attention * attn_out +
                                         node.mix_bilinear * binding +
                                         node.mix_diff * diff +
                                         node.mix_gating * gating +
                                         node.mix_phase * phase_wave;

                        // Non-linear Attractor Snapping
                        float nonlin = (1.0f - node.lambda_snap) * composed +
                                       node.lambda_snap * (composed / std::sqrt(1.0f + composed * composed));

                        // Dynamic Memory Integration
                        node.state_s[i] = node.alpha_decay * node.state_s[i] + (1.0f - node.alpha_decay) * nonlin;
                        node.latest_out[i] = std::tanh(node.state_s[i]);

                        if (node.type != NodeType::MOTOR_HEAD) {
                            field_accel[i] += node.coefficient * node.latest_out[i];
                        }
                    }
                }
            }

            // Field Integration
            float kinetic = 0.0f;
            for (size_t i = 0; i < dim_; ++i) kinetic += dfield_[i] * dfield_[i];
            float mean_kin = kinetic / static_cast<float>(dim_);
            dt_eff_ = std::clamp(0.040f / (1.0f + 2.0f * mean_kin), 0.010f, 0.080f);

            for (size_t i = 0; i < dim_; ++i) {
                dfield_[i] = (dfield_[i] + field_accel[i] * dt_eff_) * 0.980f;
                unified_field_[i] = std::tanh(unified_field_[i] + dfield_[i] * dt_eff_);
            }
        }

        // === SOVEREIGN OMNI-LAYER MOTOR READOUT ===
        // Motor output is synthesized by aggregating the volitional Motor Heads!
        std::array<float, CHANNEL_IO_DIM> aggregated_logits{};
        aggregated_logits.fill(0.0f);

        size_t active_motor_heads = 0;
        for (const auto& node : nodes_) {
            if (node.type == NodeType::MOTOR_HEAD && std::abs(node.coefficient) > 1e-4f) {
                active_motor_heads++;
                const float* m_proj = node.motor_proj.data();
                const float* head_out = node.latest_out.data();

                for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                    float dot = 0.0f;
                    const float* w_ptr = &m_proj[c * dim_];
                    for (size_t i = 0; i < dim_; ++i) dot += w_ptr[i] * head_out[i];
                    aggregated_logits[c] += node.coefficient * (dot / std::sqrt(static_cast<float>(dim_)));
                }
            }
        }

        if (active_motor_heads == 0) {
            aggregated_logits.fill(0.0f);
        }

        // Resonances with Refractory Momentum
        float max_res = -1e9f;
        std::array<float, CHANNEL_IO_DIM> resonance{};
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float r_c = aggregated_logits[c] - anchor_momentum_[c];
            resonance[c] = r_c;
            if (r_c > max_res) max_res = r_c;
        }

        // Dynamic Temperature
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = resonance[c] - max_res;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(1.5f + 3.0f * std::sqrt(variance), 1.0f, 20.0f);

        // Softmax
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        // Volitional Sampling
        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) { chosen = c; break; }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f;

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.04f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float f_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) f_sq += unified_field_[i] * unified_field_[i];
        float mean_f_sq = f_sq / static_cast<float>(dim_);

        // 1. Motor Error Gradient Backpropagation across Motor Heads
        std::vector<float> G_motor(dim_, 0.0f);

        for (auto& node : nodes_) {
            if (node.type == NodeType::MOTOR_HEAD) {
                float plastic = (1.0f - node.epigenetic_lock);
                const float* head_out = node.latest_out.data();

                for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                    float target_val = (c == target_byte) ? 1.0f : 0.0f;
                    float err = target_val - latest_probabilities_[c];

                    if (std::abs(err) > 0.0005f) {
                        float alpha = lr * err * node.coefficient;
                        float* m_ptr = &node.motor_proj[c * dim_];

                        for (size_t i = 0; i < dim_; ++i) {
                            G_motor[i] += err * node.coefficient * m_ptr[i];
                            float oja_decay = 0.02f * latest_probabilities_[c] * mean_f_sq * m_ptr[i];
                            m_ptr[i] += alpha * head_out[i] - plastic * oja_decay;
                        }
                    }
                }
            }
        }

        // Normalize backprojected causal vector
        float g_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) g_norm_sq += G_motor[i] * G_motor[i];
        float g_inv_norm = 1.0f / (std::sqrt(g_norm_sq) + SAFETY_EPSILON);

        // 2. Individual Vector Causal Credit & Totipotent Basis Adaptation
        for (auto& node : nodes_) {
            float alignment = 0.0f;
            float out_sq = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                alignment += node.latest_out[i] * G_motor[i];
                out_sq += node.latest_out[i] * node.latest_out[i];
            }
            float out_norm = std::sqrt(out_sq) + SAFETY_EPSILON;
            float local_causal = (alignment / out_norm) * g_inv_norm * 5.0f;

            node.causal_credit = 0.985f * node.causal_credit + 0.015f * local_causal;

            // Epigenetic Protection
            if (node.causal_credit > 0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock + 0.002f, 0.0f, 0.98f);
            } else if (node.causal_credit < -0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock - 0.005f, 0.0f, 0.98f);
            }

            float plastic = (1.0f - node.epigenetic_lock);
            node.coefficient += lr * 0.05f * plastic * node.causal_credit;
            node.coefficient = std::clamp(node.coefficient, 0.05f, 2.0f);

            // Memory Timescale Plasticity
            node.alpha_decay = std::clamp(node.alpha_decay + lr * 0.01f * plastic * local_causal, 0.05f, 0.9995f);

            // Adaptation of Extended Totipotent Basis Coordinates!
            node.mix_attention = std::clamp(node.mix_attention + lr * 0.02f * plastic * local_causal, 0.0f, 1.0f);
            node.mix_bilinear = std::clamp(node.mix_bilinear + lr * 0.02f * plastic * local_causal, 0.0f, 1.0f);
            node.mix_diff = std::clamp(node.mix_diff + lr * 0.02f * plastic * local_causal, 0.0f, 1.0f);
            node.mix_gating = std::clamp(node.mix_gating + lr * 0.02f * plastic * local_causal, 0.0f, 1.0f);
            node.mix_phase = std::clamp(node.mix_phase + lr * 0.02f * plastic * local_causal, 0.0f, 1.0f);
            node.lambda_snap = std::clamp(node.lambda_snap + lr * 0.02f * plastic * local_causal, 0.0f, 1.0f);

            // Substrate Projection Weights with Oja normalization
            if (node.type == NodeType::SENSORY) {
                const size_t in_offset = static_cast<size_t>(input_byte) * dim_;
                for (size_t i = 0; i < dim_; ++i) {
                    size_t s_idx = in_offset + i;
                    float oja_decay = 0.02f * mean_f_sq * node.W_a[s_idx];
                    node.W_a[s_idx] += lr * 0.05f * plastic * (G_motor[i] - oja_decay);
                }
            } else {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                const float* u_a = nodes_[p_a].latest_out.data();
                for (size_t i = 0; i < dim_; ++i) {
                    for (size_t j = 0; j < dim_; ++j) {
                        size_t idx = i * dim_ + j;
                        float oja_decay = mean_f_sq * node.W_a[idx];
                        node.W_a[idx] += lr * 0.03f * plastic * (G_motor[i] * u_a[j] - 0.02f * oja_decay);
                    }
                }
            }
        }

        // 3. Periodic Morphogenetic Audit: Genesis & Targeted Pruning
        if (total_steps_ % 20000 == 0) {
            evaluate_and_evolve();
        }
    }

    void evaluate_and_evolve() {
        std::cout << "\n--- [OMNI-LAYER TOTIPOTENT AUDIT @ Step " << total_steps_ << "] ---\n";
        std::cout << "  Free Energy: " << free_energy_
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
                      << " | Attn: " << node.mix_attention
                      << " | Gate: " << node.mix_gating
                      << " | Diff: " << node.mix_diff << "\n";
        }

        prune_harmful_operators();

        if (free_energy_ > 1.8f && nodes_.size() < 18) {
            size_t p_a = rng_() % nodes_.size();
            size_t p_b = rng_() % nodes_.size();
            sprout_new_operator(p_a, p_b);
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }
    size_t get_active_nodes() const noexcept { return nodes_.size(); }
    size_t get_dim() const noexcept { return dim_; }
    size_t get_sprouts() const noexcept { return total_sprouts_; }
    size_t get_prunes() const noexcept { return total_prunes_; }

private:
    size_t dim_;
    size_t total_steps_;
    size_t total_sub_ticks_;
    size_t total_sprouts_;
    size_t total_prunes_;
    float free_energy_;
    float prev_free_energy_;
    float dt_eff_;
    float endogenous_beta_;
    std::mt19937 rng_;

    std::vector<float> unified_field_;
    std::vector<float> dfield_;

    std::vector<OmniSuperPrimitive> nodes_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-594: OMNI-LAYER TOTIPOTENT SUPER-PRIMITIVE FIELD (OL-TSPF)   ===\n";
    std::cout << "=== Sovereign Control Over All Layers + Full Algebraic Continuum     ===\n";
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

    const size_t train_steps = 300000;

    std::cout << ">>> Initializing Omni-Layer Organism (Dim: 256)...\n";
    karyon::OmniOrganism omni_org(256);

    std::cout << ">>> Launching Omni-Layer Autopoietic Learning (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        omni_org.step(in_b);
        omni_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 50000 == 0) {
            std::cout << "  >>> Step " << (step + 1)
                      << " | Free Energy: " << omni_org.get_free_energy()
                      << " | Beta: " << omni_org.get_beta()
                      << " | Nodes: " << omni_org.get_active_nodes()
                      << " | Sprouts: " << omni_org.get_sprouts()
                      << " | Prunes: " << omni_org.get_prunes() << "\n";
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
        size_t next_act = omni_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-594 vs EXP-593 vs EXP-591 vs EXP-589) ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | Active Nodes | Sprouts | Prunes | Alphabet %\n";
    std::cout << "--------------------------------|--------------------|-----------------|--------------|---------|--------|-----------\n";
    std::cout << "OL-TSPF Omni-Layer (EXP-594)    | " << throughput << "          | " << omni_org.get_free_energy() << "          | " << omni_org.get_active_nodes() << "            | " << omni_org.get_sprouts() << "       | " << omni_org.get_prunes() << "      | " << (metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "EC-SP-DAG Conserved (EXP-593)   | 394.04             | 3.1598          | 15           | 15      | 3      | 79.56%\n";
    std::cout << "LVCC Individual Credit (EXP-591)| 937.84             | 3.4218          | 5            | 14      | 11     | 77.33%\n";
    std::cout << "CO-MTAR Operadic (EXP-589)      | 675.92             | 3.5126          | 6            | 0       | 0      | 76.89%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
