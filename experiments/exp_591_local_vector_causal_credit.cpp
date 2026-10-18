// ============================================================================
// EXP-591: LOCAL VECTOR CAUSAL CREDIT GRAPH & INDIVIDUAL STRUCTURAL AUDIT (LVCC-ISA)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 8, 27, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
// Core Philosophy:
// "Crucially: self-evolution cannot be blind dice-throwing. The organism must know its
// successes and failures — it must understand what structure leads to success or error,
// why deleting an element was a bad idea, and in what direction to evolve to achieve
// breakthrough mastery." (Bazilevs Manifesto)
//
// Root Cause of EXP-590 Failure:
// EXP-590 used a single global scalar delta_FE = FE_t - FE_{t-1} for all nodes.
// When an unexpected byte arrived, ALL nodes received negative credit (-0.519),
// inverting the sensory inflow (Coeff = -0.57) and triggering systemic collapse.
//
// Mathematical Solution in EXP-591:
// 1. LOCAL VECTOR CAUSAL CREDIT ASSIGNMENT (Principle 5):
//    Let e_c = (1_{c=target} - p_c) be the motor prediction error.
//    The back-projected error vector pointing toward Free Energy reduction in state space is:
//    G_motor = sum_c e_c * W_motor[c] in R^D.
//    Each node k outputs an acceleration force vector: Phi_k in R^D.
//    The EXACT local causal efficacy of node k on this step is the projection:
//    CausalContribution_k = < Phi_k, G_motor > = sum_i Phi_{k, i} * G_{motor, i}
//    - If CausalContribution_k > 0: THIS specific node pushed the trajectory toward the target!
//    - If CausalContribution_k < 0: THIS specific node pushed the trajectory away from the target!
//
// 2. DIRECTION-AWARE CAUSAL SPROUTING & EPIGENETIC SHIELDING:
//    - Node coefficient updates: delta_c_k = lr * CausalContribution_k.
//    - Positive efficacy -> Epigenetic invariant locking (alpha_lock -> 1.0).
//    - Persistent negative efficacy -> Targeted pruning without harming sensory/stable nodes.
//    - Sensory node is clamped non-negative (Coeff >= 0.1) so perception can never invert!
//
// 3. CONTINUOUS SUPER-PRIMITIVE OPERADIC GRAPH (No enum switch-cases):
//    Differentiable nodes blending linear, bilinear, and parametric softsign attractors.
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

struct SuperPrimitiveNode {
    size_t node_id;
    size_t input_node_a;
    size_t input_node_b;

    float coefficient;      // Node activation strength
    float causal_credit;    // Running individual causal efficacy: <Phi_k, G_motor>
    float epigenetic_lock;  // 0.0 (plastic) -> 1.0 (crystallized invariant)

    // Continuous parameter coordinates
    std::vector<float> W_linear;   // Linear projection (dim * dim)
    std::vector<float> W_bilinear; // Bilinear projection (rank * dim)
    float lambda_nonlin;           // 0.0 (linear) -> 1.0 (saturating attractor)
    bool is_sensory_coupled;

    // Output buffer from the latest forward pass: Phi_k in R^D
    std::vector<float> latest_output;

    SuperPrimitiveNode(size_t id, size_t dim, size_t in_a, size_t in_b, bool sensory, std::mt19937& rng)
        : node_id(id), input_node_a(in_a), input_node_b(in_b),
          coefficient(0.2f), causal_credit(0.0f), epigenetic_lock(0.0f),
          lambda_nonlin(0.5f), is_sensory_coupled(sensory) {

        latest_output.assign(dim, 0.0f);
        std::normal_distribution<float> dist(0.0f, 0.02f);

        if (sensory) {
            W_linear.assign(CHANNEL_IO_DIM * dim, 0.0f);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>(c + 1) / static_cast<float>(CHANNEL_IO_DIM);
                for (size_t i = 0; i < dim; ++i) {
                    W_linear[c * dim + i] = std::sin(phase * static_cast<float>(i + 1)) * 0.15f + dist(rng);
                }
            }
        } else {
            W_linear.assign(dim * dim, 0.0f);
            for (auto& w : W_linear) w = dist(rng);
        }

        size_t rank = 8;
        W_bilinear.assign(rank * dim, 0.0f);
        for (auto& w : W_bilinear) w = dist(rng);
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

class LocalVectorCausalCreditOrganism {
public:
    LocalVectorCausalCreditOrganism(size_t dim = 256) noexcept
        : dim_(dim), rng_(5910) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        dt_eff_ = 0.040f;
        endogenous_beta_ = 4.0f;
        total_sub_ticks_ = 0;
        total_sprouts_ = 0;
        total_prunes_ = 0;

        z_.assign(dim_, 0.0f);
        dz_.assign(dim_, 0.0f);

        motor_readout_.assign(CHANNEL_IO_DIM * dim_, 0.0f);
        std::normal_distribution<float> init_dist(0.0f, 0.02f);
        for (auto& w : motor_readout_) w = init_dist(rng_);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        // Primordial Nodes:
        // Node 0: Sensory Inflow (permanently positive coefficient)
        nodes_.emplace_back(0, dim_, 0, 0, true, rng_);
        nodes_[0].coefficient = 1.0f;
        nodes_[0].epigenetic_lock = 0.8f;

        // Node 1: Recurrent Tensor Field
        nodes_.emplace_back(1, dim_, 1, 1, false, rng_);
        nodes_[1].coefficient = 0.5f;
        nodes_[1].epigenetic_lock = 0.5f;
    }

    void sprout_new_operator(size_t parent_a, size_t parent_b) {
        if (nodes_.size() >= 20) return;
        size_t new_id = nodes_.size();
        nodes_.emplace_back(new_id, dim_, parent_a, parent_b, false, rng_);
        auto& new_node = nodes_.back();

        // Net2Net smooth birth: small initial amplitude so it doesn't disturb existing attractors
        new_node.coefficient = 0.05f;
        new_node.causal_credit = 0.0f;
        new_node.epigenetic_lock = 0.0f;
        total_sprouts_++;

        std::cout << "\n[+] AUTONOMOUS GENESIS: Synthesized Super-Primitive Node [" << new_id
                  << "] from (" << parent_a << " x " << parent_b << ") | Free Energy: "
                  << free_energy_ << "\n";
    }

    void prune_harmful_operators() {
        if (nodes_.size() <= 2) return;

        for (auto it = nodes_.begin() + 2; it != nodes_.end(); ) {
            // Precise local triage: prune ONLY if its personal causal credit is strongly negative
            // and it has no epigenetic lock
            if (it->causal_credit < -0.04f && it->epigenetic_lock < 0.15f) {
                std::cout << "\n[-] AUTONOMOUS PRUNING: Dissolved harmful Node [" << it->node_id
                          << "] (Individual Credit: " << it->causal_credit
                          << ", Lock: " << (it->epigenetic_lock * 100.0f) << "%)\n";
                it = nodes_.erase(it);
                total_prunes_++;
            } else {
                ++it;
            }
        }
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * dim_;

        // Dynamic relaxation sub-ticks (KEP Principle 29)
        size_t sub_ticks = 1;
        if (free_energy_ > 3.8f) sub_ticks = 4;
        else if (free_energy_ > 3.0f) sub_ticks = 3;
        else if (free_energy_ > 2.0f) sub_ticks = 2;

        total_sub_ticks_ += sub_ticks;

        for (size_t tick = 0; tick < sub_ticks; ++tick) {
            std::vector<float> total_accel(dim_, 0.0f);

            for (auto& node : nodes_) {
                if (std::abs(node.coefficient) < 1e-5f) continue;

                if (node.is_sensory_coupled) {
                    for (size_t i = 0; i < dim_; ++i) {
                        float val = node.W_linear[in_offset + i];
                        node.latest_output[i] = val;
                        total_accel[i] += node.coefficient * val;
                    }
                } else {
                    std::vector<float> lin_out(dim_, 0.0f);
                    for (size_t i = 0; i < dim_; ++i) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < dim_; ++j) {
                            sum += node.W_linear[i * dim_ + j] * z_[j];
                        }
                        lin_out[i] = sum;
                    }

                    size_t rank = 8;
                    std::vector<float> bilin_u(rank, 0.0f), bilin_v(rank, 0.0f);
                    for (size_t r = 0; r < rank; ++r) {
                        for (size_t j = 0; j < dim_; ++j) {
                            bilin_u[r] += node.W_bilinear[r * dim_ + j] * z_[j];
                            bilin_v[r] += node.W_bilinear[r * dim_ + j] * dz_[j];
                        }
                        bilin_v[r] = 1.0f / (1.0f + std::exp(-bilin_v[r]));
                    }

                    for (size_t i = 0; i < dim_; ++i) {
                        float raw = lin_out[i];
                        for (size_t r = 0; r < rank; ++r) {
                            raw += 0.2f * (bilin_u[r] * bilin_v[r]);
                        }
                        float activated = (1.0f - node.lambda_nonlin) * raw +
                                          node.lambda_nonlin * (raw / std::sqrt(1.0f + raw * raw));
                        node.latest_output[i] = activated;
                        total_accel[i] += node.coefficient * activated;
                    }
                }
            }

            // Dynamic Time Dilation
            float total_kinetic = 0.0f;
            for (size_t i = 0; i < dim_; ++i) total_kinetic += dz_[i] * dz_[i];
            float mean_kinetic = total_kinetic / static_cast<float>(dim_);
            dt_eff_ = std::clamp(0.040f / (1.0f + 2.0f * mean_kinetic), 0.010f, 0.080f);

            // Sub-tick integration
            for (size_t i = 0; i < dim_; ++i) {
                dz_[i] = (dz_[i] + total_accel[i] * dt_eff_) * 0.982f;
                z_[i] = std::tanh(z_[i] + dz_[i] * dt_eff_);
            }
        }

        // Volitional Motor Readout
        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = 0.0f;
            const float* w_ptr = &motor_readout_[c * dim_];
            for (size_t i = 0; i < dim_; ++i) dot += w_ptr[i] * z_[i];

            float norm_dot = dot / (std::sqrt(static_cast<float>(dim_)) + SAFETY_EPSILON);
            float eff_res = norm_dot - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // Endogenous Inverse Temperature
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

        // Volitional selection
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

        float z_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) z_sq += z_[i] * z_[i];
        float mean_z_sq = z_sq / static_cast<float>(dim_);

        // 1. Motor Plasticity and Back-Projected Error Vector G_motor in R^D
        std::vector<float> G_motor(dim_, 0.0f);

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];

            if (std::abs(err) > 0.0005f) {
                float alpha = lr * err;
                float oja_motor = 0.02f * latest_probabilities_[c] * mean_z_sq;

                const float* m_read = &motor_readout_[c * dim_];
                for (size_t i = 0; i < dim_; ++i) {
                    // Accumulate back-projected direction: G_motor = sum_c err_c * W_c
                    G_motor[i] += err * m_read[i];

                    // Update motor weights via Oja
                    size_t m_idx = c * dim_ + i;
                    motor_readout_[m_idx] += alpha * z_[i] - oja_motor * motor_readout_[m_idx];
                }
            }
        }

        // Normalize G_motor to compute cosine alignment
        float g_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) g_norm_sq += G_motor[i] * G_motor[i];
        float g_inv_norm = 1.0f / (std::sqrt(g_norm_sq) + SAFETY_EPSILON);

        // 2. INDIVIDUAL LOCAL VECTOR CAUSAL CREDIT ASSIGNMENT (Bazilevs Principle 5)
        // Each node is judged strictly by its OWN output alignment with G_motor!
        for (auto& node : nodes_) {
            // Compute projection of this node's output onto the motor gradient direction:
            float alignment = 0.0f;
            float out_sq = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                alignment += node.latest_output[i] * G_motor[i];
                out_sq += node.latest_output[i] * node.latest_output[i];
            }
            float out_norm = std::sqrt(out_sq) + SAFETY_EPSILON;
            float local_causal_signal = (alignment / out_norm) * g_inv_norm * 5.0f;

            // Running individual causal credit
            node.causal_credit = 0.985f * node.causal_credit + 0.015f * local_causal_signal;

            // Epigenetic Lock: Nodes with consistent positive contribution are protected
            if (node.causal_credit > 0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock + 0.002f, 0.0f, 0.98f);
            } else if (node.causal_credit < -0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock - 0.005f, 0.0f, 0.98f);
            }

            float plastic = (1.0f - node.epigenetic_lock);

            // Update node coefficient based on ITS OWN credit
            node.coefficient += lr * 0.05f * plastic * node.causal_credit;

            // CRITICAL INVARIANT: Sensory inflow cannot invert!
            if (node.is_sensory_coupled) {
                node.coefficient = std::clamp(node.coefficient, 0.20f, 2.0f);
            } else {
                node.coefficient = std::clamp(node.coefficient, -1.5f, 2.0f);
            }

            // Adapt node parameters
            if (node.is_sensory_coupled) {
                const size_t in_offset = static_cast<size_t>(input_byte) * dim_;
                for (size_t i = 0; i < dim_; ++i) {
                    size_t s_idx = in_offset + i;
                    float oja_decay = 0.02f * mean_z_sq * node.W_linear[s_idx];
                    node.W_linear[s_idx] += lr * 0.05f * plastic * (G_motor[i] - oja_decay);
                }
            } else {
                for (size_t i = 0; i < dim_; ++i) {
                    for (size_t j = 0; j < dim_; ++j) {
                        size_t idx = i * dim_ + j;
                        float oja_decay = mean_z_sq * node.W_linear[idx];
                        node.W_linear[idx] += lr * 0.04f * plastic * (G_motor[i] * z_[j] - 0.02f * oja_decay);
                    }
                }
                node.lambda_nonlin = std::clamp(node.lambda_nonlin + lr * 0.02f * plastic * local_causal_signal, 0.0f, 1.0f);
            }
        }

        // 3. Periodic Morphogenetic Audit: Genesis & Targeted Pruning
        if (total_steps_ % 20000 == 0) {
            evaluate_and_evolve();
        }
    }

    void evaluate_and_evolve() {
        std::cout << "\n--- [INDIVIDUAL CAUSAL AUDIT @ Step " << total_steps_ << "] ---\n";
        std::cout << "  Free Energy: " << free_energy_
                  << " | Active Nodes: " << nodes_.size()
                  << " | Sprouts: " << total_sprouts_
                  << " | Prunes: " << total_prunes_ << "\n";

        for (const auto& node : nodes_) {
            std::cout << "  Node [" << node.node_id << "] "
                      << (node.is_sensory_coupled ? "Sensory" : "RecurrentField")
                      << " | Coeff: " << node.coefficient
                      << " | IndivCredit: " << node.causal_credit
                      << " | Lock: " << (node.epigenetic_lock * 100.0f) << "%"
                      << " | LambdaNonlin: " << node.lambda_nonlin << "\n";
        }

        prune_harmful_operators();

        if (free_energy_ > 3.0f && nodes_.size() < 16) {
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

    std::vector<float> z_;
    std::vector<float> dz_;

    std::vector<SuperPrimitiveNode> nodes_;
    std::vector<float> motor_readout_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-591: LOCAL VECTOR CAUSAL CREDIT GRAPH & INDIVIDUAL AUDIT    ===\n";
    std::cout << "=== Bazilevs Principle 5: The Death of Blind Dice-Throwing in Genesis ===\n";
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

    std::cout << ">>> Initializing Local Vector Causal Credit Organism (Dim: 256)...\n";
    karyon::LocalVectorCausalCreditOrganism lvcc_org(256);

    std::cout << ">>> Launching Individual Vector Credit Learning (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        lvcc_org.step(in_b);
        lvcc_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 50000 == 0) {
            std::cout << "  >>> Step " << (step + 1)
                      << " | Free Energy: " << lvcc_org.get_free_energy()
                      << " | Beta: " << lvcc_org.get_beta()
                      << " | Nodes: " << lvcc_org.get_active_nodes()
                      << " | Sprouts: " << lvcc_org.get_sprouts()
                      << " | Prunes: " << lvcc_org.get_prunes() << "\n";
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
        size_t next_act = lvcc_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-591 vs EXP-590 vs EXP-589)             ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | Active Nodes | Sprouts | Prunes | Alphabet %\n";
    std::cout << "--------------------------------|--------------------|-----------------|--------------|---------|--------|-----------\n";
    std::cout << "LVCC Individual Credit (EXP-591)| " << throughput << "          | " << lvcc_org.get_free_energy() << "          | " << lvcc_org.get_active_nodes() << "            | " << lvcc_org.get_sprouts() << "       | " << lvcc_org.get_prunes() << "      | " << (metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "TTF-CG Global Credit (EXP-590)  | 541.87             | 5.4650          | 8            | 15      | 9      | 27.11%\n";
    std::cout << "CO-MTAR Operadic (EXP-589)      | 675.92             | 3.5126          | 6 (fixed)    | 0       | 0      | 76.89%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
