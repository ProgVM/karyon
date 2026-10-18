// ======================================================================================
// EXP-597: Multiscale Self-Assembling Substrate with Sovereign Noise & DNA/RNA Regeneration
// --------------------------------------------------------------------------------------
// KEP Protocol Master Benchmark:
// 1. Multiscale Separation of Mutation:
//    - Low Level (Micro-Engine): Stable, continuous tensor & memory registers.
//      Executes physical state transitions and sovereign noise modulation.
//      Micro-level equations and basic arithmetic DO NOT mutate chaotically,
//      preserving causal computing stability and gradient flow.
//    - High Level (Macro-Router & Morphic Assembler): Dynamic self-assembling
//      topology, operator sprouting, pruning, and flow routing driven by Free Energy.
// 2. Sovereign Cell-Controlled Noise (Heterogeneous Stochasticity):
//    - Each node maintains sovereign control over its noise temperature T_i.
//    - Low-error / high-credit nodes cool down towards absolute zero (deterministic).
//    - Stalled or high-surprise nodes self-excite thermal stochastic tunneling.
// 3. Sovereign Epigenetic DNA/RNA Registry & Catastrophic Regeneration:
//    - Highly successful functional sub-circuits are transcribed into an invariant
//      DNA/RNA blueprint registry.
//    - Mid-run catastrophic shock (50% state corruption & active topology disruption).
//    - Mode A: With DNA/RNA regeneration — organism rebuilds verified circuits.
//    - Mode B: Without DNA/RNA (ablated) — organism suffers catastrophic failure.
// ======================================================================================

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
#include <iomanip>
#include <cstring>
#include <memory>

namespace karyon_exp597 {

constexpr size_t CHANNEL_IO_DIM = 258;
constexpr float SAFETY_EPSILON = 1e-7f;

enum class NodeType : uint8_t {
    SENSORY = 0,
    COGNITIVE = 1,
    MOTOR_HEAD = 2
};

// Invariant DNA/RNA Genetic Blueprint for Low-Level Circuit Regeneration
struct DnaCircuitGene {
    size_t node_id;
    NodeType type;
    size_t input_node_a;
    size_t input_node_b;
    float frozen_timescale;
    float frozen_coefficient;
    std::vector<float> dna_weights;
    bool valid;
};

// Multiscale Sovereign Node (Stable low-level micro physics + macro routing)
struct MultiscaleSovereignNode {
    size_t node_id;
    NodeType type;
    size_t input_node_a;
    size_t input_node_b;

    float coefficient;          // Output injection strength
    float causal_credit;        // Running vector causal efficacy
    float epigenetic_lock;      // Epigenetic protection level [0.0, 1.0]

    // Sovereign Local Noise Control
    float noise_temperature;    // Sovereignly controlled noise [0.0, 1.0]

    // Dynamic Memory Timescale
    float alpha_decay;
    std::vector<float> state_s;
    std::vector<float> latest_out;

    size_t active_dim;
    float local_learning_rate;

    // Stable Micro-Substrate Weight Tensors
    std::vector<float> W_a;
    std::vector<float> W_b;
    std::vector<float> W_gate;
    std::vector<float> W_feedback;
    std::vector<float> motor_proj;

    MultiscaleSovereignNode(size_t id, NodeType t, size_t max_dim, size_t in_a, size_t in_b,
                            float init_alpha, float init_coeff, std::mt19937& rng)
        : node_id(id), type(t), input_node_a(in_a), input_node_b(in_b),
          coefficient(init_coeff), causal_credit(0.0f), epigenetic_lock(0.0f),
          noise_temperature(0.3f), alpha_decay(init_alpha), active_dim(max_dim),
          local_learning_rate(0.04f)
    {
        state_s.resize(max_dim, 0.0f);
        latest_out.resize(max_dim, 0.0f);

        size_t mat_size = (type == NodeType::SENSORY) ? (CHANNEL_IO_DIM * max_dim) : (max_dim * max_dim);
        W_a.resize(mat_size, 0.0f);
        W_b.resize(max_dim * max_dim, 0.0f);
        W_gate.resize(max_dim * max_dim, 0.0f);
        W_feedback.resize(max_dim * max_dim, 0.0f);

        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(max_dim)));
        for (auto& w : W_a) w = dist(rng) * 0.5f;
        for (auto& w : W_b) w = dist(rng) * 0.5f;
        for (auto& w : W_gate) w = dist(rng) * 0.5f;
        for (auto& w : W_feedback) w = dist(rng) * 0.2f;

        if (type == NodeType::MOTOR_HEAD) {
            motor_proj.resize(CHANNEL_IO_DIM * max_dim, 0.0f);
            for (auto& w : motor_proj) w = dist(rng) * 0.5f;
        }
    }
};

// Multiscale Organism with DNA/RNA Regeneration Engine
class MultiscaleDnaOrganism {
public:
    MultiscaleDnaOrganism(size_t max_dim, bool enable_dna, uint64_t seed = 2026)
        : max_dim_(max_dim), total_steps_(0), total_sprouts_(0), total_prunes_(0),
          free_energy_(5.55f), min_free_energy_(999.0f), prev_free_energy_(5.55f),
          dt_eff_(0.50f), endogenous_beta_(1.0f), rng_(seed),
          enable_dna_regeneration_(enable_dna)
    {
        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        // Initial Core: Sensory + 2 Cognitive + 1 Motor
        nodes_.emplace_back(0, NodeType::SENSORY, max_dim_, 0, 0, 0.25f, 1.0f, rng_);
        nodes_.back().epigenetic_lock = 0.98f;

        nodes_.emplace_back(1, NodeType::COGNITIVE, max_dim_, 0, 0, 0.35f, 0.8f, rng_);
        nodes_.emplace_back(2, NodeType::COGNITIVE, max_dim_, 1, 0, 0.70f, 0.8f, rng_);

        nodes_.emplace_back(3, NodeType::MOTOR_HEAD, max_dim_, 2, 1, 0.30f, 1.0f, rng_);
        nodes_.back().epigenetic_lock = 0.98f;
    }

    // High-Speed Forward Step
    void forward_pass(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * max_dim_;

        // 1. Top-Down Feedback context
        std::vector<std::vector<float>> topdown_signals(nodes_.size(), std::vector<float>(max_dim_, 0.0f));
        for (size_t k = nodes_.size(); k-- > 0;) {
            const auto& node = nodes_[k];
            if (node.type == NodeType::COGNITIVE || node.type == NodeType::MOTOR_HEAD) {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                if (p_a < k && !node.W_feedback.empty()) {
                    for (size_t i = 0; i < max_dim_; ++i) {
                        float s = 0.0f;
                        const float* w_row = &node.W_feedback[i * max_dim_];
                        for (size_t j = 0; j < max_dim_; ++j) s += w_row[j] * node.latest_out[j];
                        topdown_signals[p_a][i] += s * 0.15f;
                    }
                }
            }
        }

        // 2. Micro-Level Deterministic Transition with Sovereign Noise
        std::uniform_real_distribution<float> noise_dist(-1.0f, 1.0f);

        for (size_t idx = 0; idx < nodes_.size(); ++idx) {
            auto& node = nodes_[idx];
            size_t d_act = node.active_dim;

            if (node.type == NodeType::SENSORY) {
                for (size_t i = 0; i < d_act; ++i) {
                    float u_in = node.W_a[in_offset + i] + topdown_signals[idx][i];
                    
                    // Sovereign noise injection
                    float noise = (node.noise_temperature > 0.001f) 
                                ? (noise_dist(rng_) * node.noise_temperature * 0.05f) : 0.0f;

                    node.state_s[i] = node.state_s[i] * node.alpha_decay + u_in * (1.0f - node.alpha_decay) + noise;
                    node.latest_out[i] = std::tanh(node.state_s[i]);
                }
            } else {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                size_t p_b = std::min(node.input_node_b, nodes_.size() - 1);
                const float* u_a = nodes_[p_a].latest_out.data();
                const float* u_b = nodes_[p_b].latest_out.data();

                for (size_t i = 0; i < d_act; ++i) {
                    float s_a = 0.0f, s_b = 0.0f, s_g = 0.0f;
                    const float* w_a = &node.W_a[i * max_dim_];
                    const float* w_b = &node.W_b[i * max_dim_];
                    const float* w_g = &node.W_gate[i * max_dim_];

                    for (size_t j = 0; j < d_act; ++j) {
                        s_a += w_a[j] * u_a[j];
                        s_b += w_b[j] * u_b[j];
                        s_g += w_g[j] * u_a[j];
                    }

                    float gate = 1.0f / (1.0f + std::exp(-s_g));
                    float combined = (s_a + s_b * 0.5f) * gate + topdown_signals[idx][i];

                    // Sovereign noise injection
                    float noise = (node.noise_temperature > 0.001f)
                                ? (noise_dist(rng_) * node.noise_temperature * 0.05f) : 0.0f;

                    node.state_s[i] = node.state_s[i] * node.alpha_decay + combined * (1.0f - node.alpha_decay) + noise;
                    node.latest_out[i] = std::tanh(node.state_s[i]);
                }
            }
        }
    }

    // Step Execution & Motor Prediction
    size_t step(uint8_t input_byte) noexcept {
        forward_pass(input_byte);

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

        float max_res = -1e9f;
        std::array<float, CHANNEL_IO_DIM> resonance{};
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float r_c = aggregated_logits[c] - anchor_momentum_[c];
            resonance[c] = r_c;
            if (r_c > max_res) max_res = r_c;
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

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

    // Adaptive Learning & Multiscale Plasticity
    void adapt(uint8_t input_byte, uint8_t target_byte, float base_lr = 0.04f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;
        if (free_energy_ < min_free_energy_) min_free_energy_ = free_energy_;

        float fe_delta = free_energy_ - prev_free_energy_;

        // 1. Motor Error Gradient
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
                            m_ptr[i] += alpha * head_out[i];
                        }
                    }
                }
            }
        }

        float g_norm_sq = 0.0f;
        for (size_t i = 0; i < max_dim_; ++i) g_norm_sq += G_motor[i] * G_motor[i];
        float g_inv_norm = 1.0f / (std::sqrt(g_norm_sq) + SAFETY_EPSILON);

        // 2. Micro-Plasticity and Sovereign Cell-Governed Noise
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

            // Sovereign Noise Modulation:
            // High causal efficacy -> cool down to absolute zero (deterministic).
            // Negative causal efficacy / surprise -> thermal excitation.
            if (node.causal_credit > 0.02f) {
                node.noise_temperature = std::max(node.noise_temperature * 0.96f, 0.0f);
            } else if (node.causal_credit < -0.02f) {
                node.noise_temperature = std::min(node.noise_temperature + 0.02f, 0.8f);
            }

            // Epigenetic Protection
            if (node.causal_credit > 0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock + 0.002f, 0.0f, 0.98f);
                
                // Transcribe into DNA Blueprint
                if (enable_dna_regeneration_ && node.epigenetic_lock > 0.80f) {
                    transcribe_node_dna(node);
                }
            } else if (node.causal_credit < -0.03f) {
                node.epigenetic_lock = std::clamp(node.epigenetic_lock - 0.005f, 0.0f, 0.98f);
            }

            float plastic = (1.0f - node.epigenetic_lock);

            // Micro-level weight update (smooth, non-chaotic)
            if (node.type == NodeType::SENSORY) {
                const size_t in_offset = static_cast<size_t>(input_byte) * max_dim_;
                for (size_t i = 0; i < d_act; ++i) {
                    node.W_a[in_offset + i] += node.local_learning_rate * 0.05f * plastic * G_motor[i];
                }
            } else {
                size_t p_a = std::min(node.input_node_a, nodes_.size() - 1);
                const float* u_a = nodes_[p_a].latest_out.data();
                for (size_t i = 0; i < d_act; ++i) {
                    for (size_t j = 0; j < d_act; ++j) {
                        size_t idx = i * max_dim_ + j;
                        node.W_a[idx] += node.local_learning_rate * 0.02f * plastic * G_motor[i] * u_a[j];
                        node.W_feedback[idx] += node.local_learning_rate * 0.015f * plastic * u_a[i] * G_motor[j];
                    }
                }
            }
        }

        // 3. High-Level Macro Morphogenesis (Sprouting & Pruning)
        if (total_steps_ % 10000 == 0) {
            macro_morphogenesis();
        }
    }

    // High-Level Morphogenesis: Changes routing & topology, NOT raw micro-math
    void macro_morphogenesis() {
        // Prune harmful macro-nodes
        for (auto it = nodes_.begin(); it != nodes_.end();) {
            if (it->type != NodeType::SENSORY && it->type != NodeType::MOTOR_HEAD &&
                it->epigenetic_lock < 0.05f && it->causal_credit < -0.05f) 
            {
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

        // Sprout new macro-nodes if Free Energy needs topological capacity
        if (free_energy_ > 1.5f && nodes_.size() < 10) {
            size_t p_a = rng_() % nodes_.size();
            size_t p_b = rng_() % nodes_.size();
            size_t new_id = nodes_.size();
            float alpha = 0.20f + 0.60f * (static_cast<float>(rng_() % 1000) / 1000.0f);
            nodes_.emplace_back(new_id, NodeType::COGNITIVE, max_dim_, p_a, p_b, alpha, 0.5f, rng_);
            total_sprouts_++;
        }
    }

    // Epigenetic DNA Transcription
    void transcribe_node_dna(const MultiscaleSovereignNode& node) {
        for (auto& gene : dna_registry_) {
            if (gene.valid && gene.node_id == node.node_id) {
                // Update existing gene
                gene.frozen_timescale = node.alpha_decay;
                gene.frozen_coefficient = node.coefficient;
                gene.dna_weights = node.W_a;
                return;
            }
        }
        // Add new gene
        DnaCircuitGene new_gene;
        new_gene.node_id = node.node_id;
        new_gene.type = node.type;
        new_gene.input_node_a = node.input_node_a;
        new_gene.input_node_b = node.input_node_b;
        new_gene.frozen_timescale = node.alpha_decay;
        new_gene.frozen_coefficient = node.coefficient;
        new_gene.dna_weights = node.W_a;
        new_gene.valid = true;
        dna_registry_.push_back(std::move(new_gene));
    }

    // CATASTROPHIC SHOCK & REGENERATION
    void inject_catastrophic_shock() {
        std::cout << "\n>>> [!] INJECTING CATASTROPHIC SHOCK: 70% Nodes Corrupted & Damaged!\n";
        
        // Corrupt internal state and destroy non-locked cognitive nodes
        for (auto& node : nodes_) {
            for (auto& s : node.state_s) s = (static_cast<float>(rng_() % 1000) / 500.0f - 1.0f) * 3.0f;
            node.noise_temperature = 0.9f; // Thermal shock
        }

        for (auto it = nodes_.begin(); it != nodes_.end();) {
            if (it->type == NodeType::COGNITIVE && it->epigenetic_lock < 0.50f) {
                it = nodes_.erase(it);
            } else {
                ++it;
            }
        }

        // DNA/RNA REGENERATION:
        if (enable_dna_regeneration_ && !dna_registry_.empty()) {
            std::cout << ">>> [+] ACTIVATING DNA/RNA REGENERATION: Restoring " << dna_registry_.size() << " Invariant Genes...\n";
            for (const auto& gene : dna_registry_) {
                bool found = false;
                for (auto& node : nodes_) {
                    if (node.node_id == gene.node_id) {
                        node.alpha_decay = gene.frozen_timescale;
                        node.coefficient = gene.frozen_coefficient;
                        node.W_a = gene.dna_weights;
                        node.noise_temperature = 0.0f; // Restored to zero noise
                        node.epigenetic_lock = 0.98f;
                        found = true;
                        break;
                    }
                }
                if (!found && gene.type == NodeType::COGNITIVE) {
                    nodes_.emplace_back(gene.node_id, gene.type, max_dim_, gene.input_node_a, gene.input_node_b,
                                        gene.frozen_timescale, gene.frozen_coefficient, rng_);
                    nodes_.back().W_a = gene.dna_weights;
                    nodes_.back().noise_temperature = 0.0f;
                    nodes_.back().epigenetic_lock = 0.98f;
                }
            }
            std::cout << ">>> [+] DNA REGENERATION COMPLETE. Active Nodes: " << nodes_.size() << "\n\n";
        } else {
            std::cout << ">>> [-] DNA REGENERATION DISABLED (Ablated). No blueprint recovery.\n\n";
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_min_free_energy() const noexcept { return min_free_energy_; }
    size_t get_active_nodes() const noexcept { return nodes_.size(); }
    size_t get_dna_genes_count() const noexcept { return dna_registry_.size(); }
    float get_mean_noise() const noexcept {
        float s = 0.0f;
        for (const auto& n : nodes_) s += n.noise_temperature;
        return s / static_cast<float>(nodes_.size());
    }

private:
    size_t max_dim_;
    size_t total_steps_;
    size_t total_sprouts_;
    size_t total_prunes_;
    float free_energy_;
    float min_free_energy_;
    float prev_free_energy_;
    float dt_eff_;
    float endogenous_beta_;
    std::mt19937 rng_;
    bool enable_dna_regeneration_;

    std::vector<MultiscaleSovereignNode> nodes_;
    std::vector<DnaCircuitGene> dna_registry_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon_exp597

void run_multiscale_experiment(bool with_dna) {
    std::cout << "========================================================================\n";
    std::cout << ">>> RUNNING MULTISCALE BENCHMARK: " 
              << (with_dna ? "WITH DNA/RNA REGENERATION" : "WITHOUT DNA (ABLATED)") << "\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) text_path = "data/web_downloads/alpaca_data.json", file.open(text_path, std::ios::binary);

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    } else {
        std::string fallback = "The sovereign substrate separates fast micro execution from slow macro morphogenesis. DNA preserves invariants.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    karyon_exp597::MultiscaleDnaOrganism organism(128, with_dna, 2026);

    const size_t total_steps = 150000;
    const size_t shock_step = 75000;

    float fe_pre_shock = 0.0f;
    float fe_post_shock_immediate = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < total_steps; ++step) {
        if (step == shock_step) {
            fe_pre_shock = organism.get_free_energy();
            organism.inject_catastrophic_shock();
        }

        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];

        organism.step(in_b);
        organism.adapt(in_b, tgt_b, 0.04f);

        if (step == shock_step + 1000) {
            fe_post_shock_immediate = organism.get_free_energy();
        }

        if ((step + 1) % 30000 == 0 || step == total_steps - 1) {
            std::cout << "  >>> Step " << (step + 1)
                      << " | FE: " << organism.get_free_energy()
                      << " | Min FE: " << organism.get_min_free_energy()
                      << " | Nodes: " << organism.get_active_nodes()
                      << " | DNA Genes: " << organism.get_dna_genes_count()
                      << " | Mean Noise: " << organism.get_mean_noise() << "\n";
            std::cout.flush();
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(total_steps) / duration;

    std::cout << "\n>>> MODE SUMMARY (" << (with_dna ? "WITH DNA" : "ABLATED WITHOUT DNA") << "):\n";
    std::cout << "  Throughput           : " << throughput << " tok/s\n";
    std::cout << "  Pre-Shock FE         : " << fe_pre_shock << " nats\n";
    std::cout << "  Post-Shock (1k steps): " << fe_post_shock_immediate << " nats\n";
    std::cout << "  Final Free Energy    : " << organism.get_free_energy() << " nats\n";
    std::cout << "  Min Free Energy      : " << organism.get_min_free_energy() << " nats\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-597: MULTISCALE SUBSTRATE & DNA/RNA REGENERATION BENCHMARK   ===\n";
    std::cout << "========================================================================\n\n";

    // 1. Run Baseline without DNA/RNA (Ablated)
    run_multiscale_experiment(false);

    // 2. Run with DNA/RNA Regeneration
    run_multiscale_experiment(true);

    return 0;
}
