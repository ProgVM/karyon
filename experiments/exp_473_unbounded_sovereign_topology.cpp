// experiments/exp_473_unbounded_sovereign_topology.cpp
// EXP-473: Unbounded Sovereign Topology (Autopoietic Continuous Morphogenesis & Dynamic Head Sprouting)
// Evaluates true unbounded topological freedom:
// 1. Memory heads H are not fixed at H=4: when Free Energy / prediction entropy stays elevated
//    (H_entropy > tau), the substrate dynamically sprouts new orthogonal heads (H -> H+1 -> H+2 -> ...).
// 2. Continuous dimensional widening: embedding manifold dynamically expands its orthogonal basis.
// 3. Tests convergence to near-zero loss on long algorithmic & structured byte streams.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iomanip>

struct MorphogenesisMetrics {
    std::string name;
    float initial_loss;
    float final_loss;
    float loss_delta;
    float accuracy;
    size_t final_heads;
    size_t memory_nodes;
    double steps_per_sec;
};

class SovereignMorphogenicEngine {
public:
    size_t dim = 258;
    size_t dim_k = 16;
    size_t dim_v = 16;

    // Unbounded Topology: Dynamic heads and dimension
    size_t num_heads = 2; // Starts minimal (2 heads) and evolves on demand!
    size_t dim_emb = 32;   // Starts at 32 (num_heads * dim_v)

    std::vector<float> byte_emb;     // [dim x dim_emb]
    std::vector<float> S_memory;     // [num_heads x dim_k x dim_v]
    std::vector<float> theta_heads;  // [num_heads] timescale coordinates

    struct AttractorNode {
        std::vector<float> key;
        uint8_t target_byte;
        float usage_count;
    };
    std::vector<AttractorNode> unbounded_memory;

    float surprise_threshold = 2.0f;
    float beta_attractor = 16.0f;
    float beta_scale = 12.0f;

    // Morphogenesis parameters
    float entropy_sprout_threshold = 3.5f; // If entropy remains high, sprout new head!
    size_t max_unbounded_heads = 16;        // Substrate bound for this benchmark
    float rolling_entropy = 4.5f;

    std::mt19937 rng{42};
    std::normal_distribution<float> norm_dist{0.0f, 1.0f};

    SovereignMorphogenicEngine(size_t init_heads = 2) : num_heads(init_heads) {
        dim_emb = num_heads * dim_v;
        init_substrate();
    }

    void init_substrate() {
        byte_emb.resize(dim * dim_emb);
        for (size_t b = 0; b < dim; ++b) {
            float* eb = byte_emb.data() + b * dim_emb;
            for (size_t d = 0; d < dim_emb; ++d) eb[d] = norm_dist(rng);
            normalize(eb, dim_emb);
        }
        S_memory.assign(num_heads * dim_k * dim_v, 0.0f);
        theta_heads.resize(num_heads);
        for (size_t h = 0; h < num_heads; ++h) {
            theta_heads[h] = -1.5f + (4.0f * h) / (num_heads > 1 ? num_heads - 1 : 1);
        }
    }

    void reset(size_t init_heads = 2) {
        num_heads = init_heads;
        dim_emb = num_heads * dim_v;
        unbounded_memory.clear();
        init_substrate();
    }

    void normalize(float* ptr, size_t n) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) sum += ptr[i] * ptr[i];
        float norm = std::sqrt(sum);
        if (norm > 1e-6f) {
            float inv = 1.0f / norm;
            for (size_t i = 0; i < n; ++i) ptr[i] *= inv;
        }
    }

    float dot_product(const float* a, const float* b, size_t n) const {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

    // Sovereign Autopoietic Head Sprouting (Net2Net Zero-Shock Growth)
    void sprout_new_head() {
        if (num_heads >= max_unbounded_heads) return;

        size_t old_heads = num_heads;
        size_t new_heads = old_heads + 1;
        size_t old_dim_emb = dim_emb;
        size_t new_dim_emb = new_heads * dim_v;

        // 1. Expand byte embeddings by adding new orthogonal dimensions (zero-shock)
        std::vector<float> new_byte_emb(dim * new_dim_emb, 0.0f);
        for (size_t b = 0; b < dim; ++b) {
            // Copy old dimensions
            for (size_t d = 0; d < old_dim_emb; ++d) {
                new_byte_emb[b * new_dim_emb + d] = byte_emb[b * old_dim_emb + d];
            }
            // Infant dimensions initialized with small orthogonal variance
            for (size_t d = old_dim_emb; d < new_dim_emb; ++d) {
                new_byte_emb[b * new_dim_emb + d] = 0.05f * norm_dist(rng);
            }
            normalize(new_byte_emb.data() + b * new_dim_emb, new_dim_emb);
        }
        byte_emb = std::move(new_byte_emb);

        // 2. Expand S_memory: append zero-initialized new head tensor
        S_memory.resize(new_heads * dim_k * dim_v, 0.0f);

        // 3. Append new timescale coordinate
        theta_heads.push_back(1.0f); // Starts as medium-slow memory channel

        // 4. Zero-pad existing attractor keys in unbounded memory
        for (auto& node : unbounded_memory) {
            node.key.resize(new_dim_emb, 0.0f);
            normalize(node.key.data(), new_dim_emb);
        }

        num_heads = new_heads;
        dim_emb = new_dim_emb;
    }

    std::pair<float, bool> step(uint8_t byte_in, uint8_t byte_target, bool enable_morphogenesis = true) {
        float* e_in = byte_emb.data() + byte_in * dim_emb;

        // 1. Keys Partitioning
        std::vector<float> k(num_heads * dim_k);
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < dim_k; ++i) {
                k[h * dim_k + i] = e_in[h * dim_k + i];
            }
            normalize(k.data() + h * dim_k, dim_k);
        }

        // 2. Associative Continuous Field Readout
        std::vector<float> h_field(dim_emb, 0.0f);
        std::vector<float> head_outputs(num_heads * dim_v, 0.0f);

        for (size_t h = 0; h < num_heads; ++h) {
            const float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            float* hh = head_outputs.data() + h * dim_v;
            for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                    sum += kh[k_idx] * s_head[k_idx * dim_v + v_idx];
                }
                hh[v_idx] = sum;
                h_field[h * dim_v + v_idx] = sum;
            }
        }

        for (size_t i = 0; i < dim_emb; ++i) {
            h_field[i] = 0.70f * h_field[i] + 0.30f * e_in[i];
        }
        normalize(h_field.data(), dim_emb);

        // 3. Unbounded Attractor Manifold Resonance Readout
        std::vector<float> attractor_logits(dim, 0.0f);
        float best_match_score = -1.0f;
        size_t best_match_idx = 0;

        if (!unbounded_memory.empty()) {
            for (size_t idx = 0; idx < unbounded_memory.size(); ++idx) {
                float sim = dot_product(h_field.data(), unbounded_memory[idx].key.data(), dim_emb);
                if (sim > best_match_score) {
                    best_match_score = sim;
                    best_match_idx = idx;
                }
            }

            if (best_match_score > 0.70f) {
                uint8_t recall_byte = unbounded_memory[best_match_idx].target_byte;
                attractor_logits[recall_byte] += beta_attractor * (best_match_score - 0.70f);
                unbounded_memory[best_match_idx].usage_count += 0.01f;
            }
        }

        // 4. Manifold Cosine Projection & Logits
        std::vector<float> logits(dim, 0.0f);
        float max_l = -1e9f;
        for (size_t b = 0; b < dim; ++b) {
            const float* eb = byte_emb.data() + b * dim_emb;
            float dot = dot_product(h_field.data(), eb, dim_emb);
            float l = beta_scale * dot + attractor_logits[b];
            logits[b] = l;
            if (l > max_l) max_l = l;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(dim);
        for (size_t b = 0; b < dim; ++b) {
            probs[b] = std::exp(logits[b] - max_l);
            sum_exp += probs[b];
        }
        float inv_sum = 1.0f / sum_exp;
        uint8_t best_byte = 0;
        float max_p = 0.0f;
        float entropy = 0.0f;
        for (size_t b = 0; b < dim; ++b) {
            probs[b] *= inv_sum;
            if (probs[b] > max_p) {
                max_p = probs[b];
                best_byte = static_cast<uint8_t>(b);
            }
            if (probs[b] > 1e-7f) {
                entropy -= probs[b] * std::log(probs[b]);
            }
        }

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));
        bool is_correct = (best_byte == byte_target);

        // Update rolling entropy
        rolling_entropy = 0.95f * rolling_entropy + 0.05f * entropy;

        // 5. Autopoietic Morphogenesis: Check if substrate needs to sprout new head
        if (enable_morphogenesis && rolling_entropy > entropy_sprout_threshold) {
            sprout_new_head();
            // Re-fetch e_in pointer as byte_emb may have reallocated
            e_in = byte_emb.data() + byte_in * dim_emb;
        }

        // 6. Attractor Consolidation in Unbounded Memory
        if (!is_correct || loss > surprise_threshold) {
            if (best_match_score > 0.85f && unbounded_memory[best_match_idx].target_byte == byte_target) {
                for (size_t d = 0; d < dim_emb; ++d) {
                    unbounded_memory[best_match_idx].key[d] =
                        0.90f * unbounded_memory[best_match_idx].key[d] + 0.10f * h_field[d];
                }
                normalize(unbounded_memory[best_match_idx].key.data(), dim_emb);
            } else {
                AttractorNode new_node;
                new_node.key = h_field;
                new_node.target_byte = byte_target;
                new_node.usage_count = 1.0f;
                unbounded_memory.push_back(std::move(new_node));
            }
        }

        // 7. Field Matrix Relaxation
        float* e_target = byte_emb.data() + byte_target * dim_emb;
        std::vector<float> error_wave(dim_emb);
        for (size_t d = 0; d < dim_emb; ++d) error_wave[d] = e_target[d] - h_field[d];

        for (size_t h = 0; h < num_heads; ++h) {
            float alpha_h = 1.0f / (1.0f + std::exp(-theta_heads[h]));
            float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            const float* err_h = error_wave.data() + h * dim_v;

            for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                float kval = kh[k_idx];
                float* s_row = s_head + k_idx * dim_v;
                for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                    s_row[v_idx] = alpha_h * s_row[v_idx] + 0.45f * (kval * err_h[v_idx]);
                }
            }
        }

        // 8. Manifold Plasticity
        for (size_t d = 0; d < dim_emb; ++d) e_target[d] += 0.05f * h_field[d];
        normalize(e_target, dim_emb);

        return {loss, is_correct};
    }
};

MorphogenesisMetrics run_experiment(SovereignMorphogenicEngine& engine, const std::string& name,
                                    const std::vector<uint8_t>& stream, bool enable_morpho) {
    engine.reset(enable_morpho ? 2 : 4);
    size_t n = stream.size() - 1;
    size_t q_size = n / 4;

    float q1_loss = 0.0f;
    float q4_loss = 0.0f;
    size_t correct = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; ++i) {
        auto [loss, is_corr] = engine.step(stream[i], stream[i + 1], enable_morpho);
        if (i < q_size) q1_loss += loss;
        if (i >= n - q_size) q4_loss += loss;
        if (is_corr) correct++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    q1_loss /= q_size;
    q4_loss /= (n - (n - q_size));

    MorphogenesisMetrics m;
    m.name = name;
    m.initial_loss = q1_loss;
    m.final_loss = q4_loss;
    m.loss_delta = q4_loss - q1_loss;
    m.accuracy = (100.0f * correct) / n;
    m.final_heads = engine.num_heads;
    m.memory_nodes = engine.unbounded_memory.size();
    m.steps_per_sec = n / (elapsed_sec + 1e-6);
    return m;
}

int main() {
    std::cout << "=== EXP-473: Unbounded Sovereign Topology & Dynamic Autopoiesis Benchmark ===" << std::endl;

    std::ifstream f_cpp("karyon_native_core.cpp", std::ios::binary);
    std::vector<uint8_t> data_cpp((std::istreambuf_iterator<char>(f_cpp)),
                                   std::istreambuf_iterator<char>());
    f_cpp.close();

    std::vector<uint8_t> stream;
    for (int r = 0; r < 3; ++r) {
        stream.insert(stream.end(), data_cpp.begin(), data_cpp.end());
    }

    std::cout << "Stream length: " << stream.size() << " bytes." << std::endl;

    SovereignMorphogenicEngine engine;

    // 1. Static Topology Baseline (Fixed H=4 heads, No sprouting)
    auto m_static = run_experiment(engine, "Static Topology (Fixed H=4)", stream, false);

    // 2. Unbounded Sovereign Morphogenic Engine (Starts at H=2, Sprouts heads on high entropy)
    auto m_dynamic = run_experiment(engine, "Unbounded Morphogenic Topology", stream, true);

    std::cout << "\n=== DETAILED COMPARATIVE METRICS (EXP-473) ===\n";
    std::cout << std::left << std::setw(34) << "Topology Architecture"
              << std::setw(14) << "Q1 Loss"
              << std::setw(14) << "Q4 Loss"
              << std::setw(14) << "Loss Delta"
              << std::setw(14) << "Accuracy"
              << std::setw(14) << "Heads"
              << std::setw(16) << "Attractors"
              << std::setw(14) << "Throughput" << "\n";
    std::cout << std::string(130, '-') << "\n";

    printf("%-34s | %10.4f | %10.4f | %10.4f | %9.2f%% | %11zu | %13zu | %10.1f sps\n",
           m_static.name.c_str(), m_static.initial_loss, m_static.final_loss,
           m_static.loss_delta, m_static.accuracy, m_static.final_heads,
           m_static.memory_nodes, m_static.steps_per_sec);

    printf("%-34s | %10.4f | %10.4f | %10.4f | %9.2f%% | %11zu | %13zu | %10.1f sps\n",
           m_dynamic.name.c_str(), m_dynamic.initial_loss, m_dynamic.final_loss,
           m_dynamic.loss_delta, m_dynamic.accuracy, m_dynamic.final_heads,
           m_dynamic.memory_nodes, m_dynamic.steps_per_sec);

    return 0;
}
