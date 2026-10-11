// experiments/exp_458_hyperbolic_memory_graph.cpp — EXP-458.1: Dynamic Hyperbolic Memory Graph & Epigenetic Sprouting
// Standard: KEP v16.0 Sovereign Master & Principle 15 (Epigenetic Morphogenesis) & Principle 3 (Non-Euclidean Physics)
// Zero-Python / Zero-Torch / Pure C++20 Sovereign Architecture

#include "../karyon_native_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace karyon {

struct HyperbolicNode {
    std::vector<float> center; // Poincaré coordinates (||c|| < 1.0)
    std::vector<float> value;  // Stored value vector [dim_v]
    float novelty_energy = 1.0f;
    size_t access_count = 1;
};

class SovereignHyperbolicGraphEngine : public SovereignNativeEngine {
public:
    SovereignHyperbolicGraphEngine(const NativeConfig& cfg) : SovereignNativeEngine(cfg) {
        // Initialize with root attractor node at origin
        HyperbolicNode root;
        root.center.assign(cfg.dim_k, 0.0f);
        root.value.assign(cfg.dim_v, 0.0f);
        nodes_.push_back(root);
    }

    // Poincaré ball distance: d_H(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
    float poincare_distance(const float* u, const float* v, size_t dim) const {
        float norm_u_sq = 0.0f, norm_v_sq = 0.0f, diff_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            norm_u_sq += u[i] * u[i];
            norm_v_sq += v[i] * v[i];
            float d = u[i] - v[i];
            diff_sq += d * d;
        }
        norm_u_sq = std::min(norm_u_sq, 0.999f);
        norm_v_sq = std::min(norm_v_sq, 0.999f);

        float denom = (1.0f - norm_u_sq) * (1.0f - norm_v_sq);
        float arg = 1.0f + 2.0f * diff_sq / std::max(denom, 1e-6f);
        return std::acosh(std::max(arg, 1.000001f));
    }

    // Step with coupled hyperbolic episodic retrieval
    StepMetrics step_with_hyperbolic(uint8_t byte_in, uint8_t byte_target) {
        // Standard forward step
        auto m = step(byte_in, byte_target);

        // Map key vector into Poincaré Ball
        const float* emb_ptr = byte_emb_.data() + byte_in * config_.dim_emb;
        std::vector<float> emb(config_.dim_emb);
        for (size_t e = 0; e < config_.dim_emb; ++e) emb[e] = emb_ptr[e];

        std::vector<float> k_proj(config_.num_heads * config_.dim_k);
        matvec_mul(W_k_.data(), emb.data(), k_proj.data(), k_proj.size(), config_.dim_emb);
        normalize_heads(k_proj.data(), config_.num_heads, config_.dim_k);

        std::vector<float> v_proj(config_.num_heads * config_.dim_v);
        matvec_mul(W_v_.data(), emb.data(), v_proj.data(), v_proj.size(), config_.dim_emb);

        std::vector<float> q_poincare(config_.dim_k);
        float q_sq = 0.0f;
        for (size_t i = 0; i < config_.dim_k; ++i) {
            q_poincare[i] = k_proj[i];
            q_sq += q_poincare[i] * q_poincare[i];
        }
        float q_norm = std::sqrt(q_sq);
        if (q_norm >= 0.95f) {
            float scale = 0.90f / q_norm;
            for (size_t i = 0; i < config_.dim_k; ++i) q_poincare[i] *= scale;
        }

        // Search nearest node
        float min_dist = 1e9f;
        size_t closest_node = 0;
        for (size_t n = 0; n < nodes_.size(); ++n) {
            float dist = poincare_distance(q_poincare.data(), nodes_[n].center.data(), config_.dim_k);
            if (dist < min_dist) {
                min_dist = dist;
                closest_node = n;
            }
        }

        // Active Sprouting: If loss > 1.0 (Surprise) and distance to existing cluster > 0.4, sprout!
        if (m.loss > 1.0f && min_dist > 0.40f && nodes_.size() < max_nodes_) {
            HyperbolicNode new_node;
            new_node.center = q_poincare;
            new_node.value.assign(config_.dim_v, 0.0f);
            for (size_t v = 0; v < config_.dim_v; ++v) {
                new_node.value[v] = v_proj[v];
            }
            new_node.novelty_energy = m.loss;
            nodes_.push_back(new_node);
        } else {
            // Reinforce nearest attractor
            for (size_t v = 0; v < config_.dim_v; ++v) {
                nodes_[closest_node].value[v] = 0.92f * nodes_[closest_node].value[v] + 0.15f * v_proj[v];
            }
            nodes_[closest_node].access_count++;
        }

        return m;
    }

    size_t get_node_count() const { return nodes_.size(); }

private:
    std::vector<HyperbolicNode> nodes_;
    size_t max_nodes_ = 256;
};

} // namespace karyon

int main() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE EXP-458.1: DYNAMIC HYPERBOLIC MEMORY GRAPH BENCHMARK\n";
    std::cout << "   Poincare Ball Manifold & Epigenetic Attractor Sprouting (H^n)\n";
    std::cout << "==================================================================\n";

    std::string text = 
        "struct StateVector { float x; float y; float z; };\n"
        "void compute_trajectory(StateVector& s, float dt) {\n"
        "    s.x += 0.5f * dt * (s.y - s.x);\n"
        "    s.y += 0.2f * dt * (s.x * (28.0f - s.z) - s.y);\n"
        "    s.z += 0.8f * dt * (s.x * s.y - (8.0f / 3.0f) * s.z);\n"
        "}\n"
        "int main() {\n"
        "    StateVector sv{1.0f, 1.0f, 1.0f};\n"
        "    for (int t = 0; t < 1000; ++t) compute_trajectory(sv, 0.01f);\n"
        "    return 0;\n"
        "}\n";

    std::string full_stream;
    for (int i = 0; i < 8; ++i) full_stream += text;

    karyon::NativeConfig cfg;
    cfg.dim = 258;
    cfg.num_heads = 4;
    cfg.dim_k = 16;
    cfg.dim_v = 32;
    cfg.dim_emb = 128;
    cfg.lr = 0.015f;
    cfg.max_thinking_turns = 8;
    cfg.entropy_satisfy = 0.35f;
    cfg.eps_halt = 0.01f;
    cfg.use_rust_backend = false;

    karyon::SovereignNativeEngine baseline_eng(cfg);
    karyon::SovereignHyperbolicGraphEngine hyp_eng(cfg);

    size_t total_steps = full_stream.size() - 1;
    size_t baseline_correct = 0;
    size_t hyp_correct = 0;
    float baseline_loss_sum = 0.0f;
    float hyp_loss_sum = 0.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "[INFO] Stream length: " << full_stream.size() << " bytes.\n";
    std::cout << "[INFO] Active Neurogenesis: Tracking Poincare Attractor Sprouting under Stream Surprise...\n\n";

    for (size_t i = 0; i < total_steps; ++i) {
        uint8_t byte_in = static_cast<uint8_t>(full_stream[i]);
        uint8_t byte_target = static_cast<uint8_t>(full_stream[i + 1]);

        auto base_m = baseline_eng.step(byte_in, byte_target);
        baseline_loss_sum += base_m.loss;
        if (base_m.is_correct) baseline_correct++;

        auto hyp_m = hyp_eng.step_with_hyperbolic(byte_in, byte_target);
        hyp_loss_sum += hyp_m.loss;
        if (hyp_m.is_correct) hyp_correct++;

        if ((i + 1) % 500 == 0 || i == total_steps - 1) {
            float base_acc = 100.0f * static_cast<float>(baseline_correct) / (i + 1);
            float hyp_acc = 100.0f * static_cast<float>(hyp_correct) / (i + 1);
            float base_mean_l = baseline_loss_sum / (i + 1);
            float hyp_mean_l = hyp_loss_sum / (i + 1);

            std::cout << "  Step " << std::setw(4) << (i + 1) << "/" << total_steps 
                      << " | Base Loss: " << std::fixed << std::setprecision(4) << base_mean_l
                      << " | Hyp Loss: " << hyp_mean_l
                      << " | Base Acc: " << std::setprecision(2) << base_acc << "%"
                      << " | Hyp Acc: " << hyp_acc << "%"
                      << " | Sprouted Nodes: " << hyp_eng.get_node_count() << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    double throughput = static_cast<double>(total_steps) / elapsed;

    float final_base_acc = 100.0f * static_cast<float>(baseline_correct) / total_steps;
    float final_hyp_acc = 100.0f * static_cast<float>(hyp_correct) / total_steps;
    float final_base_loss = baseline_loss_sum / total_steps;
    float final_hyp_loss = hyp_loss_sum / total_steps;

    std::cout << "\n=============================================================\n";
    std::cout << "   EXP-458.1 BENCHMARK TELEMETRY & DECISION VERDICT\n";
    std::cout << "=============================================================\n";
    std::cout << "Total Steps Evaluated      : " << total_steps << "\n";
    std::cout << "Elapsed Time               : " << elapsed << " s\n";
    std::cout << "Throughput                 : " << throughput << " steps/s\n";
    std::cout << "Sprouted Hyperbolic Nodes  : " << hyp_eng.get_node_count() << " attractors\n";
    std::cout << "Baseline Greedy Accuracy   : " << final_base_acc << " %\n";
    std::cout << "Hyperbolic Graph Accuracy  : " << final_hyp_acc << " %\n";
    std::cout << "Baseline Final Loss        : " << final_base_loss << " nats\n";
    std::cout << "Hyperbolic Final Loss      : " << final_hyp_loss << " nats\n";
    std::cout << "=============================================================\n";

    std::ofstream out_json("experiments/exp_458_telemetry.json");
    out_json << "{\n"
             << "  \"exp_id\": \"EXP-458\",\n"
             << "  \"baseline_accuracy\": " << final_base_acc << ",\n"
             << "  \"hyp_accuracy\": " << final_hyp_acc << ",\n"
             << "  \"baseline_loss\": " << final_base_loss << ",\n"
             << "  \"hyp_loss\": " << final_hyp_loss << ",\n"
             << "  \"sprouted_nodes\": " << hyp_eng.get_node_count() << ",\n"
             << "  \"throughput_steps_per_sec\": " << throughput << ",\n"
             << "  \"elapsed_time\": " << elapsed << "\n"
             << "}\n";
    out_json.close();

    return 0;
}
