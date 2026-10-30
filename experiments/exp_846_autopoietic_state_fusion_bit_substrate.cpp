// ============================================================================
// KARYON-CORE EXP-846: Autopoietic Bottom-Up State Fusion & Graph Collapse
// KEP v30.0 Master | V = 2 Bits | Autonomous Branch Merging into State Graph
// Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <fstream>

// Autopoietic Graph Node capable of online State Fusion
struct FusionNode {
    uint32_t id;
    uint32_t level;                    // Context depth
    std::string pattern;               // Representative pattern string
    
    uint64_t count_0 = 0;
    uint64_t count_1 = 0;
    uint64_t total = 0;

    int next_0 = -1;                   // Outgoing edge for bit 0
    int next_1 = -1;                   // Outgoing edge for bit 1

    int fusion_target = -1;            // If fused into an equivalent state
    bool epigenetically_locked = false;// Immutable deterministic invariant

    FusionNode(uint32_t n_id, uint32_t lvl, const std::string& pat)
        : id(n_id), level(lvl), pattern(pat) {}

    int get_canonical_id() const {
        return (fusion_target != -1) ? fusion_target : id;
    }

    double get_p1() const {
        if (total == 0) return 0.5;
        return static_cast<double>(count_1) / static_cast<double>(total);
    }

    void evaluate_epigenetic_lock() {
        if (epigenetically_locked) return;
        if (total >= 1) {
            double p1 = get_p1();
            if (p1 == 1.0 || p1 == 0.0) {
                epigenetically_locked = true;
            }
        }
    }
};

class AutopoieticStateFusionSubstrate {
private:
    std::vector<FusionNode> nodes;
    std::unordered_map<std::string, uint32_t> pattern_map;
    
    // Equivalence Signature Map for Online State Fusion
    // Signature key: "next_0_canonical:next_1_canonical:deterministic_p1"
    std::unordered_map<std::string, uint32_t> state_signature_map;

    std::vector<uint8_t> bit_context;
    size_t max_context_depth;
    size_t fused_branches_count = 0;

public:
    AutopoieticStateFusionSubstrate(size_t max_depth = 128)
        : max_context_depth(max_depth) {
        
        nodes.reserve(2000000);
        pattern_map.reserve(2000000);

        // Root Primitive Nodes
        nodes.emplace_back(0, 0, "0");
        nodes.emplace_back(1, 0, "1");
        pattern_map["0"] = 0;
        pattern_map["1"] = 1;
    }

    size_t get_fused_count() const { return fused_branches_count; }
    size_t get_total_nodes() const { return nodes.size(); }

    // Predict next bit by traversing the autopoietic fused graph
    float predict_next_bit_prob() {
        if (bit_context.empty()) return 0.5f;

        std::string current_str = "";
        size_t ctx_len = bit_context.size();

        int deepest_locked_level = -1;
        double deepest_locked_p1 = -1.0;

        double weighted_p1 = 0.0;
        double total_weight = 0.0;

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = pattern_map.find(current_str);
            if (it != pattern_map.end()) {
                uint32_t orig_id = it->second;
                uint32_t canon_id = nodes[orig_id].get_canonical_id();
                const auto& node = nodes[canon_id];

                double p1 = node.get_p1();

                // Epigenetically locked deterministic overrule
                if (node.epigenetically_locked && (p1 == 1.0 || p1 == 0.0)) {
                    if (static_cast<int>(node.level) > deepest_locked_level) {
                        deepest_locked_level = node.level;
                        deepest_locked_p1 = p1;
                    }
                }

                double weight = std::pow(3.0, static_cast<double>(node.level + 1));
                weighted_p1 += p1 * weight;
                total_weight += weight;
            }
        }

        // DEEPEST DETERMINISTIC OVERRULE ON FUSED GRAPH
        if (deepest_locked_level >= 1 && deepest_locked_p1 >= 0.0) {
            return (deepest_locked_p1 == 1.0) ? 0.999999f : 0.000001f;
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

    // Process actual bit, update node statistics, and execute State Fusion
    void update_and_fuse(float prob_1, uint8_t actual_bit, bool& is_correct, float& free_energy) {
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        free_energy = -std::log(p_target);

        is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        std::string current_str = "";
        size_t ctx_len = bit_context.size();
        int prev_node_id = -1;

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = pattern_map.find(current_str);
            if (it != pattern_map.end()) {
                uint32_t node_id = nodes[it->second].get_canonical_id();
                auto& node = nodes[node_id];

                node.total++;
                if (actual_bit == 0) node.count_0++;
                else node.count_1++;

                node.evaluate_epigenetic_lock();

                // Connect transition in graph
                if (prev_node_id != -1) {
                    if (actual_bit == 0) nodes[prev_node_id].next_0 = node_id;
                    else nodes[prev_node_id].next_1 = node_id;
                }

                // Check for State Equivalence Fusion
                if (node.total >= 2 && node.epigenetically_locked) {
                    std::string sig = std::to_string(node.next_0) + ":" + 
                                      std::to_string(node.next_1) + ":" + 
                                      (node.get_p1() == 1.0 ? "1" : "0");

                    auto sig_it = state_signature_map.find(sig);
                    if (sig_it != state_signature_map.end() && sig_it->second != node_id) {
                        // MERGE / FUSE BRANCH INTO EXISTING EQUIVALENT STATE!
                        node.fusion_target = sig_it->second;
                        fused_branches_count++;
                    } else {
                        state_signature_map[sig] = node_id;
                    }
                }

                // Sprout deeper context node if missing
                std::string new_str = current_str + (actual_bit ? "1" : "0");
                if (pattern_map.find(new_str) == pattern_map.end()) {
                    uint32_t new_id = static_cast<uint32_t>(nodes.size());
                    nodes.emplace_back(new_id, node.level + 1, new_str);
                    pattern_map[new_str] = new_id;
                }

                prev_node_id = node_id;
            }
        }

        bit_context.push_back(actual_bit);
        if (bit_context.size() > max_context_depth) {
            bit_context.erase(bit_context.begin(), bit_context.end() - max_context_depth);
        }
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-846: AUTOPOIETIC BOTTOM-UP STATE FUSION\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Autonomous Tree-to-Graph Collapse\n";
    std::cout << "=================================================================\n\n";

    AutopoieticStateFusionSubstrate substrate(128);

    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Loaded " << raw_bits.size() << " raw bits from C++ source stream (" << file_content.size() << " bytes)\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Multi-pass execution to evaluate State Fusion convergence
    for (int pass = 1; pass <= 3; ++pass) {
        std::cout << "--- [PASS " << pass << ": " 
                  << (pass == 1 ? "DISCOVERY & SPROUTING" : (pass == 2 ? "STATE FUSION & COLLAPSE" : "OPTIMAL GRAPH CONVERGENCE"))
                  << "] ---\n";

        size_t pass_correct = 0;
        size_t pass_errors = 0;
        double pass_f_sum = 0.0;

        for (size_t i = 0; i < raw_bits.size(); ++i) {
            float p1 = substrate.predict_next_bit_prob();
            bool correct;
            float fe;
            substrate.update_and_fuse(p1, raw_bits[i], correct, fe);

            if (correct) pass_correct++;
            else pass_errors++;

            pass_f_sum += fe;
        }

        double acc = (static_cast<double>(pass_correct) / raw_bits.size()) * 100.0;
        double avg_f = pass_f_sum / raw_bits.size();

        std::cout << ">>> PASS " << pass << " RESULTS:\n"
                  << "  - Accuracy           : " << std::fixed << std::setprecision(6) << acc << " %\n"
                  << "  - Errors             : " << pass_errors << " / " << raw_bits.size() << "\n"
                  << "  - Free Energy (F)    : " << avg_f << " nats/bit\n"
                  << "  - Total Substrate    : " << substrate.get_total_nodes() << " nodes\n"
                  << "  - Fused Tree Branches: " << substrate.get_fused_count() << " branches merged into graph\n\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    std::cout << "=================================================================\n";
    std::cout << "  EXP-846 AUTOPOIETIC STATE FUSION COMPLETE\n";
    std::cout << "  - Total Multi-Pass Duration : " << total_sec << " seconds\n";
    std::cout << "=================================================================\n";

    return 0;
}
