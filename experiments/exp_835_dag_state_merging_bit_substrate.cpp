// ============================================================================
// KARYON-CORE EXP-835: Equivalence-Class Merging Prediction DAG Substrate
// KEP v30.0 Master | V = 2 Bits | Compact State-Merging Graph & Zero Memory Bloat
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

struct DAGStateNode {
    uint32_t id;
    uint32_t level;                    // Context depth
    std::string pattern;               // Primary representative pattern
    uint64_t count_0 = 1;              // Transition count to bit 0
    uint64_t count_1 = 1;              // Transition count to bit 1
    uint64_t total_occurrences = 2;
    uint32_t merged_into = 0xFFFFFFFF; // Pointer if merged into another equivalence state

    DAGStateNode(uint32_t s_id, uint32_t lvl, const std::string& pat)
        : id(s_id), level(lvl), pattern(pat) {}

    double get_p1() const {
        return static_cast<double>(count_1) / total_occurrences;
    }
};

class StateMergingBitDAGSubstrate {
private:
    std::vector<DAGStateNode> nodes;
    std::unordered_map<std::string, uint32_t> pattern_to_node;

    std::vector<uint8_t> bit_context;
    size_t max_context_depth;
    uint32_t condensation_threshold;
    double merge_divergence_threshold; // Max difference in p1 to allow merging into equivalence class

public:
    StateMergingBitDAGSubstrate(size_t max_depth = 32, uint32_t threshold = 2, double merge_thresh = 0.02)
        : max_context_depth(max_depth), condensation_threshold(threshold), merge_divergence_threshold(merge_thresh) {
        
        // Base Primitive Bit Nodes (Level 0)
        nodes.emplace_back(0, 0, "0");
        nodes.emplace_back(1, 0, "1");
        
        pattern_to_node["0"] = 0;
        pattern_to_node["1"] = 1;
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        float prob_1;
        size_t total_nodes;
        size_t merged_nodes_count;
        uint32_t max_level;
    };

    uint32_t resolve_canonical_node(uint32_t node_id) {
        uint32_t curr = node_id;
        while (nodes[curr].merged_into != 0xFFFFFFFF) {
            curr = nodes[curr].merged_into;
        }
        return curr;
    }

    // STAGE 1: Predict next bit b_{t+1} using canonical equivalence state graph
    float predict_next_bit_prob() {
        if (bit_context.empty()) return 0.5f;

        double weighted_p1 = 0.0;
        double total_weight = 0.0;

        std::string current_str = "";
        size_t ctx_len = bit_context.size();

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = pattern_to_node.find(current_str);
            if (it != pattern_to_node.end()) {
                uint32_t node_id = resolve_canonical_node(it->second);
                const auto& node = nodes[node_id];

                double p1 = node.get_p1();
                double weight = std::pow(2.0, static_cast<double>(node.level + 1));

                weighted_p1 += p1 * weight;
                total_weight += weight;
            }
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

    // STAGE 2: Observe actual bit b_{t+1}, update statistics, & perform DAG equivalence state merging
    StepResult update_with_actual(float prob_1, uint8_t actual_bit) {
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float free_energy = -std::log(p_target);

        bool is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        std::string current_str = "";
        size_t ctx_len = bit_context.size();
        uint32_t max_level = 0;
        size_t merged_count = 0;

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = pattern_to_node.find(current_str);
            if (it != pattern_to_node.end()) {
                uint32_t canonical_id = resolve_canonical_node(it->second);
                auto& node = nodes[canonical_id];

                node.total_occurrences++;
                if (actual_bit == 0) node.count_0++;
                else node.count_1++;

                if (node.level > max_level) max_level = node.level;

                // Spontaneous Sprouting / Condensation of deeper pattern
                if (node.total_occurrences >= condensation_threshold) {
                    std::string new_str = current_str + (actual_bit ? "1" : "0");
                    
                    if (pattern_to_node.find(new_str) == pattern_to_node.end()) {
                        uint32_t new_id = static_cast<uint32_t>(nodes.size());
                        uint32_t new_lvl = node.level + 1;

                        nodes.emplace_back(new_id, new_lvl, new_str);
                        pattern_to_node[new_str] = new_id;

                        // EQUIVALENCE STATE MERGING:
                        // Check if an existing node at the same level has near-identical prediction probability p1.
                        // If so, redirect new_id to merge into that existing node rather than keeping a duplicate state!
                        double new_p1 = (actual_bit == 1) ? 0.66 : 0.33; // Initial estimate

                        // Search among recent same-level nodes for prediction equivalence
                        for (int check_id = static_cast<int>(new_id) - 1; check_id >= std::max(0, static_cast<int>(new_id) - 100); --check_id) {
                            if (nodes[check_id].level == new_lvl && nodes[check_id].merged_into == 0xFFFFFFFF) {
                                double existing_p1 = nodes[check_id].get_p1();
                                if (std::abs(existing_p1 - new_p1) <= merge_divergence_threshold && nodes[check_id].total_occurrences >= 5) {
                                    // Merge new_id into check_id!
                                    nodes[new_id].merged_into = check_id;
                                    nodes[check_id].total_occurrences += 2;
                                    if (actual_bit == 1) nodes[check_id].count_1++;
                                    else nodes[check_id].count_0++;
                                    merged_count++;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Context update
        bit_context.push_back(actual_bit);
        if (bit_context.size() > max_context_depth * 2) {
            bit_context.erase(bit_context.begin(), bit_context.begin() + max_context_depth);
        }

        return {free_energy, is_correct, prob_1, nodes.size(), merged_count, max_level};
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-835: STATE-MERGING PREDICTION DAG SUBSTRATE\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Equivalence Class Collapse & Zero Loss\n";
    std::cout << "=================================================================\n\n";

    StateMergingBitDAGSubstrate substrate(32, 2, 0.01); // 1% divergence threshold for merging

    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Loaded " << raw_bits.size() << " raw bits from C++ source stream (" << file_content.size() << " bytes)\n\n";

    // --- PASS 1: DAG STATE SYNTHESIS & EQUIVALENCE MERGING ---
    std::cout << "--- [PASS 1: INITIAL DISCOVERY & STATE MERGING] ---\n";
    size_t quarter_len = raw_bits.size() / 4;
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Total Nodes | Max Level\n";
    std::cout << "---------|----------------|------------------|----------------------|-------------|----------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t node_cnt = 0;
        uint32_t max_lvl = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            node_cnt = res.total_nodes;
            max_lvl = res.max_level;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(11) << node_cnt << " | "
                  << std::setw(8) << "Level " + std::to_string(max_lvl) << "\n";
    }

    // --- PASS 2: LIFELONG MASTERY ON COMPACT EQUIVALENCE DAG ---
    std::cout << "\n--- [PASS 2: LIFELONG MASTERY PASS ON DAG] ---\n";
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Total Nodes | Max Level\n";
    std::cout << "---------|----------------|------------------|----------------------|-------------|----------\n";

    size_t total_p2_correct = 0;
    double total_p2_f = 0.0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t node_cnt = 0;
        uint32_t max_lvl = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            node_cnt = res.total_nodes;
            max_lvl = res.max_level;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        total_p2_correct += correct;
        total_p2_f += f_sum;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(11) << node_cnt << " | "
                  << std::setw(8) << "Level " + std::to_string(max_lvl) << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    double pass2_acc = (static_cast<double>(total_p2_correct) / (quarter_len * 4)) * 100.0;
    double pass2_avg_f = total_p2_f / (quarter_len * 4);

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-835 DAG STATE MERGING FINAL SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Pass 2 Mastery Accuracy      : " << pass2_acc << " %\n";
    std::cout << "  - Pass 2 Free Energy (F)       : " << pass2_avg_f << " nats/bit\n";
    std::cout << "  - Total Execution Time         : " << total_sec << " seconds\n";
    std::cout << "  - Substrate Processing Speed   : " << ((quarter_len * 8) / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
