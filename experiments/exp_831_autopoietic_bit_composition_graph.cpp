// ============================================================================
// KARYON-CORE EXP-831: Dynamic Autopoietic Bit-Structure Composition Graph
// KEP v30.0 Master | V = 2 Bits | Pure Bottom-Up Structure Sprouting & Pruning
// Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <memory>

static uint64_t prng_seed = 0x987654321ABCDEF0ULL;
inline uint64_t xorshift64() {
    prng_seed ^= prng_seed << 13;
    prng_seed ^= prng_seed >> 7;
    prng_seed ^= prng_seed << 17;
    return prng_seed;
}

// Structure Node representing an autonomously discovered bit pattern
struct BitPatternNode {
    uint32_t count_0 = 1;
    uint32_t count_1 = 1;
    uint32_t total_visits = 2;
    int child_0 = -1; // Index of extended pattern S + '0'
    int child_1 = -1; // Index of extended pattern S + '1'
};

class AutopoieticBitCompositionGraph {
private:
    std::vector<BitPatternNode> nodes;
    std::vector<uint8_t> bit_history;
    size_t max_history_depth;
    uint32_t sprout_threshold; // Visits required before sprouting a deeper structural child

public:
    AutopoieticBitCompositionGraph(size_t max_depth = 32, uint32_t threshold = 4)
        : max_history_depth(max_depth), sprout_threshold(threshold) {
        nodes.reserve(100000);
        nodes.push_back(BitPatternNode()); // Root node #0
    }

    void reset() {
        nodes.clear();
        nodes.reserve(100000);
        nodes.push_back(BitPatternNode());
        bit_history.clear();
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        float prob_1;
        size_t active_nodes_count;
        size_t max_depth_reached;
    };

    // STAGE 1: Predict bit t BEFORE observing actual bit t
    float predict_next_bit_prob() {
        if (bit_history.empty()) return 0.5f;

        // Traverse history down the graph to find all matching active structural nodes
        double weighted_p1 = 0.0;
        double total_weight = 0.0;

        int curr_node = 0; // Start at root
        size_t depth = 0;

        // Evaluate prediction from root
        double p1_root = static_cast<double>(nodes[0].count_1) / nodes[0].total_visits;
        weighted_p1 += p1_root * 1.0;
        total_weight += 1.0;

        // Traverse backwards from most recent bit
        size_t hist_len = bit_history.size();
        for (size_t i = 0; i < hist_len && i < max_history_depth; ++i) {
            uint8_t past_bit = bit_history[hist_len - 1 - i];
            int next_node = (past_bit == 0) ? nodes[curr_node].child_0 : nodes[curr_node].child_1;

            if (next_node == -1) break; // End of sprouted path

            curr_node = next_node;
            depth++;

            double p1_node = static_cast<double>(nodes[curr_node].count_1) / nodes[curr_node].total_visits;
            // Higher depth nodes represent longer context matches, weighted higher (depth squared weight)
            double weight = static_cast<double>((depth + 1) * (depth + 1));
            weighted_p1 += p1_node * weight;
            total_weight += weight;
        }

        return static_cast<float>(weighted_p1 / total_weight);
    }

    // STAGE 2: Observe actual bit t, update structure transition counts, and sprout new nodes bottom-up
    StepResult update_with_actual(float prob_1, uint8_t actual_bit) {
        // Bound probability to prevent numerical log(0)
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_actual = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float free_energy = -std::log(p_actual); // Cross-entropy Free Energy in nats

        bool is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        // Traverse matching nodes and update transition counts + sprout children
        int curr_node = 0;
        nodes[0].total_visits++;
        if (actual_bit == 0) nodes[0].count_0++;
        else nodes[0].count_1++;

        size_t hist_len = bit_history.size();
        size_t max_depth_used = 0;

        for (size_t i = 0; i < hist_len && i < max_history_depth; ++i) {
            uint8_t past_bit = bit_history[hist_len - 1 - i];
            int next_node = (past_bit == 0) ? nodes[curr_node].child_0 : nodes[curr_node].child_1;

            if (next_node == -1) {
                // Check if current node is visited frequently enough to sprout a new child node bottom-up
                if (nodes[curr_node].total_visits >= sprout_threshold) {
                    int new_node_idx = static_cast<int>(nodes.size());
                    nodes.push_back(BitPatternNode());

                    if (past_bit == 0) nodes[curr_node].child_0 = new_node_idx;
                    else nodes[curr_node].child_1 = new_node_idx;

                    next_node = new_node_idx;
                } else {
                    break; // Do not sprout further yet
                }
            }

            curr_node = next_node;
            max_depth_used = i + 1;

            nodes[curr_node].total_visits++;
            if (actual_bit == 0) nodes[curr_node].count_0++;
            else nodes[curr_node].count_1++;
        }

        // Record bit into history
        bit_history.push_back(actual_bit);
        if (bit_history.size() > max_history_depth * 2) {
            bit_history.erase(bit_history.begin(), bit_history.begin() + max_history_depth);
        }

        return {free_energy, is_correct, prob_1, nodes.size(), max_depth_used};
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-831: DYNAMIC AUTOPOIETIC BIT COMPOSITION GRAPH\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Dynamic Sprouting & Structure Synthesis\n";
    std::cout << "=================================================================\n\n";

    AutopoieticBitCompositionGraph graph(24, 3); // Max depth 24 bits (3 bytes), sprout after 3 visits

    // --- TEST 1: Pure Unpredictable Random Noise Control ---
    std::cout << "[TEST 1] Pure Unpredictable Random Noise Control (100,000 bits):\n";
    graph.reset();

    size_t correct_rand = 0;
    double f_rand_sum = 0.0;
    const size_t TEST_LEN = 100000;

    for (size_t i = 0; i < TEST_LEN; ++i) {
        uint8_t actual_bit = static_cast<uint8_t>(xorshift64() & 1);

        float p1 = graph.predict_next_bit_prob();
        auto res = graph.update_with_actual(p1, actual_bit);

        if (res.is_correct) correct_rand++;
        f_rand_sum += res.free_energy;
    }

    double acc_rand = (static_cast<double>(correct_rand) / TEST_LEN) * 100.0;
    double f_rand_avg = f_rand_sum / TEST_LEN;

    std::cout << "  -> Random Stream Accuracy   : " << std::fixed << std::setprecision(4) << acc_rand << " % (Expected: ~50.00%)\n";
    std::cout << "  -> Random Free Energy (F)   : " << f_rand_avg << " nats (Expected: ln(2) = 0.6931)\n";
    std::cout << "  -> Sprouted Structure Nodes : " << graph.predict_next_bit_prob() << " (Nodes count)\n\n";

    // --- TEST 2: Real Non-Repeating Source Code Stream ---
    std::cout << "[TEST 2] Non-Repeating Real Source Code Stream (Reading C++ code bit-by-bit):\n";
    graph.reset();

    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Total Raw Stream Length: " << raw_bits.size() << " bits (" << file_content.size() << " bytes)\n\n";

    size_t quarter_len = raw_bits.size() / 4;

    std::cout << " Quarter | Processed Bits | Accuracy (%) | Free Energy F (nats) | Active Nodes | Speed (bits/sec)\n";
    std::cout << "---------|----------------|--------------|----------------------|--------------|-----------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t total_correct = 0;
    double total_f_sum = 0.0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t nodes_cnt = 0;

        auto q_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = graph.predict_next_bit_prob();
            auto res = graph.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            nodes_cnt = res.active_nodes_count;
        }

        auto q_end = std::chrono::high_resolution_clock::now();
        double q_sec = std::chrono::duration<double>(q_end - q_start).count();

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;
        double speed = quarter_len / q_sec;

        total_correct += correct;
        total_f_sum += f_sum;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(12) << nodes_cnt << " | "
                  << std::setw(15) << std::setprecision(1) << speed << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    double total_acc = (static_cast<double>(total_correct) / (quarter_len * 4)) * 100.0;
    double total_avg_f = total_f_sum / (quarter_len * 4);

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-831 AUTOPOIETIC BIT GRAPH FINAL TELEMETRY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Random Stream Accuracy   : " << acc_rand << " % (Strict 50.0% Limit)\n";
    std::cout << "  - Overall Bit Accuracy     : " << total_acc << " %\n";
    std::cout << "  - Overall Free Energy (F)  : " << total_avg_f << " nats/bit\n";
    std::cout << "  - Total Execution Time     : " << total_sec << " seconds\n";
    std::cout << "  - Processing Speed         : " << ((quarter_len * 4) / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
