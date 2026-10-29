// ============================================================================
// KARYON-CORE EXP-822: Sovereign Compositional Interaction Substrate
// KEP v30.0 Master | V = 2 Bits | Scale-Free Recursive Composition
// Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <fstream>
#include <iomanip>

// Simple LCG PRNG for reproducible synthetic/text stream evaluation
static uint64_t lcg_state = 0x1234567891011121ULL;
inline uint64_t next_lcg() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state;
}

// Compositional Structure Node
struct CompositionNode {
    uint32_t id;
    uint32_t level;
    std::vector<uint32_t> child_ids; // IDs of constituent simpler structures
    float wave_weight;               // Active energy / weight
    float phase;                     // Dynamic phase s_t \in [-1, +1]
    uint64_t usage_count;            // Number of activations
    float free_energy_contrib;       // Causal contribution to F reduction

    CompositionNode(uint32_t node_id, uint32_t lvl)
        : id(node_id), level(lvl), wave_weight(1.0f), phase(0.0f), usage_count(1), free_energy_contrib(0.0f) {}
};

class SovereignCompositionalSubstrate {
private:
    std::vector<CompositionNode> nodes;
    std::unordered_map<uint64_t, uint32_t> pair_to_node; // Maps hash(child_a, child_b) -> parent_node_id
    std::unordered_map<uint64_t, uint64_t> pair_frequency; // Maps hash(child_a, child_b) -> co-occurrence count
    
    // Interactions matrix between nodes (sparse modulation links)
    std::unordered_map<uint64_t, float> interaction_weights; // hash(node_a, node_b) -> weight

    uint32_t next_id;
    uint32_t max_active_level;
    size_t max_nodes_limit;

    inline uint64_t hash_pair(uint32_t a, uint32_t b) const {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    }

public:
    SovereignCompositionalSubstrate(size_t max_nodes = 65536)
        : next_id(0), max_active_level(0), max_nodes_limit(max_nodes) {
        // Initialize base level (Level 0): Bit 0 and Bit 1 primitive nodes
        nodes.emplace_back(next_id++, 0); // Node 0: represents bit 0
        nodes.emplace_back(next_id++, 0); // Node 1: represents bit 1
        nodes[0].phase = -1.0f;
        nodes[1].phase = +1.0f;
    }

    // Process a bit and update compositional structures and interactions
    struct StepResult {
        float predicted_bit_prob; // Probability of bit 1
        float local_free_energy;  // F = -log(p)
        bool correct_prediction;
    };

    StepResult process_bit(int actual_bit, const std::vector<int>& recent_bit_history) {
        // 1. Gather active primitive nodes from recent history
        uint32_t current_primitive_id = (actual_bit == 1) ? 1 : 0;
        
        // Compute predicted bit probability based on active structures and interactions
        float logit = 0.0f;
        float total_weight = 0.0f;

        // Active context tracking
        std::vector<uint32_t> active_chain;
        active_chain.push_back(current_primitive_id);

        // Traversal of recent history to activate composition nodes
        if (!recent_bit_history.empty()) {
            uint32_t last_bit_id = (recent_bit_history.back() == 1) ? 1 : 0;
            uint64_t key = hash_pair(last_bit_id, current_primitive_id);
            
            // Track pair frequency for potential composition genesis
            pair_frequency[key]++;

            // Check if composite node exists for this pair
            auto it = pair_to_node.find(key);
            if (it != pair_to_node.end()) {
                uint32_t composite_id = it->second;
                nodes[composite_id].usage_count++;
                active_chain.push_back(composite_id);

                // Interactions: check modulation from other active nodes
                for (uint32_t act_id : active_chain) {
                    if (act_id != composite_id) {
                        uint64_t link_key = hash_pair(act_id, composite_id);
                        logit += interaction_weights[link_key] * nodes[act_id].wave_weight;
                        total_weight += std::abs(interaction_weights[link_key]);
                    }
                }
            }

            // Base transition logit contribution
            float p_base = (nodes[current_primitive_id].usage_count + 1.0f) / 
                           (nodes[0].usage_count + nodes[1].usage_count + 2.0f);
            logit += (current_primitive_id == 1) ? p_base : -p_base;
            total_weight += 1.0f;
        }

        // Convert logit to probability via sigmoid
        float prob_1 = 1.0f / (1.0f + std::exp(-logit));
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        int predicted_bit = (prob_1 >= 0.5f) ? 1 : 0;
        bool is_correct = (predicted_bit == actual_bit);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float local_f = -std::log(p_target);

        // Update node usage
        nodes[current_primitive_id].usage_count++;

        // 2. Autopoietic Genesis of Composite Structures (Level k > 0)
        if (recent_bit_history.size() >= 2 && next_id < max_nodes_limit) {
            uint32_t prev_id = (recent_bit_history.back() == 1) ? 1 : 0;
            uint64_t key = hash_pair(prev_id, current_primitive_id);

            // Trigger composition if frequency threshold reached
            if (pair_frequency[key] >= 8 && pair_to_node.find(key) == pair_to_node.end()) {
                uint32_t parent_lvl = std::max(nodes[prev_id].level, nodes[current_primitive_id].level) + 1;
                uint32_t new_id = next_id++;
                
                nodes.emplace_back(new_id, parent_lvl);
                nodes[new_id].child_ids = {prev_id, current_primitive_id};
                pair_to_node[key] = new_id;

                if (parent_lvl > max_active_level) {
                    max_active_level = parent_lvl;
                }

                // Initialize interaction link between child and new composite
                uint64_t link_key = hash_pair(prev_id, new_id);
                interaction_weights[link_key] = 0.5f; // Initial cooperative coupling
            }
        }

        // 3. Dynamic Interaction Plasticity: strengthen/weaken interaction links based on F
        if (!active_chain.empty() && active_chain.size() > 1) {
            float feedback = is_correct ? +0.05f : -0.05f;
            for (size_t i = 0; i < active_chain.size() - 1; ++i) {
                uint64_t link_key = hash_pair(active_chain[i], active_chain[i+1]);
                interaction_weights[link_key] += feedback;
                interaction_weights[link_key] = std::clamp(interaction_weights[link_key], -2.0f, 2.0f);
            }
        }

        return {prob_1, local_f, is_correct};
    }

    size_t get_node_count() const { return nodes.size(); }
    uint32_t get_max_level() const { return max_active_level; }
    size_t get_interaction_count() const { return interaction_weights.size(); }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-822: Sovereign Compositional Interaction\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Scale-Free Composition\n";
    std::cout << "=================================================================\n\n";

    // Build sample structured bit stream from text / PGN pattern
    std::string text_source = 
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O "
        "9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 "
        "16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 ";

    std::vector<int> bit_stream;
    for (char c : text_source) {
        for (int b = 7; b >= 0; --b) {
            bit_stream.push_back((c >> b) & 1);
        }
    }

    // Expand bit stream to 200,000 bits by repeating pattern with slight LCG noise
    const size_t TOTAL_BITS = 200000;
    std::vector<int> full_stream;
    full_stream.reserve(TOTAL_BITS);

    size_t idx = 0;
    for (size_t i = 0; i < TOTAL_BITS; ++i) {
        int bit = bit_stream[idx % bit_stream.size()];
        // 2% noise injection to test robustness
        if ((next_lcg() % 100) < 2) {
            bit ^= 1;
        }
        full_stream.push_back(bit);
        idx++;
    }

    SovereignCompositionalSubstrate substrate(65536);

    std::vector<int> bit_history;
    size_t correct_count = 0;
    double total_free_energy = 0.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < full_stream.size(); ++t) {
        int actual_bit = full_stream[t];
        auto res = substrate.process_bit(actual_bit, bit_history);

        if (res.correct_prediction) correct_count++;
        total_free_energy += res.local_free_energy;

        bit_history.push_back(actual_bit);
        if (bit_history.size() > 64) {
            bit_history.erase(bit_history.begin());
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    double bit_accuracy = (static_cast<double>(correct_count) / TOTAL_BITS) * 100.0;
    double avg_free_energy = total_free_energy / TOTAL_BITS;
    double throughput_bits_sec = TOTAL_BITS / elapsed_sec;

    std::cout << "================================================ catastrophe ====\n";
    std::cout << "  EXP-822 SOVEREIGN COMPOSITIONAL INTERACTION TELEMETRY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Bit Predictive Accuracy     : " << std::fixed << std::setprecision(4) << bit_accuracy << " %\n";
    std::cout << "  - Average Bit Free Energy (F) : " << avg_free_energy << "\n";
    std::cout << "  - Total Composition Nodes     : " << substrate.get_node_count() << "\n";
    std::cout << "  - Max Composition Hierarchy   : Level " << substrate.get_max_level() << "\n";
    std::cout << "  - Active Inter-Node Links     : " << substrate.get_interaction_count() << "\n";
    std::cout << "  - Evaluated Bit Stream Length : " << TOTAL_BITS << " bits\n";
    std::cout << "  - Substrate Bit Throughput    : " << throughput_bits_sec << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
