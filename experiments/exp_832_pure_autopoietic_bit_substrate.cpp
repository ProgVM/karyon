// ============================================================================
// KARYON-CORE EXP-832: Pure Autopoietic Bit-Entity Condensation Substrate
// KEP v30.0 Master | V = 2 Bits | Pure Bottom-Up Entity Condensation
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

struct CondensedEntity {
    uint32_t id;
    uint32_t level;                    // 0 = primitive bit (0/1), 1 = 2-bit pair, 2 = 4-bit, 3 = byte, ...
    std::vector<uint8_t> bit_sequence; // The exact binary sequence represented by this entity
    uint64_t count_0 = 1;              // Transitions to next bit 0
    uint64_t count_1 = 1;              // Transitions to next bit 1
    uint64_t total_occurrences = 2;

    CondensedEntity(uint32_t e_id, uint32_t lvl, const std::vector<uint8_t>& seq)
        : id(e_id), level(lvl), bit_sequence(seq) {}
};

class PureAutopoieticBitSubstrate {
private:
    std::vector<CondensedEntity> entities;
    
    // Fast lookup from exact bit sequence to entity ID
    std::unordered_map<std::string, uint32_t> sequence_to_entity;

    std::vector<uint8_t> bit_context;
    size_t max_context_depth;
    uint32_t condensation_threshold; // Occurrences before condensing a longer sequence into an entity

public:
    PureAutopoieticBitSubstrate(size_t max_depth = 32, uint32_t threshold = 3)
        : max_context_depth(max_depth), condensation_threshold(threshold) {
        // Base Primitive Bit Entities (Level 0)
        entities.emplace_back(0, 0, std::vector<uint8_t>{0});
        entities.emplace_back(1, 0, std::vector<uint8_t>{1});
        
        sequence_to_entity["0"] = 0;
        sequence_to_entity["1"] = 1;
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        float prob_1;
        size_t total_entities;
        uint32_t max_level;
    };

    void reset() {
        entities.clear();
        sequence_to_entity.clear();
        bit_context.clear();

        entities.emplace_back(0, 0, std::vector<uint8_t>{0});
        entities.emplace_back(1, 0, std::vector<uint8_t>{1});
        sequence_to_entity["0"] = 0;
        sequence_to_entity["1"] = 1;
    }

    // STAGE 1: Predict next bit $b_{t+1}$ using highest-level active condensed entity
    float predict_next_bit_prob() {
        if (bit_context.empty()) return 0.5f;

        double weighted_p1 = 0.0;
        double total_weight = 0.0;

        std::string current_str = "";
        size_t ctx_len = bit_context.size();

        // Search from longest matching active suffix (highest hierarchical entity)
        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = sequence_to_entity.find(current_str);
            if (it != sequence_to_entity.end()) {
                uint32_t ent_id = it->second;
                const auto& ent = entities[ent_id];

                double p1 = static_cast<double>(ent.count_1) / ent.total_occurrences;
                
                // Weight exponentially increases with entity level (sequence length)
                double weight = std::pow(2.0, static_cast<double>(ent.level + 1));
                weighted_p1 += p1 * weight;
                total_weight += weight;
            }
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

    // STAGE 2: Observe actual bit $b_{t+1}$, update transition stats & condense new bit-entities bottom-up
    StepResult update_with_actual(float prob_1, uint8_t actual_bit) {
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float free_energy = -std::log(p_target); // Variational Free Energy in nats

        bool is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        // Update statistics for all matching active entities in current bit context
        std::string current_str = "";
        size_t ctx_len = bit_context.size();

        uint32_t max_level = 0;

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = sequence_to_entity.find(current_str);
            if (it != sequence_to_entity.end()) {
                uint32_t ent_id = it->second;
                entities[ent_id].total_occurrences++;
                if (actual_bit == 0) entities[ent_id].count_0++;
                else entities[ent_id].count_1++;

                if (entities[ent_id].level > max_level) {
                    max_level = entities[ent_id].level;
                }

                // Autopoietic Condensation:
                // If this entity appears frequently, condense it + actual_bit into a NEW higher-level entity!
                if (entities[ent_id].total_occurrences >= condensation_threshold) {
                    std::string new_str = current_str + (actual_bit ? "1" : "0");
                    if (sequence_to_entity.find(new_str) == sequence_to_entity.end()) {
                        uint32_t new_id = static_cast<uint32_t>(entities.size());
                        uint32_t new_lvl = entities[ent_id].level + 1;

                        std::vector<uint8_t> new_seq = entities[ent_id].bit_sequence;
                        new_seq.push_back(actual_bit);

                        entities.emplace_back(new_id, new_lvl, new_seq);
                        sequence_to_entity[new_str] = new_id;
                    }
                }
            }
        }

        // Add actual_bit to context window
        bit_context.push_back(actual_bit);
        if (bit_context.size() > max_context_depth * 2) {
            bit_context.erase(bit_context.begin(), bit_context.begin() + max_context_depth);
        }

        return {free_energy, is_correct, prob_1, entities.size(), max_level};
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-832: PURE AUTOPOIETIC BIT-ENTITY SUBSTRATE\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Spontaneous Hierarchical Synthesis\n";
    std::cout << "=================================================================\n\n";

    PureAutopoieticBitSubstrate substrate(32, 2); // Max depth 32 bits, condense after 2 occurrences

    // Read real non-repeating source code stream
    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Loaded " << raw_bits.size() << " raw bits from C++ source stream (" << file_content.size() << " bytes)\n\n";

    // --- STREAM PASS 1: Initial Discovery & Condensation ---
    std::cout << "--- [PASS 1: INITIAL DISCOVERY & ENTITY CONDENSATION] ---\n";
    
    size_t quarter_len = raw_bits.size() / 4;
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Condensed Entities | Max Level\n";
    std::cout << "---------|----------------|------------------|----------------------|--------------------|----------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t ent_cnt = 0;
        uint32_t max_lvl = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            ent_cnt = res.total_entities;
            max_lvl = res.max_level;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(18) << ent_cnt << " | "
                  << std::setw(8) << "Level " + std::to_string(max_lvl) << "\n";
    }

    // --- STREAM PASS 2: Lifelong Memory Mastery Pass ---
    std::cout << "\n--- [PASS 2: LIFELONG MEMORY MASTERY PASS] ---\n";
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Condensed Entities | Max Level\n";
    std::cout << "---------|----------------|------------------|----------------------|--------------------|----------\n";

    size_t total_p2_correct = 0;
    double total_p2_f = 0.0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t ent_cnt = 0;
        uint32_t max_lvl = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            ent_cnt = res.total_entities;
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
                  << std::setw(18) << ent_cnt << " | "
                  << std::setw(8) << "Level " + std::to_string(max_lvl) << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    double pass2_acc = (static_cast<double>(total_p2_correct) / (quarter_len * 4)) * 100.0;
    double pass2_avg_f = total_p2_f / (quarter_len * 4);

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-832 PURE AUTOPOIETIC SUBSTRATE FINAL SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Pass 2 Mastery Accuracy      : " << pass2_acc << " %\n";
    std::cout << "  - Pass 2 Free Energy (F)       : " << pass2_avg_f << " nats/bit\n";
    std::cout << "  - Total Execution Time         : " << total_sec << " seconds\n";
    std::cout << "  - Substrate Processing Speed   : " << ((quarter_len * 8) / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
