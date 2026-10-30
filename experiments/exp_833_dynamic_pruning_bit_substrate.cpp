// ============================================================================
// KARYON-CORE EXP-833: Bounded Autopoietic Bit Substrate with Dynamic F-Pruning
// KEP v30.0 Master | V = 2 Bits | Causal Free Energy Utility Pruning & Zero Bloat
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

struct DynamicBitEntity {
    uint32_t id;
    uint32_t level;                    // Entity hierarchy depth (0 = bit, 1 = pair, 2 = quad, ...)
    std::string bit_pattern;           // Structural binary pattern string
    uint64_t count_0 = 1;              // Transition counts to next bit 0
    uint64_t count_1 = 1;              // Transition counts to next bit 1
    uint64_t total_occurrences = 2;
    double utility_score = 1.0;        // Causal Free Energy reduction utility
    uint64_t last_used_step = 0;       // Step index of last match for decay/pruning

    DynamicBitEntity(uint32_t e_id, uint32_t lvl, const std::string& pattern, uint64_t step = 0)
        : id(e_id), level(lvl), bit_pattern(pattern), last_used_step(step) {}
};

class BoundedAutopoieticBitSubstrate {
private:
    std::vector<DynamicBitEntity> entities;
    std::unordered_map<std::string, uint32_t> pattern_to_entity;

    std::vector<uint8_t> bit_context;
    size_t max_context_depth;
    uint32_t condensation_threshold;
    size_t max_allowed_entities;        // Sovereign hard memory bound
    uint64_t global_step = 0;

public:
    BoundedAutopoieticBitSubstrate(size_t max_depth = 32, uint32_t threshold = 2, size_t max_entities = 50000)
        : max_context_depth(max_depth), condensation_threshold(threshold), max_allowed_entities(max_entities) {
        
        // Base Primitive Bit Entities (Level 0)
        entities.emplace_back(0, 0, "0", 0);
        entities.emplace_back(1, 0, "1", 0);
        
        pattern_to_entity["0"] = 0;
        pattern_to_entity["1"] = 1;
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        float prob_1;
        size_t total_entities;
        uint32_t max_level;
        size_t pruned_count;
    };

    void reset() {
        entities.clear();
        pattern_to_entity.clear();
        bit_context.clear();
        global_step = 0;

        entities.emplace_back(0, 0, "0", 0);
        entities.emplace_back(1, 0, "1", 0);
        pattern_to_entity["0"] = 0;
        pattern_to_entity["1"] = 1;
    }

    // STAGE 1: Predict next bit b_{t+1} using soft Bayesian weighting over active matching entities
    float predict_next_bit_prob() {
        if (bit_context.empty()) return 0.5f;

        double weighted_p1 = 0.0;
        double total_weight = 0.0;

        std::string current_str = "";
        size_t ctx_len = bit_context.size();

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = pattern_to_entity.find(current_str);
            if (it != pattern_to_entity.end()) {
                uint32_t ent_id = it->second;
                const auto& ent = entities[ent_id];

                // Laplace-smoothed probability estimate
                double p1 = static_cast<double>(ent.count_1) / (ent.total_occurrences + 2.0);
                
                // Weight combines level exponent + causal utility score
                double weight = std::pow(2.2, static_cast<double>(ent.level)) * std::max(0.1, ent.utility_score);
                weighted_p1 += p1 * weight;
                total_weight += weight;
            }
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

    // STAGE 2: Observe actual bit b_{t+1}, update utility, condense new entities, & prune stale/low-utility nodes
    StepResult update_with_actual(float prob_1, uint8_t actual_bit) {
        global_step++;
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float free_energy = -std::log(p_target); // Variational Free Energy

        bool is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        std::string current_str = "";
        size_t ctx_len = bit_context.size();
        uint32_t max_level = 0;

        for (size_t i = 0; i < ctx_len && i < max_context_depth; ++i) {
            uint8_t past_bit = bit_context[ctx_len - 1 - i];
            current_str.insert(current_str.begin(), past_bit ? '1' : '0');

            auto it = pattern_to_entity.find(current_str);
            if (it != pattern_to_entity.end()) {
                uint32_t ent_id = it->second;
                auto& ent = entities[ent_id];

                ent.total_occurrences++;
                if (actual_bit == 0) ent.count_0++;
                else ent.count_1++;

                ent.last_used_step = global_step;

                // Update Causal Utility Score:
                // If prediction was accurate, boost entity utility; if wrong, penalize it.
                if (is_correct) {
                    ent.utility_score += 0.05 * (1.0 - free_energy);
                } else {
                    ent.utility_score -= 0.02 * free_energy;
                }
                ent.utility_score = std::clamp(ent.utility_score, 0.01, 100.0);

                if (ent.level > max_level) max_level = ent.level;

                // Spontaneous Condensation of deeper bit-entity
                if (ent.total_occurrences >= condensation_threshold) {
                    std::string new_str = current_str + (actual_bit ? "1" : "0");
                    if (pattern_to_entity.find(new_str) == pattern_to_entity.end()) {
                        uint32_t new_id = static_cast<uint32_t>(entities.size());
                        uint32_t new_lvl = ent.level + 1;

                        entities.emplace_back(new_id, new_lvl, new_str, global_step);
                        pattern_to_entity[new_str] = new_id;
                    }
                }
            }
        }

        // AUTOPOIETIC PRUNING ENGINE:
        // When entities budget exceeds max_allowed_entities, perform causal pruning of lowest utility & oldest entities
        size_t pruned_count = 0;
        if (entities.size() > max_allowed_entities) {
            pruned_count = prune_low_utility_entities();
        }

        // Append actual_bit to context window
        bit_context.push_back(actual_bit);
        if (bit_context.size() > max_context_depth * 2) {
            bit_context.erase(bit_context.begin(), bit_context.begin() + max_context_depth);
        }

        return {free_energy, is_correct, prob_1, entities.size(), max_level, pruned_count};
    }

private:
    size_t prune_low_utility_entities() {
        // Collect candidate entities for pruning (must NOT prune Level 0 base bit entities 0 and 1)
        struct PruneCandidate {
            uint32_t id;
            double score;
        };

        std::vector<PruneCandidate> candidates;
        candidates.reserve(entities.size());

        for (size_t i = 2; i < entities.size(); ++i) {
            uint64_t age = global_step - entities[i].last_used_step;
            double recency_penalty = std::exp(-static_cast<double>(age) / 50000.0);
            double composite_score = entities[i].utility_score * recency_penalty;

            candidates.push_back({static_cast<uint32_t>(i), composite_score});
        }

        // Sort candidates ascending by composite score (lowest utility first)
        std::sort(candidates.begin(), candidates.end(), [](const PruneCandidate& a, const PruneCandidate& b) {
            return a.score < b.score;
        });

        // Prune the lowest 20% candidates
        size_t target_to_prune = candidates.size() / 5;
        if (target_to_prune == 0) return 0;

        std::unordered_map<uint32_t, bool> to_remove;
        for (size_t i = 0; i < target_to_prune; ++i) {
            to_remove[candidates[i].id] = true;
        }

        // Rebuild entities vector and map
        std::vector<DynamicBitEntity> new_entities;
        new_entities.reserve(entities.size() - target_to_prune);

        // Preserve base primitives
        new_entities.push_back(entities[0]);
        new_entities.push_back(entities[1]);

        pattern_to_entity.clear();
        pattern_to_entity["0"] = 0;
        pattern_to_entity["1"] = 1;

        for (size_t i = 2; i < entities.size(); ++i) {
            if (to_remove.find(static_cast<uint32_t>(i)) == to_remove.end()) {
                uint32_t new_id = static_cast<uint32_t>(new_entities.size());
                new_entities.push_back(entities[i]);
                new_entities.back().id = new_id;
                pattern_to_entity[new_entities.back().bit_pattern] = new_id;
            }
        }

        size_t actual_pruned = entities.size() - new_entities.size();
        entities = std::move(new_entities);
        return actual_pruned;
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-833: BOUNDED AUTOPOIETIC SUBSTRATE (F-PRUNING)\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Dynamic Causal Utility & Zero Bloat\n";
    std::cout << "=================================================================\n\n";

    // Substrate bounded to max 50,000 entities to guarantee constant O(1) memory footprint
    BoundedAutopoieticBitSubstrate substrate(32, 2, 50000);

    // Read real non-repeating source code stream
    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Loaded " << raw_bits.size() << " raw bits from C++ source stream (" << file_content.size() << " bytes)\n";
    std::cout << "[+] Sovereign Memory Bound Enforced: Max 50,000 Active Condensed Entities\n\n";

    // --- PASS 1: INITIAL DISCOVERY & CAUSAL PRUNING ---
    std::cout << "--- [PASS 1: INITIAL DISCOVERY & CAUSAL UTILITY PRUNING] ---\n";
    
    size_t quarter_len = raw_bits.size() / 4;
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Active Entities | Total Pruned\n";
    std::cout << "---------|----------------|------------------|----------------------|-----------------|-------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t cumulative_pruned = 0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t ent_cnt = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            ent_cnt = res.total_entities;
            cumulative_pruned += res.pruned_count;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(15) << ent_cnt << " | "
                  << std::setw(11) << cumulative_pruned << "\n";
    }

    // --- PASS 2: LIFELONG MEMORY MASTERY UNDER BOUNDED CAPACITY ---
    std::cout << "\n--- [PASS 2: BOUNDED LIFELONG MEMORY MASTERY PASS] ---\n";
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Active Entities | Total Pruned\n";
    std::cout << "---------|----------------|------------------|----------------------|-----------------|-------------\n";

    size_t total_p2_correct = 0;
    double total_p2_f = 0.0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t ent_cnt = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            ent_cnt = res.total_entities;
            cumulative_pruned += res.pruned_count;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        total_p2_correct += correct;
        total_p2_f += f_sum;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(15) << ent_cnt << " | "
                  << std::setw(11) << cumulative_pruned << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    double pass2_acc = (static_cast<double>(total_p2_correct) / (quarter_len * 4)) * 100.0;
    double pass2_avg_f = total_p2_f / (quarter_len * 4);

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-833 BOUNDED AUTOPOIETIC SUBSTRATE FINAL SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Pass 2 Mastery Accuracy      : " << pass2_acc << " %\n";
    std::cout << "  - Pass 2 Free Energy (F)       : " << pass2_avg_f << " nats/bit\n";
    std::cout << "  - Final Active Entities Count  : " << substrate.predict_next_bit_prob() << " (Bounded <= 50,000)\n";
    std::cout << "  - Total Entities Pruned        : " << cumulative_pruned << "\n";
    std::cout << "  - Total Execution Time         : " << total_sec << " seconds\n";
    std::cout << "  - Substrate Processing Speed   : " << ((quarter_len * 8) / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
