// ============================================================================
// KARYON-CORE EXP-834: Epigenetically Shielded Bounded Autopoietic Substrate
// KEP v30.0 Master | V = 2 Bits | Epigenetic Protection + Adaptive Sprouting
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

struct ShieldedBitEntity {
    uint32_t id;
    uint32_t level;                    // Sequence length
    std::string bit_pattern;           // Binary sequence
    uint64_t count_0 = 1;
    uint64_t count_1 = 1;
    uint64_t total_occurrences = 2;
    double free_energy_sum = 0.0;      // Accumulated Free Energy when predicting from this entity
    bool epigenetically_locked = false; // Protected from pruning if high accuracy
    uint64_t last_used_step = 0;

    ShieldedBitEntity(uint32_t e_id, uint32_t lvl, const std::string& pattern, uint64_t step = 0)
        : id(e_id), level(lvl), bit_pattern(pattern), last_used_step(step) {}
};

class ShieldedAutopoieticBitSubstrate {
private:
    std::vector<ShieldedBitEntity> entities;
    std::unordered_map<std::string, uint32_t> pattern_to_entity;

    std::vector<uint8_t> bit_context;
    size_t max_context_depth;
    size_t max_allowed_entities;
    uint64_t global_step = 0;

public:
    ShieldedAutopoieticBitSubstrate(size_t max_depth = 32, size_t max_entities = 100000)
        : max_context_depth(max_depth), max_allowed_entities(max_entities) {
        
        // Base Primitive Bit Entities (Level 0)
        entities.emplace_back(0, 0, "0", 0);
        entities.emplace_back(1, 0, "1", 0);
        entities[0].epigenetically_locked = true;
        entities[1].epigenetically_locked = true;
        
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
        size_t locked_count;
    };

    void reset() {
        entities.clear();
        pattern_to_entity.clear();
        bit_context.clear();
        global_step = 0;

        entities.emplace_back(0, 0, "0", 0);
        entities.emplace_back(1, 0, "1", 0);
        entities[0].epigenetically_locked = true;
        entities[1].epigenetically_locked = true;

        pattern_to_entity["0"] = 0;
        pattern_to_entity["1"] = 1;
    }

    // STAGE 1: Predict next bit b_{t+1} using weighted combination of matching active entities
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

                double p1 = static_cast<double>(ent.count_1) / ent.total_occurrences;
                
                // Higher level entities carry higher exponential weight
                double weight = std::pow(2.0, static_cast<double>(ent.level + 1));
                if (ent.epigenetically_locked) weight *= 2.0; // Boost epigenetically protected entities

                weighted_p1 += p1 * weight;
                total_weight += weight;
            }
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

    // STAGE 2: Observe actual bit b_{t+1}, update statistics, apply Epigenetic Shielding, & prune unshielded noise
    StepResult update_with_actual(float prob_1, uint8_t actual_bit) {
        global_step++;
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float free_energy = -std::log(p_target); // Variational Free Energy

        bool is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        std::string current_str = "";
        size_t ctx_len = bit_context.size();
        uint32_t max_level = 0;
        size_t locked_count = 0;

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

                ent.free_energy_sum += free_energy;
                ent.last_used_step = global_step;

                // EPIGENETIC SHIELDING (KEP Principle 6):
                // If an entity has been used >= 4 times AND its average Free Energy is low (< 0.25 nats -> high accuracy > 90%),
                // it is permanently locked against pruning!
                double avg_f = ent.free_energy_sum / ent.total_occurrences;
                if (!ent.epigenetically_locked && ent.total_occurrences >= 4 && avg_f < 0.25) {
                    ent.epigenetically_locked = true;
                }

                if (ent.epigenetically_locked) locked_count++;
                if (ent.level > max_level) max_level = ent.level;

                // Adaptive Spontaneous Condensation:
                // Sprout deeper node if current node is frequently used
                uint32_t req_occurrences = ent.epigenetically_locked ? 2 : 3;
                if (ent.total_occurrences >= req_occurrences) {
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

        // AUTOPOIETIC SMART PRUNING:
        // Only prune UNLOCKED entities when budget is exceeded
        size_t pruned_count = 0;
        if (entities.size() > max_allowed_entities) {
            pruned_count = prune_unlocked_noise();
        }

        // Append actual_bit to context window
        bit_context.push_back(actual_bit);
        if (bit_context.size() > max_context_depth * 2) {
            bit_context.erase(bit_context.begin(), bit_context.begin() + max_context_depth);
        }

        return {free_energy, is_correct, prob_1, entities.size(), max_level, pruned_count, locked_count};
    }

private:
    size_t prune_unlocked_noise() {
        // Collect candidate entities for pruning (ONLY UNLOCKED ENTITIES)
        struct PruneCandidate {
            uint32_t id;
            double score;
        };

        std::vector<PruneCandidate> candidates;
        candidates.reserve(entities.size());

        for (size_t i = 2; i < entities.size(); ++i) {
            if (!entities[i].epigenetically_locked) {
                uint64_t age = global_step - entities[i].last_used_step;
                double avg_f = entities[i].free_energy_sum / std::max((uint64_t)1, entities[i].total_occurrences);
                
                // Higher score = more valuable. Score = (Occurrences / Avg_F) * Recency
                double recency = std::exp(-static_cast<double>(age) / 20000.0);
                double score = (static_cast<double>(entities[i].total_occurrences) / (avg_f + 0.01)) * recency;

                candidates.push_back({static_cast<uint32_t>(i), score});
            }
        }

        if (candidates.empty()) return 0; // All active entities are epigenetically protected!

        // Sort candidates ascending by value score
        std::sort(candidates.begin(), candidates.end(), [](const PruneCandidate& a, const PruneCandidate& b) {
            return a.score < b.score;
        });

        // Prune up to 10% of unlocked noise nodes
        size_t target_to_prune = std::min(candidates.size(), max_allowed_entities / 10);
        if (target_to_prune == 0) return 0;

        std::unordered_map<uint32_t, bool> to_remove;
        for (size_t i = 0; i < target_to_prune; ++i) {
            to_remove[candidates[i].id] = true;
        }

        // Rebuild entities vector
        std::vector<ShieldedBitEntity> new_entities;
        new_entities.reserve(entities.size() - target_to_prune);

        pattern_to_entity.clear();

        for (size_t i = 0; i < entities.size(); ++i) {
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
    std::cout << "  KARYON-CORE EXP-834: EPIGENETICALLY SHIELDED AUTOPOIETIC SUBSTRATE\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Epigenetic Lock + Smart Noise Pruning\n";
    std::cout << "=================================================================\n\n";

    // Enforce memory ceiling of 100,000 entities with Epigenetic Shielding
    ShieldedAutopoieticBitSubstrate substrate(32, 100000);

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
    std::cout << "[+] Sovereign Memory Limit: 100,000 Entities | Epigenetic Shield Active\n\n";

    // --- PASS 1: INITIAL DISCOVERY & EPIGENETIC SHIELDING ---
    std::cout << "--- [PASS 1: INITIAL DISCOVERY & EPIGENETIC SHIELDING] ---\n";
    
    size_t quarter_len = raw_bits.size() / 4;
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Total Entities | Locked Invariants | Total Pruned\n";
    std::cout << "---------|----------------|------------------|----------------------|----------------|-------------------|-------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t cumulative_pruned = 0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t ent_cnt = 0;
        size_t locked_cnt = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            ent_cnt = res.total_entities;
            locked_cnt = res.locked_count;
            cumulative_pruned += res.pruned_count;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(14) << ent_cnt << " | "
                  << std::setw(17) << locked_cnt << " | "
                  << std::setw(11) << cumulative_pruned << "\n";
    }

    // --- PASS 2: LIFELONG MEMORY MASTERY PASS ---
    std::cout << "\n--- [PASS 2: EPIGENETICALLY SHIELDED LIFELONG MASTERY PASS] ---\n";
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | Total Entities | Locked Invariants | Total Pruned\n";
    std::cout << "---------|----------------|------------------|----------------------|----------------|-------------------|-------------\n";

    size_t total_p2_correct = 0;
    double total_p2_f = 0.0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t ent_cnt = 0;
        size_t locked_cnt = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            ent_cnt = res.total_entities;
            locked_cnt = res.locked_count;
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
                  << std::setw(14) << ent_cnt << " | "
                  << std::setw(17) << locked_cnt << " | "
                  << std::setw(11) << cumulative_pruned << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    double pass2_acc = (static_cast<double>(total_p2_correct) / (quarter_len * 4)) * 100.0;
    double pass2_avg_f = total_p2_f / (quarter_len * 4);

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-834 SHIELDED AUTOPOIETIC SUBSTRATE FINAL SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Pass 2 Mastery Accuracy      : " << pass2_acc << " %\n";
    std::cout << "  - Pass 2 Free Energy (F)       : " << pass2_avg_f << " nats/bit\n";
    std::cout << "  - Total Entities Pruned        : " << cumulative_pruned << "\n";
    std::cout << "  - Total Execution Time         : " << total_sec << " seconds\n";
    std::cout << "  - Substrate Processing Speed   : " << ((quarter_len * 8) / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
