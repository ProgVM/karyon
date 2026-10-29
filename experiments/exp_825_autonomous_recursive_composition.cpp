// ============================================================================
// KARYON-CORE EXP-825: Autonomous Recursive Entity Composition Substrate
// KEP v30.0 Master | V = 2 Bits | Pure Autonomous Substrate Self-Interaction
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

static uint64_t lcg_state = 0xFEDCBA9876543210ULL;
inline uint64_t next_lcg() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state;
}

struct AutonomousEntity {
    uint32_t id;
    uint32_t level;                    // 0 = primitive bit, 1 = compound, 2 = meta-compound, ...
    std::vector<uint32_t> components;  // Arbitrary constituent entity IDs
    
    float phase;                       // Spin/phase state \in [-1, +1]
    float binding_energy;              // Internal physical stability
    uint64_t activations;

    AutonomousEntity(uint32_t e_id, uint32_t lvl, float init_phase = 0.0f)
        : id(e_id), level(lvl), phase(init_phase), binding_energy(0.0f), activations(1) {}
};

class SovereignRecursiveSubstrate {
private:
    std::vector<AutonomousEntity> entities;
    
    // Physical coupling matrix J_ij between ANY entities (base bits OR higher-level compounds)
    std::unordered_map<uint64_t, float> coupling_matrix;
    
    // Compound registry to prevent duplicate condensation: hash(entity_a, entity_b) -> entity_id
    std::unordered_map<uint64_t, uint32_t> compound_registry;

    uint32_t next_id;
    uint32_t max_level_reached;
    float condensation_threshold;

    inline uint64_t hash_pair(uint32_t a, uint32_t b) const {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    }

public:
    SovereignRecursiveSubstrate(float threshold = 0.5f)
        : next_id(0), max_level_reached(0), condensation_threshold(threshold) {
        // Base Primitive Bit Entities (Level 0)
        entities.emplace_back(next_id++, 0, -1.0f); // Bit 0
        entities.emplace_back(next_id++, 0, +1.0f); // Bit 1
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        uint32_t new_entities_condensed;
    };

    // Process a bit autonomously: entities react, couple, and recursively self-assemble
    StepResult process_bit(int actual_bit, std::vector<uint32_t>& active_entity_window) {
        uint32_t cur_bit_id = (actual_bit == 1) ? 1 : 0;
        entities[cur_bit_id].activations++;

        // 1. Calculate continuous wave potential J_ij across ALL active entities in window
        float net_potential = 0.0f;

        for (uint32_t active_id : active_entity_window) {
            uint64_t key = hash_pair(active_id, cur_bit_id);
            auto it = coupling_matrix.find(key);
            if (it != coupling_matrix.end()) {
                // Wave interference: J_ij * phase_i
                net_potential += it->second * entities[active_id].phase;
            }
        }

        // Base frequency bias
        float p_base = (entities[1].activations + 1.0f) / 
                       (entities[0].activations + entities[1].activations + 2.0f);
        net_potential += (cur_bit_id == 1) ? p_base : -p_base;

        float prob_1 = 1.0f / (1.0f + std::exp(-net_potential));
        prob_1 = std::clamp(prob_1, 1e-7f, 1.0f - 1e-7f);

        int pred_bit = (prob_1 >= 0.5f) ? 1 : 0;
        bool is_correct = (pred_bit == actual_bit);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float local_f = -std::log(p_target);

        // 2. Autonomous Hebbian Coupling & Recursive Self-Assembly
        // NO external intervention! ANY two entities active in context can couple and condense.
        uint32_t new_condensations = 0;

        if (!active_entity_window.empty()) {
            // Pairwise physical coupling between all recent active entities
            for (size_t i = 0; i < active_entity_window.size(); ++i) {
                uint32_t ent_a = active_entity_window[i];
                uint32_t ent_b = cur_bit_id;

                if (ent_a == ent_b) continue;

                uint64_t key = hash_pair(ent_a, ent_b);

                // Physical resonance: update coupling strength J_ab
                float delta_j = is_correct ? (+0.20f * (1.0f - local_f)) : (-0.15f * local_f);
                coupling_matrix[key] += delta_j;

                // Spontaneous Physical Condensation:
                // When J_ab exceeds binding energy threshold, ent_a and ent_b bind into a NEW recursive entity!
                if (coupling_matrix[key] > condensation_threshold) {
                    if (compound_registry.find(key) == compound_registry.end() && entities.size() < 50000) {
                        uint32_t new_id = next_id++;
                        uint32_t new_level = std::max(entities[ent_a].level, entities[ent_b].level) + 1;

                        entities.emplace_back(new_id, new_level);
                        entities[new_id].components = {ent_a, ent_b};
                        entities[new_id].phase = (entities[ent_a].phase + entities[ent_b].phase) * 0.5f;
                        entities[new_id].binding_energy = coupling_matrix[key];

                        compound_registry[key] = new_id;
                        new_condensations++;

                        if (new_level > max_level_reached) {
                            max_level_reached = new_level;
                        }

                        // The newly formed higher-level entity immediately enters active context!
                        active_entity_window.push_back(new_id);
                    }
                }
            }
        }

        // Add current bit entity to active window
        active_entity_window.push_back(cur_bit_id);

        // Maintain physical context window size
        if (active_entity_window.size() > 48) {
            active_entity_window.erase(active_entity_window.begin());
        }

        return {local_f, is_correct, new_condensations};
    }

    size_t get_node_count() const { return entities.size(); }
    uint32_t get_max_level() const { return max_level_reached; }
    size_t get_coupling_count() const { return coupling_matrix.size(); }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-825: Autonomous Recursive Entity Composition\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Pure Autonomous Recursive Field\n";
    std::cout << "=================================================================\n\n";

    // Structured bit stream
    std::string text_source = 
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O "
        "9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 ";

    std::vector<int> bit_pattern;
    for (char c : text_source) {
        for (int b = 7; b >= 0; --b) {
            bit_pattern.push_back((c >> b) & 1);
        }
    }

    SovereignRecursiveSubstrate substrate(0.45f); // Autonomous physical condensation threshold

    std::vector<uint32_t> active_entity_window;

    const size_t CHUNK_SIZE = 100000;
    const size_t MAX_CHUNKS = 10; // 1,000,000 bits

    auto global_start = std::chrono::high_resolution_clock::now();

    size_t global_bit_count = 0;
    size_t pattern_idx = 0;

    std::cout << " Chunk | Processed Bits | Chunk Acc (%) | Avg Free Energy (F) | Entities | Max Hierarchy Level | Couplings J_ij\n";
    std::cout << "-------|----------------|---------------|---------------------|----------|---------------------|---------------\n";

    for (size_t chunk = 1; chunk <= MAX_CHUNKS; ++chunk) {
        size_t chunk_correct = 0;
        double chunk_f_sum = 0.0;

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            int actual_bit = bit_pattern[pattern_idx % bit_pattern.size()];
            pattern_idx++;
            global_bit_count++;

            auto res = substrate.process_bit(actual_bit, active_entity_window);

            if (res.is_correct) chunk_correct++;
            chunk_f_sum += res.free_energy;
        }

        double chunk_acc = (static_cast<double>(chunk_correct) / CHUNK_SIZE) * 100.0;
        double chunk_f_avg = chunk_f_sum / CHUNK_SIZE;

        std::cout << std::setw(6) << chunk << " | "
                  << std::setw(14) << global_bit_count << " | "
                  << std::setw(13) << std::fixed << std::setprecision(4) << chunk_acc << " | "
                  << std::setw(19) << std::setprecision(6) << chunk_f_avg << " | "
                  << std::setw(8) << substrate.get_node_count() << " | "
                  << std::setw(14) << "Level " + std::to_string(substrate.get_max_level()) << " | "
                  << std::setw(14) << substrate.get_coupling_count() << "\n";

        if (chunk_acc >= 99.99 && chunk_f_avg < 0.001) {
            std::cout << "\n[!] ZERO FREE ENERGY / PERFECT ACCURACY REACHED AT CHUNK " << chunk << "!\n";
            break;
        }
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(global_end - global_start).count();

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-825 AUTONOMOUS RECURSIVE SUBSTRATE FINAL SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Total Execution Time         : " << total_sec << " seconds\n";
    std::cout << "  - Total Stream Length Processed: " << global_bit_count << " bits\n";
    std::cout << "  - Final Emergent Hierarchy Depth: Level " << substrate.get_max_level() << "\n";
    std::cout << "  - Total Autonomous Entities    : " << substrate.get_node_count() << "\n";
    std::cout << "  - Active Dynamic Field Links   : " << substrate.get_coupling_count() << "\n";
    std::cout << "  - Processing Speed             : " << (global_bit_count / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
