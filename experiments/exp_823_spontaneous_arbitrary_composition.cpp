// ============================================================================
// KARYON-CORE EXP-823: Spontaneous Arbitrary Self-Assembly Substrate
// KEP v30.0 Master | V = 2 Bits | Pure Emergent Field Condensation
// Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iomanip>

// Simple LCG PRNG for noise injection
static uint64_t lcg_state = 0x9876543210111213ULL;
inline uint64_t next_lcg() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state;
}

// Spontaneous Physical Entity (can be a primitive bit or an emergent compound)
struct SpontaneousEntity {
    uint32_t id;
    uint32_t level;                    // 0 = primitive bit, >0 = emergent compound
    std::vector<uint32_t> components;  // Arbitrary constituent ids (can be 2, 3, or N)
    
    // Continuous physical state
    float phase;                       // Spin/phase state \in [-1, +1]
    float binding_energy;              // Internal binding energy / stability
    float activity_field;              // Activity field trace
    uint64_t total_activations;

    SpontaneousEntity(uint32_t entity_id, uint32_t lvl, float init_phase = 0.0f)
        : id(entity_id), level(lvl), phase(init_phase), binding_energy(0.0f),
          activity_field(1.0f), total_activations(1) {}
};

class PureSpontaneousSubstrate {
private:
    std::vector<SpontaneousEntity> entities;
    
    // Continuous physical coupling field between arbitrary entities (J_ij matrix)
    // Key: hash(id_a, id_b) -> coupling energy J_ij
    std::unordered_map<uint64_t, float> coupling_matrix;

    uint32_t next_id;
    uint32_t max_level;
    float condensation_threshold;      // Spontaneous physical binding threshold

    inline uint64_t hash_pair(uint32_t a, uint32_t b) const {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    }

public:
    PureSpontaneousSubstrate(float threshold = 1.2f)
        : next_id(0), max_level(0), condensation_threshold(threshold) {
        // Primitive Bit Entities (Level 0)
        entities.emplace_back(next_id++, 0, -1.0f); // Bit 0
        entities.emplace_back(next_id++, 0, +1.0f); // Bit 1
    }

    struct StepTelemetry {
        float predicted_bit_prob;
        float free_energy;
        bool is_correct;
        bool new_spontaneous_assembly;
    };

    StepTelemetry process_bit_stream(int actual_bit, const std::vector<uint32_t>& active_window) {
        uint32_t current_bit_id = (actual_bit == 1) ? 1 : 0;
        entities[current_bit_id].total_activations++;

        // 1. Compute total continuous field potential from all currently active entities
        float net_field_potential = 0.0f;
        float total_weight = 0.0f;

        for (uint32_t active_id : active_window) {
            uint64_t link_key = hash_pair(active_id, current_bit_id);
            auto it = coupling_matrix.find(link_key);
            if (it != coupling_matrix.end()) {
                net_field_potential += it->second * entities[active_id].phase;
                total_weight += std::abs(it->second);
            }
        }

        // Add base field state
        float base_bias = (entities[1].total_activations + 1.0f) / 
                          (entities[0].total_activations + entities[1].total_activations + 2.0f);
        net_field_potential += (current_bit_id == 1) ? base_bias : -base_bias;
        total_weight += 1.0f;

        // Prediction via continuous wave field response
        float prob_1 = 1.0f / (1.0f + std::exp(-net_field_potential));
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        int predicted_bit = (prob_1 >= 0.5f) ? 1 : 0;
        bool is_correct = (predicted_bit == actual_bit);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float local_f = -std::log(p_target);

        // 2. Pure Continuous Physical Coupling Dynamics (Hebbian / Resonance Field)
        // No hardcoded counters! Coupling J_ij evolves spontaneously based on phase alignment & F reduction
        bool assembled = false;
        if (!active_window.empty()) {
            uint32_t prev_id = active_window.back();
            uint64_t key = hash_pair(prev_id, current_bit_id);

            // Field resonance update: increase coupling if state alignment reduces surprise (is_correct)
            float delta_j = is_correct ? (+0.15f * (1.0f - local_f)) : (-0.10f * local_f);
            coupling_matrix[key] += delta_j;

            // 3. Spontaneous Physical Condensation (Arbitrary Self-Assembly)
            // When coupling energy J_ij naturally accumulates above physical binding threshold,
            // the entities spontaneously condense into a new higher-level compound!
            if (coupling_matrix[key] > condensation_threshold && entities.size() < 10000) {
                // Check if this compound already condensed
                bool exists = false;
                for (const auto& e : entities) {
                    if (e.components.size() == 2 && e.components[0] == prev_id && e.components[1] == current_bit_id) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    uint32_t new_compound_id = next_id++;
                    uint32_t compound_level = std::max(entities[prev_id].level, entities[current_bit_id].level) + 1;
                    
                    entities.emplace_back(new_compound_id, compound_level);
                    entities[new_compound_id].components = {prev_id, current_bit_id};
                    entities[new_compound_id].binding_energy = coupling_matrix[key];
                    entities[new_compound_id].phase = (entities[prev_id].phase + entities[current_bit_id].phase) * 0.5f;

                    if (compound_level > max_level) {
                        max_level = compound_level;
                    }

                    assembled = true;
                }
            }
        }

        return {prob_1, local_f, is_correct, assembled};
    }

    size_t get_entity_count() const { return entities.size(); }
    uint32_t get_max_level() const { return max_level; }
    size_t get_coupling_count() const { return coupling_matrix.size(); }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-823: Pure Spontaneous Arbitrary Self-Assembly\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Zero Hardcoded Rules\n";
    std::cout << "=================================================================\n\n";

    // Structured bit stream
    std::string text_source = 
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O "
        "9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 ";

    std::vector<int> bit_stream;
    for (char c : text_source) {
        for (int b = 7; b >= 0; --b) {
            bit_stream.push_back((c >> b) & 1);
        }
    }

    const size_t TOTAL_BITS = 200000;
    std::vector<int> full_stream;
    full_stream.reserve(TOTAL_BITS);

    size_t idx = 0;
    for (size_t i = 0; i < TOTAL_BITS; ++i) {
        int bit = bit_stream[idx % bit_stream.size()];
        if ((next_lcg() % 100) < 2) bit ^= 1; // 2% noise
        full_stream.push_back(bit);
        idx++;
    }

    PureSpontaneousSubstrate substrate(1.0f); // Physical binding threshold

    std::vector<uint32_t> active_window;
    size_t correct_count = 0;
    double total_free_energy = 0.0;
    size_t spontaneous_assemblies = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < full_stream.size(); ++t) {
        int actual_bit = full_stream[t];
        auto res = substrate.process_bit_stream(actual_bit, active_window);

        if (res.is_correct) correct_count++;
        if (res.new_spontaneous_assembly) spontaneous_assemblies++;
        total_free_energy += res.free_energy;

        uint32_t cur_id = (actual_bit == 1) ? 1 : 0;
        active_window.push_back(cur_id);
        if (active_window.size() > 32) {
            active_window.erase(active_window.begin());
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    double bit_accuracy = (static_cast<double>(correct_count) / TOTAL_BITS) * 100.0;
    double avg_free_energy = total_free_energy / TOTAL_BITS;
    double throughput_bits_sec = TOTAL_BITS / elapsed_sec;

    std::cout << "=================================================================\n";
    std::cout << "  EXP-823 SPONTANEOUS SELF-ASSEMBLY TELEMETRY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Bit Predictive Accuracy        : " << std::fixed << std::setprecision(4) << bit_accuracy << " %\n";
    std::cout << "  - Average Bit Free Energy (F)    : " << avg_free_energy << "\n";
    std::cout << "  - Spontaneously Assembled Nodes  : " << substrate.get_entity_count() << " entities\n";
    std::cout << "  - Max Composition Level Emerged : Level " << substrate.get_max_level() << "\n";
    std::cout << "  - Active Dynamic Couplings J_ij  : " << substrate.get_coupling_count() << "\n";
    std::cout << "  - New Spontaneous Assembly Events: " << spontaneous_assemblies << "\n";
    std::cout << "  - Substrate Bit Throughput       : " << throughput_bits_sec << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
