// ============================================================================
// KARYON-CORE EXP-854: Endoscopic Audit of 0.52% Error Leakage in Deterministic Code
//
// Question: In a 100% deterministic file (C++ source file), where do the 431 errors
// (0.52% out of 82,264 bits) leak from in Pass 2?
//
// Hypotheses:
//  H1: Finite Horizon Limit (horizon = 128 bits = 16 bytes). Two distant locations
//      in the C++ file share identical 16-byte history, but differ at byte 17.
//  H2: Novel Contexts (New N-grams introduced late in the file that were seen 0 times prior).
//  H3: Equivalence Merge Collisions (Bisimulation merged non-identical histories).
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <fstream>
#include <iomanip>

struct AuditNode {
    uint32_t id;
    uint32_t depth;
    std::string pattern;

    uint64_t count_0 = 0;
    uint64_t count_1 = 0;
    uint64_t total = 0;

    int canonical_id = -1;

    AuditNode(uint32_t n_id, uint32_t d, const std::string& pat)
        : id(n_id), depth(d), pattern(pat) {}

    uint32_t get_canonical() const {
        return (canonical_id != -1) ? canonical_id : id;
    }

    double get_p1() const {
        if (total == 0) return 0.5;
        return static_cast<double>(count_1) / static_cast<double>(total);
    }

    double get_free_energy() const {
        if (total == 0) return 1.0;
        double p1 = get_p1();
        if (p1 <= 0.0 || p1 >= 1.0) return 0.0;
        return -(p1 * std::log2(p1) + (1.0 - p1) * std::log2(1.0 - p1));
    }
};

class AuditSubstrate {
private:
    std::vector<AuditNode> nodes;
    std::unordered_map<std::string, uint32_t> pattern_map;
    std::vector<uint8_t> bit_window;
    size_t max_horizon;

public:
    AuditSubstrate(size_t horizon = 128) : max_horizon(horizon) {
        nodes.emplace_back(0, 0, "0");
        nodes.emplace_back(1, 0, "1");
        pattern_map["0"] = 0;
        pattern_map["1"] = 1;
    }

    float predict_next_bit(int& out_max_depth, bool& out_is_ambiguous) {
        out_max_depth = -1;
        out_is_ambiguous = false;
        if (bit_window.empty()) return 0.5f;

        std::string pattern = "";
        size_t win_len = bit_window.size();

        int max_deterministic_depth = -1;
        double deterministic_p1 = -1.0;

        for (size_t i = 0; i < win_len && i < max_horizon; ++i) {
            uint8_t past_bit = bit_window[win_len - 1 - i];
            pattern.insert(pattern.begin(), past_bit ? '1' : '0');

            auto it = pattern_map.find(pattern);
            if (it != pattern_map.end()) {
                uint32_t canon_id = nodes[it->second].get_canonical();
                const auto& node = nodes[canon_id];

                double p1 = node.get_p1();
                double fe = node.get_free_energy();

                if (fe == 0.0 && node.total >= 2) {
                    if (static_cast<int>(node.depth) > max_deterministic_depth) {
                        max_deterministic_depth = node.depth;
                        deterministic_p1 = p1;
                    }
                } else if (fe > 0.0) {
                    // This context is inherently ambiguous in the dataset given the current depth horizon!
                    if (static_cast<int>(node.depth) > max_deterministic_depth) {
                        out_is_ambiguous = true;
                    }
                }
            }
        }

        out_max_depth = max_deterministic_depth;

        if (max_deterministic_depth >= 1 && deterministic_p1 >= 0.0) {
            return (deterministic_p1 == 1.0) ? 0.999999f : 0.000001f;
        }

        return 0.5f;
    }

    void update(uint8_t actual_bit) {
        std::string pattern = "";
        size_t win_len = bit_window.size();

        for (size_t i = 0; i < win_len && i < max_horizon; ++i) {
            uint8_t past_bit = bit_window[win_len - 1 - i];
            pattern.insert(pattern.begin(), past_bit ? '1' : '0');

            auto it = pattern_map.find(pattern);
            if (it != pattern_map.end()) {
                uint32_t node_id = nodes[it->second].get_canonical();
                auto& node = nodes[node_id];

                node.total++;
                if (actual_bit == 0) node.count_0++;
                else node.count_1++;

                std::string child_pattern = pattern + (actual_bit ? "1" : "0");
                if (pattern_map.find(child_pattern) == pattern_map.end()) {
                    uint32_t new_id = static_cast<uint32_t>(nodes.size());
                    nodes.emplace_back(new_id, node.depth + 1, child_pattern);
                    pattern_map[child_pattern] = new_id;
                }
            }
        }

        bit_window.push_back(actual_bit);
        if (bit_window.size() > max_horizon) {
            bit_window.erase(bit_window.begin(), bit_window.end() - max_horizon);
        }
    }
};

int main() {
    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string code_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : code_data) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "=== ENDOSCOPIC AUDIT OF 0.52% ERROR LEAKAGE ===\n";
    std::cout << "Total file size: " << raw_bits.size() << " bits (" << code_data.size() << " bytes)\n\n";

    // Test across different Horizon Depths (128 bits, 256 bits, 512 bits, 1024 bits)
    std::vector<size_t> horizons = {64, 128, 256, 512, 1024};

    for (size_t h : horizons) {
        AuditSubstrate substrate(h);

        // Pass 1: Build Substrate
        for (size_t i = 0; i < raw_bits.size(); ++i) {
            int depth;
            bool amb;
            substrate.predict_next_bit(depth, amb);
            substrate.update(raw_bits[i]);
        }

        // Pass 2: Audit Exact Errors
        size_t errors = 0;
        size_t horizon_collision_errors = 0;
        size_t unseen_context_errors = 0;

        for (size_t i = 0; i < raw_bits.size(); ++i) {
            int depth;
            bool amb;
            float pred = substrate.predict_next_bit(depth, amb);
            uint8_t actual = raw_bits[i];

            bool correct = ((pred >= 0.5f && actual == 1) || (pred < 0.5f && actual == 0));
            if (!correct) {
                errors++;
                if (depth == -1) {
                    unseen_context_errors++;
                } else {
                    horizon_collision_errors++;
                }
            }
            substrate.update(actual);
        }

        double acc = (1.0 - (double)errors / raw_bits.size()) * 100.0;
        std::cout << "Horizon Depth = " << h << " bits (" << (h/8) << " bytes):\n"
                  << "  - Accuracy                    : " << std::fixed << std::setprecision(6) << acc << " %\n"
                  << "  - Total Errors                : " << errors << " / " << raw_bits.size() << "\n"
                  << "  - Horizon Collision Errors (H1): " << horizon_collision_errors << " (Identical " << (h/8) << "B prefix -> different next bit)\n"
                  << "  - Unseen Context Errors (H2)   : " << unseen_context_errors << "\n\n";
    }

    return 0;
}
