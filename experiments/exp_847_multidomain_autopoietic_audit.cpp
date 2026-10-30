// ============================================================================
// KARYON-CORE EXP-847: Multi-Domain Autopoietic Substrate Benchmark Audit
// KEP v30.0 Master | V = 2 Bits | Generalization Audit Across 4 Modalities
// Modalities:
//  1. Chess PGN Stream (Structured Strategic Rules)
//  2. Periodic Wave Signal (Pure Periodicity / Harmonic Law)
//  3. High-Entropy Noise Stream (Chaos / Uncompressibility Limit)
//  4. Natural Language UTF-8 Text (Complex Human Knowledge)
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
#include <random>

struct FusionNode {
    uint32_t id;
    uint32_t level;
    std::string pattern;
    
    uint64_t count_0 = 0;
    uint64_t count_1 = 0;
    uint64_t total = 0;

    int next_0 = -1;
    int next_1 = -1;

    int fusion_target = -1;
    bool epigenetically_locked = false;

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
    std::unordered_map<std::string, uint32_t> state_signature_map;

    std::vector<uint8_t> bit_context;
    size_t max_context_depth;
    size_t fused_branches_count = 0;

public:
    AutopoieticStateFusionSubstrate(size_t max_depth = 64)
        : max_context_depth(max_depth) {
        
        nodes.reserve(1000000);
        pattern_map.reserve(1000000);

        nodes.emplace_back(0, 0, "0");
        nodes.emplace_back(1, 0, "1");
        pattern_map["0"] = 0;
        pattern_map["1"] = 1;
    }

    size_t get_fused_count() const { return fused_branches_count; }
    size_t get_total_nodes() const { return nodes.size(); }

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

        if (deepest_locked_level >= 1 && deepest_locked_p1 >= 0.0) {
            return (deepest_locked_p1 == 1.0) ? 0.999999f : 0.000001f;
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

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

                if (prev_node_id != -1) {
                    if (actual_bit == 0) nodes[prev_node_id].next_0 = node_id;
                    else nodes[prev_node_id].next_1 = node_id;
                }

                if (node.total >= 2 && node.epigenetically_locked) {
                    std::string sig = std::to_string(node.next_0) + ":" + 
                                      std::to_string(node.next_1) + ":" + 
                                      (node.get_p1() == 1.0 ? "1" : "0");

                    auto sig_it = state_signature_map.find(sig);
                    if (sig_it != state_signature_map.end() && sig_it->second != node_id) {
                        node.fusion_target = sig_it->second;
                        fused_branches_count++;
                    } else {
                        state_signature_map[sig] = node_id;
                    }
                }

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

struct AuditResult {
    std::string domain;
    size_t total_bits;
    double pass1_acc;
    double pass2_acc;
    double pass3_acc;
    double pass3_fe;
    size_t nodes_created;
    size_t branches_fused;
};

AuditResult run_domain_audit(const std::string& domain_name, const std::vector<uint8_t>& raw_bits) {
    AutopoieticStateFusionSubstrate substrate(64);

    AuditResult res;
    res.domain = domain_name;
    res.total_bits = raw_bits.size();

    std::vector<double> pass_accs(3, 0.0);
    double final_fe = 0.0;

    for (int pass = 1; pass <= 3; ++pass) {
        size_t correct = 0;
        double fe_sum = 0.0;

        for (size_t i = 0; i < raw_bits.size(); ++i) {
            float p1 = substrate.predict_next_bit_prob();
            bool is_corr;
            float fe;
            substrate.update_and_fuse(p1, raw_bits[i], is_corr, fe);

            if (is_corr) correct++;
            fe_sum += fe;
        }

        pass_accs[pass - 1] = (static_cast<double>(correct) / raw_bits.size()) * 100.0;
        if (pass == 3) final_fe = fe_sum / raw_bits.size();
    }

    res.pass1_acc = pass_accs[0];
    res.pass2_acc = pass_accs[1];
    res.pass3_acc = pass_accs[2];
    res.pass3_fe = final_fe;
    res.nodes_created = substrate.get_total_nodes();
    res.branches_fused = substrate.get_fused_count();

    return res;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-847: MULTI-DOMAIN AUTOPOIETIC SUBSTRATE AUDIT\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Multi-Task Generalization Check\n";
    std::cout << "=================================================================\n\n";

    // 1. Periodic Wave Signal (Period = 16 bits, repeated 4000 times = 64,000 bits)
    std::vector<uint8_t> periodic_bits;
    periodic_bits.reserve(64000);
    std::vector<uint8_t> period_pattern = {1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0};
    for (size_t i = 0; i < 4000; ++i) {
        periodic_bits.insert(periodic_bits.end(), period_pattern.begin(), period_pattern.end());
    }

    // 2. High-Entropy Random Stream (64,000 bits)
    std::vector<uint8_t> random_bits;
    random_bits.reserve(64000);
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < 64000; ++i) {
        random_bits.push_back(static_cast<uint8_t>(dist(rng)));
    }

    // 3. Chess PGN Game Notation Stream (Simulated 50 PGN games string)
    std::vector<uint8_t> chess_bits;
    std::string pgn_sample = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0 ";
    std::string chess_pgn_full = "";
    for (int g = 0; g < 30; ++g) chess_pgn_full += pgn_sample;
    for (char c : chess_pgn_full) {
        for (int b = 7; b >= 0; --b) {
            chess_bits.push_back((c >> b) & 1);
        }
    }

    // 4. Natural Language UTF-8 Text (Karyon Philosophical Manifesto excerpt)
    std::vector<uint8_t> text_bits;
    std::string text_sample = "Karyon-CoRE operates on a continuous wavefield substrate without discrete subword dictionaries. Mind and intelligence are substrate-independent. Variational Free Energy minimization governs self-organization, structure sprouting, and state fusion. Continuous physical wave fields dissolve artificial token boundaries, achieving sovereign autopoiesis across all modalities.";
    std::string text_full = "";
    for (int t = 0; t < 50; ++t) text_full += text_sample;
    for (char c : text_full) {
        for (int b = 7; b >= 0; --b) {
            text_bits.push_back((c >> b) & 1);
        }
    }

    std::vector<AuditResult> results;
    
    std::cout << "[+] Auditing Domain 1: Periodic Harmonic Wave Stream...\n";
    results.push_back(run_domain_audit("1. Periodic Wave Signal", periodic_bits));

    std::cout << "[+] Auditing Domain 2: High-Entropy Random Noise...\n";
    results.push_back(run_domain_audit("2. High-Entropy Noise", random_bits));

    std::cout << "[+] Auditing Domain 3: Chess PGN Notation Stream...\n";
    results.push_back(run_domain_audit("3. Chess PGN Games", chess_bits));

    std::cout << "[+] Auditing Domain 4: Natural Language Text...\n";
    results.push_back(run_domain_audit("4. Natural Language Text", text_bits));

    std::cout << "\n=========================================================================================================\n";
    std::cout << "  EXP-847 MULTI-DOMAIN AUDIT SUMMARY RESULTS\n";
    std::cout << "=========================================================================================================\n";
    std::cout << std::left << std::setw(26) << "Domain Modality"
              << std::setw(12) << "Total Bits"
              << std::setw(13) << "Pass 1 Acc"
              << std::setw(13) << "Pass 2 Acc"
              << std::setw(13) << "Pass 3 Acc"
              << std::setw(15) << "Free Energy"
              << std::setw(14) << "Fused States" << "\n";
    std::cout << "---------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(26) << r.domain
                  << std::setw(12) << r.total_bits
                  << std::fixed << std::setprecision(2)
                  << (std::to_string(r.pass1_acc) + "%").substr(0, 7) << "      "
                  << (std::to_string(r.pass2_acc) + "%").substr(0, 7) << "      "
                  << (std::to_string(r.pass3_acc) + "%").substr(0, 7) << "      "
                  << std::setprecision(6) << std::setw(15) << r.pass3_fe
                  << std::setw(14) << r.branches_fused << "\n";
    }
    std::cout << "=========================================================================================================\n";

    return 0;
}
