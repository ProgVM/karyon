// ============================================================================
// KARYON-CORE EXP-837: Suffix-Tree Compact State Automaton Substrate
// KEP v30.0 Master | V = 2 Bits | Suffix Automaton Equivalence & Zero-Loss Compression
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

struct SuffixAutomatonState {
    int len;
    int link;
    int next[2]; // Transitions on bit 0 and bit 1
    uint64_t count_0 = 1;
    uint64_t count_1 = 1;
    uint64_t total = 2;

    SuffixAutomatonState(int l = 0, int lk = -1) : len(l), link(lk) {
        next[0] = -1;
        next[1] = -1;
    }

    double get_p1() const {
        return static_cast<double>(count_1) / total;
    }
};

class SuffixAutomatonBitSubstrate {
private:
    std::vector<SuffixAutomatonState> st;
    int last;

public:
    SuffixAutomatonBitSubstrate() {
        st.emplace_back(0, -1);
        last = 0;
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        float prob_1;
        size_t total_states;
    };

    void sa_extend(uint8_t c) {
        int cur = static_cast<int>(st.size());
        st.emplace_back(st[last].len + 1);

        int p = last;
        while (p != -1 && st[p].next[c] == -1) {
            st[p].next[c] = cur;
            p = st[p].link;
        }

        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = static_cast<int>(st.size());
                st.push_back(st[q]);
                st[clone].len = st[p].len + 1;
                st[clone].count_0 = st[q].count_0;
                st[clone].count_1 = st[q].count_1;
                st[clone].total = st[q].total;

                while (p != -1 && st[p].next[c] == q) {
                    st[p].next[c] = clone;
                    p = st[p].link;
                }
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;
    }

    float predict_next_bit_prob() {
        if (last < 0 || last >= st.size()) return 0.5f;

        // Use current active suffix automaton state to predict next transition
        double p1 = st[last].get_p1();
        
        // Also walk up suffix links for smoothed multi-scale prediction
        double weighted_p1 = p1 * st[last].len;
        double total_weight = st[last].len;

        int curr = st[last].link;
        while (curr > 0 && total_weight < 64) {
            double link_p1 = st[curr].get_p1();
            double w = st[curr].len;
            weighted_p1 += link_p1 * w;
            total_weight += w;
            curr = st[curr].link;
        }

        if (total_weight == 0.0) return 0.5f;
        return static_cast<float>(weighted_p1 / total_weight);
    }

    StepResult update_with_actual(float prob_1, uint8_t actual_bit) {
        prob_1 = std::clamp(prob_1, 1e-6f, 1.0f - 1e-6f);

        float p_target = (actual_bit == 1) ? prob_1 : (1.0f - prob_1);
        float free_energy = -std::log(p_target);

        bool is_correct = ((prob_1 >= 0.5f && actual_bit == 1) || (prob_1 < 0.5f && actual_bit == 0));

        // Update transition stats for current active state and its suffix links
        int curr = last;
        while (curr >= 0) {
            st[curr].total++;
            if (actual_bit == 0) st[curr].count_0++;
            else st[curr].count_1++;
            curr = st[curr].link;
        }

        // Extend Suffix Automaton with new bit actual_bit
        sa_extend(actual_bit);

        return {free_energy, is_correct, prob_1, st.size()};
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-837: SUFFIX AUTOMATON EQUIVALENCE SUBSTRATE\n";
    std::cout << "  KEP v30.0 Master | V = 2 Bits | Theoretical Minimal State DFA\n";
    std::cout << "=================================================================\n\n";

    SuffixAutomatonBitSubstrate substrate;

    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> raw_bits;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Loaded " << raw_bits.size() << " raw bits from C++ source stream (" << file_content.size() << " bytes)\n\n";

    // --- PASS 1: SUFFIX AUTOMATON CONSTRUCTION ---
    std::cout << "--- [PASS 1: SUFFIX AUTOMATON CONSTRUCTION] ---\n";
    size_t quarter_len = raw_bits.size() / 4;
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | DFA Minimal States\n";
    std::cout << "---------|----------------|------------------|----------------------|-------------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t dfa_states = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            dfa_states = res.total_states;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(18) << dfa_states << "\n";
    }

    // --- PASS 2: LIFELONG MASTERY ON MINIMAL SUFFIX AUTOMATON ---
    std::cout << "\n--- [PASS 2: LIFELONG MASTERY PASS ON SUFFIX DFA] ---\n";
    std::cout << " Quarter | Processed Bits | Bit Accuracy (%) | Free Energy F (nats) | DFA Minimal States\n";
    std::cout << "---------|----------------|------------------|----------------------|-------------------\n";

    size_t total_p2_correct = 0;
    double total_p2_f = 0.0;

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;
        size_t dfa_states = 0;

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            uint8_t actual_bit = raw_bits[idx];

            float p1 = substrate.predict_next_bit_prob();
            auto res = substrate.update_with_actual(p1, actual_bit);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
            dfa_states = res.total_states;
        }

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;

        total_p2_correct += correct;
        total_p2_f += f_sum;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(14) << ((q + 1) * quarter_len) << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(18) << dfa_states << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    double pass2_acc = (static_cast<double>(total_p2_correct) / (quarter_len * 4)) * 100.0;
    double pass2_avg_f = total_p2_f / (quarter_len * 4);

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-837 SUFFIX AUTOMATON FINAL SUMMARY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Pass 2 Mastery Accuracy      : " << pass2_acc << " %\n";
    std::cout << "  - Pass 2 Free Energy (F)       : " << pass2_avg_f << " nats/bit\n";
    std::cout << "  - Total Execution Time         : " << total_sec << " seconds\n";
    std::cout << "  - Substrate Processing Speed   : " << ((quarter_len * 8) / total_sec) << " bits/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
