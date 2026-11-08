#ifndef KARYON_CORE_WAVEFIELD_HPP
#define KARYON_CORE_WAVEFIELD_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <string>

// ============================================================================
// KARYON-CORE SOVEREIGN PURE BIT-TRIE SUBSTRATE (V=2)
// Standard: KEP v40.0 Master, Principle 0 (Zero Imposed Mechanisms), Principle 4 (V=2), Principle 13
// Lowest-level fundamental mechanism: Bit-Trie (Context Tree) with Exact Transition Counting.
// Zero human magic constants, zero FNV-1a hashes, zero pre-baked arrays, zero wave math.
// ============================================================================

namespace karyon {

struct BitTrieNode {
    int child[2];
    uint32_t count[2]; // Frequency of target bit 0 and target bit 1

    BitTrieNode() {
        child[0] = -1;
        child[1] = -1;
        count[0] = 0;
        count[1] = 0;
    }
};

class ContinuousWavefieldEngine {
private:
    std::vector<uint8_t> bit_tape;
    std::vector<BitTrieNode> trie;

    uint64_t total_bits_evaluated;
    uint64_t bit_errors;

public:
    ContinuousWavefieldEngine() 
        : total_bits_evaluated(0), bit_errors(0) {
        bit_tape.reserve(2000000);
        trie.reserve(2000000);
        // Root node at index 0
        trie.emplace_back();
    }

    void ResetEvaluationStats() {
        total_bits_evaluated = 0;
        bit_errors = 0;
    }

    void ResetHistory() {
        bit_tape.clear();
        trie.clear();
        trie.emplace_back();
        total_bits_evaluated = 0;
        bit_errors = 0;
    }

    inline uint8_t StepBit(uint8_t target_bit, bool learn = true, bool enable_jitter = false) {
        size_t current_pos = bit_tape.size();

        // 1. Prediction: Traverse down Bit-Trie following deepest matching historical bit context
        int current_node = 0;
        int best_prediction = -1;
        size_t max_matched_depth = 0;

        for (size_t lookback = 1; lookback <= current_pos; ++lookback) {
            uint8_t past_bit = bit_tape[current_pos - lookback] & 1;
            int next_node = trie[current_node].child[past_bit];
            if (next_node == -1) break;

            current_node = next_node;
            uint32_t c0 = trie[current_node].count[0];
            uint32_t c1 = trie[current_node].count[1];

            if (c0 > 0 || c1 > 0) {
                best_prediction = (c1 >= c0) ? 1 : 0;
                max_matched_depth = lookback;
            }
        }

        uint8_t predicted_bit = (best_prediction != -1) ? static_cast<uint8_t>(best_prediction) : 0;
        uint8_t err = (predicted_bit != target_bit) ? 1 : 0;
        total_bits_evaluated++;
        bit_errors += err;

        // 2. Autopoietic Learning: Update transition counts and sprout nodes for full historical path
        if (learn) {
            current_node = 0;
            trie[0].count[target_bit & 1]++;

            for (size_t lookback = 1; lookback <= current_pos; ++lookback) {
                uint8_t past_bit = bit_tape[current_pos - lookback] & 1;
                if (trie[current_node].child[past_bit] == -1) {
                    trie[current_node].child[past_bit] = trie.size();
                    trie.emplace_back();
                }
                current_node = trie[current_node].child[past_bit];
                trie[current_node].count[target_bit & 1]++;
            }
        }

        bit_tape.push_back(target_bit);
        return predicted_bit;
    }

    void IngestContinuousByteStream(const uint8_t* raw_bytes, size_t total_bytes, bool learn = true, bool enable_jitter = false) {
        for (size_t i = 0; i < total_bytes; ++i) {
            uint8_t byte_val = raw_bytes[i];
            for (int b = 7; b >= 0; --b) {
                uint8_t bit_val = (byte_val >> b) & 1;
                StepBit(bit_val, learn, enable_jitter);
            }
        }
    }

    double GetFreeEnergyNats() const {
        if (total_bits_evaluated == 0) return 0.0;
        double err_rate = static_cast<double>(bit_errors) / static_cast<double>(total_bits_evaluated);
        if (err_rate <= 0.0) return 0.0;
        if (err_rate >= 1.0) return 1.0;
        return - (err_rate * std::log2(err_rate) + (1.0 - err_rate) * std::log2(1.0 - err_rate));
    }

    size_t GetSproutedTapCount() const { return trie.size(); }
    size_t GetSproutedNodeCount() const { return trie.size(); }
    uint64_t GetTotalBitsEvaluated() const { return total_bits_evaluated; }
    uint64_t GetTotalBitErrors() const { return bit_errors; }
};

} // namespace karyon

#endif // KARYON_CORE_WAVEFIELD_HPP
