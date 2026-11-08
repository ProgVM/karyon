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
// Absolute lowest-level fundamental mechanism: Endogenous Bit-Trie (Context Tree).
// Zero human magic constants, zero FNV-1a hashes, zero pre-baked arrays, zero wave math.
// Resolves state collisions strictly bottom-up by sprouting context depth on divergence.
// ============================================================================

namespace karyon {

struct BitTrieNode {
    int child[2];
    int predicted_bit; // -1 if unassigned, 0 or 1
    uint32_t frequency;

    BitTrieNode() : predicted_bit(-1), frequency(0) {
        child[0] = -1;
        child[1] = -1;
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
        trie.reserve(1000000);
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

        // Traverse down the Bit-Trie following historical bits (b_{t-1}, b_{t-2}, ...)
        int current_node = 0; // Root
        int deepest_matching_node = 0;
        int prediction = -1;

        for (size_t lookback = 1; lookback <= current_pos; ++lookback) {
            uint8_t past_bit = bit_tape[current_pos - lookback] & 1;
            int next_node = trie[current_node].child[past_bit];
            if (next_node == -1) {
                break;
            }
            current_node = next_node;
            if (trie[current_node].predicted_bit != -1) {
                deepest_matching_node = current_node;
                prediction = trie[current_node].predicted_bit;
            }
        }

        uint8_t predicted_bit = (prediction != -1) ? static_cast<uint8_t>(prediction) : 0;
        uint8_t err = (predicted_bit != target_bit) ? 1 : 0;
        total_bits_evaluated++;
        bit_errors += err;

        if (learn) {
            // Endogenous Collision Resolution:
            // Traverse history and sprout nodes until target_bit is deterministically recorded
            current_node = 0;
            for (size_t lookback = 1; lookback <= current_pos + 1; ++lookback) {
                if (trie[current_node].predicted_bit == -1) {
                    trie[current_node].predicted_bit = target_bit;
                    trie[current_node].frequency = 1;
                    break;
                } else if (trie[current_node].predicted_bit == target_bit) {
                    trie[current_node].frequency++;
                    if (lookback <= current_pos) {
                        uint8_t past_bit = bit_tape[current_pos - lookback] & 1;
                        if (trie[current_node].child[past_bit] == -1) {
                            trie[current_node].child[past_bit] = trie.size();
                            trie.emplace_back();
                        }
                        current_node = trie[current_node].child[past_bit];
                    }
                } else { // State collision: trie node recorded bit X, but target is Y
                    // Sprout context depth by 1 bit to resolve ambiguity
                    trie[current_node].predicted_bit = target_bit; // Update dominant transition
                    if (lookback <= current_pos) {
                        uint8_t past_bit = bit_tape[current_pos - lookback] & 1;
                        if (trie[current_node].child[past_bit] == -1) {
                            trie[current_node].child[past_bit] = trie.size();
                            trie.emplace_back();
                        }
                        current_node = trie[current_node].child[past_bit];
                    } else {
                        break;
                    }
                }
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
