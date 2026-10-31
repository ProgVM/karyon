#pragma once
#ifndef KARYON_CORE_DP_SAE_HPP
#define KARYON_CORE_DP_SAE_HPP

// ============================================================================
// KARYON-CORE SOVEREIGN ENGINE v39.3 — karyon_core_dp_sae.hpp
// Core Module: Pure V=2 Dynamic Bit-Spin Engine with Soft Attractor Basin Snapping
// Standard: KEP v39.0 Master Protocol | Principles 2, 3, 8, 27, 28, 29, 30, 31, 32
//
// Features:
// - Dynamic Longest-Match Bit-Path Attractor Retrieval (DP-SAE)
// - Threshold-Gated Soft Spin-Glass / Hamming Basin Snapping (EXP-917)
//   If exact match length is shallow (e.g. < MIN_EXACT_MATCH_BITS), triggers STAGE 2
//   soft resonance scan to snap back into deep long-range attractor basins!
// - Closed-loop autoregressive bit-level generation
// - High-speed Binary State Persistence (.kcore serialization/deserialization)
//
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <limits>

namespace karyon {

struct BitPathHash {
    std::size_t operator()(const std::vector<uint8_t>& bits) const {
        std::size_t hash = bits.size();
        for (uint8_t b : bits) {
            hash ^= std::hash<uint8_t>()(b) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

class SovereignV2Engine {
private:
    std::vector<uint8_t> bit_history; // Pure V=2 bit stream
    std::unordered_map<std::vector<uint8_t>, double, BitPathHash> bit_attractors;
    std::vector<std::pair<std::vector<uint8_t>, double>> attractor_list; // For soft Hamming scan
    size_t max_stored_depth = 0;
    static constexpr size_t MAX_LOOKBACK_BITS = 1024; // Upper limit for stored bit-paths
    static constexpr size_t MIN_EXACT_MATCH_BITS = 16; // Threshold below which exact match is deemed shallow

public:
    SovereignV2Engine() {}

    // Predict next spin in {-1.0, +1.0} via dynamic longest-match search with Soft Resonance fallback
    double PredictNextSpin(double max_noise_threshold = 0.20, bool enable_soft_resonance = true) const {
        if (bit_history.empty()) return 0.0;

        size_t available = bit_history.size();
        size_t max_search_len = std::min(available, max_stored_depth > 0 ? max_stored_depth : MAX_LOOKBACK_BITS);

        size_t exact_match_len = 0;
        double exact_spin = 0.0;

        // STAGE 1: Exact Longest-Match Retrieval
        for (size_t len = max_search_len; len >= 1; --len) {
            std::vector<uint8_t> sub_path(bit_history.end() - len, bit_history.end());
            auto it = bit_attractors.find(sub_path);
            if (it != bit_attractors.end()) {
                exact_match_len = len;
                exact_spin = it->second;
                break;
            }
        }

        // If exact match is deep (e.g. >= 16 bits), it's unperturbed confidence!
        if (exact_match_len >= MIN_EXACT_MATCH_BITS || !enable_soft_resonance) {
            return exact_spin;
        }

        // STAGE 2: Soft Spin-Glass / Hamming Basin Snapping (Triggered when exact match is shallow due to noise)
        if (enable_soft_resonance && !attractor_list.empty()) {
            double best_resonance = -1.0;
            double best_spin = exact_spin; // Fallback to shallow exact match if soft fails

            for (const auto& entry : attractor_list) {
                const auto& path = entry.first;
                size_t p_len = path.size();
                if (p_len < MIN_EXACT_MATCH_BITS) continue;

                size_t cmp_len = std::min(available, p_len);
                if (cmp_len < MIN_EXACT_MATCH_BITS) continue;

                size_t diffs = 0;
                size_t hist_start = available - cmp_len;
                size_t path_start = p_len - cmp_len;

                for (size_t i = 0; i < cmp_len; ++i) {
                    if (bit_history[hist_start + i] != path[path_start + i]) {
                        diffs++;
                    }
                }

                double dist = static_cast<double>(diffs) / static_cast<double>(cmp_len);
                if (dist <= max_noise_threshold) {
                    // Exponential penalty for noise and linear reward for deep matching length
                    double score = (1.0 - dist * 2.0) * static_cast<double>(cmp_len);
                    if (score > best_resonance) {
                        best_resonance = score;
                        best_spin = entry.second;
                    }
                }
            }

            if (best_resonance > 0.0) {
                return best_spin; // Attractor basin snapped!
            }
        }

        return exact_spin;
    }

    double StepBit(double target_spin, bool learn = true, double max_noise_threshold = 0.20, bool enable_soft_resonance = true) {
        uint8_t target_bit = (target_spin > 0.0) ? 1 : 0;
        double pred_spin = PredictNextSpin(max_noise_threshold, enable_soft_resonance);
        double err = target_spin - pred_spin;
        double fe = 0.5 * err * err;

        if (learn && !bit_history.empty()) {
            size_t available = bit_history.size();
            size_t max_store = std::min(available, MAX_LOOKBACK_BITS); // Up to 1024 bits dynamic path
            std::vector<uint8_t> full_path(bit_history.end() - max_store, bit_history.end());
            
            if (bit_attractors.find(full_path) == bit_attractors.end()) {
                bit_attractors[full_path] = target_spin;
                attractor_list.push_back({full_path, target_spin});
            } else {
                bit_attractors[full_path] = target_spin;
            }
            
            if (max_store > max_stored_depth) {
                max_stored_depth = max_store;
            }
        }

        bit_history.push_back(target_bit);
        if (bit_history.size() > 4096) {
            bit_history.erase(bit_history.begin(), bit_history.begin() + 2048);
        }

        return fe;
    }

    void IngestRawBytes(const uint8_t* data, size_t size, bool learn = true, double noise_thresh = 0.20, bool enable_soft = true) {
        for (size_t i = 0; i < size; ++i) {
            uint8_t byte = data[i];
            for (int b = 7; b >= 0; --b) {
                double spin = ((byte >> b) & 1) ? +1.0 : -1.0;
                StepBit(spin, learn, noise_thresh, enable_soft);
            }
        }
    }

    std::vector<uint8_t> GenerateAutoregressiveBytes(size_t num_bytes, double noise_thresh = 0.20, bool enable_soft = true) {
        std::vector<uint8_t> gen_bytes;

        for (size_t i = 0; i < num_bytes; ++i) {
            uint8_t current_byte = 0;
            for (int b = 7; b >= 0; --b) {
                double pred = PredictNextSpin(noise_thresh, enable_soft);
                uint8_t next_bit = (pred >= 0.0) ? 1 : 0;
                if (next_bit) {
                    current_byte |= (1 << b);
                }
                double next_spin = next_bit ? +1.0 : -1.0;
                StepBit(next_spin, false, noise_thresh, enable_soft);
            }
            gen_bytes.push_back(current_byte);
        }

        return gen_bytes;
    }

    void ResetHistory() {
        bit_history.clear();
    }

    size_t GetMaxStoredDepthBits() const { return max_stored_depth; }
    size_t GetAttractorCount() const { return bit_attractors.size(); }

    // ============================================================================
    // STATE PERSISTENCE: High-Speed Binary .kcore Serialization & Deserialization
    // ============================================================================

    bool SaveState(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        const char magic[8] = {'K', 'C', 'O', 'R', 'E', 'v', '3', '9'};
        out.write(magic, 8);

        uint64_t max_depth = static_cast<uint64_t>(max_stored_depth);
        out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));

        uint64_t history_size = static_cast<uint64_t>(bit_history.size());
        out.write(reinterpret_cast<const char*>(&history_size), sizeof(history_size));
        if (history_size > 0) {
            out.write(reinterpret_cast<const char*>(bit_history.data()), history_size * sizeof(uint8_t));
        }

        uint64_t num_attractors = static_cast<uint64_t>(bit_attractors.size());
        out.write(reinterpret_cast<const char*>(&num_attractors), sizeof(num_attractors));

        for (const auto& kv : bit_attractors) {
            const auto& path = kv.first;
            double spin = kv.second;

            uint32_t path_len = static_cast<uint32_t>(path.size());
            out.write(reinterpret_cast<const char*>(&path_len), sizeof(path_len));
            out.write(reinterpret_cast<const char*>(path.data()), path_len * sizeof(uint8_t));
            out.write(reinterpret_cast<const char*>(&spin), sizeof(spin));
        }

        out.close();
        return true;
    }

    bool LoadState(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;

        char magic[8];
        in.read(magic, 8);
        if (std::string(magic, 8) != "KCOREv39") {
            in.close();
            return false;
        }

        uint64_t max_depth = 0;
        in.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
        max_stored_depth = static_cast<size_t>(max_depth);

        uint64_t history_size = 0;
        in.read(reinterpret_cast<char*>(&history_size), sizeof(history_size));
        bit_history.resize(history_size);
        if (history_size > 0) {
            in.read(reinterpret_cast<char*>(bit_history.data()), history_size * sizeof(uint8_t));
        }

        uint64_t num_attractors = 0;
        in.read(reinterpret_cast<char*>(&num_attractors), sizeof(num_attractors));
        
        bit_attractors.clear();
        attractor_list.clear();
        bit_attractors.reserve(num_attractors);
        attractor_list.reserve(num_attractors);

        for (uint64_t i = 0; i < num_attractors; ++i) {
            uint32_t path_len = 0;
            in.read(reinterpret_cast<char*>(&path_len), sizeof(path_len));
            
            std::vector<uint8_t> path(path_len);
            in.read(reinterpret_cast<char*>(path.data()), path_len * sizeof(uint8_t));
            
            double spin = 0.0;
            in.read(reinterpret_cast<char*>(&spin), sizeof(spin));

            bit_attractors[path] = spin;
            attractor_list.push_back({path, spin});
        }

        in.close();
        return true;
    }
};

} // namespace karyon

#endif // KARYON_CORE_DP_SAE_HPP
