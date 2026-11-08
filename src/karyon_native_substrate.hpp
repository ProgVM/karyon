// Karyon-CoRE Native Bit Substrate & Engine Architecture
// Specification: KEP v40.0 Master & Principle 13 (Universal Anti-Staticity)
// - Pure 64-bit hardware register words and physical bit-spins (V=2).
// - Zero human-imposed spaces or static vector tables.
// - Endogenous memory sprouting (growth only on Free Energy > 0).
// - Clean .kcore v41 serialization: Manifest Header + Kernel Section + Bit Substrate Section.

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>

#pragma pack(push, 1)
struct KCoreManifest {
    char magic[8];             // "KCOREv41"
    uint64_t timestamp_epoch;  // Nanoseconds since epoch
    uint64_t total_couplings;  // Number of active sprouted couplings
    uint64_t total_words_seen; // Total 64-bit words processed
    double final_free_energy;  // Final Free Energy F in nats
    char reserved[32];         // Reserved alignment padding
};

struct KCoreKernelConfig {
    uint64_t tap_mask_0;       // Hardware bit-tap mask 0
    uint64_t tap_mask_1;       // Hardware bit-tap mask 1
    uint32_t shift_a;          // Register shift A
    uint32_t shift_b;          // Register shift B
    uint32_t shift_c;          // Register shift C
    uint32_t shift_d;          // Register shift D
};
#pragma pack(pop)

class KaryonNativeSubstrate {
public:
    // Dynamic bit-tap mapping: hardware signature -> predicted 64-bit physical word
    // Endogenous growth: zero pre-allocated limits.
    std::unordered_map<uint64_t, uint64_t> bit_couplings;

    KCoreKernelConfig kernel;
    uint64_t hw_register;
    uint64_t processed_words;
    uint64_t bit_errors;
    size_t peak_couplings;

    KaryonNativeSubstrate() {
        // Default physical hardware kernel invariants (bit-taps)
        kernel.tap_mask_0 = 0xAAAAAAAAAAAAAAAAULL;
        kernel.tap_mask_1 = 0x5555555555555555ULL;
        kernel.shift_a = 7;
        kernel.shift_b = 13;
        kernel.shift_c = 19;
        kernel.shift_d = 29;

        hw_register = 0ULL;
        processed_words = 0;
        bit_errors = 0;
        peak_couplings = 0;
    }

    // Pure hardware step function: zero abstraction layers
    inline uint64_t Step(uint64_t target_word, bool enable_sprouting) {
        // Physical signature from register bit-taps
        uint64_t signature = hw_register ^ 
                             (hw_register >> kernel.shift_a) ^ 
                             (hw_register >> kernel.shift_b) ^ 
                             (hw_register >> kernel.shift_c) ^
                             (hw_register >> kernel.shift_d);

        auto it = bit_couplings.find(signature);
        uint64_t predicted_word = (it != bit_couplings.end()) ? it->second : 0ULL;

        // Bitwise difference
        uint64_t diff = predicted_word ^ target_word;
        int errors = __builtin_popcountll(diff);

        processed_words++;
        bit_errors += errors;

        // Endogenous Sprouting: born ONLY on divergence (Free Energy > 0)
        if (errors > 0 && enable_sprouting) {
            bit_couplings[signature] = target_word;
            if (bit_couplings.size() > peak_couplings) {
                peak_couplings = bit_couplings.size();
            }
        }

        // Advance physical register state
        hw_register = (hw_register << 1) | (target_word & 1ULL);
        return predicted_word;
    }

    double ComputeFreeEnergy() const {
        uint64_t total_bits = processed_words * 64;
        if (total_bits == 0) return 0.0;
        double error_rate = (double)bit_errors / (double)total_bits;
        if (error_rate <= 0.0) return 0.0;
        if (error_rate >= 1.0) return 1.0;
        return - (error_rate * std::log2(error_rate) + (1.0 - error_rate) * std::log2(1.0 - error_rate));
    }

    void ResetStepCounters() {
        processed_words = 0;
        bit_errors = 0;
    }

    // Save container (.kcore v41)
    bool Save(const std::string &filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        KCoreManifest manifest;
        std::memcpy(manifest.magic, "KCOREv41", 8);
        manifest.timestamp_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        manifest.total_couplings = bit_couplings.size();
        manifest.total_words_seen = processed_words;
        manifest.final_free_energy = ComputeFreeEnergy();
        std::memset(manifest.reserved, 0, sizeof(manifest.reserved));

        // 1. Manifest Section
        out.write(reinterpret_cast<const char*>(&manifest), sizeof(manifest));

        // 2. Kernel Section
        out.write(reinterpret_cast<const char*>(&kernel), sizeof(kernel));

        // 3. Dynamic Bit Substrate Section (Key-Value stream)
        for (const auto &pair : bit_couplings) {
            uint64_t key = pair.first;
            uint64_t val = pair.second;
            out.write(reinterpret_cast<const char*>(&key), sizeof(key));
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }

        out.close();
        return true;
    }

    // Load container (.kcore v41)
    bool Load(const std::string &filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;

        KCoreManifest manifest;
        in.read(reinterpret_cast<char*>(&manifest), sizeof(manifest));
        if (std::memcmp(manifest.magic, "KCOREv41", 8) != 0) {
            in.close();
            return false;
        }

        // 2. Kernel Section
        in.read(reinterpret_cast<char*>(&kernel), sizeof(kernel));

        // 3. Dynamic Bit Substrate Section
        bit_couplings.clear();
        bit_couplings.reserve(manifest.total_couplings);

        for (uint64_t i = 0; i < manifest.total_couplings; ++i) {
            uint64_t key, val;
            in.read(reinterpret_cast<char*>(&key), sizeof(key));
            in.read(reinterpret_cast<char*>(&val), sizeof(val));
            bit_couplings[key] = val;
        }

        peak_couplings = bit_couplings.size();
        in.close();
        return true;
    }
};
