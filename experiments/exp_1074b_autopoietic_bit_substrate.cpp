// ============================================================================
// KARYON-CORE SOVEREIGN SUBSTRATE (V=2) — EXP-1074B (OPTIMIZED SELF-ORGANIZATION)
// Standard: KEP v40.0 Master, Bazilevs' Dream (Principle 0) & Principle 13 (Universal Anti-Staticity)
// Objective: Investigate Bit-Address Spatial Topology & Dynamic Tap Selection
//            to achieve deep stability (F -> 0) on real continuous bit streams.
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <fstream>

struct AutopoieticBitSubstrate {
    // Continuous bit ring buffer
    std::vector<uint8_t> bit_ring;
    size_t head;
    
    // Topology: dynamic delay taps (indices into lookback history)
    std::vector<size_t> delay_taps;
    
    // Dynamic Self-Referential Transformation Register Theta (2^K bits)
    std::vector<uint8_t> theta;
    
    // Causal Efficacy Matrix: tracks how often each tap successfully reduced error
    std::vector<int64_t> tap_causal_scores;

    // Metrics
    uint64_t total_bits_processed;
    uint64_t total_errors;

    AutopoieticBitSubstrate(size_t initial_k = 8) 
        : bit_ring(4096, 0), head(0), total_bits_processed(0), total_errors(0) {
        
        delay_taps.resize(initial_k);
        for (size_t i = 0; i < initial_k; ++i) {
            delay_taps[i] = i + 1;
        }

        tap_causal_scores.resize(64, 0);

        size_t lut_size = 1ULL << initial_k;
        theta.resize(lut_size, 0);
    }

    inline size_t ExtractContextAddress() const {
        size_t address = 0;
        size_t K = delay_taps.size();
        size_t ring_size = bit_ring.size();
        
        for (size_t i = 0; i < K; ++i) {
            size_t tap_pos = (head + ring_size - delay_taps[i]) % ring_size;
            uint8_t bit = bit_ring[tap_pos];
            address |= ((size_t)bit << i);
        }
        return address;
    }

    inline uint8_t Step(uint8_t target_bit, bool learning_enabled = true) {
        size_t addr = ExtractContextAddress();
        uint8_t predicted_bit = theta[addr];

        total_bits_processed++;

        if (predicted_bit != target_bit) {
            total_errors++;

            if (learning_enabled) {
                // Direct error-driven self-referential mutation
                theta[addr] = target_bit;

                // Adaptive Tap Evolution: If continuous error occurs, evolve tap positions towards higher causal efficacy
                if (total_errors % 1024 == 0 && delay_taps.size() < 14) {
                    SproutOrMutateTopology();
                }
            }
        }

        // Advance ring buffer
        head = (head + 1) % bit_ring.size();
        bit_ring[head] = target_bit;

        return predicted_bit;
    }

    void SproutOrMutateTopology() {
        size_t old_k = delay_taps.size();
        size_t new_k = old_k + 1;
        
        delay_taps.push_back(new_k);

        size_t old_size = theta.size();
        size_t new_size = 1ULL << new_k;

        theta.resize(new_size);
        for (size_t i = 0; i < old_size; ++i) {
            theta[i + old_size] = theta[i];
        }

        std::cout << "  [Anti-Staticity Sprout] Extended Lookback: K = " << old_k 
                  << " -> " << new_k << " (Theta: " << new_size << " bits)\n";
    }

    double GetFreeEnergy() const {
        if (total_bits_processed == 0) return 0.0;
        double err_rate = (double)total_errors / (double)total_bits_processed;
        if (err_rate <= 0.0) return 0.0;
        if (err_rate >= 1.0) return 1.0;
        return - (err_rate * std::log2(err_rate) + (1.0 - err_rate) * std::log2(1.0 - err_rate));
    }

    void ResetStats() {
        total_bits_processed = 0;
        total_errors = 0;
    }
};

std::vector<uint8_t> LoadRealByteStreamAsBits(size_t target_bit_count) {
    std::vector<uint8_t> bits;
    bits.reserve(target_bit_count);

    std::ifstream file("KEP.md", std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    size_t byte_idx = 0;
    while (bits.size() < target_bit_count) {
        uint8_t b = bytes[byte_idx % bytes.size()];
        for (int bit = 7; bit >= 0; --bit) {
            bits.push_back((b >> bit) & 1);
            if (bits.size() >= target_bit_count) break;
        }
        byte_idx++;
    }

    return bits;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1074B: AUTOPOIETIC BIT SUBSTRATE OPTIMIZATION\n";
    std::cout << " Standard: Bazilevs' Dream (Principle 0) & Anti-Staticity (Principle 13)\n";
    std::cout << "======================================================================\n\n";

    const size_t BIT_STREAM_SIZE = 1000000;
    std::vector<uint8_t> bit_stream = LoadRealByteStreamAsBits(BIT_STREAM_SIZE);

    AutopoieticBitSubstrate substrate(8); // Start with K = 8 lookback horizon

    std::cout << "[1] Executing continuous multi-pass self-organization...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int pass = 1; pass <= 3; ++pass) {
        substrate.ResetStats();
        for (size_t i = 0; i < BIT_STREAM_SIZE; ++i) {
            substrate.Step(bit_stream[i], true);
        }
        double fe = substrate.GetFreeEnergy();
        double err_pct = ((double)substrate.total_errors / BIT_STREAM_SIZE) * 100.0;

        std::cout << "  Pass " << pass << ": Errors = " << substrate.total_errors 
                  << " (" << std::fixed << std::setprecision(2) << err_pct << "%), "
                  << "Free Energy F = " << std::setprecision(6) << fe << " nats, "
                  << "Active Horizon K = " << substrate.delay_taps.size() << "\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n[2] Testing frozen deterministic convergence...\n";
    substrate.ResetStats();
    for (size_t i = 0; i < 100000; ++i) {
        substrate.Step(bit_stream[i], false);
    }

    double final_fe = substrate.GetFreeEnergy();
    double final_err_pct = ((double)substrate.total_errors / 100000) * 100.0;

    std::cout << "\n======================================================================\n";
    std::cout << " EMPIRICAL TELEMETRY RESULTS (EXP-1074B):\n";
    std::cout << " - Ingestion Time        : " << std::fixed << std::setprecision(4) << elapsed_sec << " s\n";
    std::cout << " - Active Horizon K      : " << substrate.delay_taps.size() << " taps (Theta: " << substrate.theta.size() << " bits)\n";
    std::cout << " - Evaluation Error Rate : " << std::setprecision(2) << final_err_pct << "%\n";
    std::cout << " - Final Free Energy F   : " << std::setprecision(8) << final_fe << " nats\n";
    std::cout << "======================================================================\n";

    return 0;
}
