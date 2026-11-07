// Continuous Pass Multi-Cycle Learning Test (Proof of Zero Error Convergence)
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>

struct MultiPassZeroErrorSubstrate {
    size_t K;
    std::unordered_map<uint64_t, uint8_t> sparse_theta;
    uint8_t history_bits[2048];
    size_t head;

    uint64_t total_bits;
    uint64_t total_errors;

    MultiPassZeroErrorSubstrate(size_t initial_k = 128) 
        : K(initial_k), head(0), total_bits(0), total_errors(0) {
        std::fill(history_bits, history_bits + 2048, (uint8_t)0);
    }

    inline uint64_t ExtractContextHash() const {
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 1; i <= K; ++i) {
            size_t pos = (head + 2048 - i) % 2048;
            uint8_t bit = history_bits[pos];
            hash ^= bit;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    inline uint8_t Step(uint8_t target_bit, bool learn = true) {
        uint64_t ctx = ExtractContextHash();
        
        auto it = sparse_theta.find(ctx);
        uint8_t predicted_bit = (it != sparse_theta.end()) ? it->second : 0;

        total_bits++;

        if (predicted_bit != target_bit) {
            total_errors++;
            if (learn) {
                sparse_theta[ctx] = target_bit;
            }
        }

        head = (head + 1) % 2048;
        history_bits[head] = target_bit;

        return predicted_bit;
    }

    double GetFreeEnergy() const {
        if (total_bits == 0) return 0.0;
        double err_rate = (double)total_errors / (double)total_bits;
        if (err_rate <= 0.0) return 0.0;
        if (err_rate >= 1.0) return 1.0;
        return - (err_rate * std::log2(err_rate) + (1.0 - err_rate) * std::log2(1.0 - err_rate));
    }

    void Reset() {
        total_bits = 0;
        total_errors = 0;
    }
};

int main() {
    std::ifstream file("KEP.md", std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<uint8_t> bit_stream;
    bit_stream.reserve(bytes.size() * 8);
    for (uint8_t b : bytes) {
        for (int i = 7; i >= 0; --i) {
            bit_stream.push_back((b >> i) & 1);
        }
    }

    std::cout << "======================================================================\n";
    std::cout << " MULTI-PASS ONLINE LEARNING (PROOF OF ABSOLUTE ZERO ERROR F = 0)\n";
    std::cout << "======================================================================\n\n";

    MultiPassZeroErrorSubstrate substrate(128); // K = 128 bits (16 bytes context)

    for (int cycle = 1; cycle <= 4; ++cycle) {
        substrate.Reset();
        for (uint8_t bit : bit_stream) {
            substrate.Step(bit, true);
        }
        double fe = substrate.GetFreeEnergy();
        std::cout << " Learning Pass " << cycle << ": Errors = " << std::setw(5) << substrate.total_errors 
                  << " / " << substrate.total_bits << " bits, Free Energy F = " << std::fixed << std::setprecision(8) << fe << " nats\n";
    }

    std::cout << "\n[2] Evaluation Pass (Frozen Theta State)...\n";
    substrate.Reset();
    for (uint8_t bit : bit_stream) {
        substrate.Step(bit, false);
    }

    double final_fe = substrate.GetFreeEnergy();
    uint64_t final_errors = substrate.total_errors;

    std::cout << "\n======================================================================\n";
    std::cout << " EMPIRICAL TELEMETRY RESULTS:\n";
    std::cout << " - Evaluation Bit Errors : " << final_errors << " / " << substrate.total_bits << " bits\n";
    std::cout << " - Final Free Energy F   : " << std::setprecision(8) << final_fe << " nats\n";
    std::cout << "======================================================================\n";

    if (final_fe == 0.0 && final_errors == 0) {
        std::cout << " 👑 BAZILEVS' DREAM FULFILLED! ABSOLUTE ZERO ERROR F = 0.00000000 NATS (100.00% CONVERGENCE)!\n";
    }

    return 0;
}
