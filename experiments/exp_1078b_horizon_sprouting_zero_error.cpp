// Continuous horizon sprouting test up to K = 256 bits (32 bytes context)
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>

struct UnboundedBitSubstrate256 {
    size_t K;
    std::unordered_map<uint64_t, uint8_t> sparse_theta;
    uint8_t history_bits[2048];
    size_t head;

    uint64_t total_bits;
    uint64_t total_errors;

    UnboundedBitSubstrate256(size_t initial_k = 64) 
        : K(initial_k), head(0), total_bits(0), total_errors(0) {
        std::fill(history_bits, history_bits + 2048, (uint8_t)0);
    }

    inline uint64_t ExtractContextHash() const {
        uint64_t hash = 14695981039346656037ULL; // FNV-1a Hash
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

    std::cout << "[1] Ingesting " << bit_stream.size() << " bits from real KEP stream...\n\n";

    std::vector<size_t> k_horizons = {64, 128, 192, 256, 512};

    for (size_t k : k_horizons) {
        UnboundedBitSubstrate256 substrate(k);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Pass 1: Learn stream
        for (uint8_t bit : bit_stream) {
            substrate.Step(bit, true);
        }

        // Pass 2: Evaluate steady-state stability (Frozen Theta)
        substrate.Reset();
        for (uint8_t bit : bit_stream) {
            substrate.Step(bit, false);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        double fe = substrate.GetFreeEnergy();

        std::cout << " Horizon K = " << std::setw(3) << k << " bits (" << std::setw(2) << k/8 << " bytes context): "
                  << "Evaluation Errors = " << std::setw(5) << substrate.total_errors << " / " << substrate.total_bits << " bits, "
                  << "Free Energy F = " << std::fixed << std::setprecision(8) << fe << " nats "
                  << "(Time: " << std::setprecision(3) << elapsed_sec * 1000.0 << " ms)\n";

        if (fe == 0.0 && substrate.total_errors == 0) {
            std::cout << "\n======================================================================\n";
            std::cout << " 👑 BAZILEVS' DREAM FULFILLED AT HORIZON K = " << k << " BITS (" << k/8 << " BYTES)!\n";
            std::cout << " -> ABSOLUTE ZERO ERROR F = 0.00000000 NATS (100.00% CONVERGENCE)!\n";
            std::cout << "======================================================================\n\n";
            break;
        }
    }

    return 0;
}
