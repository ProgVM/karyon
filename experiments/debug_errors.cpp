// Exact Bit-Stream Autopoietic Alignment Audit (Finding exact cause of 44 bit errors)
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>

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

    size_t K = 128; // 16 bytes context
    std::unordered_map<uint64_t, uint8_t> sparse_theta;
    std::vector<uint8_t> ring(2048, 0);
    size_t head = 0;

    auto get_hash = [&](size_t h_pos) {
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 1; i <= K; ++i) {
            size_t pos = (h_pos + 2048 - i) % 2048;
            hash ^= ring[pos];
            hash *= 1099511628211ULL;
        }
        return hash;
    };

    // Train 2 passes
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < bit_stream.size(); ++i) {
            uint64_t h = get_hash(head);
            sparse_theta[h] = bit_stream[i];
            head = (head + 1) % 2048;
            ring[head] = bit_stream[i];
        }
    }

    // Evaluate
    size_t errors = 0;
    for (size_t i = 0; i < bit_stream.size(); ++i) {
        uint64_t h = get_hash(head);
        uint8_t pred = sparse_theta[h];
        if (pred != bit_stream[i]) {
            if (errors < 10) {
                std::cout << "Mismatch at bit " << i << ": target=" << (int)bit_stream[i] << ", pred=" << (int)pred << "\n";
            }
            errors++;
        }
        head = (head + 1) % 2048;
        ring[head] = bit_stream[i];
    }

    std::cout << "\nTotal Errors: " << errors << " / " << bit_stream.size() << " bits\n";
    return 0;
}
