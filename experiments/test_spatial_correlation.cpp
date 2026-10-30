#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

int main() {
    std::ifstream file("data/ultrahd_1024.bin", std::ios::binary);
    if (!file.is_open()) return 1;
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    size_t total_bits = raw.size() * 8;
    int stride_bits = 1024 * 3 * 8; // 24576 bits (1 row)

    uint64_t direct_matches = 0;
    uint64_t valid_evals = 0;

    for (size_t t = stride_bits; t < total_bits; ++t) {
        size_t byte_curr = t / 8;
        int bit_curr = 7 - (t % 8);
        uint8_t b1 = (raw[byte_curr] >> bit_curr) & 1;

        size_t t_prev = t - stride_bits;
        size_t byte_prev = t_prev / 8;
        int bit_prev = 7 - (t_prev % 8);
        uint8_t b0 = (raw[byte_prev] >> bit_prev) & 1;

        if (b1 == b0) direct_matches++;
        valid_evals++;
    }

    double acc = (double)direct_matches / valid_evals * 100.0;
    std::cout << "Direct Spatial Stride s = " << stride_bits << " bits (1 row):\n"
              << "  - Valid Evals: " << valid_evals << "\n"
              << "  - Direct Bit Matches: " << direct_matches << "\n"
              << "  - Direct Match Accuracy: " << acc << "%\n";

    return 0;
}
