// Deep Collision Audit: Checking if exact identical 128-bit contexts produce different next bits
#include <iostream>
#include <vector>
#include <fstream>
#include <map>

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

    // Check multiple K horizons
    for (size_t K : {64, 128, 256, 512, 1024}) {
        std::map<std::vector<uint8_t>, uint8_t> context_map;
        size_t collisions = 0;

        for (size_t i = K; i < bit_stream.size(); ++i) {
            std::vector<uint8_t> ctx(bit_stream.begin() + i - K, bit_stream.begin() + i);
            uint8_t next_bit = bit_stream[i];

            auto it = context_map.find(ctx);
            if (it != context_map.end()) {
                if (it->second != next_bit) {
                    collisions++;
                }
            } else {
                context_map[ctx] = next_bit;
            }
        }

        std::cout << "K = " << K << " bits: Collisions = " << collisions << "\n";
    }

    return 0;
}
