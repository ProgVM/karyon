// Dissection tool for .kcore v42 dissipative attractor crystals
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <algorithm>

#pragma pack(push, 1)
struct KCoreManifest {
    char magic[8];             // "KCOREv42"
    uint64_t timestamp_epoch;  // Nanoseconds since epoch
    uint64_t total_couplings;  // Sprouted crystallized bit couplings
    uint64_t total_words_seen; // Total 64-bit words processed
    double bit_error_rate;     // Final bit discrepancy rate
    char reserved[32];         // Reserved alignment padding
};
#pragma pack(pop)

void PrintWordAsAscii(uint64_t word) {
    char bytes[9];
    std::memcpy(bytes, &word, 8);
    bytes[8] = '\0';
    std::cout << " [ASCII: \"";
    for (int i = 0; i < 8; ++i) {
        char c = bytes[i];
        if (c >= 32 && c <= 126) std::cout << c;
        else if (c == '\n') std::cout << "\\n";
        else std::cout << '.';
    }
    std::cout << "\"]";
}

int main(int argc, char* argv[]) {
    std::string path = (argc > 1) ? argv[1] : "build/karyon_exp1094_dissipative.kcore";
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }

    KCoreManifest m;
    in.read(reinterpret_cast<char*>(&m), sizeof(m));

    std::cout << "======================================================================\n";
    std::cout << " .KCORE v42 ENDOSCOPIC AUTOPSY & TOPOLOGICAL DISSECTION\n";
    std::cout << " File Path          : " << path << "\n";
    std::cout << " Magic              : " << std::string(m.magic, 8) << "\n";
    std::cout << " Timestamp Epoch    : " << m.timestamp_epoch << " ns\n";
    std::cout << " Total Couplings    : " << m.total_couplings << " nodes\n";
    std::cout << " Total Words Seen   : " << m.total_words_seen << "\n";
    std::cout << " Discrepancy Rate   : " << std::fixed << std::setprecision(6) << (m.bit_error_rate * 100.0) << " %\n";
    std::cout << "======================================================================\n\n";

    std::vector<std::pair<uint64_t, uint64_t>> nodes;
    nodes.reserve(m.total_couplings);

    for (uint64_t i = 0; i < m.total_couplings; ++i) {
        uint64_t k, v;
        in.read(reinterpret_cast<char*>(&k), sizeof(k));
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        nodes.push_back({k, v});
    }

    std::cout << " [1] Sampling 15 Crystallized Attractor Basins from Substrate Memory:\n";
    for (size_t i = 0; i < std::min((size_t)15, nodes.size()); ++i) {
        std::cout << "  #" << std::setw(2) << (i + 1) << " Basin Signature: 0x" 
                  << std::hex << std::setw(16) << std::setfill('0') << nodes[i].first
                  << " -> Crystallized Word: 0x" << std::setw(16) << std::setfill('0') << nodes[i].second;
        PrintWordAsAscii(nodes[i].second);
        std::cout << std::dec << "\n";
    }

    // Topological Analysis: Hamming distance distribution of signatures
    std::cout << "\n [2] Internal Crystal Topology Statistics:\n";
    uint64_t total_popcount = 0;
    for (const auto &p : nodes) {
        total_popcount += __builtin_popcountll(p.second);
    }
    double avg_spin_density = (double)total_popcount / (double)(nodes.size() * 64);
    std::cout << "  - Average Bit-Spin Polarization (+1 ratio): " 
              << std::fixed << std::setprecision(4) << avg_spin_density << " (Expected ~0.50)\n";
    std::cout << "  - Substrate Memory Footprint              : " 
              << (nodes.size() * 16) << " bytes pure topology\n";
    std::cout << "======================================================================\n";

    return 0;
}
