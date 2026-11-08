// Deep Autopsy of EXP-1096 Bit Natural Selection Failure
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <algorithm>

#pragma pack(push, 1)
struct KCoreManifestV44 {
    char magic[8];             // "KCOREv44"
    uint64_t timestamp_epoch;  // Nanoseconds since epoch
    uint64_t total_micro_nodes;// Active crystallized micro-couplings
    uint64_t total_bits_seen;  // Total raw bits processed
    double bit_friction_rate;  // Final bit error rate
    double free_energy_nats;   // Final Free Energy
    char reserved[32];         // Reserved alignment padding
};

struct BitCoupling {
    int offset_a;      // Relative historical bit offset A
    int offset_b;      // Relative historical bit offset B
    uint8_t logic_rule;// Elementary 4-bit lookup truth table
    int32_t stability; // Physical energy / stability score
};
#pragma pack(pop)

int main() {
    std::ifstream in("build/karyon_exp1096_bit_natural.kcore", std::ios::binary);
    if (!in.is_open()) return 1;

    KCoreManifestV44 m;
    in.read(reinterpret_cast<char*>(&m), sizeof(m));

    std::cout << "======================================================================\n";
    std::cout << " EXP-1096 AUTOPSY: WHY DID INDEPENDENT RANDOM COUPLES FAIL TO SINK F?\n";
    std::cout << " Total Micro-Nodes : " << m.total_micro_nodes << "\n";
    std::cout << " Final Bit Error   : " << (m.bit_friction_rate * 100.0) << " %\n";
    std::cout << " Final Free Energy : " << m.free_energy_nats << " nats\n";
    std::cout << "======================================================================\n\n";

    std::vector<BitCoupling> nodes(m.total_micro_nodes);
    for (size_t i = 0; i < m.total_micro_nodes; ++i) {
        in.read(reinterpret_cast<char*>(&nodes[i]), sizeof(BitCoupling));
    }

    std::cout << " [1] Inspecting First 10 Micro-Couplings:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "  Node #" << i << ": offsets=(" << nodes[i].offset_a << ", " << nodes[i].offset_b 
                  << "), rule=0x" << std::hex << (int)nodes[i].logic_rule << std::dec 
                  << ", stability=" << nodes[i].stability << "\n";
    }

    return 0;
}
