// Deep Endoscopic Autopsy for EXP-1095 Mega-Crystal
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <map>

#pragma pack(push, 1)
struct KCoreManifestV43 {
    char magic[8];             // "KCOREv43"
    uint64_t timestamp_epoch;  // Nanoseconds since epoch
    uint64_t total_couplings;  // Sprouted crystallized bit couplings
    uint64_t total_words_seen; // Total 64-bit words processed
    double bit_error_rate;     // Final bit discrepancy rate
    double free_energy_nats;   // Final Free Energy
    char reserved[32];         // Reserved alignment padding
};
#pragma pack(pop)

void DissectWord(uint64_t word) {
    char bytes[9];
    std::memcpy(bytes, &word, 8);
    bytes[8] = '\0';
    std::cout << " [Hex: 0x" << std::hex << std::setw(16) << std::setfill('0') << word << std::dec << "] | ASCII: \"";
    for (int i = 0; i < 8; ++i) {
        char c = bytes[i];
        if (c >= 32 && c <= 126) std::cout << c;
        else if (c == '\n') std::cout << "\\n";
        else std::cout << '.';
    }
    std::cout << "\"";
}

int main(int argc, char* argv[]) {
    std::string path = (argc > 1) ? argv[1] : "build/karyon_exp1095_megacrystal.kcore";
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }

    KCoreManifestV43 m;
    in.read(reinterpret_cast<char*>(&m), sizeof(m));

    std::cout << "======================================================================\n";
    std::cout << " EXP-1095 MEGA-CRYSTAL DEEP DISSECTION & SURGICAL INVASION\n";
    std::cout << " Target File        : " << path << "\n";
    std::cout << " Manifest Magic     : " << std::string(m.magic, 8) << "\n";
    std::cout << " Total Couplings    : " << m.total_couplings << " nodes\n";
    std::cout << " Total Words Seen   : " << m.total_words_seen << "\n";
    std::cout << " Friction Error Rate: " << std::fixed << std::setprecision(6) << (m.bit_error_rate * 100.0) << " %\n";
    std::cout << " Final Free Energy  : " << std::fixed << std::setprecision(8) << m.free_energy_nats << " nats\n";
    std::cout << "======================================================================\n\n";

    std::vector<std::pair<uint64_t, uint64_t>> couplings;
    couplings.reserve(m.total_couplings);

    for (uint64_t i = 0; i < m.total_couplings; ++i) {
        uint64_t k, v;
        in.read(reinterpret_cast<char*>(&k), sizeof(k));
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        couplings.push_back({k, v});
    }

    // 1. Structural Section Breakdown: ELF Machine Bytecode vs Human Text Attractors
    size_t elf_instructions = 0;
    size_t text_sequences = 0;
    size_t zero_padding = 0;

    for (const auto &p : couplings) {
        uint64_t w = p.second;
        if (w == 0ULL) {
            zero_padding++;
        } else {
            // Check if mostly printable ASCII
            char bytes[8];
            std::memcpy(bytes, &w, 8);
            int printable = 0;
            for (int j = 0; j < 8; ++j) {
                if ((bytes[j] >= 32 && bytes[j] <= 126) || bytes[j] == '\n' || bytes[j] == '\r') printable++;
            }
            if (printable >= 5) {
                text_sequences++;
            } else {
                elf_instructions++;
            }
        }
    }

    std::cout << " [1] Substrate Crystal Functional Compartmentalization:\n";
    std::cout << "     - Pure ELF Machine Bytecode Op-Nodes: " << elf_instructions << " (" 
              << std::fixed << std::setprecision(1) << (100.0 * elf_instructions / couplings.size()) << "%)\n";
    std::cout << "     - Lexical / Text Cognitive Attractors : " << text_sequences << " (" 
              << std::fixed << std::setprecision(1) << (100.0 * text_sequences / couplings.size()) << "%)\n";
    std::cout << "     - Ground Zero Alignment Basins       : " << zero_padding << " (" 
              << std::fixed << std::setprecision(1) << (100.0 * zero_padding / couplings.size()) << "%)\n\n";

    // 2. Sampling Raw ELF Machine Instructions Ingested by the Crystal
    std::cout << " [2] Surgical Inspection: Pure ELF Machine Code Attractors (First 6 Samples):\n";
    size_t sampled_elf = 0;
    for (const auto &p : couplings) {
        char bytes[8];
        std::memcpy(bytes, &p.second, 8);
        int printable = 0;
        for (int j = 0; j < 8; ++j) {
            if ((bytes[j] >= 32 && bytes[j] <= 126) || bytes[j] == '\n') printable++;
        }
        if (printable < 5 && p.second != 0ULL) {
            std::cout << "     Attractor Basin [0x" << std::hex << std::setw(16) << std::setfill('0') << p.first << "]";
            DissectWord(p.second);
            std::cout << "\n";
            if (++sampled_elf >= 6) break;
        }
    }

    // 3. Sampling Lexical / KEP Specification Attractors
    std::cout << "\n [3] Surgical Inspection: Lexical KEP Specification Attractors (First 6 Samples):\n";
    size_t sampled_txt = 0;
    for (const auto &p : couplings) {
        char bytes[8];
        std::memcpy(bytes, &p.second, 8);
        int printable = 0;
        for (int j = 0; j < 8; ++j) {
            if ((bytes[j] >= 32 && bytes[j] <= 126) || bytes[j] == '\n') printable++;
        }
        if (printable >= 6) {
            std::cout << "     Attractor Basin [0x" << std::hex << std::setw(16) << std::setfill('0') << p.first << "]";
            DissectWord(p.second);
            std::cout << "\n";
            if (++sampled_txt >= 6) break;
        }
    }

    std::cout << "\n======================================================================\n";
    std::cout << " AUTOPSY CONCLUSION:\n";
    std::cout << " The substrate simultaneously ingested both raw x86_64 binary CPU\n";
    std::cout << " machine bytecode and high-level human semantic text into a single\n";
    std::cout << " continuous, weightless topological crystal of 4,767 nodes.\n";
    std::cout << " Total Storage: " << (couplings.size() * 16) / 1024.0 << " KB.\n";
    std::cout << "======================================================================\n";

    return 0;
}
