// ============================================================================
// KARYON-CORE EXPERIMENT EXP-1034: CONTINUOUS CAUSAL RELAXATION & CREDIT PRUNING
// Standard: KEP v40.0 Master Specification
// Principles:
//   Principle 0 (Bazilevs' Axiom): Drive toward stability (F -> 0.0 nats)
//   Principle 5: Weightless Topological Memory Taps
//   Principle 6: Real-Time Unbiased Causal Credit Assignment
//   Principle 9: Sovereign Self-Tuning without Imposed Top-Down Rules
//
// HYPOTHESIS:
// The remaining 0.39 nats Free Energy is caused by destructive interference
// among accumulated laws XORing into the same target register, and lack of
// iterative internal relaxation.
// Adding continuous causal credit pruning (decaying laws that increase FE)
// and multi-step internal state relaxation will drive F significantly lower toward 0.
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include "../include/karyon_monad_v2.h"

namespace fs = std::filesystem;

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1034: CONTINUOUS CAUSAL RELAXATION & CREDIT PRUNING\n";
    std::cout << "  Standard: KEP v40.0 Master Specification | Raw Bit-Spin V=2 Substrate\n";
    std::cout << "==================================================================================\n\n";

    std::string data_dir = "data/web_bootstrap_corpus";
    if (!fs::exists(data_dir)) {
        std::cerr << "[!] Corpus directory not found: " << data_dir << "\n";
        return 1;
    }

    std::vector<uint64_t> stream;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_regular_file()) continue;
        std::ifstream in(entry.path(), std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        size_t blocks = content.size() / sizeof(uint64_t);
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(content.data());
        stream.insert(stream.end(), ptr, ptr + blocks);
    }

    std::cout << "[+] Loaded " << stream.size() << " 64-bit physical stream blocks (" << (stream.size() * 8) << " bytes).\n\n";

    karyon::SovereignMonadV2 monad(32, 16, 512);

    std::cout << "  [1/2] Simulating Continuous Learning with Causal Credit Pruning...\n";
    double sum_fe_first_half = 0.0;
    double sum_fe_second_half = 0.0;
    size_t half = stream.size() / 2;

    for (size_t i = 0; i < stream.size(); ++i) {
        uint64_t in_spin = stream[i];
        uint64_t tgt_spin = (i + 1 < stream.size()) ? stream[i + 1] : stream[0];

        double step_fe = monad.Step(in_spin, tgt_spin, i);
        if (i < half) {
            sum_fe_first_half += step_fe;
        } else {
            sum_fe_second_half += step_fe;
        }
    }

    double mean_fe_first = sum_fe_first_half / half;
    double mean_fe_second = sum_fe_second_half / (stream.size() - half);
    double total_mean_fe = monad.GetMeanFreeEnergy();

    std::cout << "  Results Summary:\n";
    std::cout << "    - First Half Mean Free Energy ($F_1$) : " << std::fixed << std::setprecision(7) << mean_fe_first << " nats\n";
    std::cout << "    - Second Half Mean Free Energy ($F_2$): " << mean_fe_second << " nats\n";
    std::cout << "    - Overall Stream Mean Free Energy ($F$): " << total_mean_fe << " nats\n";
    std::cout << "    - Synthesized Laws (K)                : " << monad.GetLawCount() << " laws\n";
    std::cout << "    - Sprouted Registers (D)              : " << monad.GetDimension() << " registers\n\n";

    return 0;
}
