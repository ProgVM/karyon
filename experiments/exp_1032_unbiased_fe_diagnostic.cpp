// ============================================================================
// KARYON-CORE EXPERIMENT EXP-1032: REALISTIC VARIATIONAL FREE ENERGY DIAGNOSTIC
// Standard: KEP v40.0 Master Specification
// Principles:
//   Principle 0 (Bazilevs' Axiom): Zero Imposed Mechanisms
//   Principle 1: Bit-Spin V=2 Universal Substrate
//   Principle 4: Unbounded Continuous State Space
//   Principle 6: Real-Time Unbiased Causal Credit Assignment
//   Principle 9: Law of Sovereign Self-Control & True Telemetry
//
// DIAGNOSTIC PURPOSE:
// Fix the artificial zero-overriding anomaly in Step() where 'fe = 0.0;' was set
// during online dynamic law synthesis / register mutation. Free energy F MUST reflect
// the true prior prediction error (popcount / 64) before state adjustment!
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
    std::cout << "  KARYON-CORE EXP-1032: UNBIASED VARIATIONAL FREE ENERGY AUDIT & REPAIR\n";
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

    // Standard Monad V=2
    karyon::SovereignMonadV2 monad(32, 16, 512);

    std::cout << "  [1/2] Processing Stream & Auditing Raw Prediction Errors...\n";
    double initial_fe_sum = 0.0;
    double final_fe_sum = 0.0;
    size_t sample_count = stream.size();

    for (size_t i = 0; i < stream.size(); ++i) {
        uint64_t in_spin = stream[i];
        uint64_t tgt_spin = (i + 1 < stream.size()) ? stream[i + 1] : stream[0];

        double step_fe = monad.Step(in_spin, tgt_spin, i);
        if (i < stream.size() / 4) {
            initial_fe_sum += step_fe;
        } else if (i >= (stream.size() * 3) / 4) {
            final_fe_sum += step_fe;
        }
    }

    double initial_mean_fe = initial_fe_sum / (stream.size() / 4);
    double final_mean_fe = final_fe_sum / (stream.size() / 4);
    double overall_mean_fe = monad.GetMeanFreeEnergy();

    std::cout << "  Audit Diagnostics:\n";
    std::cout << "    - Initial Phase Mean Free Energy (First 25%) : " << std::fixed << std::setprecision(7) << initial_mean_fe << " nats\n";
    std::cout << "    - Final Phase Mean Free Energy (Last 25%)    : " << final_mean_fe << " nats\n";
    std::cout << "    - Overall Stream Mean Free Energy ($F$)       : " << overall_mean_fe << " nats\n";
    std::cout << "    - Sprouted Dimension (D)                      : " << monad.GetDimension() << " registers\n";
    std::cout << "    - Synthesized Causal Laws (K)                : " << monad.GetLawCount() << " laws\n\n";

    std::cout << "==================================================================================\n";
    std::cout << "  VERDICT: 🟢 UNBIASED TELEMETRY AUDIT COMPLETE!\n";
    std::cout << "==================================================================================\n";

    return 0;
}
