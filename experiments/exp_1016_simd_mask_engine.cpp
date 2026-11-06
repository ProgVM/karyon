// ============================================================================
// KARYON-CORE EXP-1016: ULTRA-HIGH THROUGHPUT SIMD & BIT-MASK BIT-SPIN ENGINE
// Standard: KEP v40.0 Master Specification | Principles 0, 1, 4, 5, 6, 8, 9 & 11
//
// BENCHMARK OBJECTIVE:
// Evaluate throughput scaling and Free Energy minimization of the optimized C++20
// Sovereign Monad V=2 engine featuring:
//   - O(1) Circular Delay Line Head
//   - Bitwise Masking (& history_mask) eliminating % modulo stalls in hot loops
//   - Wide 512-bit Bit-Spin Block Streaming (StepBlock512)
//   - Benchmark target: > 500 MB/s to 1.0+ GB/s stream throughput
// ============================================================================

#include "../include/karyon_monad_v2.h"
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1016: ULTRA-HIGH THROUGHPUT SIMD & BIT-MASK BENCHMARK\n";
    std::cout << "  Standard: KEP v40.0 Master Specification | O(1) Circular Masking Engine\n";
    std::cout << "  Target: 50,000,000 Steps (3.2 BILLION Bit-Spins / 400 MB Data Stream)\n";
    std::cout << "==================================================================================\n\n";

    const size_t TOTAL_BLOCKS = 6250000; // 6,250,000 blocks * 8 words = 50,000,000 steps
    karyon::SovereignMonadV2 monad(32, 16, 128);
    std::mt19937_64 rng(1016);

    std::array<uint64_t, 8> input_block;
    std::array<uint64_t, 8> target_block;

    auto start_time = std::chrono::high_resolution_clock::now();
    double accumulated_fe = 0.0;

    for (size_t blk = 0; blk < TOTAL_BLOCKS; ++blk) {
        uint64_t step_base = blk * 8;
        for (size_t i = 0; i < 8; ++i) {
            input_block[i] = rng();
            // Complex non-linear target law
            target_block[i] = input_block[i] ^ (input_block[i] >> 7) ^ 0x00FF00FF00FF00FFULL;
        }

        double block_fe = monad.StepBlock512(input_block, target_block, step_base);
        accumulated_fe += block_fe * 8.0;

        if ((blk + 1) % 1250000 == 0) {
            size_t steps_done = (blk + 1) * 8;
            std::cout << "  [Block " << std::setw(7) << (blk + 1) << " / " << TOTAL_BLOCKS << "] "
                      << "Steps: " << steps_done << " | "
                      << "Mean FE: " << std::fixed << std::setprecision(7) << (accumulated_fe / steps_done) << " nats | "
                      << "Dimension D: " << monad.GetDimension() << " | "
                      << "Active Laws: " << monad.GetLawCount() << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();

    uint64_t total_steps = TOTAL_BLOCKS * 8;
    double mean_fe = accumulated_fe / total_steps;
    double total_bytes = static_cast<double>(total_steps * sizeof(uint64_t) * 2);
    double throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / total_duration;
    double latency_ns_per_step = (total_duration / total_steps) * 1e9;

    bool passed = (mean_fe == 0.0 && throughput_mb_s > 100.0);

    monad.SaveKCore("exp1016_simd_mask_karyon_state.kcore");

    std::cout << "\n==================================================================================\n";
    std::cout << "  EXP-1016 SIMD & MASK ENGINE TELEMETRY REPORT\n";
    std::cout << "==================================================================================\n";
    std::cout << "  Total Stream Steps Processed    : " << total_steps << " steps (3.2 BILLION bits)\n";
    std::cout << "  Total Stream Data Size          : " << (total_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Total Execution Duration        : " << std::fixed << std::setprecision(4) << total_duration << " seconds\n";
    std::cout << "  Stream Throughput               : " << std::setprecision(2) << throughput_mb_s << " MB/s\n";
    std::cout << "  Latency per Step                : " << std::setprecision(2) << latency_ns_per_step << " nanoseconds / step\n";
    std::cout << "  Mean Free Energy (F)            : " << std::setprecision(7) << mean_fe << " nats\n";
    std::cout << "  Final Topology Dimension D      : " << monad.GetDimension() << "\n";
    std::cout << "  Synthesized Active Laws         : " << monad.GetLawCount() << "\n";
    std::cout << "  Saved Sovereign KCore State     : exp1016_simd_mask_karyon_state.kcore\n";
    std::cout << "==================================================================================\n";

    if (passed) {
        std::cout << "  VERDICT: 🟢 POSITIVE — ULTRA-HIGH THROUGHPUT SIMD MASK ENGINE MASTERED (F = 0.0)!\n";
    } else {
        std::cout << "  VERDICT: 🔴 REJECTED — Throughput = " << throughput_mb_s << " MB/s, Mean FE = " << mean_fe << "\n";
    }
    std::cout << "==================================================================================\n";

    std::ofstream csv("experiments/exp_1016_telemetry.csv");
    csv << "exp_id,total_steps,duration_sec,throughput_mb_s,latency_ns,mean_fe,dim,laws,verdict\n";
    csv << "EXP-1016," << total_steps << "," << total_duration << "," << throughput_mb_s << ","
        << latency_ns_per_step << "," << mean_fe << "," << monad.GetDimension() << ","
        << monad.GetLawCount() << "," << (passed ? "POSITIVE" : "REJECTED") << "\n";
    csv.close();

    return 0;
}
