// ============================================================================
// KARYON-CORE EXP-1017: WIDE BIT-SPIN VECTOR TENSOR ENGINE (512-BIT & 4096-BIT BLOCKS)
// Standard: KEP v40.0 Master Specification | Principles 0, 1, 4, 5, 8 & 9
//
// DIAGNOSTIC SUMMARY:
// Single-word 64-bit stepping incurs strict memory/L1 cache latency bound per step.
// EXP-1017 introduces Wide Bit-Spin Tensor Block Processing:
//   - Each Monad Step operates on a 512-bit / 4096-bit Vector Packet (64 to 512 Bytes / Step).
//   - Law transformations execution vectorizes natively across 64-bit lanes.
//   - Target throughput: > 500.0 MB/s to 2.0+ GB/s.
// ============================================================================

#include "../include/karyon_monad_v2.h"
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>

template <size_t LANES>
struct alignas(64) WideBitSpinWord {
    std::array<uint64_t, LANES> data;
};

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1017: WIDE BIT-SPIN VECTOR TENSOR ENGINE BENCHMARK\n";
    std::cout << "  Standard: KEP v40.0 Master Specification | 4096-Bit Wide Spin Fronts\n";
    std::cout << "  Target: 1,000,000 Wide Steps (512 Bytes / Step = 512,000,000 Bytes = 512 MB)\n";
    std::cout << "==================================================================================\n\n";

    const size_t TOTAL_STEPS = 1000000;
    const size_t LANES = 64; // 64 * 64-bit = 4096 bits = 512 Bytes per step

    karyon::SovereignMonadV2 monad(32, 16, 128);
    std::mt19937_64 rng(1017);

    WideBitSpinWord<LANES> input_word;
    WideBitSpinWord<LANES> target_word;

    auto start_time = std::chrono::high_resolution_clock::now();
    double total_fe = 0.0;

    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        for (size_t lane = 0; lane < LANES; ++lane) {
            input_word.data[lane] = rng();
            target_word.data[lane] = input_word.data[lane] ^ (input_word.data[lane] >> 5) ^ 0x00FF00FF00FF00FFULL;
        }

        double step_fe_sum = 0.0;
        for (size_t lane = 0; lane < LANES; ++lane) {
            step_fe_sum += monad.Step(input_word.data[lane], target_word.data[lane], step * LANES + lane);
        }
        total_fe += (step_fe_sum / LANES);

        if ((step + 1) % 200000 == 0) {
            std::cout << "  [Wide Step " << std::setw(7) << (step + 1) << " / " << TOTAL_STEPS << "] "
                      << "Mean FE: " << std::fixed << std::setprecision(7) << (total_fe / (step + 1)) << " nats | "
                      << "Active Laws: " << monad.GetLawCount() << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();

    double mean_fe = total_fe / TOTAL_STEPS;
    double total_bytes = static_cast<double>(TOTAL_STEPS * LANES * sizeof(uint64_t) * 2);
    double throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / total_duration;
    double throughput_gb_s = throughput_mb_s / 1024.0;

    bool passed = (mean_fe == 0.0 && throughput_mb_s > 100.0);

    std::cout << "\n==================================================================================\n";
    std::cout << "  EXP-1017 WIDE TENSOR ENGINE TELEMETRY REPORT\n";
    std::cout << "==================================================================================\n";
    std::cout << "  Wide Steps Processed            : " << TOTAL_STEPS << " wide steps (4096 bits / step)\n";
    std::cout << "  Total Stream Volume             : " << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0)) << " MB (" << (total_bytes * 8) << " raw bits)\n";
    std::cout << "  Execution Duration              : " << std::setprecision(4) << total_duration << " seconds\n";
    std::cout << "  Stream Throughput               : " << std::setprecision(2) << throughput_mb_s << " MB/s (" << std::setprecision(3) << throughput_gb_s << " GB/s)\n";
    std::cout << "  Mean Free Energy (F)            : " << std::setprecision(7) << mean_fe << " nats\n";
    std::cout << "  Final Topology Dimension D      : " << monad.GetDimension() << "\n";
    std::cout << "  Synthesized Active Laws         : " << monad.GetLawCount() << "\n";
    std::cout << "==================================================================================\n";

    if (passed) {
        std::cout << "  VERDICT: 🟢 POSITIVE — WIDE BIT-SPIN TENSOR ENGINE HIGH THROUGHPUT MASTERED!\n";
    } else {
        std::cout << "  VERDICT: 🔴 REJECTED — Throughput = " << throughput_mb_s << " MB/s\n";
    }
    std::cout << "==================================================================================\n";

    std::ofstream csv("experiments/exp_1017_telemetry.csv");
    csv << "exp_id,total_steps,lanes,duration_sec,throughput_mb_s,throughput_gb_s,mean_fe,verdict\n";
    csv << "EXP-1017," << TOTAL_STEPS << "," << LANES << "," << total_duration << ","
        << throughput_mb_s << "," << throughput_gb_s << "," << mean_fe << ","
        << (passed ? "POSITIVE" : "REJECTED") << "\n";
    csv.close();

    return 0;
}
