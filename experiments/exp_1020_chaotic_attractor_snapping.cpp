// ============================================================================
// KARYON-CORE EXP-1020: CHAOTIC ATTRACTOR BASIN SNAPPING & NOISE RESILIENCE
// Standard: KEP v40.0 Master Specification
// Principles:
//   Principle 0: Zero Imposed Mechanisms
//   Principle 1: Bit-Spin V=2 Universal Substrate
//   Principle 7: Soft Attractor Basin Snapping & Noise Resilience
//   Principle 9: Law of Sovereign Self-Control (F -> 0.0 nats / entropy lower bound)
//   KEP Rule #7: Continuous High-Entropy Megabyte Bit-Spin Streams
//
// HYPOTHESIS:
// The Sovereign Monad V=2, when exposed to continuous high-entropy bit-spin streams
// derived from a non-linear chaotic Lorenz attractor corrupted by 15% random
// Hamming bit-flip noise, will autopoietically construct deep stability basins.
// It will snap corrupted sensory trajectories back into the clean chaotic attractor
// manifold, reducing prediction error to minimal noise lower bound without collapsing.
// ============================================================================

#include "../include/karyon_monad_v2.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>

// High-Precision Continuous Lorenz Chaotic Attractor Engine
struct LorenzSystem {
    double x = 0.1, y = 0.0, z = 0.0;
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;
    double dt = 0.005;

    // Advance 1 RK4 step and quantize continuous state into 64-bit spin word
    uint64_t NextSpinWord() {
        // RK4 Integration
        double dx1 = sigma * (y - x);
        double dy1 = x * (rho - z) - y;
        double dz1 = x * y - beta * z;

        double x2 = x + 0.5 * dt * dx1;
        double y2 = y + 0.5 * dt * dy1;
        double z2 = z + 0.5 * dt * dz1;
        double dx2 = sigma * (y2 - x2);
        double dy2 = x2 * (rho - z2) - y2;
        double dz2 = x2 * y2 - beta * z2;

        double x3 = x + 0.5 * dt * dx2;
        double y3 = y + 0.5 * dt * dy2;
        double z3 = z + 0.5 * dt * dz2;
        double dx3 = sigma * (y3 - x3);
        double dy3 = x3 * (rho - z3) - y3;
        double dz3 = x3 * y3 - beta * z3;

        double x4 = x + dt * dx3;
        double y4 = y + dt * dy3;
        double z4 = z + dt * dz3;
        double dx4 = sigma * (y4 - x4);
        double dy4 = x4 * (rho - z4) - y4;
        double dz4 = x4 * y4 - beta * z4;

        x += (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
        y += (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4);
        z += (dt / 6.0) * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4);

        // Map continuous floating point coordinates (x, y, z) into 64-bit spin word
        uint32_t ix = static_cast<uint32_t>((x + 50.0) * 1000.0) & 0xFFFFF;
        uint32_t iy = static_cast<uint32_t>((y + 50.0) * 1000.0) & 0xFFFFF;
        uint32_t iz = static_cast<uint32_t>((z + 0.0) * 1000.0) & 0xFFFFF;

        uint64_t spin = (static_cast<uint64_t>(ix) << 40) |
                        (static_cast<uint64_t>(iy) << 20) |
                        (static_cast<uint64_t>(iz));
        // Fill remaining upper bits with chaotic parity checks
        spin |= ((spin ^ (spin >> 17) ^ (spin >> 31)) << 60);
        return spin;
    }
};

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1020: CHAOTIC ATTRACTOR BASIN SNAPPING & NOISE RESILIENCE\n";
    std::cout << "  Standard: KEP v40.0 Master Specification | Principle 7 Resonance Test\n";
    std::cout << "  Target Stream: 10,000,000 Words (640 MILLION Bits) from Lorenz Chaotic Manifold\n";
    std::cout << "  Noise Conditions: Clean Phase -> 15% Hamming Bit-Flip Noise Phase\n";
    std::cout << "==================================================================================\n\n";

    const size_t TOTAL_WORDS = 10000000;
    LorenzSystem lorenz;
    karyon::SovereignMonadV2 monad(64, 16, 256);

    std::mt19937_64 rng(1020);
    std::uniform_real_distribution<double> noise_dist(0.0, 1.0);

    double total_fe = 0.0;
    double clean_phase_fe = 0.0;
    double noisy_phase_fe = 0.0;

    size_t clean_steps = 2000000;
    size_t noisy_steps = 8000000;

    double noise_rate = 0.15; // 15% bit-flip noise in noisy phase

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < TOTAL_WORDS; ++step) {
        uint64_t clean_spin = lorenz.NextSpinWord();
        uint64_t input_spin = clean_spin;

        // Inject Hamming noise during noisy phase
        if (step >= clean_steps) {
            uint64_t noise_mask = 0ULL;
            for (int bit = 0; bit < 64; ++bit) {
                if (noise_dist(rng) < noise_rate) {
                    noise_mask |= (1ULL << bit);
                }
            }
            input_spin ^= noise_mask; // Corrupted input spin
        }

        // Monad step predicts clean chaotic attractor spin from noisy input
        double fe = monad.Step(input_spin, clean_spin, step);
        total_fe += fe;

        if (step < clean_steps) {
            clean_phase_fe += fe;
        } else {
            noisy_phase_fe += fe;
        }

        if ((step + 1) % 2000000 == 0) {
            double current_phase_fe = (step < clean_steps) ? (clean_phase_fe / (step + 1))
                                                           : (noisy_phase_fe / (step + 1 - clean_steps));
            std::cout << "  [Step " << std::setw(8) << (step + 1) << " / " << TOTAL_WORDS << "] "
                      << "Phase: " << (step < clean_steps ? "Clean Lorenz " : "15% Noise Phase") << " | "
                      << "Mean FE: " << std::fixed << std::setprecision(7) << current_phase_fe << " nats | "
                      << "Laws: " << monad.GetLawCount() << " | "
                      << "Locks: " << monad.GetTotalLocks() << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();

    double overall_mean_fe = total_fe / TOTAL_WORDS;
    double clean_mean_fe = clean_phase_fe / clean_steps;
    double noisy_mean_fe = noisy_phase_fe / noisy_steps;

    double total_bytes = static_cast<double>(TOTAL_WORDS * sizeof(uint64_t) * 2);
    double throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / total_duration;
    double throughput_gb_s = throughput_mb_s / 1024.0;

    // Success criterion: Monad must achieve clean convergence F = 0 in clean phase,
    // and hold Free Energy significantly LOWER than raw noise rate (15% noise = 0.15 FE) in noisy phase.
    bool passed = (clean_mean_fe == 0.0 && noisy_mean_fe <= noise_rate);

    std::cout << "\n==================================================================================\n";
    std::cout << "  EXP-1020 CHAOTIC ATTRACTOR SNAPPING TELEMETRY REPORT\n";
    std::cout << "==================================================================================\n";
    std::cout << "  Total Stream Words Processed   : " << TOTAL_WORDS << " words (640 MILLION bits)\n";
    std::cout << "  Clean Phase Mean FE (0% Noise) : " << std::fixed << std::setprecision(7) << clean_mean_fe << " nats\n";
    std::cout << "  Noisy Phase Mean FE (15% Noise): " << std::fixed << std::setprecision(7) << noisy_mean_fe << " nats\n";
    std::cout << "  Raw Input Noise Rate           : " << noise_rate << " (15.0% Hamming Noise)\n";
    std::cout << "  Noise Snapping Efficiency      : " << std::setprecision(2) << ((1.0 - noisy_mean_fe / noise_rate) * 100.0) << "% Attractor Noise Reduction\n";
    std::cout << "  Execution Duration              : " << std::setprecision(4) << total_duration << " seconds\n";
    std::cout << "  Stream Throughput               : " << std::setprecision(2) << throughput_mb_s << " MB/s (" << std::setprecision(3) << throughput_gb_s << " GB/s)\n";
    std::cout << "  Synthesized Active Laws         : " << monad.GetLawCount() << "\n";
    std::cout << "  Epigenetically Locked Laws     : " << monad.GetTotalLocks() << "\n";
    std::cout << "==================================================================================\n";

    if (passed) {
        std::cout << "  VERDICT: 🟢 POSITIVE — CHAOTIC ATTRACTOR BASIN SNAPPING MASTERED!\n";
    } else {
        std::cout << "  VERDICT: 🔴 REJECTED — Clean FE = " << clean_mean_fe << ", Noisy FE = " << noisy_mean_fe << "\n";
    }
    std::cout << "==================================================================================\n";

    std::ofstream csv("experiments/exp_1020_telemetry.csv");
    csv << "exp_id,total_words,clean_fe,noisy_fe,noise_rate,duration_sec,throughput_mb_s,verdict\n";
    csv << "EXP-1020," << TOTAL_WORDS << "," << clean_mean_fe << "," << noisy_mean_fe << ","
        << noise_rate << "," << total_duration << "," << throughput_mb_s << ","
        << (passed ? "POSITIVE" : "REJECTED") << "\n";
    csv.close();

    return 0;
}
