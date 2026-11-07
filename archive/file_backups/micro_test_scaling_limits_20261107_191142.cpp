// ============================================================================
// KARYON-CORE MICRO-TEST 10: SCALING LIMITS & STRESS BENCHMARK (EXP-1073)
// Objective: Evaluate substrate adaptation and scaling under:
//            1. 4-Input Logic Streams (65,536 LUT space)
//            2. Non-Stationary Target Streams (Rule shifts mid-stream)
//            3. Noisy Non-Deterministic Bitstreams (5% random bit flips)
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct Universal4InputMutableSubstrate {
    uint64_t ring[128];
    size_t head;

    uint8_t tap_A, tap_B, tap_C, tap_D;
    uint16_t theta; // 16-bit Truth Table LUT (65,536 possible 4-input Boolean functions)

    Universal4InputMutableSubstrate() 
        : head(0), tap_A(1), tap_B(2), tap_C(3), tap_D(4), theta(0) {
        std::fill(ring, ring + 128, 0ULL);
    }

    inline uint64_t Evaluate16BitLUT(uint64_t A, uint64_t B, uint64_t C, uint64_t D, uint16_t lut) const {
        uint64_t notA = ~A, notB = ~B, notC = ~C, notD = ~D;
        uint64_t res = 0ULL;
        
        // Bit-parallel evaluation of all 16 minterms
        if (lut & 0x0001) res |= (notA & notB & notC & notD);
        if (lut & 0x0002) res |= (notA & notB & notC & D);
        if (lut & 0x0004) res |= (notA & notB & C    & notD);
        if (lut & 0x0008) res |= (notA & notB & C    & D);
        if (lut & 0x0010) res |= (notA & B    & notC & notD);
        if (lut & 0x0020) res |= (notA & B    & notC & D);
        if (lut & 0x0040) res |= (notA & B    & C    & notD);
        if (lut & 0x0080) res |= (notA & B    & C    & D);
        if (lut & 0x0100) res |= (A    & notB & notC & notD);
        if (lut & 0x0200) res |= (A    & notB & notC & D);
        if (lut & 0x0400) res |= (A    & notB & C    & notD);
        if (lut & 0x0800) res |= (A    & notB & C    & D);
        if (lut & 0x1000) res |= (A    & B    & notC & notD);
        if (lut & 0x2000) res |= (A    & B    & notC & D);
        if (lut & 0x4000) res |= (A    & B    & C    & notD);
        if (lut & 0x8000) res |= (A    & B    & C    & D);
        return res;
    }

    inline double Step(uint64_t in_spin, uint64_t target_spin) {
        head = (head + 127) & 127;
        ring[head] = in_spin;

        uint64_t A = ring[(head + tap_A) & 127];
        uint64_t B = ring[(head + tap_B) & 127];
        uint64_t C = ring[(head + tap_C) & 127];
        uint64_t D = ring[(head + tap_D) & 127];

        uint64_t pred = Evaluate16BitLUT(A, B, C, D, theta);
        int err = __builtin_popcountll(pred ^ target_spin);
        double fe = (double)err / 64.0;

        if (fe > 0.0) {
            // Sample/search candidates to find minimum error configuration
            uint16_t best_t = theta;
            int min_err = err;
            for (int cand = 0; cand < 65536; ++cand) {
                uint64_t p = Evaluate16BitLUT(A, B, C, D, (uint16_t)cand);
                int e = __builtin_popcountll(p ^ target_spin);
                if (e < min_err) {
                    min_err = e;
                    best_t = (uint16_t)cand;
                    if (e == 0) break;
                }
            }
            theta = best_t;
            fe = (double)min_err / 64.0;
        }
        return fe;
    }
};

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE MICRO-TEST 10: SCALING LIMITS & COMPLEXITY BENCHMARK\n";
    std::cout << "==================================================================================\n\n";

    size_t steps = 500000;
    uint64_t state = 0x123456789ABCDEF0ULL;

    // ------------------------------------------------------------------------
    // TEST 1: 4-Input Deterministic Logic Stream (XOR4: A ^ B ^ C ^ D)
    // ------------------------------------------------------------------------
    Universal4InputMutableSubstrate sub1;
    sub1.tap_A = 1; sub1.tap_B = 2; sub1.tap_C = 3; sub1.tap_D = 4;
    double fe_accum1 = 0.0;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;

        uint64_t A = sub1.ring[(sub1.head + 1) & 127];
        uint64_t B = sub1.ring[(sub1.head + 2) & 127];
        uint64_t C = sub1.ring[(sub1.head + 3) & 127];
        uint64_t D = sub1.ring[(sub1.head + 4) & 127];
        uint64_t target = A ^ B ^ C ^ D;

        double fe = sub1.Step(prev, target);
        if (i >= steps - 50000) fe_accum1 += fe;
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    double dur1 = std::chrono::duration<double>(end1 - start1).count();
    double fe1 = fe_accum1 / 50000.0;

    // ------------------------------------------------------------------------
    // TEST 2: Non-Stationary Target (Rule switches every 100,000 steps)
    // ------------------------------------------------------------------------
    Universal4InputMutableSubstrate sub2;
    sub2.tap_A = 1; sub2.tap_B = 2; sub2.tap_C = 3; sub2.tap_D = 4;
    double fe_accum2 = 0.0;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;

        uint64_t A = sub2.ring[(sub2.head + 1) & 127];
        uint64_t B = sub2.ring[(sub2.head + 2) & 127];
        uint64_t C = sub2.ring[(sub2.head + 3) & 127];
        uint64_t D = sub2.ring[(sub2.head + 4) & 127];

        uint64_t target;
        if ((i / 100000) % 2 == 0) {
            target = (A & B) | (C & D);
        } else {
            target = A ^ B ^ C ^ D;
        }

        double fe = sub2.Step(prev, target);
        if (i >= steps - 50000) fe_accum2 += fe;
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    double dur2 = std::chrono::duration<double>(end2 - start2).count();
    double fe2 = fe_accum2 / 50000.0;

    // ------------------------------------------------------------------------
    // TEST 3: Noisy Non-Deterministic Stream (5% random bit noise)
    // ------------------------------------------------------------------------
    Universal4InputMutableSubstrate sub3;
    sub3.tap_A = 1; sub3.tap_B = 2; sub3.tap_C = 3; sub3.tap_D = 4;
    double fe_accum3 = 0.0;

    std::mt19937_64 rng(42);
    std::bernoulli_dist_fast noise_dist(0.05); // 5% bit flip probability

    auto start3 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;

        uint64_t A = sub3.ring[(sub3.head + 1) & 127];
        uint64_t B = sub3.ring[(sub3.head + 2) & 127];
        uint64_t C = sub3.ring[(sub3.head + 3) & 127];
        uint64_t D = sub3.ring[(sub3.head + 4) & 127];
        
        uint64_t clean_target = (A & B) | (~C & D);
        
        // Inject 5% random bit noise
        uint64_t noise_mask = 0ULL;
        for (int b = 0; b < 64; ++b) {
            if (rng() % 100 < 5) noise_mask |= (1ULL << b);
        }
        uint64_t noisy_target = clean_target ^ noise_mask;

        double fe = sub3.Step(prev, noisy_target);
        if (i >= steps - 50000) fe_accum3 += fe;
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    double dur3 = std::chrono::duration<double>(end3 - start3).count();
    double fe3 = fe_accum3 / 50000.0;

    // ------------------------------------------------------------------------
    // RESULTS & TELEMETRY
    // ------------------------------------------------------------------------
    std::cout << "--- [MICRO-TEST 10 TELEMETRY RESULTS] ---\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "1. 4-Input Parity Stream (XOR4) : Final F = " << fe1 << " nats | Discovered Theta = 0x" 
              << std::hex << sub1.theta << std::dec << " | Time: " << dur1 << " s\n";
    std::cout << "2. Non-Stationary Shift Stream  : Final F = " << fe2 << " nats | Discovered Theta = 0x" 
              << std::hex << sub2.theta << std::dec << " | Time: " << dur2 << " s\n";
    std::cout << "3. Noisy Stream (5% Noise)      : Final F = " << fe3 << " nats | Discovered Theta = 0x" 
              << std::hex << sub3.theta << std::dec << " | Time: " << dur3 << " s\n\n";

    std::cout << "==================================================================================\n";
    return 0;
}
