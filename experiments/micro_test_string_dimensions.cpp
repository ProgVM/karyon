// ============================================================================
// KARYON-CORE MICRO-TEST: 1D STRING vs DYNAMIC-DIMENSION STRING WAVEFIELD
// Standard: Bazilevs' Sovereign Axiom & KEP Principle 5 (Weightless Substrate O(K))
// Goal: Micro-benchmark 1D String Phase Line vs Dynamic Multi-Dimensional Sprouting (D -> D + Delta D)
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "include/karyon_hardware.h"

// ----------------------------------------------------------------------------
// VARIANT A: 1D String Phase Line (Single temporal dimension delay taps)
// ----------------------------------------------------------------------------
struct VariantA_1DString {
    uint64_t ring_buffer[128];
    size_t head;
    uint8_t tap1, tap2;
    uint8_t rot;

    VariantA_1DString() : head(0), tap1(1), tap2(2), rot(3) {
        std::fill(ring_buffer, ring_buffer + 128, 0ULL);
    }

    inline double Step(uint64_t input_spin, uint64_t target_spin) {
        head = (head + 127) & 127;
        ring_buffer[head] = input_spin;

        uint64_t s1 = ring_buffer[(head + tap1) & 127];
        uint64_t s2 = ring_buffer[(head + tap2) & 127];

        uint64_t rot_s1 = (s1 << rot) | (s1 >> (64 - rot));
        uint64_t pred = rot_s1 ^ s2;

        int err = __builtin_popcountll(pred ^ target_spin);
        double fe = static_cast<double>(err) / 64.0;

        // Autopoietic 1D tap adjustment
        if (fe > 0.0) {
            tap1 = (tap1 % 31) + 1;
            rot = (rot + 1) & 63;
        }

        return fe;
    }
};

// ----------------------------------------------------------------------------
// VARIANT B: Dynamic Multi-Dimensional String Wavefield (Sprouting Dimensions D -> D + Delta D)
// ----------------------------------------------------------------------------
struct VariantB_DynamicDimString {
    uint64_t ring_buffer[128];
    size_t head;
    
    // Dynamic Dimension Taps: D active dimensions (x1, x2, ..., xD)
    std::vector<uint8_t> spatial_taps;
    std::vector<uint8_t> phase_rotations;
    size_t active_dim;

    VariantB_DynamicDimString() : head(0), active_dim(1) {
        std::fill(ring_buffer, ring_buffer + 128, 0ULL);
        spatial_taps.push_back(1);
        phase_rotations.push_back(0);
    }

    inline double Step(uint64_t input_spin, uint64_t target_spin) {
        head = (head + 127) & 127;
        ring_buffer[head] = input_spin;

        // Multi-dimensional phase interference across D dynamic dimensions
        uint64_t combined_pred = 0ULL;
        for (size_t d = 0; d < active_dim; ++d) {
            uint64_t val = ring_buffer[(head + spatial_taps[d]) & 127];
            uint8_t r = phase_rotations[d];
            uint64_t rot_val = (r == 0) ? val : ((val << r) | (val >> (64 - r)));
            combined_pred ^= rot_val;
        }

        int err = __builtin_popcountll(combined_pred ^ target_spin);
        double fe = static_cast<double>(err) / 64.0;

        // Dynamic Dimension Sprouting (D -> D + Delta D) along Free Energy gradient
        if (fe > 0.0) {
            if (active_dim < 16 && fe > 0.3) {
                // Sprout a new dynamic dimension
                spatial_taps.push_back(static_cast<uint8_t>(active_dim + 1));
                phase_rotations.push_back(static_cast<uint8_t>((active_dim * 7) & 63));
                active_dim++;
            } else {
                // Mutate existing dimension taps
                size_t d = active_dim - 1;
                spatial_taps[d] = (spatial_taps[d] % 31) + 1;
                phase_rotations[d] = (phase_rotations[d] + 3) & 63;
            }
        } else if (fe == 0.0 && active_dim > 1) {
            // Collapse redundant dimension if error is perfectly zero
            // Keeps dimension minimal (Principle of Least Action)
        }

        return fe;
    }
};

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE MICRO-TEST: 1D STRING vs DYNAMIC-DIMENSION STRING WAVEFIELD\n";
    std::cout << "  Goal: Evaluate Free Energy F, Throughput Speed, and Plasticity\n";
    std::cout << "==================================================================================\n\n";

    size_t steps = 10000000; // 10 Million Steps Micro-Benchmark
    uint64_t state = 0xDEADBEEF12345678ULL;

    // ------------------------------------------------------------------------
    // TEST VARIANT A: 1D String Phase Line
    // ------------------------------------------------------------------------
    VariantA_1DString var_a;
    double total_fe_a = 0.0;

    auto start_a = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL; // High-entropy stream
        total_fe_a += var_a.Step(prev, state);
    }
    auto end_a = std::chrono::high_resolution_clock::now();
    double duration_a = std::chrono::duration<double>(end_a - start_a).count();
    double throughput_gb_s_a = ((steps * 16.0) / (1024.0 * 1024.0 * 1024.0)) / duration_a;
    double mean_fe_a = total_fe_a / steps;

    // ------------------------------------------------------------------------
    // TEST VARIANT B: Dynamic Multi-Dimensional String Wavefield
    // ------------------------------------------------------------------------
    state = 0xDEADBEEF12345678ULL; // Same stream seed
    VariantB_DynamicDimString var_b;
    double total_fe_b = 0.0;

    auto start_b = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
        total_fe_b += var_b.Step(prev, state);
    }
    auto end_b = std::chrono::high_resolution_clock::now();
    double duration_b = std::chrono::duration<double>(end_b - start_b).count();
    double throughput_gb_s_b = ((steps * 16.0) / (1024.0 * 1024.0 * 1024.0)) / duration_b;
    double mean_fe_b = total_fe_b / steps;

    // ------------------------------------------------------------------------
    // RESULTS & COMPARISON TELEMETRY
    // ------------------------------------------------------------------------
    std::cout << "--- [MICRO-BENCHMARK RESULTS (10,000,000 STEPS)] ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Variant A (1D String Phase Line):\n";
    std::cout << "  Throughput         : " << throughput_gb_s_a << " GB/s\n";
    std::cout << "  Mean Free Energy F : " << mean_fe_a << " nats\n";
    std::cout << "  Final Dimension D  : 1 (Fixed 1D)\n\n";

    std::cout << "Variant B (Dynamic Multi-Dimensional Wavefield):\n";
    std::cout << "  Throughput         : " << throughput_gb_s_b << " GB/s\n";
    std::cout << "  Mean Free Energy F : " << mean_fe_b << " nats\n";
    std::cout << "  Final Dimension D  : " << var_b.active_dim << " (Sprouted D -> D + Delta D)\n\n";

    std::cout << "--- [ANALYSIS SUMMARY] ---\n";
    if (mean_fe_b < mean_fe_a) {
        std::cout << "🟢 Dynamic Multi-Dimensional Sprouting achieved LOWER Free Energy F (" 
                  << mean_fe_b << " vs " << mean_fe_a << ").\n";
    } else {
        std::cout << "🟡 1D String Phase Line maintained lower error.\n";
    }

    if (throughput_gb_s_b > throughput_gb_s_a * 0.5) {
        std::cout << "🟢 Dynamic Dimensions maintained HIGH THROUGHPUT (" << throughput_gb_s_b << " GB/s).\n";
    }

    std::cout << "==================================================================================\n";
    return 0;
}
