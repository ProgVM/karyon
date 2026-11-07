// ============================================================================
// KARYON-CORE MICRO-TEST 2: DETERMINISTIC & PATTERNED WAVE RESOLUTION
// Goal: Benchmark 1D vs Dynamic-Dimension Wavefield on Structured/Patterned Streams
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>

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

        if (fe > 0.0) {
            tap1 = (tap1 % 31) + 1;
            rot = (rot + 1) & 63;
        }

        return fe;
    }
};

struct VariantB_DynamicDimString {
    uint64_t ring_buffer[128];
    size_t head;
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

        uint64_t combined_pred = 0ULL;
        for (size_t d = 0; d < active_dim; ++d) {
            uint64_t val = ring_buffer[(head + spatial_taps[d]) & 127];
            uint8_t r = phase_rotations[d];
            uint64_t rot_val = (r == 0) ? val : ((val << r) | (val >> (64 - r)));
            combined_pred ^= rot_val;
        }

        int err = __builtin_popcountll(combined_pred ^ target_spin);
        double fe = static_cast<double>(err) / 64.0;

        if (fe > 0.0) {
            if (active_dim < 8 && fe > 0.1) {
                spatial_taps.push_back(static_cast<uint8_t>(active_dim + 1));
                phase_rotations.push_back(static_cast<uint8_t>((active_dim * 7) & 63));
                active_dim++;
            } else {
                size_t d = active_dim - 1;
                spatial_taps[d] = (spatial_taps[d] % 31) + 1;
                phase_rotations[d] = (phase_rotations[d] + 1) & 63;
            }
        }

        return fe;
    }
};

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE MICRO-TEST 2: PATTERNED STREAM CONVERGENCE (1,000,000 STEPS)\n";
    std::cout << "==================================================================================\n\n";

    size_t steps = 1000000;
    uint64_t state = 0x123456789ABCDEF0ULL;

    // Patterned Stream Generator (Periodic Text/Formant Multi-Frequency Harmonic)
    auto generate_next = [](uint64_t curr, size_t idx) {
        uint64_t p1 = (curr << 13) | (curr >> 51);
        uint64_t p2 = (idx % 8 == 0) ? 0xAAAAAAAAAAAAAAAAULL : 0x5555555555555555ULL;
        return p1 ^ p2;
    };

    // Variant A
    VariantA_1DString var_a;
    double fe_late_a = 0.0;
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = generate_next(state, i);
        double fe = var_a.Step(prev, state);
        if (i >= steps - 100000) fe_late_a += fe;
    }
    fe_late_a /= 100000.0;

    // Variant B
    state = 0x123456789ABCDEF0ULL;
    VariantB_DynamicDimString var_b;
    double fe_late_b = 0.0;
    for (size_t i = 0; i < steps; ++i) {
        uint64_t prev = state;
        state = generate_next(state, i);
        double fe = var_b.Step(prev, state);
        if (i >= steps - 100000) fe_late_b += fe;
    }
    fe_late_b /= 100000.0;

    std::cout << "--- [PATTERNED CONVERGENCE RESULTS] ---\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Variant A (1D String Phase Line)   Late F: " << fe_late_a << " nats\n";
    std::cout << "Variant B (Dynamic Dimension)       Late F: " << fe_late_b << " nats (Sprouted D = " << var_b.active_dim << ")\n\n";

    std::cout << "==================================================================================\n";
    return 0;
}
