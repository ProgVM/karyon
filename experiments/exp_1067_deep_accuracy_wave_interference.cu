// ============================================================================
// KARYON-CORE EXP-1067: ULTRA-LOW FREE ENERGY CONVERGENCE (F -> 0.000x)
// Substrate: Pure Sovereign Monad V=2 (Superposition of Dynamic Scalar-Phase Wavefields)
// Standard: Bazilevs' Axiom & KEP Principle 9 (Zero Imposed Mechanisms + Sufficient Relaxation)
// Goal: Push non-deterministic & semi-deterministic Free Energy error down to 0.0000x!
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <random>

#include "include/karyon_hardware.h"
#include "include/karyon_monad_v2.h"

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1067: DEEP ACCURACY & ULTRA-LOW FREE ENERGY AUDIT (F -> 0.000x)\n";
    std::cout << "  Substrate : Sovereign Monad V=2 (Pure Autopoietic Superposition)\n";
    std::cout << "  Standard  : Bazilevs' Axiom (Zero Error Standard on Deterministic & Semi-Det)\n";
    std::cout << "==================================================================================\n\n";

    karyon::SovereignMonadV2 monad(128, 128);
    std::mt19937_64 rng(42);

    // ------------------------------------------------------------------------
    // TEST 1: DETERMINISTIC STREAM CONVERGENCE AUDIT
    // ------------------------------------------------------------------------
    std::cout << "[TEST 1: DETERMINISTIC STREAM AUDIT]\n";
    std::vector<uint64_t> det_stream(100000);
    uint64_t state = 0x1234567890ABCDEFULL;
    for (size_t i = 0; i < 100000; ++i) {
        state = ((state << 13) | (state >> 51)) ^ 0x9E3779B97F4A7C15ULL;
        det_stream[i] = state;
    }

    double det_fe_accum = 0.0;
    size_t det_eval_count = 0;
    for (size_t i = 1; i < det_stream.size(); ++i) {
        double fe = monad.Step(det_stream[i - 1], det_stream[i], i);
        if (i > 20000) {
            det_fe_accum += fe;
            det_eval_count++;
        }
    }
    double det_final_fe = det_fe_accum / det_eval_count;
    std::cout << "  Deterministic Free Energy F : " << std::fixed << std::setprecision(8) << det_final_fe << " nats\n";
    std::cout << "  Deterministic Status        : " << (det_final_fe == 0.0 ? "🟢 PERFECT ZERO (F = 0.00000000)" : "🔴 FAILED") << "\n\n";

    // ------------------------------------------------------------------------
    // TEST 2: NOISE-INJECTED SEMI-DETERMINISTIC STREAM AUDIT WITH SUFFICIENT RELAXATION
    // ------------------------------------------------------------------------
    std::cout << "[TEST 2: NOISY SEMI-DETERMINISTIC STREAM AUDIT (SUFFICIENT RELAXATION TIME)]\n";
    std::vector<uint64_t> noisy_stream(200000);
    state = 0x1234567890ABCDEFULL;
    for (size_t i = 0; i < 200000; ++i) {
        state = ((state << 13) | (state >> 51)) ^ 0x9E3779B97F4A7C15ULL;
        if (i % 100 == 0) state ^= (rng() & 0x000000000000000FULL); // 4-bit noise perturbation
        noisy_stream[i] = state;
    }

    karyon::SovereignMonadV2 monad_noisy(128, 128);
    double noisy_fe_accum = 0.0;
    size_t noisy_eval_count = 0;

    for (size_t i = 1; i < noisy_stream.size(); ++i) {
        double fe = monad_noisy.Step(noisy_stream[i - 1], noisy_stream[i], i);
        if (i > 100000) { // Evaluate after sufficient relaxation time
            noisy_fe_accum += fe;
            noisy_eval_count++;
        }
    }
    double noisy_final_fe = noisy_fe_accum / noisy_eval_count;
    std::cout << "  Noisy Stream Free Energy F  : " << std::fixed << std::setprecision(8) << noisy_final_fe << " nats\n";
    std::cout << "  Active Dynamic Couplings    : " << monad_noisy.GetActiveCouplingsCount() << "\n";
    std::cout << "  Epigenetic Locked Invariants: " << monad_noisy.GetTotalLocks() << "\n";
    std::cout << "  Noisy Stream Convergence    : " << (noisy_final_fe < 0.01 ? "🟢 ULTRA-LOW ERROR (F < 0.01)" : "🟡 NEEDS DEEPER WAVE INTERFERENCE") << "\n\n";

    return 0;
}
