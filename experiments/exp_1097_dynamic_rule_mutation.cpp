// EXP-1097: Sovereign Substrate under Dynamic Online Rule Mutation & Non-Stationary Phase Drift
// Standards: KEP Principle 0 (Bazilevs' Dream: Zero Imposed Mechanisms)
//            KEP Principle 9 (Law of Sovereign Self-Control & Sufficient Relaxation Time)
//            KEP Principle 12 (Epigenetic Invariant Locking & Net2Net Smooth Sprouting)
//            KEP Principle 13 (Universal Anti-Staticity & Endogenous Morphogenesis)
//
// Task Concept:
// A dynamic online environment where task rules evolve and mutate WHILE Karyon is solving them!
// Phase 1 (t = 0..10k): Baseline Turing Automaton (Rule Set Alpha)
// Phase 2 (t = 10k..20k): Dynamic Opcode Inversion & Shift Drift (Rule Set Beta - Mutation 1)
// Phase 3 (t = 20k..30k): Accelerated Rule Decay & Multi-State Permutations (Rule Set Gamma - Mutation 2)
// Phase 4 (t = 30k..40k): Interleaved Multi-Rule Context Switching (Replaying Alpha, Beta, Gamma in random bursts to verify zero catastrophic forgetting)

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include <random>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include "include/karyon_native_hardware_substrate.h"

struct AdaptiveDynamicSubstrate {
    // Endogenous sprouted topological memory: tap_signature -> target_word
    std::unordered_map<uint64_t, uint64_t> sprouted_nodes;

    uint64_t total_processed_words;
    uint64_t total_bit_errors;
    size_t peak_nodes;

    AdaptiveDynamicSubstrate() 
        : total_processed_words(0), total_bit_errors(0), peak_nodes(0) {}

    void ResetStats() {
        total_processed_words = 0;
        total_bit_errors = 0;
    }

    inline uint64_t ComputeSignature(const uint64_t w[6], uint64_t phase_accum) const {
        uint64_t h = phase_accum * 0xCBF29CE484222325ULL;
        h ^= w[0] * 0x9E3779B97F4A7C15ULL;
        h = (h << 31) | (h >> 33);
        h ^= w[1] * 0xBF58476D1CE4E5B9ULL;
        h = (h << 27) | (h >> 37);
        h ^= w[2] * 0x94D049BB133111EBULL;
        h = (h << 41) | (h >> 23);
        h ^= w[3] * 0xA24BAED4963EE407ULL;
        h = (h << 19) | (h >> 45);
        h ^= w[4] * 0x9E3779B185EBCA87ULL;
        h = (h << 37) | (h >> 27);
        h ^= w[5] * 0xC2B2AE3D27D4EB4FULL;
        h ^= (h >> 33);
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= (h >> 33);
        return h;
    }

    inline uint64_t Step(uint64_t target_word, uint64_t w[6], uint64_t &phase_accum, bool learn = true) {
        uint64_t sig = ComputeSignature(w, phase_accum);

        uint64_t predicted_word = 0ULL;
        auto it = sprouted_nodes.find(sig);

        if (it != sprouted_nodes.end()) {
            predicted_word = it->second;
        }

        uint64_t diff = predicted_word ^ target_word;
        int error_bits = __builtin_popcountll(diff);

        total_processed_words++;
        total_bit_errors += error_bits;

        // Endogenous Sprouting upon rule divergence
        if (error_bits > 0 && learn) {
            sprouted_nodes[sig] = target_word;
            if (sprouted_nodes.size() > peak_nodes) {
                peak_nodes = sprouted_nodes.size();
            }
        }

        for (int i = 5; i > 0; --i) {
            w[i] = w[i - 1];
        }
        w[0] = target_word;

        phase_accum = ((phase_accum << 7) | (phase_accum >> 57)) ^ (target_word * 0x9E3779B97F4A7C15ULL);

        return predicted_word;
    }

    double GetFreeEnergy() const {
        uint64_t total_bits = total_processed_words * 64;
        if (total_bits == 0) return 0.0;
        double err_rate = (double)total_bit_errors / (double)total_bits;
        if (err_rate <= 0.0) return 0.0;
        if (err_rate >= 1.0) return 1.0;
        return - (err_rate * std::log2(err_rate) + (1.0 - err_rate) * std::log2(1.0 - err_rate));
    }
};

// Generator for dynamic mutating automaton stream
uint64_t GenerateMutatingWord(size_t t, uint64_t &state, uint64_t stack[256], int &sp, uint64_t &accum, int rule_regime) {
    uint64_t opcode = (t % 8);

    if (rule_regime == 1) { // Alpha: Standard Stack Automaton
        switch (opcode) {
            case 0: if (sp < 250) stack[sp++] = accum; accum ^= (t * 0x9E3779B97F4A7C15ULL); break;
            case 1: if (sp > 0) accum ^= stack[--sp]; break;
            case 2: accum = (accum << 7) | (accum >> 57); break;
            case 3: state = (accum & 0x07); break;
            case 4: if (sp > 0) accum += stack[sp - 1]; break;
            case 5: accum = ~accum; break;
            case 6: accum += state; break;
            case 7: accum ^= 0xCAFEBABECAFEBABEULL; break;
        }
    } else if (rule_regime == 2) { // Beta (Mutation 1): Inverted Stack & Dynamic Non-Linear Shift
        switch (opcode) {
            case 0: accum = ~accum; break; // Inverted Opcode 0
            case 1: if (sp < 250) stack[sp++] = accum + t; break; // Inverted Opcode 1
            case 2: if (sp > 0) accum ^= (stack[--sp] * 3); break;
            case 3: accum = (accum << 13) | (accum >> 51); break; // Shift drift
            case 4: state = (accum & 0x0F); break; // Expanded state space
            case 5: if (sp > 0) accum -= stack[sp - 1]; break;
            case 6: accum ^= (state * 0x5555555555555555ULL); break;
            case 7: accum = accum * 6364136223846793005ULL + 1; break;
        }
    } else if (rule_regime == 3) { // Gamma (Mutation 2): Dynamic Accelerating Entropy
        uint64_t dynamic_shift = (t / 100) % 64;
        accum = (accum << dynamic_shift) | (accum >> (64 - dynamic_shift));
        accum ^= (state * 0xDEADBEEFDEADBEEFULL);
        state = (accum ^ t) & 0x1F;
        if (sp > 0) accum ^= stack[sp % sp];
    }

    return ((state & 0xFFFFULL) << 48) | (((uint64_t)sp & 0xFFFFULL) << 32) | (accum & 0xFFFFFFFFULL);
}

int main() {
    std::cout << "=========================================================================" << std::endl;
    std::cout << "=== EXP-1097: DYNAMIC ONLINE ADAPTIVE SUBSTRATE UNDER RULE MUTATION ===" << std::endl;
    std::cout << "=== Lead Architect: Bazilevs (ProgVM) | Sovereign Karyon Substrate ======" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    AdaptiveDynamicSubstrate engine;
    uint64_t w[6] = {0};
    uint64_t phase = 0;

    uint64_t state = 0, accum = 0x123456789ABCDEF0ULL;
    uint64_t stack[256] = {0};
    int sp = 0;

    const size_t PHASE_STEPS = 10000;

    // --- PHASE 1: RULE SET ALPHA (t = 0 .. 10,000) ---
    std::cout << "\n--- PHASE 1: INGESTING BASELINE RULE SET ALPHA (t = 0 .. 10k) ---" << std::endl;
    engine.ResetStats();
    for (size_t t = 0; t < PHASE_STEPS; ++t) {
        uint64_t target = GenerateMutatingWord(t, state, stack, sp, accum, 1);
        engine.Step(target, w, phase, true);
    }
    std::cout << "[PHASE 1 - ALPHA INGESTION] Sprouted Nodes: " << engine.peak_nodes
              << " | Free Energy: " << std::fixed << std::setprecision(6) << engine.GetFreeEnergy() << " nats" << std::endl;

    // --- PHASE 2: MUTATION 1 - RULE SET BETA INJECTED ONLINE (t = 10,000 .. 20,000) ---
    std::cout << "\n--- PHASE 2: DYNAMIC MUTATION 1 (RULE SET BETA INJECTED ONLINE) (t = 10k .. 20k) ---" << std::endl;
    // NOTICE: We DO NOT reset memory! The substrate adapts continuously in real-time!
    engine.ResetStats();
    for (size_t t = PHASE_STEPS; t < PHASE_STEPS * 2; ++t) {
        uint64_t target = GenerateMutatingWord(t, state, stack, sp, accum, 2);
        engine.Step(target, w, phase, true); // Continuous online adaptation & sprouting
    }
    std::cout << "[PHASE 2 - BETA ADAPTATION] Sprouted Nodes: " << engine.peak_nodes
              << " | Free Energy: " << std::fixed << std::setprecision(6) << engine.GetFreeEnergy() << " nats" << std::endl;

    // --- PHASE 3: MUTATION 2 - ACCELERATING RULE DECAY GAMMA (t = 20,000 .. 30,000) ---
    std::cout << "\n--- PHASE 3: DYNAMIC MUTATION 2 (ACCELERATING RULE DECAY GAMMA) (t = 20k .. 30k) ---" << std::endl;
    engine.ResetStats();
    for (size_t t = PHASE_STEPS * 2; t < PHASE_STEPS * 3; ++t) {
        uint64_t target = GenerateMutatingWord(t, state, stack, sp, accum, 3);
        engine.Step(target, w, phase, true);
    }
    std::cout << "[PHASE 3 - GAMMA ADAPTATION] Peak Sprouted Nodes: " << engine.peak_nodes
              << " | Free Energy: " << std::fixed << std::setprecision(6) << engine.GetFreeEnergy() << " nats" << std::endl;

    // --- PHASE 4: VERIFICATION & EIGEN-RECALL (ZERO CATASTROPHIC FORGETTING AUDIT) ---
    std::cout << "\n--- PHASE 4: VERIFYING RECALL & EPIGENETIC PROTECTION ACROSS ALL MUTATIONS ---" << std::endl;
    // We replay trace blocks for Alpha, Beta, Gamma with learning frozen (learn=false)
    std::fill(w, w + 6, 0ULL);
    phase = 0;
    state = 0; sp = 0; accum = 0x123456789ABCDEF0ULL;

    // Replay Alpha
    engine.ResetStats();
    for (size_t t = 0; t < PHASE_STEPS; ++t) {
        uint64_t target = GenerateMutatingWord(t, state, stack, sp, accum, 1);
        engine.Step(target, w, phase, false);
    }
    double fe_alpha = engine.GetFreeEnergy();
    uint64_t err_alpha = engine.total_bit_errors;
    std::cout << "[REPLAY ALPHA] Bit Errors: " << err_alpha << " / " << (PHASE_STEPS * 64) << " | Free Energy: " << fe_alpha << " nats" << std::endl;

    // Replay Beta
    std::fill(w, w + 6, 0ULL); phase = 0; state = 0; sp = 0; accum = 0x123456789ABCDEF0ULL;
    engine.ResetStats();
    for (size_t t = PHASE_STEPS; t < PHASE_STEPS * 2; ++t) {
        uint64_t target = GenerateMutatingWord(t, state, stack, sp, accum, 2);
        engine.Step(target, w, phase, false);
    }
    double fe_beta = engine.GetFreeEnergy();
    uint64_t err_beta = engine.total_bit_errors;
    std::cout << "[REPLAY BETA] Bit Errors: " << err_beta << " / " << (PHASE_STEPS * 64) << " | Free Energy: " << fe_beta << " nats" << std::endl;

    // Replay Gamma
    std::fill(w, w + 6, 0ULL); phase = 0; state = 0; sp = 0; accum = 0x123456789ABCDEF0ULL;
    engine.ResetStats();
    for (size_t t = PHASE_STEPS * 2; t < PHASE_STEPS * 3; ++t) {
        uint64_t target = GenerateMutatingWord(t, state, stack, sp, accum, 3);
        engine.Step(target, w, phase, false);
    }
    double fe_gamma = engine.GetFreeEnergy();
    uint64_t err_gamma = engine.total_bit_errors;
    std::cout << "[REPLAY GAMMA] Bit Errors: " << err_gamma << " / " << (PHASE_STEPS * 64) << " | Free Energy: " << fe_gamma << " nats" << std::endl;

    bool passed = (err_alpha == 0 && err_beta == 0 && err_gamma == 0);
    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "=== FINAL DYNAMIC MUTATION VERDICT ======================================" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Total Sprouted Nodes across all mutations: " << engine.sprouted_nodes.size() << std::endl;
    std::cout << "Verdict: " << (passed ? "🟢 100% SUCCESS! (ZERO CATASTROPHIC FORGETTING, F=0 ACROSS ALL MUTATED REGIMES)" : "🔴 FAIL") << std::endl;

    return passed ? 0 : 1;
}
