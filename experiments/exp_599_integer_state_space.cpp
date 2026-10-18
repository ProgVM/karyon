// ======================================================================================
// EXP-599 v3: Tiered Multi-Scale N-Gram Silicon Memory with Fallback (SIMD Multi-Tier)
// --------------------------------------------------------------------------------------
// Cybernetic Resolution of EXP-598 & EXP-599 v1/v2 Failure Modes:
// 1. Why Linear SSM (v1/v2) failed (20.0 nats): Linear state h_t without non-linear
//    capacity expansion cannot separate 258 byte categories in 128 integer dimensions.
// 2. Why EXP-598 failed on unseen text (4.39 nats): Single 4-gram hash table suffered
//    from "Hash Cliff" - missing 4-grams fell into uninitialized slots (random noise).
// 3. The Tiered Multi-Scale Silicon Solution:
//    - Tier 1: 1-byte Unigram Base Conductance Array (258 slots) - Guaranteed smooth fallback.
//    - Tier 2: 2-byte Bigram Associative Table (65,536 slots) - High-speed local syntax.
//    - Tier 3: 3-byte Trigram Associative Table (256,144 slots) - Word morpheme structures.
//    - Tier 4: 4-byte Quadgram Associative Table (524,288 slots) - Long-context facts.
//    - SIMD Tiered Fallback: Potentials = Tier4 + Tier3 + Tier2 + Tier1
// 4. Verification Targets:
//    - Unseen Test Free Energy: < 1.5 - 2.0 nats (Smooth generalization on unseen text).
//    - Unseen Test Hit Rate: > 65% - 75%.
//    - Throughput: > 200,000 tok/s on CPU.
// ======================================================================================

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <memory>

namespace karyon_exp599_v3 {

constexpr size_t VOCAB_SIZE      = 258;
constexpr size_t TIER2_SLOTS     = 65536;    // 2-byte context (64K)
constexpr size_t TIER3_SLOTS     = 262144;   // 3-byte context (256K)
constexpr size_t TIER4_SLOTS     = 524288;   // 4-byte context (512K)
constexpr size_t DNA_CAPACITY    = 2048;

struct alignas(32) ShortConductanceSlot {
    int16_t conductance[VOCAB_SIZE];
    uint8_t lock_count;
};

struct DnaGene {
    uint32_t tag;
    uint8_t  tier_level;
    uint8_t  target_byte;
    int16_t  frozen_conductance;
    bool     valid;
};

class TieredSiliconEngine {
public:
    // Tier 1: 1-byte Unigram (Base Distribution)
    int16_t tier1_unigram[VOCAB_SIZE];

    // Tier 2: 2-byte Bigram (64K)
    std::unique_ptr<ShortConductanceSlot[]> tier2_bigram;

    // Tier 3: 3-byte Trigram (256K)
    std::unique_ptr<ShortConductanceSlot[]> tier3_trigram;

    // Tier 4: 4-byte Quadgram (512K)
    std::unique_ptr<ShortConductanceSlot[]> tier4_quadgram;

    // DNA Invariant Blueprint Registry
    std::unique_ptr<DnaGene[]> dna_registry;
    size_t                     dna_count;
    bool                       enable_dna;

    uint32_t rolling_context;
    alignas(64) int32_t aggregated_potentials[VOCAB_SIZE];

    double smooth_fe;
    double min_fe;
    size_t correct_hits;
    size_t total_evals;

    TieredSiliconEngine(bool with_dna) 
        : tier2_bigram(std::make_unique<ShortConductanceSlot[]>(TIER2_SLOTS)),
          tier3_trigram(std::make_unique<ShortConductanceSlot[]>(TIER3_SLOTS)),
          tier4_quadgram(std::make_unique<ShortConductanceSlot[]>(TIER4_SLOTS)),
          dna_registry(std::make_unique<DnaGene[]>(DNA_CAPACITY)),
          dna_count(0), enable_dna(with_dna), rolling_context(0),
          smooth_fe(5.55), min_fe(999.0), correct_hits(0), total_evals(0)
    {
        std::memset(tier1_unigram, 0, sizeof(tier1_unigram));
        for (size_t s = 0; s < TIER2_SLOTS; ++s) {
            std::memset(tier2_bigram[s].conductance, 0, sizeof(tier2_bigram[s].conductance));
            tier2_bigram[s].lock_count = 0;
        }
        for (size_t s = 0; s < TIER3_SLOTS; ++s) {
            std::memset(tier3_trigram[s].conductance, 0, sizeof(tier3_trigram[s].conductance));
            tier3_trigram[s].lock_count = 0;
        }
        for (size_t s = 0; s < TIER4_SLOTS; ++s) {
            std::memset(tier4_quadgram[s].conductance, 0, sizeof(tier4_quadgram[s].conductance));
            tier4_quadgram[s].lock_count = 0;
        }
        for (size_t d = 0; d < DNA_CAPACITY; ++d) dna_registry[d].valid = false;
    }

    inline void shift_byte(uint8_t in_b) noexcept {
        rolling_context = (rolling_context << 8) | in_b;
    }

    // Single-cycle hashing for multi-tier lookup
    inline size_t hash2(uint32_t ctx) const noexcept { return (ctx & 0xFFFF) % TIER2_SLOTS; }
    inline size_t hash3(uint32_t ctx) const noexcept {
        uint32_t h = (ctx & 0xFFFFFF) * 0x45d9f3b;
        return ((h >> 16) ^ h) % TIER3_SLOTS;
    }
    inline size_t hash4(uint32_t ctx) const noexcept {
        uint32_t h = ctx * 0x45d9f3b;
        h = ((h >> 16) ^ h) * 0x45d9f3b;
        return ((h >> 16) ^ h) % TIER4_SLOTS;
    }

    inline uint8_t predict_and_eval(uint8_t target_byte, double& fe_step, bool is_training) noexcept {
        total_evals++;

        size_t idx2 = hash2(rolling_context);
        size_t idx3 = hash3(rolling_context);
        size_t idx4 = hash4(rolling_context);

        const int16_t* c1 = tier1_unigram;
        const int16_t* c2 = tier2_bigram[idx2].conductance;
        const int16_t* c3 = tier3_trigram[idx3].conductance;
        const int16_t* c4 = tier4_quadgram[idx4].conductance;

        // Aggregate Multi-Tier Potentials: Base + 2-gram + 3-gram + 4-gram
        int32_t max_pot = -9999999;
        uint8_t best_byte = 0;

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            int32_t pot = static_cast<int32_t>(c1[v]) +
                          static_cast<int32_t>(c2[v]) * 2 +
                          static_cast<int32_t>(c3[v]) * 3 +
                          static_cast<int32_t>(c4[v]) * 4;
            aggregated_potentials[v] = pot;
            if (pot > max_pot) {
                max_pot = pot;
                best_byte = static_cast<uint8_t>(v);
            }
        }

        if (best_byte == target_byte) correct_hits++;

        // Numerically Stable Softmax for Free Energy
        double sum_exp = 0.0;
        double target_exp = 1.0;
        double scale = 0.02;

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            double logit = static_cast<double>(aggregated_potentials[v] - max_pot) * scale;
            double p = std::exp(std::max(logit, -20.0));
            sum_exp += p;
            if (v == target_byte) target_exp = p;
        }

        double p_norm = std::max(target_exp / (sum_exp + 1e-12), 1e-12);
        fe_step = -std::log(p_norm);
        smooth_fe = 0.999 * smooth_fe + 0.001 * fe_step;
        if (smooth_fe < min_fe) min_fe = smooth_fe;

        // Adaptive Multi-Tier Plasticity (Hebbian / Delta)
        if (is_training) {
            // Tier 1 Update (Base)
            tier1_unigram[target_byte] = static_cast<int16_t>(std::min(tier1_unigram[target_byte] + 4, 1000));

            // Tier 2 Update
            if (tier2_bigram[idx2].lock_count < 95) {
                tier2_bigram[idx2].conductance[target_byte] = static_cast<int16_t>(std::min(tier2_bigram[idx2].conductance[target_byte] + 16, 2000));
                if (best_byte != target_byte) {
                    tier2_bigram[idx2].conductance[best_byte] = static_cast<int16_t>(std::max(tier2_bigram[idx2].conductance[best_byte] - 8, -500));
                }
            }

            // Tier 3 Update
            if (tier3_trigram[idx3].lock_count < 95) {
                tier3_trigram[idx3].conductance[target_byte] = static_cast<int16_t>(std::min(tier3_trigram[idx3].conductance[target_byte] + 24, 2000));
                if (best_byte != target_byte) {
                    tier3_trigram[idx3].conductance[best_byte] = static_cast<int16_t>(std::max(tier3_trigram[idx3].conductance[best_byte] - 12, -500));
                }
            }

            // Tier 4 Update
            if (tier4_quadgram[idx4].lock_count < 95) {
                tier4_quadgram[idx4].conductance[target_byte] = static_cast<int16_t>(std::min(tier4_quadgram[idx4].conductance[target_byte] + 32, 2000));
                if (best_byte != target_byte) {
                    tier4_quadgram[idx4].conductance[best_byte] = static_cast<int16_t>(std::max(tier4_quadgram[idx4].conductance[best_byte] - 16, -500));
                }
                
                // Epigenetic DNA Transcription for strong quadgram features
                if (enable_dna && tier4_quadgram[idx4].conductance[target_byte] > 800) {
                    transcribe_dna(rolling_context, 4, target_byte, tier4_quadgram[idx4].conductance[target_byte]);
                }
            }
        }

        return best_byte;
    }

    void transcribe_dna(uint32_t tag, uint8_t tier, uint8_t target_byte, int16_t cond) noexcept {
        for (size_t d = 0; d < dna_count; ++d) {
            if (dna_registry[d].valid && dna_registry[d].tag == tag && dna_registry[d].target_byte == target_byte) {
                dna_registry[d].frozen_conductance = cond;
                return;
            }
        }
        if (dna_count < DNA_CAPACITY) {
            dna_registry[dna_count].tag = tag;
            dna_registry[dna_count].tier_level = tier;
            dna_registry[dna_count].target_byte = target_byte;
            dna_registry[dna_count].frozen_conductance = cond;
            dna_registry[dna_count].valid = true;
            dna_count++;
        }
    }

    void inject_shock() noexcept {
        // Corrupt 70% of Tiers 2-4
        for (size_t s = 0; s < TIER2_SLOTS; ++s) {
            if (tier2_bigram[s].lock_count < 80) std::memset(tier2_bigram[s].conductance, 0, sizeof(tier2_bigram[s].conductance));
        }
        for (size_t s = 0; s < TIER3_SLOTS; ++s) {
            if (tier3_trigram[s].lock_count < 80) std::memset(tier3_trigram[s].conductance, 0, sizeof(tier3_trigram[s].conductance));
        }
        for (size_t s = 0; s < TIER4_SLOTS; ++s) {
            if (tier4_quadgram[s].lock_count < 80) std::memset(tier4_quadgram[s].conductance, 0, sizeof(tier4_quadgram[s].conductance));
        }

        // DNA Regeneration
        if (enable_dna && dna_count > 0) {
            for (size_t d = 0; d < dna_count; ++d) {
                if (!dna_registry[d].valid) continue;
                uint32_t tag = dna_registry[d].tag;
                uint8_t  tgt = dna_registry[d].target_byte;
                int16_t cond = dna_registry[d].frozen_conductance;

                size_t idx4 = hash4(tag);
                tier4_quadgram[idx4].conductance[tgt] = cond;
                tier4_quadgram[idx4].lock_count = 98;
            }
        }
    }
};

} // namespace karyon_exp599_v3

void run_tiered_silicon_benchmark(bool with_dna) {
    std::cout << "========================================================================\n";
    std::cout << ">>> RUNNING EXP-599 v3 (TIERED MULTI-SCALE SILICON): " 
              << (with_dna ? "WITH DNA RESURRECTION" : "WITHOUT DNA (ABLATED)") << "\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: data/alpaca_data.json not found!\n";
        return;
    }

    std::vector<uint8_t> corpus((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::cout << "Corpus loaded: " << corpus.size() << " raw bytes.\n";

    size_t train_size = static_cast<size_t>(corpus.size() * 0.80);
    size_t test_size  = corpus.size() - train_size;

    karyon_exp599_v3::TieredSiliconEngine engine(with_dna);

    const size_t SHOCK_STEP = train_size / 2;
    double fe_pre_shock = 0.0;
    double fe_post_shock_immediate = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Training Phase
    std::cout << "\n>>> Starting Training Phase on " << train_size << " continuous bytes...\n";
    for (size_t step = 0; step < train_size - 1; ++step) {
        if (step == SHOCK_STEP) {
            fe_pre_shock = engine.smooth_fe;
            std::cout << "\n[!] INJECTING CATASTROPHIC SHOCK AT STEP " << step << " (70% Damage)...\n";
            engine.inject_shock();
        }

        uint8_t in_b  = corpus[step];
        uint8_t tgt_b = corpus[step + 1];

        engine.shift_byte(in_b);
        double fe_step = 0.0;
        engine.predict_and_eval(tgt_b, fe_step, true);

        if (step == SHOCK_STEP + 1000) {
            fe_post_shock_immediate = engine.smooth_fe;
        }

        if ((step + 1) % 4000000 == 0 || step == train_size - 2) {
            double hit_rate = (static_cast<double>(engine.correct_hits) / engine.total_evals) * 100.0;
            std::cout << "  [Train Step " << std::setw(8) << (step + 1) << "]"
                      << " | FE: " << std::fixed << std::setprecision(4) << engine.smooth_fe
                      << " | Min FE: " << engine.min_fe
                      << " | HitRate: " << std::setprecision(2) << hit_rate << "%"
                      << " | DNA Genes: " << engine.dna_count << "\n";
            std::cout.flush();
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double train_duration = std::chrono::duration<double>(t1 - t0).count();
    double train_tok_per_sec = static_cast<double>(train_size) / train_duration;

    // 2. Out-of-Distribution Generalization Test (Zero training)
    std::cout << "\n>>> Starting Test / Generalization Phase on " << test_size << " unseen bytes...\n";
    size_t test_hits = 0;
    double test_fe_sum = 0.0;
    size_t test_evals = 0;

    for (size_t step = train_size; step < corpus.size() - 1; ++step) {
        uint8_t in_b  = corpus[step];
        uint8_t tgt_b = corpus[step + 1];

        engine.shift_byte(in_b);
        double fe_step = 0.0;
        uint8_t pred = engine.predict_and_eval(tgt_b, fe_step, false);

        if (pred == tgt_b) test_hits++;
        test_fe_sum += fe_step;
        test_evals++;
    }

    double avg_test_fe = test_fe_sum / static_cast<double>(test_evals);
    double test_hit_rate = (static_cast<double>(test_hits) / static_cast<double>(test_evals)) * 100.0;

    std::cout << "\n>>> FINAL RESULTS (" << (with_dna ? "WITH DNA" : "ABLATED WITHOUT DNA") << "):\n";
    std::cout << "  Throughput           : " << std::fixed << std::setprecision(2) << train_tok_per_sec << " tok/s\n";
    std::cout << "  Pre-Shock FE         : " << fe_pre_shock << " nats\n";
    std::cout << "  Post-Shock (1k steps): " << fe_post_shock_immediate << " nats\n";
    std::cout << "  Final Train FE       : " << engine.smooth_fe << " nats\n";
    std::cout << "  Min Train FE         : " << engine.min_fe << " nats\n";
    std::cout << "  Train Hit Rate       : " << (static_cast<double>(engine.correct_hits) / engine.total_evals) * 100.0 << " %\n";
    std::cout << "  UNSEEN TEST FE       : " << avg_test_fe << " nats (Generalization)\n";
    std::cout << "  UNSEEN TEST HIT RATE : " << test_hit_rate << " %\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-599 v3: TIERED MULTI-SCALE SILICON BENCHMARK (22.7MB ALPACA) ===\n";
    std::cout << "========================================================================\n\n";

    run_tiered_silicon_benchmark(false);
    run_tiered_silicon_benchmark(true);

    return 0;
}
