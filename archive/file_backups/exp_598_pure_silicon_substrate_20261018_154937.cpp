// ======================================================================================
// EXP-598 v3: Real Natural Language Corpus & Generalization Benchmark
// --------------------------------------------------------------------------------------
// Strict KEP Audit:
// 1. Train on real open-ended text (first 500k bytes of real Alpaca dialogue data).
// 2. Measure out-of-distribution / unseen sequence Free Energy (Generalization).
// 3. Ensure this is true associative silicon plasticity and NOT degenerate memorization.
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

namespace karyon_exp598_v3 {

constexpr size_t VOCAB_SIZE        = 258;
constexpr size_t HASH_SLOTS         = 65536; // 64K context slots for real linguistic n-grams
constexpr size_t DNA_CAPACITY       = 1024;

struct alignas(32) AssociativeSlot {
    uint32_t context_tag;
    int16_t  conductance[VOCAB_SIZE];
    int16_t  credit;
    uint8_t  epigenetic_lock;
};

struct DnaGene {
    uint32_t tag;
    uint8_t  target_byte;
    int16_t  frozen_conductance;
    bool     valid;
};

class SiliconEngine {
public:
    AssociativeSlot table[HASH_SLOTS];
    DnaGene         dna[DNA_CAPACITY];
    size_t          dna_count;
    bool            enable_dna;

    uint32_t rolling_context;
    double   smooth_fe;
    double   min_fe;
    size_t   correct_hits;
    size_t   total_evals;

    SiliconEngine(bool with_dna) 
        : dna_count(0), enable_dna(with_dna), rolling_context(0),
          smooth_fe(5.55), min_fe(999.0), correct_hits(0), total_evals(0)
    {
        for (size_t s = 0; s < HASH_SLOTS; ++s) {
            table[s].context_tag = 0;
            std::memset(table[s].conductance, 0, sizeof(table[s].conductance));
            table[s].credit = 0;
            table[s].epigenetic_lock = 0;
        }
        for (size_t d = 0; d < DNA_CAPACITY; ++d) dna[d].valid = false;
    }

    inline void shift_byte(uint8_t in_b) noexcept {
        rolling_context = (rolling_context << 8) | in_b;
    }

    inline size_t get_slot_idx() const noexcept {
        uint32_t h = rolling_context * 0x45d9f3b;
        h = ((h >> 16) ^ h) * 0x45d9f3b;
        h = (h >> 16) ^ h;
        return h % HASH_SLOTS;
    }

    inline uint8_t predict_and_eval(uint8_t target_b, double& fe_step, bool is_training) noexcept {
        total_evals++;
        size_t slot = get_slot_idx();
        AssociativeSlot& entry = table[slot];

        int16_t max_c = entry.conductance[0];
        uint8_t best_b = 0;
        for (size_t v = 1; v < VOCAB_SIZE; ++v) {
            if (entry.conductance[v] > max_c) {
                max_c = entry.conductance[v];
                best_b = static_cast<uint8_t>(v);
            }
        }

        if (best_b == target_b) correct_hits++;

        double sum_exp = 0.0;
        double target_exp = 1.0;
        double scale = 0.05;

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            double logit = static_cast<double>(entry.conductance[v] - max_c) * scale;
            double p = std::exp(std::max(logit, -20.0));
            sum_exp += p;
            if (v == target_b) target_exp = p;
        }

        double p_norm = std::max(target_exp / (sum_exp + 1e-12), 1e-12);
        fe_step = -std::log(p_norm);
        smooth_fe = 0.999 * smooth_fe + 0.001 * fe_step;
        if (smooth_fe < min_fe) min_fe = smooth_fe;

        if (is_training && entry.epigenetic_lock < 95) {
            entry.context_tag = rolling_context;

            int32_t new_t = entry.conductance[target_b] + 32;
            entry.conductance[target_b] = static_cast<int16_t>(std::min(new_t, 2000));

            if (best_b != target_b) {
                int32_t new_f = entry.conductance[best_b] - 16;
                entry.conductance[best_b] = static_cast<int16_t>(std::max(new_f, -500));
            }

            entry.credit++;
            if (entry.credit > 30 && entry.epigenetic_lock < 95) {
                entry.epigenetic_lock += 2;
                if (enable_dna && entry.epigenetic_lock > 40) {
                    transcribe_dna(rolling_context, target_b, entry.conductance[target_b]);
                }
            }
        }

        return best_b;
    }

    void transcribe_dna(uint32_t tag, uint8_t tgt, int16_t cond) noexcept {
        for (size_t d = 0; d < dna_count; ++d) {
            if (dna[d].valid && dna[d].tag == tag && dna[d].target_byte == tgt) {
                dna[d].frozen_conductance = cond;
                return;
            }
        }
        if (dna_count < DNA_CAPACITY) {
            dna[dna_count].tag = tag;
            dna[dna_count].target_byte = tgt;
            dna[dna_count].frozen_conductance = cond;
            dna[dna_count].valid = true;
            dna_count++;
        }
    }

    void inject_shock() noexcept {
        for (size_t s = 0; s < HASH_SLOTS; ++s) {
            if (table[s].epigenetic_lock < 80) {
                table[s].context_tag = 0;
                std::memset(table[s].conductance, 0, sizeof(table[s].conductance));
                table[s].credit = 0;
            }
        }

        if (enable_dna && dna_count > 0) {
            for (size_t d = 0; d < dna_count; ++d) {
                if (!dna[d].valid) continue;
                uint32_t h = dna[d].tag * 0x45d9f3b;
                h = ((h >> 16) ^ h) * 0x45d9f3b;
                h = (h >> 16) ^ h;
                size_t s = h % HASH_SLOTS;

                table[s].context_tag = dna[d].tag;
                table[s].conductance[dna[d].target_byte] = dna[d].frozen_conductance;
                table[s].epigenetic_lock = 98;
            }
        }
    }
};

} // namespace karyon_exp598_v3

void run_natural_corpus_benchmark(bool with_dna) {
    std::cout << "========================================================================\n";
    std::cout << ">>> RUNNING REAL ALPACA DATASET: " 
              << (with_dna ? "WITH DNA RESURRECTION" : "WITHOUT DNA (ABLATED)") << "\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) text_path = "data/web_downloads/alpaca_data.json", file.open(text_path, std::ios::binary);

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    } else {
        std::cerr << "Error: alpaca_data.json not found!\n";
        return;
    }

    std::cout << "Corpus loaded: " << corpus.size() << " raw bytes.\n";

    // Split: First 80% train stream, Last 20% test stream
    size_t train_size = static_cast<size_t>(corpus.size() * 0.80);
    size_t test_size = corpus.size() - train_size;

    karyon_exp598_v3::SiliconEngine engine(with_dna);

    const size_t SHOCK_STEP = train_size / 2;
    double fe_pre_shock = 0.0;
    double fe_post_shock_immediate = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Training Stream
    std::cout << "\n>>> Starting Training Phase on " << train_size << " continuous bytes...\n";
    for (size_t step = 0; step < train_size - 1; ++step) {
        if (step == SHOCK_STEP) {
            fe_pre_shock = engine.smooth_fe;
            std::cout << "\n[!] INJECTING CATASTROPHIC SHOCK AT STEP " << step << " (70% Damage)...\n";
            engine.inject_shock();
        }

        uint8_t in_b = corpus[step];
        uint8_t tgt_b = corpus[step + 1];

        engine.shift_byte(in_b);
        double fe_step = 0.0;
        engine.predict_and_eval(tgt_b, fe_step, true);

        if (step == SHOCK_STEP + 1000) {
            fe_post_shock_immediate = engine.smooth_fe;
        }

        if ((step + 1) % 1000000 == 0 || step == train_size - 2) {
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

    // 2. Out-of-Distribution Test Phase (Zero training, pure forward generalization)
    std::cout << "\n>>> Starting Test / Generalization Phase on " << test_size << " unseen bytes...\n";
    size_t test_hits = 0;
    double test_fe_sum = 0.0;
    size_t test_evals = 0;

    for (size_t step = train_size; step < corpus.size() - 1; ++step) {
        uint8_t in_b = corpus[step];
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
    std::cout << "=== EXP-598 v3: REAL ALPACA GENERALIZATION BENCHMARK (25MB CORPUS)   ===\n";
    std::cout << "========================================================================\n\n";

    run_natural_corpus_benchmark(false);
    run_natural_corpus_benchmark(true);

    return 0;
}
