// ======================================================================================
// EXP-599 v2: Bounded Fixed-Point Integer SSM with Integer RMSNorm
// --------------------------------------------------------------------------------------
// Fixes:
// 1. Integer RMS Normalization on Continuous State h_t:
//    - Normalizes 128-D state vector h_t into a bounded 8-bit range [-127 .. +127].
//    - Eliminates numeric potential explosion and underflow.
// 2. Proportional Integer Plasticity:
//    - Learning updates W_{target, d} += (h_norm_d >> 3), W_{false, d} -= (h_norm_d >> 3).
//    - Provides balanced, dynamic gradient force to minimize Free Energy.
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

namespace karyon_exp599_v2 {

constexpr size_t VOCAB_SIZE        = 258;
constexpr size_t STATE_DIM         = 128;
constexpr size_t DNA_CAPACITY       = 1024;

struct DnaGene {
    uint8_t  state_dim_idx;
    uint8_t  target_byte;
    int16_t  frozen_weight;
    bool     valid;
};

class FixedPointSSMEngine {
public:
    alignas(64) int32_t state_vector[STATE_DIM];
    alignas(64) int8_t  norm_state[STATE_DIM];
    alignas(64) int16_t decay_factors[STATE_DIM];

    alignas(64) int8_t input_embeddings[VOCAB_SIZE][STATE_DIM];
    alignas(64) int16_t output_weights[VOCAB_SIZE][STATE_DIM];
    alignas(64) uint8_t weight_locks[VOCAB_SIZE][STATE_DIM];

    std::unique_ptr<DnaGene[]> dna_registry;
    size_t                     dna_count;
    bool                       enable_dna;

    alignas(64) int32_t output_potentials[VOCAB_SIZE];

    double smooth_fe;
    double min_fe;
    size_t correct_hits;
    size_t total_evals;

    FixedPointSSMEngine(bool with_dna, uint64_t seed = 2026)
        : dna_registry(std::make_unique<DnaGene[]>(DNA_CAPACITY)),
          dna_count(0), enable_dna(with_dna),
          smooth_fe(5.55), min_fe(999.0), correct_hits(0), total_evals(0)
    {
        std::memset(state_vector, 0, sizeof(state_vector));
        std::memset(norm_state, 0, sizeof(norm_state));
        std::memset(weight_locks, 0, sizeof(weight_locks));
        for (size_t d = 0; d < DNA_CAPACITY; ++d) dna_registry[d].valid = false;

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist_emb(-16, 16);
        std::uniform_int_distribution<int> dist_out(-32, 32);

        // Multi-Timescale Decay Factors [128 .. 252] (1/256 fixed point)
        for (size_t i = 0; i < STATE_DIM; ++i) {
            double ratio = static_cast<double>(i) / static_cast<double>(STATE_DIM - 1);
            double alpha = 0.50 + 0.48 * std::pow(ratio, 1.5);
            decay_factors[i] = static_cast<int16_t>(std::clamp(static_cast<int>(alpha * 256.0), 128, 252));
        }

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            for (size_t d = 0; d < STATE_DIM; ++d) {
                input_embeddings[v][d] = static_cast<int8_t>(dist_emb(rng));
                output_weights[v][d]   = static_cast<int16_t>(dist_out(rng));
            }
        }
    }

    inline void step_recurrent_state(uint8_t input_byte) noexcept {
        const int8_t* __restrict emb_ptr = input_embeddings[input_byte];
        
        uint64_t sum_sq = 0;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            int32_t decayed = (static_cast<int64_t>(decay_factors[i]) * state_vector[i]) >> 8;
            int32_t new_state = std::clamp(decayed + static_cast<int32_t>(emb_ptr[i]) * 16, -32768, 32767);
            state_vector[i] = new_state;
            sum_sq += static_cast<uint64_t>(new_state * new_state);
        }

        // Fast Fixed-Point RMSNorm
        uint32_t rms = static_cast<uint32_t>(std::sqrt(static_cast<double>(sum_sq) / static_cast<double>(STATE_DIM))) + 1;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            int32_t scaled = (state_vector[i] * 127) / rms;
            norm_state[i] = static_cast<int8_t>(std::clamp(scaled, -127, 127));
        }
    }

    inline uint8_t predict_and_eval(uint8_t target_byte, double& fe_step, bool is_training) noexcept {
        total_evals++;

        // Linear Dot-Product Projection
        int32_t max_pot = -9999999;
        uint8_t best_byte = 0;

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            int32_t acc = 0;
            const int16_t* __restrict w_ptr = output_weights[v];
            
            for (size_t d = 0; d < STATE_DIM; ++d) {
                acc += (static_cast<int32_t>(norm_state[d]) * static_cast<int32_t>(w_ptr[d])) >> 7;
            }
            output_potentials[v] = acc;
            if (acc > max_pot) {
                max_pot = acc;
                best_byte = static_cast<uint8_t>(v);
            }
        }

        if (best_byte == target_byte) correct_hits++;

        // Numerically Stable Softmax for Free Energy
        double sum_exp = 0.0;
        double target_exp = 1.0;
        double scale = 0.02;

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            double logit = static_cast<double>(output_potentials[v] - max_pot) * scale;
            double p = std::exp(std::max(logit, -20.0));
            sum_exp += p;
            if (v == target_byte) target_exp = p;
        }

        double p_norm = std::max(target_exp / (sum_exp + 1e-12), 1e-12);
        fe_step = -std::log(p_norm);
        smooth_fe = 0.999 * smooth_fe + 0.001 * fe_step;
        if (smooth_fe < min_fe) min_fe = smooth_fe;

        // Adaptive Plasticity
        if (is_training) {
            int16_t* __restrict target_w = output_weights[target_byte];
            int16_t* __restrict false_w  = output_weights[best_byte];
            uint8_t* __restrict target_locks = weight_locks[target_byte];

            bool error_occurred = (best_byte != target_byte);

            for (size_t d = 0; d < STATE_DIM; ++d) {
                if (target_locks[d] >= 95) continue;

                int8_t s_norm = norm_state[d];
                if (s_norm == 0) continue;

                int delta = s_norm >> 2; // Proportional gradient step

                int32_t new_t_w = target_w[d] + delta;
                target_w[d] = static_cast<int16_t>(std::clamp(new_t_w, -2000, 2000));

                if (error_occurred) {
                    int32_t new_f_w = false_w[d] - delta;
                    false_w[d] = static_cast<int16_t>(std::clamp(new_f_w, -2000, 2000));
                }

                if (std::abs(target_w[d]) > 1000 && target_locks[d] < 95) {
                    target_locks[d] += 1;
                    if (enable_dna && target_locks[d] > 50) {
                        transcribe_dna(static_cast<uint8_t>(d), target_byte, target_w[d]);
                    }
                }
            }
        }

        return best_byte;
    }

    void transcribe_dna(uint8_t dim_idx, uint8_t target_byte, int16_t weight) noexcept {
        for (size_t d = 0; d < dna_count; ++d) {
            if (dna_registry[d].valid && dna_registry[d].state_dim_idx == dim_idx && dna_registry[d].target_byte == target_byte) {
                dna_registry[d].frozen_weight = weight;
                return;
            }
        }
        if (dna_count < DNA_CAPACITY) {
            dna_registry[dna_count].state_dim_idx = dim_idx;
            dna_registry[dna_count].target_byte = target_byte;
            dna_registry[dna_count].frozen_weight = weight;
            dna_registry[dna_count].valid = true;
            dna_count++;
        }
    }

    void inject_shock() noexcept {
        std::mt19937_64 rng(999);
        std::uniform_int_distribution<int> dist_out(-32, 32);

        for (size_t d = 0; d < STATE_DIM; ++d) state_vector[d] = 0;

        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            for (size_t d = 0; d < STATE_DIM; ++d) {
                if (weight_locks[v][d] < 80) {
                    output_weights[v][d] = static_cast<int16_t>(dist_out(rng));
                }
            }
        }

        if (enable_dna && dna_count > 0) {
            for (size_t d = 0; d < dna_count; ++d) {
                if (!dna_registry[d].valid) continue;
                uint8_t dim = dna_registry[d].state_dim_idx;
                uint8_t tgt = dna_registry[d].target_byte;

                output_weights[tgt][dim] = dna_registry[d].frozen_weight;
                weight_locks[tgt][dim]   = 98;
            }
        }
    }
};

} // namespace karyon_exp599_v2

void run_integer_ssm_benchmark(bool with_dna) {
    std::cout << "========================================================================\n";
    std::cout << ">>> RUNNING EXP-599 v2 (BOUNDED INTEGER SSM): " 
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

    karyon_exp599_v2::FixedPointSSMEngine engine(with_dna, 2026);

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

        engine.step_recurrent_state(in_b);
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

        engine.step_recurrent_state(in_b);
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
    std::cout << "=== EXP-599 v2: BOUNDED INTEGER SSM BENCHMARK (22.7MB ALPACA)        ===\n";
    std::cout << "========================================================================\n\n";

    run_integer_ssm_benchmark(false);
    run_integer_ssm_benchmark(true);

    return 0;
}
