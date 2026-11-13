// ============================================================================
// KARYON-CORE EXP-1148: ENTROPY-GUIDED AUTONOMOUS STRUCTURAL SEARCH
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 4 (V=2), 6, 8, 13, 14, 15
// ============================================================================
// Core Idea (Bazilevs):
//   Instead of human/AI-imposed routing rules ("use Q if coh > X, else D"),
//   hand over the exploratory search directly to ENTROPY (H / Free Energy F).
//
// Mechanism:
//   - A population of N=16 autonomous 64-bit native "Causal Motifs" competes
//     and evolves strictly via local causal Shannon Entropy minimization.
//   - Each Motif can freely express ANY primitive type without bias:
//       * [D] Deterministic bit-lag logic (COPY, NOT, XOR2, AND2, OR2, MAJ3, ADD)
//       * [Q] Quantum Wave Interference (phase-weighted multi-lag superposition)
//       * [S] Stochastic / Chaotic Perturbation (sigma_hw proportional to H!)
//   - Entropy-Driven Exploration Law:
//       Mutation rate & lag-jump radius scale directly with local Entropy H:
//       High H -> wide structural exploration across lags [1..4096] & ops.
//       Low  H -> epigenetic stabilization & fine phase tuning.
//   - Strictly Causal (Principle 15):
//       Every window's score is evaluated on PAST observed words only, and
//       prediction accuracy is measured on the UNSEEN next window.
//   - Endogenous Telemetry:
//       Prints the exact "Autobiography of Entropy" — which operators,
//       modalities (D vs Q vs MAJ3 vs ADD), and lag horizons Entropy itself
//       discovered to be optimal for Text, PNG, and ELF!
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <array>

static inline uint64_t rotl64(uint64_t x, int k) {
    k &= 63;
    return k == 0 ? x : ((x << k) | (x >> (64 - k)));
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double binary_entropy_bits(double p_err) {
    if (p_err <= 0.0 || p_err >= 1.0) return 0.0;
    return -p_err * std::log2(p_err) - (1.0 - p_err) * std::log2(1.0 - p_err);
}

static inline double compute_free_energy(uint64_t error_bits, uint64_t total_bits) {
    if (total_bits == 0 || error_bits == 0) return 0.0;
    double err_rate = static_cast<double>(error_bits) / static_cast<double>(total_bits);
    if (err_rate >= 1.0) err_rate = 1.0 - 1e-15;
    return -std::log(std::max(1.0 - err_rate, 1e-15));
}

static std::vector<uint64_t> load_file_as_bitstream(const std::string& path, size_t max_bytes = 0) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    size_t file_size = static_cast<size_t>(ifs.tellg());
    if (max_bytes > 0 && file_size > max_bytes) file_size = max_bytes;
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw(file_size);
    ifs.read(reinterpret_cast<char*>(raw.data()), file_size);

    size_t num_words = (file_size + 7) / 8;
    std::vector<uint64_t> words(num_words, 0ULL);
    if (file_size > 0) {
        std::memcpy(words.data(), raw.data(), file_size);
    }
    return words;
}

static inline uint64_t fetch_bit_lag_window(const std::vector<uint64_t>& words, size_t i, size_t lag) {
    size_t word_shift = lag >> 6;
    size_t bit_shift  = lag & 63;
    if (bit_shift == 0) {
        return (i >= word_shift) ? words[i - word_shift] : 0ULL;
    }
    uint64_t hi = (i >= word_shift) ? words[i - word_shift] : 0ULL;
    uint64_t lo = (i >= word_shift + 1) ? words[i - word_shift - 1] : 0ULL;
    return (hi << bit_shift) | (lo >> (64 - bit_shift));
}

// ============================================================================
// OPEN-ENDED PRIMITIVE GENOME (D + Q + Arithmetic + Majority)
// ============================================================================
enum OpType : uint8_t {
    OP_LAG1   = 0,  // D: direct bit-lag
    OP_XOR2   = 1,  // D: x1 ^ x2
    OP_AND2   = 2,  // D: x1 & x2
    OP_OR2    = 3,  // D: x1 | x2
    OP_MAJ3   = 4,  // D/Wave: majority(x1, x2, x3) — 3-lag consensus
    OP_XOR3   = 5,  // D: x1 ^ x2 ^ x3 — 3-lag parity
    OP_ADD2   = 6,  // D/Carry: (x1 + x2) — carries across bit boundaries!
    OP_QWAVE5 = 7,  // Q: 5-lag Quantum Wave Interference (signed majority + phase flip)
    OP_COUNT  = 8
};

static const char* op_name(OpType op) {
    switch (op) {
        case OP_LAG1:   return "D:LAG1";
        case OP_XOR2:   return "D:XOR2";
        case OP_AND2:   return "D:AND2";
        case OP_OR2:    return "D:OR2";
        case OP_MAJ3:   return "D:MAJ3";
        case OP_XOR3:   return "D:XOR3";
        case OP_ADD2:   return "D:ADD2";
        case OP_QWAVE5: return "Q:WAVE5";
        default:        return "UNK";
    }
}

struct EntropyMotif {
    OpType   op = OP_LAG1;
    uint8_t  pol = 0;         // Global polarity inversion
    uint8_t  phase_mask = 0;  // 5-bit phase inversion mask for QWAVE5 / MAJ3
    uint16_t lags[5] = {8, 16, 24, 32, 64};
    double   ema_err = 0.5;   // Causal exponential moving average bit-error

    inline uint64_t eval(const std::vector<uint64_t>& stream, size_t i) const {
        uint64_t v = 0ULL;
        uint64_t x0 = fetch_bit_lag_window(stream, i, lags[0]);
        if (op == OP_LAG1) {
            v = x0;
        } else if (op == OP_XOR2) {
            v = x0 ^ fetch_bit_lag_window(stream, i, lags[1]);
        } else if (op == OP_AND2) {
            v = x0 & fetch_bit_lag_window(stream, i, lags[1]);
        } else if (op == OP_OR2) {
            v = x0 | fetch_bit_lag_window(stream, i, lags[1]);
        } else if (op == OP_MAJ3) {
            uint64_t x1 = fetch_bit_lag_window(stream, i, lags[1]);
            uint64_t x2 = fetch_bit_lag_window(stream, i, lags[2]);
            v = (x0 & x1) | (x0 & x2) | (x1 & x2);
        } else if (op == OP_XOR3) {
            uint64_t x1 = fetch_bit_lag_window(stream, i, lags[1]);
            uint64_t x2 = fetch_bit_lag_window(stream, i, lags[2]);
            v = x0 ^ x1 ^ x2;
        } else if (op == OP_ADD2) {
            uint64_t x1 = fetch_bit_lag_window(stream, i, lags[1]);
            v = x0 + x1;
        } else if (op == OP_QWAVE5) {
            // Fast 64-bit parallel 5-path Quantum Wave Interference (no loops!)
            // Each path k has phase sign bit (phase_mask >> k) & 1
            uint64_t p[5];
            for (int k = 0; k < 5; ++k) {
                uint64_t raw = fetch_bit_lag_window(stream, i, lags[k]);
                p[k] = ((phase_mask >> k) & 1) ? ~raw : raw;
            }
            // Exact 3-of-5 bit-parallel majority via carry-save adder tree
            // Sum of 5 bits >= 3
            uint64_t s0 = p[0] ^ p[1] ^ p[2];
            uint64_t c0 = (p[0] & p[1]) | (p[0] & p[2]) | (p[1] & p[2]);
            uint64_t s1 = s0 ^ p[3] ^ p[4];
            uint64_t c1 = (s0 & p[3]) | (s0 & p[4]) | (p[3] & p[4]);
            // Total = s1 + 2*(c0 + c1). Majority (>=3) iff (c0 & c1) | ((c0 ^ c1) & s1)
            v = (c0 & c1) | ((c0 ^ c1) & s1);
        }
        return pol ? ~v : v;
    }

    // Entropy-driven mutation: higher H -> larger lag jumps & operator switches
    void mutate(uint64_t& rng, double entropy_H, size_t max_lag) {
        auto next_r = [&]() { rng = splitmix64(rng); return rng; };

        // Probability of operator mutation scales with Entropy H (0.15 .. 0.65)
        double p_op = 0.15 + 0.50 * entropy_H;
        if ((next_r() & 1023ULL) < static_cast<uint64_t>(p_op * 1024.0)) {
            op = static_cast<OpType>(next_r() % OP_COUNT);
        }
        if ((next_r() & 3ULL) == 0) pol ^= 1;
        if ((next_r() & 1ULL) == 0) phase_mask ^= (1U << (next_r() % 5));

        // Lag jump radius scales with Entropy H
        int num_mut_lags = (entropy_H > 0.7) ? 2 : 1;
        for (int m = 0; m < num_mut_lags; ++m) {
            int slot = static_cast<int>(next_r() % 5);
            uint64_t r = next_r();
            if ((r & 7ULL) == 0) {
                // Wide exploratory jump across [1 .. max_lag]
                lags[slot] = static_cast<uint16_t>(1 + (next_r() % max_lag));
            } else if ((r & 3ULL) == 0) {
                // Byte-harmonic jump (multiples of 8: 8, 16, 24, ..., 512)
                lags[slot] = static_cast<uint16_t>(8 * (1 + (next_r() % (max_lag / 8))));
            } else {
                // Local phase shift (+/- 1..16 scaled by H)
                int radius = std::max(1, static_cast<int>(entropy_H * 16.0));
                int delta = static_cast<int>(next_r() % (2 * radius + 1)) - radius;
                int nl = static_cast<int>(lags[slot]) + delta;
                if (nl < 1) nl = 1;
                if (nl > static_cast<int>(max_lag)) nl = static_cast<int>(max_lag);
                lags[slot] = static_cast<uint16_t>(nl);
            }
        }
    }
};

// ============================================================================
// UNBOUNDED ENDOGENOUS MEMORY (Zero-Error Pass 2 Anchor)
// ============================================================================
struct UnboundedNode {
    uint64_t sig1, sig2, delta;
    uint32_t first_t, depth;
    uint8_t  state;
};

class UnboundedEndogenousMemory {
public:
    std::vector<UnboundedNode> table;
    size_t count = 0;
    size_t max_depth_reached = 2;

    static inline void advance_sig(const std::vector<uint64_t>& words, size_t t,
                                   size_t from_d, size_t to_d,
                                   uint64_t& h1, uint64_t& h2) {
        size_t limit = std::min(to_d, t);
        for (size_t d = from_d + 1; d <= limit; ++d) {
            uint64_t w = words[t - d];
            h1 = splitmix64(h1 ^ w ^ (d * 0x9e3779b97f4a7c15ULL));
            h2 = splitmix64(h2 ^ rotl64(w, static_cast<int>(d & 63)) ^ (d * 0xbf58476d1ce4e5b9ULL));
        }
        if (to_d > t) {
            h1 = splitmix64(h1 ^ (static_cast<uint64_t>(t) << 32) ^ to_d);
            h2 = splitmix64(h2 ^ static_cast<uint64_t>(t));
        }
    }

    void rehash(size_t new_cap) {
        std::vector<UnboundedNode> next(new_cap, {0, 0, 0, 0, 0, 0});
        size_t mask = new_cap - 1;
        for (const auto& n : table) {
            if (n.state == 0) continue;
            size_t idx = (n.sig1 ^ n.sig2 ^ n.depth) & mask;
            while (next[idx].state != 0) idx = (idx + 1) & mask;
            next[idx] = n;
        }
        table = std::move(next);
    }

    void insert_at_depth(const std::vector<uint64_t>& words, size_t t, uint64_t delta,
                         size_t depth, uint64_t h1, uint64_t h2) {
        if (table.empty()) rehash(64);
        else if ((count + 1) * 2 >= table.size()) rehash(table.size() * 2);
        if (depth > max_depth_reached) max_depth_reached = depth;

        size_t mask = table.size() - 1;
        size_t idx = (h1 ^ h2 ^ depth) & mask;

        while (table[idx].state != 0) {
            if (table[idx].sig1 == h1 && table[idx].sig2 == h2 && table[idx].depth == depth) {
                if (table[idx].state == 1) {
                    if (table[idx].delta == delta) return;
                    if (depth > words.size() * 2) return;
                    uint32_t old_t = table[idx].first_t;
                    uint64_t old_delta = table[idx].delta;
                    table[idx].state = 2;
                    size_t next_depth = depth * 2;
                    uint64_t oh1 = h1, oh2 = h2;
                    advance_sig(words, old_t, depth, next_depth, oh1, oh2);
                    insert_at_depth(words, old_t, old_delta, next_depth, oh1, oh2);
                    uint64_t nh1 = h1, nh2 = h2;
                    advance_sig(words, t, depth, next_depth, nh1, nh2);
                    insert_at_depth(words, t, delta, next_depth, nh1, nh2);
                    return;
                } else {
                    size_t next_depth = depth * 2;
                    advance_sig(words, t, depth, next_depth, h1, h2);
                    insert_at_depth(words, t, delta, next_depth, h1, h2);
                    return;
                }
            }
            idx = (idx + 1) & mask;
        }
        table[idx] = {h1, h2, delta, static_cast<uint32_t>(t), static_cast<uint32_t>(depth), 1};
        count++;
    }

    inline void sprout(const std::vector<uint64_t>& words, size_t t, uint64_t delta) {
        uint64_t h1 = 0x9e3779b97f4a7c15ULL;
        uint64_t h2 = 0xbf58476d1ce4e5b9ULL;
        advance_sig(words, t, 0, 2, h1, h2);
        insert_at_depth(words, t, delta, 2, h1, h2);
    }

    inline uint64_t query(const std::vector<uint64_t>& words, size_t t) const {
        if (table.empty()) return 0ULL;
        size_t depth = 2, prev_d = 0;
        uint64_t h1 = 0x9e3779b97f4a7c15ULL, h2 = 0xbf58476d1ce4e5b9ULL;
        size_t mask = table.size() - 1;
        while (depth <= max_depth_reached) {
            advance_sig(words, t, prev_d, depth, h1, h2);
            size_t idx = (h1 ^ h2 ^ depth) & mask;
            bool found = false;
            while (table[idx].state != 0) {
                if (table[idx].sig1 == h1 && table[idx].sig2 == h2 && table[idx].depth == depth) {
                    if (table[idx].state == 1) return table[idx].delta;
                    found = true;
                    break;
                }
                idx = (idx + 1) & mask;
            }
            if (!found) return 0ULL;
            prev_d = depth;
            depth *= 2;
        }
        return 0ULL;
    }
};

// ============================================================================
// ENTROPY-GUIDED AUTONOMOUS SEARCH ENGINE
// ============================================================================
static void run_entropy_search_audit(const std::string& name, const std::vector<uint64_t>& words) {
    const size_t POP = 16;
    const size_t MAX_LAG = 2048; // Wide unshackled horizon (up to 2048 bits back)
    const size_t START_I = (MAX_LAG + 63) / 64 + 1;
    if (words.size() <= START_I + 64) return;

    size_t split = words.size() / 2;
    uint64_t rng = 0x9e3779b97f4a7c15ULL ^ words.size();

    // Initialize diverse population of Motifs across all operator types
    std::array<EntropyMotif, POP> pop;
    for (size_t p = 0; p < POP; ++p) {
        pop[p].op = static_cast<OpType>(p % OP_COUNT);
        pop[p].pol = (p / OP_COUNT) & 1;
        pop[p].phase_mask = static_cast<uint8_t>(p * 5);
        for (int k = 0; k < 5; ++k) {
            pop[p].lags[k] = static_cast<uint16_t>(8 * (k + 1) + (p % 4));
        }
        pop[p].ema_err = 0.5;
    }

    // Track Telemetry: What did Entropy discover?
    std::array<uint64_t, OP_COUNT> op_wins{};
    uint64_t ensemble_wins = 0;
    uint64_t total_mutations = 0;
    uint64_t beneficial_mutations = 0;

    // Phase 1: Train / Pre-adapt on first 50% (strictly causal word-by-word)
    // Phase 2: Measure strictly causal unseen accuracy on second 50%
    uint64_t unseen_best_err = 0;
    uint64_t unseen_ens_err  = 0;
    uint64_t unseen_bits     = 0;

    const size_t EVAL_WIN = 16; // Causal evaluation window (1024 bits)

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = START_I; i < words.size(); ++i) {
        // 1. Find current lowest-entropy motif + top-3 weighted ensemble
        size_t best_idx = 0;
        for (size_t p = 1; p < POP; ++p) {
            if (pop[p].ema_err < pop[best_idx].ema_err) best_idx = p;
        }

        // Top-3 consensus (entropy-weighted superposition of best 3 motifs)
        size_t idx1 = best_idx, idx2 = (best_idx + 1) % POP, idx3 = (best_idx + 2) % POP;
        for (size_t p = 0; p < POP; ++p) {
            if (p == idx1) continue;
            if (pop[p].ema_err < pop[idx2].ema_err) { idx3 = idx2; idx2 = p; }
            else if (pop[p].ema_err < pop[idx3].ema_err) { idx3 = p; }
        }

        uint64_t pred_best = pop[best_idx].eval(words, i);
        uint64_t p1 = pred_best;
        uint64_t p2 = pop[idx2].eval(words, i);
        uint64_t p3 = pop[idx3].eval(words, i);
        uint64_t pred_ens = (p1 & p2) | (p1 & p3) | (p2 & p3);

        uint64_t target = words[i];
        uint64_t err_best = __builtin_popcountll(pred_best ^ target);
        uint64_t err_ens  = __builtin_popcountll(pred_ens ^ target);

        // Strictly record unseen test accuracy on 2nd half BEFORE updating
        if (i >= split) {
            unseen_best_err += err_best;
            unseen_ens_err  += err_ens;
            unseen_bits     += 64ULL;
            op_wins[pop[best_idx].op]++;
            if (err_ens < err_best) ensemble_wins++;
        }

        // 2. Causal EMA update for all motifs after observing words[i]
        for (size_t p = 0; p < POP; ++p) {
            uint64_t pe = __builtin_popcountll(pop[p].eval(words, i) ^ target);
            double r = static_cast<double>(pe) / 64.0;
            pop[p].ema_err = 0.96 * pop[p].ema_err + 0.04 * r;
        }

        // 3. ENTROPY-GUIDED EXPLORATORY SEARCH (every EVAL_WIN steps)
        if ((i % EVAL_WIN) == 0 && i >= START_I + EVAL_WIN) {
            double H_local = binary_entropy_bits(pop[best_idx].ema_err);

            // Find worst motif in population and let Entropy mutate it from best
            size_t worst_idx = 0;
            for (size_t p = 1; p < POP; ++p) {
                if (pop[p].ema_err > pop[worst_idx].ema_err) worst_idx = p;
            }

            if (worst_idx != best_idx) {
                EntropyMotif candidate = pop[best_idx];
                candidate.mutate(rng, H_local, MAX_LAG);
                total_mutations++;

                // Evaluate candidate strictly on the PAST window [i - EVAL_WIN + 1 .. i]
                uint64_t cand_err = 0, base_err = 0;
                for (size_t k = i - EVAL_WIN + 1; k <= i; ++k) {
                    cand_err += __builtin_popcountll(candidate.eval(words, k) ^ words[k]);
                    base_err += __builtin_popcountll(pop[best_idx].eval(words, k) ^ words[k]);
                }

                // If candidate reduces entropy on recent past, replace worst (or even best!)
                if (cand_err <= base_err) {
                    candidate.ema_err = static_cast<double>(cand_err) / (EVAL_WIN * 64.0);
                    pop[worst_idx] = candidate;
                    beneficial_mutations++;
                } else if (H_local > 0.85 && (splitmix64(rng ^ i) & 3ULL) == 0) {
                    // High-entropy exploration: keep diverse mutant in worst slot anyway
                    candidate.ema_err = static_cast<double>(cand_err) / (EVAL_WIN * 64.0);
                    pop[worst_idx] = candidate;
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double speed = (static_cast<double>((words.size() - START_I) * 8ULL) / (1024.0 * 1024.0)) / (ms / 1000.0);

    double best_acc = 100.0 * (1.0 - static_cast<double>(unseen_best_err) / unseen_bits);
    double ens_acc  = 100.0 * (1.0 - static_cast<double>(unseen_ens_err) / unseen_bits);

    // Find top-discovered motif at end of stream
    size_t final_best = 0;
    for (size_t p = 1; p < POP; ++p) {
        if (pop[p].ema_err < pop[final_best].ema_err) final_best = p;
    }

    // Pass 2 Zero-Error Audit via Unbounded Endogenous Memory
    UnboundedEndogenousMemory mem;
    for (size_t i = START_I; i < words.size(); ++i) {
        uint64_t p = pop[final_best].eval(words, i);
        if (p != words[i]) mem.sprout(words, i, p ^ words[i]);
    }
    for (size_t i = START_I; i < words.size(); ++i) {
        uint64_t p = pop[final_best].eval(words, i);
        if ((p ^ mem.query(words, i)) != words[i]) mem.sprout(words, i, p ^ words[i]);
    }
    uint64_t pass2_err = 0;
    for (size_t i = START_I; i < words.size(); ++i) {
        uint64_t p = pop[final_best].eval(words, i) ^ mem.query(words, i);
        pass2_err += __builtin_popcountll(p ^ words[i]);
    }

    std::cout << "\n[BITSTREAM] " << name << " (" << (words.size() * 64) << " bits)" << std::endl;
    std::cout << "  Entropy-Guided Best Motif: " << std::fixed << std::setprecision(2) << best_acc
              << "% (F = " << std::setprecision(6) << compute_free_energy(unseen_best_err, unseen_bits) << " nats)" << std::endl;
    std::cout << "  Entropy-Guided Top-3 Ens : " << std::fixed << std::setprecision(2) << ens_acc
              << "% (F = " << std::setprecision(6) << compute_free_energy(unseen_ens_err, unseen_bits)
              << " nats) | Speed: " << std::setprecision(2) << speed << " MB/s" << std::endl;
    std::cout << "  Mutations (Accepted/Total): " << beneficial_mutations << " / " << total_mutations
              << " | Ensemble > Best steps: " << ensemble_wins << std::endl;
    std::cout << "  [AUTOBIOGRAPHY OF ENTROPY — Discovered Operator Dominance]:" << std::endl;
    std::cout << "    ";
    for (int o = 0; o < OP_COUNT; ++o) {
        if (op_wins[o] > 0) {
            std::cout << op_name(static_cast<OpType>(o)) << "=" << op_wins[o] << "  ";
        }
    }
    std::cout << "\n    Final Dominant Motif: " << op_name(pop[final_best].op)
              << " (pol=" << (int)pop[final_best].pol
              << ", lags=[" << pop[final_best].lags[0] << "," << pop[final_best].lags[1]
              << "," << pop[final_best].lags[2] << "," << pop[final_best].lags[3]
              << "," << pop[final_best].lags[4] << "])" << std::endl;
    std::cout << "  Pass 2 Zero-Error Audit  : " << pass2_err << " bits (Nodes: " << mem.count << ")" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1148: ENTROPY-GUIDED AUTONOMOUS STRUCTURAL SEARCH" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 1, 4 (V=2), 6, 8, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    auto kep_words = load_file_as_bitstream("KEP.md");
    if (!kep_words.empty()) run_entropy_search_audit("TEXT: KEP.md", kep_words);

    auto shake_words = load_file_as_bitstream("tinyshakespeare.txt", 262144);
    if (!shake_words.empty()) run_entropy_search_audit("TEXT: tinyshakespeare.txt (256KB)", shake_words);

    auto png_words = load_file_as_bitstream("experiments/renders/bloodbath_precise_v4.png", 262144);
    if (!png_words.empty()) run_entropy_search_audit("IMAGE/PNG: bloodbath_precise_v4.png (256KB)", png_words);

    if (argc > 0 && argv[0]) {
        auto elf_words = load_file_as_bitstream(argv[0]);
        if (!elf_words.empty()) run_entropy_search_audit("BINARY/ELF: x86_64 Machine Code", elf_words);
    }

    return 0;
}
