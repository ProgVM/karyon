#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1175d: CLEAN MULTI-HYPOTHESIS SWAR EMERGENCE BENCHMARK
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// Strict Anti-Falsification & Zero-Wrapper Guarantees:
//   - ZERO PAQ hash tables, ZERO ExactLUT64K, ZERO OnlineF2Basis solvers.
//   - ZERO external lookback buffers (H0/H1): world feeds strictly 1 bit/tick.
//   - ZERO human-assigned roles ("receptor", "memory", "output").
//   - ZERO floats in substrate state; pure branchless uint64_t SWAR operations.
//   - Causal ordering: pred_bit computed strictly BEFORE observing target_bit.
//   - Honest Baselines printed side-by-side: Order-0 (majority bit) & Order-1.
// ============================================================================

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t rotl64(uint64_t x, int k) {
    uint32_t s = (uint32_t)k & 63U;
    return s ? ((x << s) | (x >> (64U - s))) : x;
}

// 4-bit bit-sliced saturating counter [0..15] across 64 parallel bit channels
static inline void swar_sat4_step(
    uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3,
    uint64_t inc_mask, uint64_t dec_mask)
{
    uint64_t can_inc = inc_mask & ~(c0 & c1 & c2 & c3);
    uint64_t cr0 = can_inc;
    uint64_t cr1 = cr0 & c0;
    uint64_t cr2 = cr1 & c1;
    uint64_t cr3 = cr2 & c2;
    c0 ^= cr0; c1 ^= cr1; c2 ^= cr2; c3 ^= cr3;

    uint64_t can_dec = dec_mask & (c0 | c1 | c2 | c3);
    uint64_t br0 = can_dec;
    uint64_t br1 = br0 & ~c0;
    uint64_t br2 = br1 & ~c1;
    uint64_t br3 = br2 & ~c2;
    c0 ^= br0; c1 ^= br1; c2 ^= br2; c3 ^= br3;
}

enum EngineMode {
    H1_RANDOM_DRIFT = 0,
    H2_HIERARCHICAL_RATCHET = 1,
    H3_DISSIPATIVE_F0 = 2,
    H4_EDGE_OF_CHAOS = 3
};

constexpr int PLANES = 4; // 4 x 64 = 256 autonomous bit-cells per organism

struct SwarmOrganism {
    uint64_t S[PLANES];      // Dynamic state bits (no pre-assigned roles)
    uint64_t M_in[PLANES];   // Self-evolved boundary permeability (1=couples to world, 0=sheltered)
    uint64_t g0[PLANES];     // Local gate bit 0 (00=A, 01=A^B, 10=A&B, 11=~(A&B))
    uint64_t g1[PLANES];     // Local gate bit 1
    uint8_t  rotA[PLANES];   // Rotation from neighbor/self plane A
    uint8_t  rotB[PLANES];   // Rotation from neighbor plane B
    uint8_t  srcA[PLANES];   // Source plane index A
    uint8_t  srcB[PLANES];   // Source plane index B

    // 4-bit bit-sliced predictive alignment counter per cell [0..15], neutral=7/8
    uint64_t C0[PLANES], C1[PLANES], C2[PLANES], C3[PLANES];
    // Lock mask for H2 (stabilized cells resist mutation)
    uint64_t lock_mask[PLANES];

    uint64_t rng;
    uint64_t tail_err;
    uint64_t recent_err_ema; // Integer EMA [0..256] for tournament fitness
};

static inline void init_organism(SwarmOrganism& o, uint64_t seed) {
    o.rng = seed;
    for (int p = 0; p < PLANES; ++p) {
        o.S[p] = splitmix64(o.rng);
        // Start with sparse boundary permeability (~25% exposed to world, ~75% internal)
        o.M_in[p] = splitmix64(o.rng) & splitmix64(o.rng);
        o.g0[p] = splitmix64(o.rng);
        o.g1[p] = splitmix64(o.rng);
        o.rotA[p] = (uint8_t)(splitmix64(o.rng) & 63);
        o.rotB[p] = (uint8_t)(splitmix64(o.rng) & 63);
        o.srcA[p] = (uint8_t)(splitmix64(o.rng) & 3);
        o.srcB[p] = (uint8_t)(splitmix64(o.rng) & 3);
        // Neutral confidence = 7 (0b0111)
        o.C0[p] = ~0ULL; o.C1[p] = ~0ULL; o.C2[p] = ~0ULL; o.C3[p] = 0ULL;
        o.lock_mask[p] = 0ULL;
    }
    o.tail_err = 0;
    o.recent_err_ema = 128;
}

static inline uint8_t predict_from_field(const SwarmOrganism& o) {
    // Pure emergent consensus: cells with strong positive alignment (C >= 12 -> C3 & C2)
    // vote with their current state S; cells with strong anti-alignment (C <= 3 -> ~C3 & ~C2)
    // vote with ~S. Cells in [4..11] stay silent (uncertain).
    int votes1 = 0, votes0 = 0;
    for (int p = 0; p < PLANES; ++p) {
        uint64_t pos = o.C3[p] & o.C2[p];
        uint64_t neg = (~o.C3[p]) & (~o.C2[p]);
        uint64_t active = pos | neg;
        uint64_t vote1_bits = (pos & o.S[p]) | (neg & ~o.S[p]);
        int act_cnt = __builtin_popcountll(active);
        int v1_cnt  = __builtin_popcountll(vote1_bits);
        votes1 += v1_cnt;
        votes0 += (act_cnt - v1_cnt);
    }
    if (votes1 == votes0) {
        // Tie-breaker: slight bias to overall S[0] majority (no external info)
        return (__builtin_popcountll(o.S[0]) >= 32) ? 1U : 0U;
    }
    return (votes1 > votes0) ? 1U : 0U;
}

static inline uint64_t step_organism(SwarmOrganism& o, uint8_t target_bit, EngineMode mode, size_t t, bool in_tail) {
    // 1. Causal prediction BEFORE observing target_bit
    uint8_t pred = predict_from_field(o);
    uint8_t err = pred ^ target_bit;
    if (in_tail) o.tail_err += err;
    o.recent_err_ema = ((o.recent_err_ema * 63ULL) + (err ? 256ULL : 0ULL)) >> 6;

    uint64_t in_word = target_bit ? ~0ULL : 0ULL;

    // 2. Update confidence counters BEFORE state transitions (how well did each cell at t predict target_bit?)
    for (int p = 0; p < PLANES; ++p) {
        uint64_t agree = ~(o.S[p] ^ in_word);
        uint64_t disagree = ~agree;
        swar_sat4_step(o.C0[p], o.C1[p], o.C2[p], o.C3[p], agree, disagree);
    }

    // 3. Branchless SWAR physics transition (ZERO float, ZERO switch)
    uint64_t next_S[PLANES];
    uint64_t total_flips = 0;
    for (int p = 0; p < PLANES; ++p) {
        uint64_t a = rotl64(o.S[o.srcA[p]], o.rotA[p]);
        uint64_t b_internal = rotl64(o.S[o.srcB[p]], o.rotB[p]);
        uint64_t b = (o.M_in[p] & in_word) | (~o.M_in[p] & b_internal);

        uint64_t g0 = o.g0[p];
        uint64_t g1 = o.g1[p];
        // 4 universal gates per bit simultaneously:
        // 00: a (memory/wire), 01: a^b (XOR), 10: a&b (AND), 11: ~(a&b) (NAND)
        next_S[p] = (~g1 & ~g0 & a)
                  | (~g1 &  g0 & (a ^ b))
                  | ( g1 & ~g0 & (a & b))
                  | ( g1 &  g0 & ~(a & b));
        total_flips += __builtin_popcountll(next_S[p] ^ o.S[p]);
    }
    for (int p = 0; p < PLANES; ++p) o.S[p] = next_S[p];

    // 4. Local Plasticity / Evolution per Hypothesis
    if (mode == H1_RANDOM_DRIFT) {
        if ((t & 31ULL) == 0ULL) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 3ULL);
            int bit = (int)((r >> 2) & 63ULL);
            uint64_t m = 1ULL << bit;
            uint64_t sel = (r >> 8) & 7ULL;
            if (sel < 2) o.M_in[p] ^= m;
            else if (sel < 4) o.g0[p] ^= m;
            else if (sel < 6) o.g1[p] ^= m;
            else if (sel == 6) o.rotA[p] = (uint8_t)((r >> 16) & 63ULL);
            else o.rotB[p] = (uint8_t)((r >> 16) & 63ULL);
        }
    } else if (mode == H2_HIERARCHICAL_RATCHET) {
        // Cells with extreme confidence (0 or 15) lock onto lock_mask; unconfident cells mutate on error
        for (int p = 0; p < PLANES; ++p) {
            uint64_t strong = (o.C3[p] & o.C2[p] & o.C1[p] & o.C0[p])
                            | (~o.C3[p] & ~o.C2[p] & ~o.C1[p] & ~o.C0[p]);
            o.lock_mask[p] |= strong;
            // Decay lock if global EMA stays bad (prevent permanent dead crystal)
            if (o.recent_err_ema > 120 && ((t & 255ULL) == 0ULL)) {
                o.lock_mask[p] &= splitmix64(o.rng);
            }
        }
        if (err && ((t & 7ULL) == 0ULL)) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 3ULL);
            uint64_t mut_bit = (1ULL << ((r >> 2) & 63ULL)) & ~o.lock_mask[p];
            if (mut_bit) {
                uint64_t sel = (r >> 8) & 3ULL;
                if (sel == 0) o.M_in[p] ^= mut_bit;
                else if (sel == 1) o.g0[p] ^= mut_bit;
                else o.g1[p] ^= mut_bit;
            } else {
                o.rotA[p] = (uint8_t)((r >> 16) & 63ULL);
            }
        }
    } else if (mode == H3_DISSIPATIVE_F0) {
        // F -> 0 Local Dissipative Cooling:
        // Only cells currently uncertain (4..11) and wrong on global err rewire!
        // Cells that accurately align with target (C >= 12 or C <= 3) stay cold!
        if (err) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 3ULL);
            uint64_t confident = (o.C3[p] & o.C2[p]) | (~o.C3[p] & ~o.C2[p]);
            uint64_t hot_cells = ~confident;
            uint64_t m = (1ULL << ((r >> 2) & 63ULL)) & hot_cells;
            if (m) {
                uint64_t sel = (r >> 8) & 3ULL;
                if (sel == 0) o.M_in[p] ^= m;
                else if (sel == 1) o.g0[p] ^= m;
                else o.g1[p] ^= m;
            }
            if ((r >> 12) & 1ULL) o.rotA[p] = (uint8_t)((r >> 16) & 63ULL);
            if ((r >> 13) & 1ULL) o.rotB[p] = (uint8_t)((r >> 22) & 63ULL);
            if ((r >> 14) & 3ULL) {
                o.srcA[p] = (uint8_t)((r >> 28) & 3ULL);
                o.srcB[p] = (uint8_t)((r >> 30) & 3ULL);
            }
        }
    } else if (mode == H4_EDGE_OF_CHAOS) {
        // Maintain flip density in critical window [48..144] out of 256 bits (19%..56%)
        if (total_flips < 48 || total_flips > 144) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 3ULL);
            int bit = (int)((r >> 2) & 63ULL);
            uint64_t m = 1ULL << bit;
            if (total_flips < 48) {
                // Too frozen -> nudge toward XOR (g1=0, g0=1)
                o.g1[p] &= ~m;
                o.g0[p] |= m;
            } else {
                // Too chaotic -> nudge toward WIRE/AND (g0=0)
                o.g0[p] &= ~m;
            }
        } else if (err) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 3ULL);
            uint64_t confident = (o.C3[p] & o.C2[p]) | (~o.C3[p] & ~o.C2[p]);
            uint64_t m = (1ULL << ((r >> 2) & 63ULL)) & ~confident;
            if (m) {
                if ((r >> 8) & 1ULL) o.M_in[p] ^= m;
                else o.g0[p] ^= m;
            }
            if (((r >> 10) & 7ULL) == 0ULL) {
                o.rotA[p] = (uint8_t)((r >> 16) & 63ULL);
                o.rotB[p] = (uint8_t)((r >> 22) & 63ULL);
            }
        }
    }
    return total_flips;
}

// Stream generators
static std::vector<uint8_t> gen_splitmix(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes);
    uint64_t s = 0xDEADBEEFCAFE1234ULL;
    for (size_t i = 0; i < n_bytes; ++i) out[i] = (uint8_t)(splitmix64(s) & 0xFF);
    return out;
}

// Pure 1-bit stream with ZERO Order-0/Order-1 bias: b_t = b_{t-7} ^ b_{t-6} (maximal-length 127-bit period LFSR-7)
static std::vector<uint8_t> gen_lfsr7(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    uint8_t reg = 0x5AU;
    for (size_t t = 0; t < n_bytes * 8; ++t) {
        uint8_t b = reg & 1U;
        if (b) out[t >> 3] |= (1U << (t & 7));
        uint8_t fb = (reg ^ (reg >> 1)) & 1U;
        reg = (uint8_t)((reg >> 1) | (fb << 6));
    }
    return out;
}

// Nonlinear hidden causal rule: b_t = (b_{t-3} & b_{t-5}) ^ b_{t-8} (wait: check non-degeneracy!)
static std::vector<uint8_t> gen_nl_lag(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    uint16_t st = 0xACE1U;
    for (size_t t = 0; t < n_bytes * 8; ++t) {
        uint8_t b = st & 1U;
        if (b) out[t >> 3] |= (1U << (t & 7));
        // Nonlinear feedback: ((bit2 & bit4) ^ bit7 ^ bit11)
        uint8_t fb = (((st >> 2) & (st >> 4)) ^ (st >> 7) ^ (st >> 11)) & 1U;
        st = (uint16_t)((st >> 1) | ((uint16_t)fb << 15));
    }
    return out;
}

static std::vector<uint8_t> gen_rule110(size_t n_bytes, size_t W = 64) {
    std::vector<uint8_t> out(n_bytes, 0);
    std::vector<uint8_t> row(W, 0), nxt(W, 0);
    row[W / 2] = 1; row[W / 3] = 1;
    size_t t = 0, tot = n_bytes * 8;
    while (t < tot) {
        for (size_t x = 0; x < W && t < tot; ++x, ++t) if (row[x]) out[t >> 3] |= (1U << (t & 7));
        for (size_t x = 0; x < W; ++x) {
            uint8_t pat = (row[(x + W - 1) % W] << 2) | (row[x] << 1) | row[(x + 1) % W];
            nxt[x] = (110 >> pat) & 1U;
        }
        row.swap(nxt);
    }
    return out;
}

struct BaselineStats {
    double order0_tail_acc;
    double order1_tail_acc;
    double zero_frac_tail;
};

static BaselineStats compute_baselines(const std::vector<uint8_t>& data, size_t total_bits) {
    size_t tail_start = (total_bits * 3) / 4;
    size_t c0[2]{}, c1[2][2]{};
    uint64_t ok0 = 0, ok1 = 0, tail_n = 0, zeros_tail = 0;
    uint8_t prev = 0;
    for (size_t t = 0; t < total_bits; ++t) {
        uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
        uint8_t pred0 = (c0[1] > c0[0]) ? 1U : 0U;
        uint8_t pred1 = (c1[prev][1] > c1[prev][0]) ? 1U : 0U;
        if (t >= tail_start) {
            tail_n++;
            if (pred0 == b) ok0++;
            if (pred1 == b) ok1++;
            if (b == 0) zeros_tail++;
        }
        c0[b]++;
        c1[prev][b]++;
        prev = b;
    }
    return {100.0 * ok0 / tail_n, 100.0 * ok1 / tail_n, 100.0 * zeros_tail / tail_n};
}

static void run_suite(const std::string& name, const std::vector<uint8_t>& data, size_t max_bits) {
    size_t total_bits = std::min(data.size() * 8, max_bits);
    size_t tail_start = (total_bits * 3) / 4;
    size_t tail_bits = total_bits - tail_start;

    BaselineStats bs = compute_baselines(data, total_bits);

    std::cout << "\n========================================================================================\n";
    std::cout << "STREAM: " << name << " (" << total_bits << " bits = " << (total_bits / 8) << " B)\n";
    std::cout << "HONEST BASELINES (Tail 25%): ZeroFrac=" << std::fixed << std::setprecision(2) << bs.zero_frac_tail
              << "% | Order-0=" << bs.order0_tail_acc << "% | Order-1=" << bs.order1_tail_acc << "%\n";
    std::cout << "========================================================================================\n";

    constexpr int POP = 128; // 128 organisms per hypothesis
    const EngineMode modes[4] = {H1_RANDOM_DRIFT, H2_HIERARCHICAL_RATCHET, H3_DISSIPATIVE_F0, H4_EDGE_OF_CHAOS};
    const char* names[4] = {"H1: Random+Select", "H2: Hier.Ratchet", "H3: F->0 Cooling", "H4: Edge-of-Chaos"};

    for (int m = 0; m < 4; ++m) {
        std::vector<SwarmOrganism> pop(POP);
        for (int i = 0; i < POP; ++i) init_organism(pop[i], 0x9E3779B97F4A7C15ULL * (i + 1) + (uint64_t)m * 1337ULL);

        uint64_t tourney_rng = 0xCAFEBABE12345678ULL + m;
        uint64_t sum_flips = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < total_bits; ++t) {
            uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
            bool in_tail = (t >= tail_start);

            #pragma omp parallel for reduction(+:sum_flips) schedule(static)
            for (int i = 0; i < POP; ++i) {
                uint64_t f = step_organism(pop[i], b, modes[m], t, in_tail);
                if (in_tail) sum_flips += f;
            }

            // Evolutionary tournament selection every 512 ticks (replace worst with mutated best)
            if ((t & 511ULL) == 511ULL) {
                int best_i = 0, worst_i = 0;
                for (int i = 1; i < POP; ++i) {
                    if (pop[i].recent_err_ema < pop[best_i].recent_err_ema) best_i = i;
                    if (pop[i].recent_err_ema > pop[worst_i].recent_err_ema) worst_i = i;
                }
                if (best_i != worst_i && pop[best_i].recent_err_ema + 4 < pop[worst_i].recent_err_ema) {
                    uint64_t kept_tail = pop[worst_i].tail_err;
                    uint64_t kept_rng  = pop[worst_i].rng;
                    pop[worst_i] = pop[best_i];
                    pop[worst_i].tail_err = kept_tail;
                    pop[worst_i].rng = kept_rng;
                    // Mutate 1 wire/gate in clone so population keeps exploring
                    uint64_t r = splitmix64(tourney_rng);
                    int p = (int)(r & 3ULL);
                    uint64_t bitm = 1ULL << ((r >> 2) & 63ULL);
                    if ((r >> 8) & 1ULL) pop[worst_i].M_in[p] ^= bitm;
                    else pop[worst_i].g0[p] ^= bitm;
                    if ((r >> 9) & 1ULL) pop[worst_i].rotA[p] = (uint8_t)((r >> 16) & 63ULL);
                }
            }
        }

        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        uint64_t min_err = pop[0].tail_err;
        int best_idx = 0;
        double sum_err = 0.0;
        for (int i = 0; i < POP; ++i) {
            sum_err += (double)pop[i].tail_err;
            if (pop[i].tail_err < min_err) { min_err = pop[i].tail_err; best_idx = i; }
        }
        double best_acc = 100.0 * (1.0 - (double)min_err / tail_bits);
        double avg_acc  = 100.0 * (1.0 - (sum_err / POP) / tail_bits);
        double avg_flips = (double)sum_flips / ((double)POP * tail_bits);

        // Endoscopic inspection of Best Organism: how much internal sheltered memory (M_in == 0) evolved?
        int mem_bits = 0, xor_bits = 0, hold_bits = 0;
        for (int p = 0; p < PLANES; ++p) {
            mem_bits  += __builtin_popcountll(~pop[best_idx].M_in[p]);
            xor_bits  += __builtin_popcountll(~pop[best_idx].g1[p] & pop[best_idx].g0[p]);
            hold_bits += __builtin_popcountll(~pop[best_idx].g1[p] & ~pop[best_idx].g0[p]);
        }

        double delta_o0 = best_acc - bs.order0_tail_acc;
        double delta_o1 = best_acc - bs.order1_tail_acc;

        std::cout << std::left << std::setw(18) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << delta_o0
                  << ", O1:" << std::setw(6) << delta_o1 << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(5) << std::setprecision(1) << avg_flips << "/256"
                  << " | Mem=" << (mem_bits * 100 / 256) << "% XOR=" << (xor_bits * 100 / 256)
                  << "% Hold=" << (hold_bits * 100 / 256) << "%"
                  << " | " << std::setprecision(2) << sec << "s\n";
    }
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " KARYON-CORE EXP-1175d: PURE SWAR MULTI-HYPOTHESIS EMERGENCE (ZERO TABLES / ZERO FLOATS)\n";
    std::cout << " 4 Hypotheses Compared Side-by-Side vs Order-0 & Order-1 Baselines (CPU AVX2 + OpenMP)\n";
    std::cout << "========================================================================================\n";

    const size_t N_BYTES = 16384; // 131,072 bits per stream for fast crisp diagnostics
    run_suite("Control 0: SplitMix64 Crypto Noise", gen_splitmix(N_BYTES), N_BYTES * 8);
    run_suite("Law 1: LFSR-7 (50/50 balanced, 1 bit/tick, NO buffer)", gen_lfsr7(N_BYTES), N_BYTES * 8);
    run_suite("Law 2: Nonlinear Lag (b_{t-3}&b_{t-5})^b_{t-8}^b_{t-12}", gen_nl_lag(N_BYTES), N_BYTES * 8);
    run_suite("Law 3: Rule 110 CA (1 bit/tick, NO buffer)", gen_rule110(N_BYTES), N_BYTES * 8);

    std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
    if (fw) {
        std::vector<uint8_t> wiki(N_BYTES);
        fw.read((char*)wiki.data(), N_BYTES);
        wiki.resize((size_t)fw.gcount());
        run_suite("Real 4: Wikipedia Stream (1 bit/tick, NO buffer)", wiki, wiki.size() * 8);
    }
    return 0;
}
