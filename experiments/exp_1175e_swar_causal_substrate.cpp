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
// KARYON-CORE EXP-1175e: PURE SWAR CAUSAL SUBSTRATE (NO ROLE HARDCODING)
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// Fixes over EXP-1175d:
//   1. Eliminated H3/H4 routing-thrashing bug: structural mutations happen at
//      a timescale slower than state-counter integration, OR strictly gated by
//      persistent cell-level error accumulation (not single-tick global noise).
//   2. True Emergent Local Readout (zero bias-drowning):
//      Each bit cell updates its OWN 4-bit causal predictive score C[0..15]
//      against target_bit; readout emerges from the most predictive cells
//      (or majority of highest tier), without forcing constant-0 cells to lock
//      when dynamic transitions fail.
//   3. Richer SWAR local topology WITHOUT human-designed history buffers:
//      State S[8] (512 bits). Each bit i can couple to shifted neighbors
//      (S shifted by 1 bit = natural spatial wave propagation), cross-plane
//      rotations, and external world bit via self-evolved M_in.
//      Zero external H0/H1 shift buffer! Any temporal depth must emerge
//      inside S[0..7] itself!
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

constexpr int PLANES = 8; // 8 x 64 = 512 autonomous bit-cells per organism

struct SwarmOrganism {
    uint64_t S[PLANES];      // 512 internal dynamic bits
    uint64_t M_in[PLANES];   // Self-evolved boundary permeability (1=couples to world, 0=internal)
    uint64_t g0[PLANES];     // Local gate bit 0 (00=A, 01=A^B, 10=A&B, 11=A|B)
    uint64_t g1[PLANES];     // Local gate bit 1
    uint64_t shift_en[PLANES]; // Self-evolved spatial coupling: 1=take shifted neighbor (+1 bit), 0=cross-plane

    uint8_t  rotA[PLANES];
    uint8_t  rotB[PLANES];
    uint8_t  srcA[PLANES];
    uint8_t  srcB[PLANES];

    // 4-bit bit-sliced predictive alignment counter per cell [0..15]
    uint64_t C0[PLANES], C1[PLANES], C2[PLANES], C3[PLANES];
    // 4-bit bit-sliced activity/flip EMA per cell (tracks whether cell is alive vs dead constant)
    uint64_t A0[PLANES], A1[PLANES], A2[PLANES], A3[PLANES];

    uint64_t lock_mask[PLANES];
    uint64_t rng;
    uint64_t tail_err;
    uint64_t recent_err_ema; // [0..256]
    uint32_t err_cooldown;
};

static inline void init_organism(SwarmOrganism& o, uint64_t seed) {
    o.rng = seed;
    for (int p = 0; p < PLANES; ++p) {
        o.S[p] = splitmix64(o.rng);
        o.M_in[p] = splitmix64(o.rng) & splitmix64(o.rng) & splitmix64(o.rng); // ~12.5% initial exposure
        o.g0[p] = splitmix64(o.rng);
        o.g1[p] = splitmix64(o.rng);
        o.shift_en[p] = splitmix64(o.rng);
        o.rotA[p] = (uint8_t)(splitmix64(o.rng) & 63);
        o.rotB[p] = (uint8_t)(splitmix64(o.rng) & 63);
        o.srcA[p] = (uint8_t)(splitmix64(o.rng) & 7);
        o.srcB[p] = (uint8_t)(splitmix64(o.rng) & 7);
        // Neutral confidence = 7 (0b0111)
        o.C0[p] = ~0ULL; o.C1[p] = ~0ULL; o.C2[p] = ~0ULL; o.C3[p] = 0ULL;
        o.A0[p] = ~0ULL; o.A1[p] = ~0ULL; o.A2[p] = 0ULL; o.A3[p] = 0ULL;
        o.lock_mask[p] = 0ULL;
    }
    o.tail_err = 0;
    o.recent_err_ema = 128;
    o.err_cooldown = 0;
}

static inline uint8_t predict_from_field(const SwarmOrganism& o) {
    // Tiered confidence voting:
    // Extreme tier (C == 15 or C == 0) gets weight 4; strong tier (C >= 12 or C <= 3) gets weight 1.
    int score = 0;
    for (int p = 0; p < PLANES; ++p) {
        uint64_t pos_strong = o.C3[p] & o.C2[p];
        uint64_t neg_strong = (~o.C3[p]) & (~o.C2[p]);
        uint64_t pos_ext = pos_strong & o.C1[p] & o.C0[p];
        uint64_t neg_ext = neg_strong & (~o.C1[p]) & (~o.C0[p]);

        int v1_strong = __builtin_popcountll(pos_strong & o.S[p]) + __builtin_popcountll(neg_strong & ~o.S[p]);
        int v0_strong = __builtin_popcountll(pos_strong & ~o.S[p]) + __builtin_popcountll(neg_strong & o.S[p]);

        int v1_ext = __builtin_popcountll(pos_ext & o.S[p]) + __builtin_popcountll(neg_ext & ~o.S[p]);
        int v0_ext = __builtin_popcountll(pos_ext & ~o.S[p]) + __builtin_popcountll(neg_ext & o.S[p]);

        score += (v1_strong - v0_strong) + 3 * (v1_ext - v0_ext);
    }
    if (score == 0) return (__builtin_popcountll(o.S[0]) >= 32) ? 1U : 0U;
    return (score > 0) ? 1U : 0U;
}

static inline uint64_t step_organism(SwarmOrganism& o, uint8_t target_bit, EngineMode mode, size_t t, bool in_tail) {
    // 1. Strictly causal prediction BEFORE observing target_bit
    uint8_t pred = predict_from_field(o);
    uint8_t err = pred ^ target_bit;
    if (in_tail) o.tail_err += err;
    o.recent_err_ema = ((o.recent_err_ema * 63ULL) + (err ? 256ULL : 0ULL)) >> 6;

    uint64_t in_word = target_bit ? ~0ULL : 0ULL;

    // 2. Cell-level causal credit assignment (which cells predicted target_bit at this exact tick?)
    for (int p = 0; p < PLANES; ++p) {
        uint64_t agree = ~(o.S[p] ^ in_word);
        uint64_t disagree = ~agree;
        swar_sat4_step(o.C0[p], o.C1[p], o.C2[p], o.C3[p], agree, disagree);
    }

    // 3. Branchless SWAR state transition
    uint64_t next_S[PLANES];
    uint64_t total_flips = 0;
    for (int p = 0; p < PLANES; ++p) {
        uint64_t local_shift = rotl64(o.S[p], 1);
        uint64_t cross_a = rotl64(o.S[o.srcA[p]], o.rotA[p]);
        uint64_t a = (o.shift_en[p] & local_shift) | (~o.shift_en[p] & cross_a);

        uint64_t b_internal = rotl64(o.S[o.srcB[p]], o.rotB[p]);
        uint64_t b = (o.M_in[p] & in_word) | (~o.M_in[p] & b_internal);

        uint64_t g0 = o.g0[p];
        uint64_t g1 = o.g1[p];
        // 00: a (wire/shift), 01: a^b (XOR), 10: a&b (AND), 11: a|b (OR)
        next_S[p] = (~g1 & ~g0 & a)
                  | (~g1 &  g0 & (a ^ b))
                  | ( g1 & ~g0 & (a & b))
                  | ( g1 &  g0 & (a | b));

        uint64_t diff = next_S[p] ^ o.S[p];
        total_flips += __builtin_popcountll(diff);
        // Track per-cell dynamic activity (A): 1 if flipped, 0 if static
        swar_sat4_step(o.A0[p], o.A1[p], o.A2[p], o.A3[p], diff, ~diff);
    }
    for (int p = 0; p < PLANES; ++p) o.S[p] = next_S[p];

    // 4. Controlled timescale plasticity (mutates at most once per ~16-64 ticks so counters C integrate signal!)
    if (mode == H1_RANDOM_DRIFT) {
        if ((t & 31ULL) == 0ULL) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 7ULL);
            uint64_t m = 1ULL << ((r >> 3) & 63ULL);
            uint64_t sel = (r >> 9) & 7ULL;
            if (sel < 2) o.M_in[p] ^= m;
            else if (sel < 4) o.g0[p] ^= m;
            else if (sel < 6) o.g1[p] ^= m;
            else if (sel == 6) o.shift_en[p] ^= m;
            else o.rotA[p] = (uint8_t)((r >> 16) & 63ULL);
        }
    } else if (mode == H2_HIERARCHICAL_RATCHET) {
        if ((t & 15ULL) == 0ULL) {
            for (int p = 0; p < PLANES; ++p) {
                uint64_t strong = (o.C3[p] & o.C2[p] & o.C1[p] & o.C0[p])
                                | (~o.C3[p] & ~o.C2[p] & ~o.C1[p] & ~o.C0[p]);
                o.lock_mask[p] |= strong;
            }
            if (err) {
                uint64_t r = splitmix64(o.rng);
                int p = (int)(r & 7ULL);
                uint64_t m = (1ULL << ((r >> 3) & 63ULL)) & ~o.lock_mask[p];
                if (m) {
                    uint64_t sel = (r >> 9) & 3ULL;
                    if (sel == 0) o.M_in[p] ^= m;
                    else if (sel == 1) o.g0[p] ^= m;
                    else if (sel == 2) o.g1[p] ^= m;
                    else o.shift_en[p] ^= m;
                }
            }
            if ((t & 511ULL) == 0ULL && o.recent_err_ema > 120) {
                int p = (int)(splitmix64(o.rng) & 7ULL);
                o.lock_mask[p] &= splitmix64(o.rng);
            }
        }
    } else if (mode == H3_DISSIPATIVE_F0) {
        // F -> 0 Dissipative Cooling with proper integration window:
        // Only mutates if organism error EMA is still elevated AND targets ONLY unconfident/dead cells!
        if (err && (t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 7ULL);
            uint64_t confident = (o.C3[p] & o.C2[p]) | (~o.C3[p] & ~o.C2[p]);
            // Cold cells (confident) stay frozen; hot cells (~confident) rewire!
            uint64_t hot = ~confident;
            if (hot) {
                int bit = __builtin_ctzll(rotl64(hot, (int)((r >> 3) & 63ULL)));
                int actual_bit = (bit - (int)((r >> 3) & 63ULL)) & 63;
                uint64_t m = 1ULL << actual_bit;
                uint64_t sel = (r >> 10) & 7ULL;
                if (sel < 2) o.M_in[p] ^= m;
                else if (sel < 4) o.g0[p] ^= m;
                else if (sel < 6) o.g1[p] ^= m;
                else o.shift_en[p] ^= m;
            }
            // Slow structural drift of plane routing ONLY when plane is globally unconfident
            if (__builtin_popcountll(confident) < 8 && ((r >> 14) & 3ULL) == 0ULL) {
                o.rotA[p] = (uint8_t)((r >> 16) & 63ULL);
                o.rotB[p] = (uint8_t)((r >> 22) & 63ULL);
                o.srcA[p] = (uint8_t)((r >> 28) & 7ULL);
                o.srcB[p] = (uint8_t)((r >> 31) & 7ULL);
            }
        }
    } else if (mode == H4_EDGE_OF_CHAOS) {
        // Criticality: keep total_flips in [96..256] out of 512 cells, plus slow F->0 rewiring
        if ((t & 15ULL) == 0ULL) {
            if (total_flips < 96 || total_flips > 256) {
                uint64_t r = splitmix64(o.rng);
                int p = (int)(r & 7ULL);
                uint64_t m = 1ULL << ((r >> 3) & 63ULL);
                if (total_flips < 96) {
                    o.g1[p] &= ~m; o.g0[p] |= m; // XOR injects dynamic activity
                } else {
                    o.g0[p] &= ~m; o.g1[p] &= ~m; // Wire/shift damps excess chaos
                }
            } else if (err) {
                uint64_t r = splitmix64(o.rng);
                int p = (int)(r & 7ULL);
                uint64_t confident = (o.C3[p] & o.C2[p]) | (~o.C3[p] & ~o.C2[p]);
                uint64_t m = (1ULL << ((r >> 3) & 63ULL)) & ~confident;
                if (m) {
                    uint64_t sel = (r >> 10) & 3ULL;
                    if (sel == 0) o.M_in[p] ^= m;
                    else if (sel == 1) o.g0[p] ^= m;
                    else if (sel == 2) o.g1[p] ^= m;
                    else o.shift_en[p] ^= m;
                }
            }
        }
    }
    return total_flips;
}

static std::vector<uint8_t> gen_splitmix(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes);
    uint64_t s = 0xDEADBEEFCAFE1234ULL;
    for (size_t i = 0; i < n_bytes; ++i) out[i] = (uint8_t)(splitmix64(s) & 0xFF);
    return out;
}

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

static std::vector<uint8_t> gen_nl_lag(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    uint16_t st = 0xACE1U;
    for (size_t t = 0; t < n_bytes * 8; ++t) {
        uint8_t b = st & 1U;
        if (b) out[t >> 3] |= (1U << (t & 7));
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

struct BaselineStats { double o0, o1, o8_markov; };

static BaselineStats compute_baselines(const std::vector<uint8_t>& data, size_t total_bits) {
    size_t tail_start = (total_bits * 3) / 4;
    size_t c0[2]{}, c1[2][2]{}, c8[256][2]{};
    uint64_t ok0 = 0, ok1 = 0, ok8 = 0, tail_n = 0;
    uint8_t prev = 0, h8 = 0;
    for (size_t t = 0; t < total_bits; ++t) {
        uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
        uint8_t p0 = (c0[1] > c0[0]) ? 1U : 0U;
        uint8_t p1 = (c1[prev][1] > c1[prev][0]) ? 1U : 0U;
        uint8_t p8 = (c8[h8][1] > c8[h8][0]) ? 1U : 0U;
        if (t >= tail_start) {
            tail_n++;
            if (p0 == b) ok0++;
            if (p1 == b) ok1++;
            if (p8 == b) ok8++;
        }
        c0[b]++; c1[prev][b]++; c8[h8][b]++;
        prev = b;
        h8 = (uint8_t)((h8 << 1) | b);
    }
    return {100.0 * ok0 / tail_n, 100.0 * ok1 / tail_n, 100.0 * ok8 / tail_n};
}

static void run_suite(const std::string& name, const std::vector<uint8_t>& data, size_t max_bits) {
    size_t total_bits = std::min(data.size() * 8, max_bits);
    size_t tail_start = (total_bits * 3) / 4;
    size_t tail_bits = total_bits - tail_start;
    BaselineStats bs = compute_baselines(data, total_bits);

    std::cout << "\n========================================================================================\n";
    std::cout << "STREAM: " << name << " (" << total_bits << " bits)\n";
    std::cout << "HONEST BASELINES (Tail 25%): Order-0=" << std::fixed << std::setprecision(2) << bs.o0
              << "% | Order-1=" << bs.o1 << "% | Order-8(256-ctx table)=" << bs.o8_markov << "%\n";
    std::cout << "========================================================================================\n";

    constexpr int POP = 128;
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

            if ((t & 511ULL) == 511ULL) {
                int best_i = 0, worst_i = 0;
                for (int i = 1; i < POP; ++i) {
                    if (pop[i].recent_err_ema < pop[best_i].recent_err_ema) best_i = i;
                    if (pop[i].recent_err_ema > pop[worst_i].recent_err_ema) worst_i = i;
                }
                if (best_i != worst_i && pop[best_i].recent_err_ema + 3 < pop[worst_i].recent_err_ema) {
                    uint64_t kept_tail = pop[worst_i].tail_err;
                    uint64_t kept_rng  = pop[worst_i].rng;
                    pop[worst_i] = pop[best_i];
                    pop[worst_i].tail_err = kept_tail;
                    pop[worst_i].rng = kept_rng;
                    uint64_t r = splitmix64(tourney_rng);
                    int p = (int)(r & 7ULL);
                    uint64_t bitm = 1ULL << ((r >> 3) & 63ULL);
                    if ((r >> 9) & 1ULL) pop[worst_i].M_in[p] ^= bitm;
                    else if ((r >> 10) & 1ULL) pop[worst_i].g0[p] ^= bitm;
                    else pop[worst_i].shift_en[p] ^= bitm;
                    if ((r >> 11) & 1ULL) pop[worst_i].rotA[p] = (uint8_t)((r >> 16) & 63ULL);
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

        int mem_bits = 0, shift_bits = 0, active_cells = 0;
        for (int p = 0; p < PLANES; ++p) {
            mem_bits     += __builtin_popcountll(~pop[best_idx].M_in[p]);
            shift_bits   += __builtin_popcountll(pop[best_idx].shift_en[p]);
            active_cells += __builtin_popcountll(pop[best_idx].A3[p] | pop[best_idx].A2[p]);
        }

        std::cout << std::left << std::setw(18) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << (best_acc - bs.o0)
                  << ", O1:" << std::setw(6) << (best_acc - bs.o1) << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(5) << std::setprecision(1) << avg_flips << "/512"
                  << " | Live=" << (active_cells * 100 / 512) << "% Mem=" << (mem_bits * 100 / 512)
                  << "% Shf=" << (shift_bits * 100 / 512) << "%"
                  << " | " << std::setprecision(2) << sec << "s\n";
    }
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " KARYON-CORE EXP-1175e: PURE SWAR CAUSAL SUBSTRATE (512-CELL SELF-ORGANIZING SHIFT/LOGIC)\n";
    std::cout << " ZERO Hash Tables | ZERO Floats | ZERO Human Role Assignment | 1 bit/tick input only\n";
    std::cout << "========================================================================================\n";

    const size_t N_BYTES = 16384; // 131,072 bits
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
