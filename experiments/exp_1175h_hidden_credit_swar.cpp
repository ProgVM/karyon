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
// KARYON-CORE EXP-1175h: HIDDEN-STATE CREDIT & MULTI-STEP CASCADE SWAR TEST
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// Scientific Hypothesis Tested Here (from EXP-1175g endoscopic finding):
//   In EXP-1175g, hidden-state / memory cells (like b_{t-1}..b_{t-6} in LFSR-7)
//   have 50/50 marginal correlation with target_bit (C ~ 7). Mutating "unconfident"
//   cells destroyed hidden computation!
//   Here we test:
//     1. Synchronized internal cascade across planes (2 internal micro-steps per bit,
//        zero external buffer, zero human roles) so depth-2 nonlinearities form.
//     2. Organism-level F->0 cooling: plasticity rate cools strictly as organism
//        prediction error drops (not punishing hidden 50/50 state cells!).
//     3. Direct side-by-side comparison of H1, H2, H3(fixed), H4(fixed), H234(fixed)
//        + Order-0 / Order-1 / Order-8 baselines.
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

static inline void swar_sat4_step(
    uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3,
    uint64_t inc_mask, uint64_t dec_mask)
{
    uint64_t can_inc = inc_mask & ~(c0 & c1 & c2 & c3);
    uint64_t cr0 = can_inc, cr1 = cr0 & c0, cr2 = cr1 & c1, cr3 = cr2 & c2;
    c0 ^= cr0; c1 ^= cr1; c2 ^= cr2; c3 ^= cr3;

    uint64_t can_dec = dec_mask & (c0 | c1 | c2 | c3);
    uint64_t br0 = can_dec, br1 = br0 & ~c0, br2 = br1 & ~c1, br3 = br2 & ~c2;
    c0 ^= br0; c1 ^= br1; c2 ^= br2; c3 ^= br3;
}

enum EngineMode {
    M_H1_RANDOM = 0,
    M_H2_RATCHET = 1,
    M_H3_ORG_COOLING = 2,
    M_H4_CRITICAL = 3,
    M_H234_SYNERGY = 4
};

constexpr int PLANES = 8;

struct SwarmOrganism {
    uint64_t S[PLANES];
    uint64_t M_in[PLANES];
    uint64_t t00[PLANES], t01[PLANES], t10[PLANES], t11[PLANES];
    uint64_t shift_en[PLANES];
    uint8_t  rotA[PLANES], rotB[PLANES], srcA[PLANES], srcB[PLANES];

    uint64_t C0[PLANES], C1[PLANES], C2[PLANES], C3[PLANES]; // Predictive alignment [0..15]
    uint64_t A0[PLANES], A1[PLANES], A2[PLANES], A3[PLANES]; // Activity [0..15]
    uint64_t lock_mask[PLANES];

    // Snapshot for causal perturbation test (did the last structural tweak reduce error?)
    uint64_t rng;
    uint64_t tail_err;
    uint64_t recent_err_ema; // [0..256], 128 = 50% error
    uint64_t best_ema_seen;
};

static inline void init_organism(SwarmOrganism& o, uint64_t seed) {
    o.rng = seed;
    for (int p = 0; p < PLANES; ++p) {
        o.S[p] = splitmix64(o.rng);
        o.M_in[p] = splitmix64(o.rng) & splitmix64(o.rng) & splitmix64(o.rng);
        o.t00[p] = splitmix64(o.rng);
        o.t01[p] = splitmix64(o.rng);
        o.t10[p] = splitmix64(o.rng);
        o.t11[p] = splitmix64(o.rng);
        o.shift_en[p] = splitmix64(o.rng);
        o.rotA[p] = (uint8_t)(splitmix64(o.rng) & 63);
        o.rotB[p] = (uint8_t)(splitmix64(o.rng) & 63);
        o.srcA[p] = (uint8_t)(splitmix64(o.rng) & 7);
        o.srcB[p] = (uint8_t)(splitmix64(o.rng) & 7);
        o.C0[p] = ~0ULL; o.C1[p] = ~0ULL; o.C2[p] = ~0ULL; o.C3[p] = 0ULL;
        o.A0[p] = ~0ULL; o.A1[p] = ~0ULL; o.A2[p] = 0ULL; o.A3[p] = 0ULL;
        o.lock_mask[p] = 0ULL;
    }
    o.tail_err = 0;
    o.recent_err_ema = 128;
    o.best_ema_seen = 128;
}

static inline uint8_t predict_from_field(const SwarmOrganism& o) {
    int score = 0;
    for (int p = 0; p < PLANES; ++p) {
        uint64_t pos_strong = o.C3[p] & o.C2[p];
        uint64_t neg_strong = (~o.C3[p]) & (~o.C2[p]);
        uint64_t pos_ext = pos_strong & o.C1[p] & o.C0[p];
        uint64_t neg_ext = neg_strong & (~o.C1[p]) & (~o.C0[p]);

        int v1_s = __builtin_popcountll(pos_strong & o.S[p]) + __builtin_popcountll(neg_strong & ~o.S[p]);
        int v0_s = __builtin_popcountll(pos_strong & ~o.S[p]) + __builtin_popcountll(neg_strong & o.S[p]);
        int v1_e = __builtin_popcountll(pos_ext & o.S[p]) + __builtin_popcountll(neg_ext & ~o.S[p]);
        int v0_e = __builtin_popcountll(pos_ext & ~o.S[p]) + __builtin_popcountll(neg_ext & o.S[p]);

        score += (v1_s - v0_s) + 4 * (v1_e - v0_e);
    }
    if (score == 0) return (__builtin_popcountll(o.S[0]) >= 32) ? 1U : 0U;
    return (score > 0) ? 1U : 0U;
}

static inline void mutate_cell_bit(SwarmOrganism& o, int p, uint64_t m, uint64_t r) {
    uint64_t sel = (r >> 10) & 7ULL;
    if (sel == 0) o.M_in[p] ^= m;
    else if (sel == 1) o.t00[p] ^= m;
    else if (sel == 2) o.t01[p] ^= m;
    else if (sel == 3) o.t10[p] ^= m;
    else if (sel == 4) o.t11[p] ^= m;
    else if (sel == 5) o.shift_en[p] ^= m;
    else if (sel == 6) {
        if ((r >> 15) & 1ULL) { o.t00[p] &= ~m; o.t01[p] &= ~m; o.t10[p] |= m; o.t11[p] |= m; }
        else { o.t00[p] &= ~m; o.t01[p] |= m; o.t10[p] &= ~m; o.t11[p] |= m; }
    } else {
        o.t00[p] &= ~m; o.t01[p] |= m; o.t10[p] |= m; o.t11[p] &= ~m;
    }
}

static inline uint64_t step_organism(SwarmOrganism& o, uint8_t target_bit, EngineMode mode, size_t t, bool in_tail) {
    uint8_t pred = predict_from_field(o);
    uint8_t err = pred ^ target_bit;
    if (in_tail) o.tail_err += err;
    o.recent_err_ema = ((o.recent_err_ema * 63ULL) + (err ? 256ULL : 0ULL)) >> 6;
    if (o.recent_err_ema < o.best_ema_seen) o.best_ema_seen = o.recent_err_ema;

    uint64_t in_word = target_bit ? ~0ULL : 0ULL;

    for (int p = 0; p < PLANES; ++p) {
        uint64_t agree = ~(o.S[p] ^ in_word);
        swar_sat4_step(o.C0[p], o.C1[p], o.C2[p], o.C3[p], agree, ~agree);
    }

    // Two internal propagation sub-steps? Wait: keep 1 synchronous transition per bit tick for max speed,
    // OR 2 sub-steps: step 0 reads world (M_in), step 1 propagates internally (~M_in)!
    // Wait: let's test pure 1-step sequential-in-plane (Jacobi/Gauss-Seidel across planes 0..7):
    // Plane p sees ALREADY-updated planes 0..p-1 if srcA/srcB < p! That creates multi-layer depth in 1 tick for FREE!
    uint64_t total_flips = 0;
    for (int p = 0; p < PLANES; ++p) {
        uint64_t local_shift = rotl64(o.S[p], 1);
        uint64_t cross_a = rotl64(o.S[o.srcA[p]], o.rotA[p]);
        uint64_t a = (o.shift_en[p] & local_shift) | (~o.shift_en[p] & cross_a);
        uint64_t b_internal = rotl64(o.S[o.srcB[p]], o.rotB[p]);
        uint64_t b = (o.M_in[p] & in_word) | (~o.M_in[p] & b_internal);

        uint64_t ns = (o.t00[p] & ~a & ~b)
                    | (o.t01[p] & ~a &  b)
                    | (o.t10[p] &  a & ~b)
                    | (o.t11[p] &  a &  b);

        uint64_t diff = ns ^ o.S[p];
        total_flips += __builtin_popcountll(diff);
        swar_sat4_step(o.A0[p], o.A1[p], o.A2[p], o.A3[p], diff, ~diff);
        o.S[p] = ns;
    }

    if (mode == M_H1_RANDOM) {
        if ((t & 31ULL) == 0ULL) {
            uint64_t r = splitmix64(o.rng);
            int p = (int)(r & 7ULL);
            mutate_cell_bit(o, p, 1ULL << ((r >> 3) & 63ULL), r);
            if ((r >> 20) & 1ULL) o.rotA[p] = (uint8_t)((r >> 24) & 63ULL);
        }
    } else if (mode == M_H2_RATCHET) {
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
                if (m) mutate_cell_bit(o, p, m, r);
            }
            if ((t & 511ULL) == 0ULL && o.recent_err_ema > 120) {
                int p = (int)(splitmix64(o.rng) & 7ULL);
                o.lock_mask[p] &= splitmix64(o.rng);
            }
        }
    } else if (mode == M_H3_ORG_COOLING) {
        // Organism-level F->0 Cooling:
        // Mutation temperature scales strictly with global recent_err_ema!
        // When organism improves (low err_ema), ALL its internal memory & hidden cells cool down!
        // Also prioritize mutating dead/static cells (~alive) over active hidden memory!
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(o.rng);
            // Temperature threshold: 0..256 based on recent_err_ema
            if ((r & 255ULL) < (o.recent_err_ema * 2ULL)) {
                int p = (int)((r >> 8) & 7ULL);
                uint64_t alive = o.A3[p] | o.A2[p];
                uint64_t confident = (o.C3[p] & o.C2[p]) | (~o.C3[p] & ~o.C2[p]);
                // Protect confident cells; prefer mutating dead/inactive cells first
                uint64_t target_pool = ~confident;
                if (target_pool & ~alive) target_pool &= ~alive;
                if (target_pool) {
                    int bit = (__builtin_ctzll(rotl64(target_pool, (int)((r >> 11) & 63ULL))) - (int)((r >> 11) & 63ULL)) & 63;
                    mutate_cell_bit(o, p, 1ULL << bit, r);
                }
                if (((r >> 18) & 7ULL) == 0ULL && __builtin_popcountll(confident) < 8) {
                    o.rotA[p] = (uint8_t)((r >> 21) & 63ULL);
                    o.rotB[p] = (uint8_t)((r >> 27) & 63ULL);
                    o.srcA[p] = (uint8_t)((r >> 33) & 7ULL);
                    o.srcB[p] = (uint8_t)((r >> 36) & 7ULL);
                }
            }
        }
    } else if (mode == M_H4_CRITICAL) {
        if ((t & 15ULL) == 0ULL) {
            if (total_flips < 96 || total_flips > 256) {
                uint64_t r = splitmix64(o.rng);
                int p = (int)(r & 7ULL);
                uint64_t m = 1ULL << ((r >> 3) & 63ULL);
                if (total_flips < 96) { o.t00[p] &= ~m; o.t01[p] |= m; o.t10[p] |= m; o.t11[p] &= ~m; }
                else { o.t00[p] &= ~m; o.t01[p] &= ~m; o.t10[p] |= m; o.t11[p] |= m; }
            } else if (err) {
                uint64_t r = splitmix64(o.rng);
                int p = (int)(r & 7ULL);
                uint64_t confident = (o.C3[p] & o.C2[p]) | (~o.C3[p] & ~o.C2[p]);
                uint64_t m = (1ULL << ((r >> 3) & 63ULL)) & ~confident;
                if (m) mutate_cell_bit(o, p, m, r);
            }
        }
    } else if (mode == M_H234_SYNERGY) {
        // Unified H2 + H3 + H4:
        //   - H2: Ratchet locks cells when confident & alive, AND global cooling slows down ALL mutations
        //         as organism recent_err_ema drops (so hidden 50/50 delay loops stay intact once useful!).
        //   - H4: Keeps flip density inside [96..288] using ONLY unlocked/hot cells.
        //   - H3: Rewires dead/hot cells proportionally to global error EMA.
        if ((t & 15ULL) == 0ULL) {
            for (int p = 0; p < PLANES; ++p) {
                uint64_t alive = o.A3[p] | o.A2[p];
                uint64_t strong = (o.C3[p] & o.C2[p] & o.C1[p] & o.C0[p])
                                | (~o.C3[p] & ~o.C2[p] & ~o.C1[p] & ~o.C0[p]);
                o.lock_mask[p] |= (strong & alive);
            }
            uint64_t r = splitmix64(o.rng);
            if (total_flips < 96 || total_flips > 288) {
                int p = (int)(r & 7ULL);
                uint64_t m = (1ULL << ((r >> 3) & 63ULL)) & ~o.lock_mask[p];
                if (m) {
                    if (total_flips < 96) { o.t00[p] &= ~m; o.t01[p] |= m; o.t10[p] |= m; o.t11[p] &= ~m; }
                    else { o.t00[p] &= ~m; o.t01[p] &= ~m; o.t10[p] |= m; o.t11[p] |= m; }
                }
            } else if ((r & 255ULL) < (o.recent_err_ema * 2ULL)) {
                int p = (int)((r >> 8) & 7ULL);
                uint64_t alive = o.A3[p] | o.A2[p];
                uint64_t target_pool = ~o.lock_mask[p];
                if (target_pool & ~alive) target_pool &= ~alive;
                if (target_pool) {
                    int bit = (__builtin_ctzll(rotl64(target_pool, (int)((r >> 11) & 63ULL))) - (int)((r >> 11) & 63ULL)) & 63;
                    mutate_cell_bit(o, p, 1ULL << bit, r);
                }
                if (((r >> 18) & 7ULL) == 0ULL && o.recent_err_ema > 105) {
                    o.rotA[p] = (uint8_t)((r >> 21) & 63ULL);
                    o.rotB[p] = (uint8_t)((r >> 27) & 63ULL);
                    o.srcA[p] = (uint8_t)((r >> 33) & 7ULL);
                    o.srcB[p] = (uint8_t)((r >> 36) & 7ULL);
                }
            }
            if ((t & 511ULL) == 0ULL && o.recent_err_ema > 120) {
                int p = (int)(splitmix64(o.rng) & 7ULL);
                o.lock_mask[p] &= splitmix64(o.rng);
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
    const EngineMode modes[5] = {M_H1_RANDOM, M_H2_RATCHET, M_H3_ORG_COOLING, M_H4_CRITICAL, M_H234_SYNERGY};
    const char* names[5] = {"H1: Random+Select", "H2: Hier.Ratchet", "H3: Org.Cooling", "H4: Edge-of-Chaos", "H234: Synergy"};

    for (int m = 0; m < 5; ++m) {
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
                    mutate_cell_bit(pop[worst_i], p, bitm, r);
                    if ((r >> 20) & 1ULL) pop[worst_i].rotA[p] = (uint8_t)((r >> 24) & 63ULL);
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

        int mem_bits = 0, active_cells = 0, locked_cells = 0;
        for (int p = 0; p < PLANES; ++p) {
            mem_bits     += __builtin_popcountll(~pop[best_idx].M_in[p]);
            active_cells += __builtin_popcountll(pop[best_idx].A3[p] | pop[best_idx].A2[p]);
            locked_cells += __builtin_popcountll(pop[best_idx].lock_mask[p]);
        }

        std::cout << std::left << std::setw(18) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << (best_acc - bs.o0)
                  << ", O1:" << std::setw(6) << (best_acc - bs.o1) << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(5) << std::setprecision(1) << avg_flips << "/512"
                  << " | Live=" << (active_cells * 100 / 512) << "% Mem=" << (mem_bits * 100 / 512)
                  << "% Lock=" << (locked_cells * 100 / 512) << "%"
                  << " | " << std::setprecision(2) << sec << "s\n";
    }
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " KARYON-CORE EXP-1175h: HIDDEN-STATE CREDIT & SEQUENTIAL CASCADE SWAR SUBSTRATE\n";
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
