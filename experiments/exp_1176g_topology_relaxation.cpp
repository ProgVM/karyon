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
// KARYON-CORE EXP-1176g: TOPOLOGY & RELAXATION TIME ON SINGLE-REGISTER PHYSICS
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// Strictly ZERO layers, ZERO masks, ZERO weighted voting, ZERO lookup tables!
//   - Entire physical universe is ONE homogeneous 64-site uint64_t field S.
//   - Prediction is literally S & 1ULL at contact site 0 before target_bit arrives.
//   - Tests the exact physical bottleneck discovered in EXP-1176..1176f:
//     Does moving from 1D Ring (2 neighbors, 1 tick/bit) to:
//       (A) 1D Ring with 2 internal relaxation ticks per world bit
//       (B) 2D 8x8 Torus (4 local neighbors N/S/E/W in 1 uint64_t, 1 tick/bit)
//       (C) 2D 8x8 Torus with 2 internal relaxation ticks per world bit
//     break through the 68% ceiling on LFSR-7, Nonlinear Lag, Rule 110, & Wiki?
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

static inline uint64_t rotr64(uint64_t x, int k) {
    uint32_t s = (uint32_t)k & 63U;
    return s ? ((x >> s) | (x << (64U - s))) : x;
}

// Pure branchless 8x8 torus shifts inside a single uint64_t (zero loops!)
static inline void torus8x8_neighbors(uint64_t S, uint64_t& N, uint64_t& So, uint64_t& E, uint64_t& W) {
    constexpr uint64_t NOT_COL0 = 0xFEFEFEFEFEFEFEFEULL;
    constexpr uint64_t COL0     = 0x0101010101010101ULL;
    constexpr uint64_t NOT_COL7 = 0x7F7F7F7F7F7F7F7FULL;
    constexpr uint64_t COL7     = 0x8080808080808080ULL;
    E = ((S & NOT_COL7) << 1) | ((S & COL7) >> 7);
    W = ((S & NOT_COL0) >> 1) | ((S & COL0) << 7);
    So = rotl64(S, 8);
    N  = rotr64(S, 8);
}

enum TopoMode {
    T_1D_1TICK = 0,
    T_1D_2TICK = 1,
    T_2D_1TICK = 2,
    T_2D_2TICK = 3
};

struct Field64 {
    uint64_t S;
    uint64_t K0, K1, J; // Homogeneous local coupling rules across 64 sites
    uint64_t err_fast, err_slow, snap_err;
    uint64_t dK0, dK1, dJ;
    uint32_t probe_timer;
    uint64_t rng;
    uint64_t tail_err, ood_err;
    uint32_t flips_ema;
};

static inline void init_field(Field64& u, uint64_t seed) {
    u.rng = seed;
    u.S = splitmix64(u.rng);
    u.K0 = splitmix64(u.rng);
    u.K1 = splitmix64(u.rng);
    u.J  = 0ULL;
    u.err_fast = 512;
    u.err_slow = 512;
    u.snap_err = 512;
    u.dK0 = 0; u.dK1 = 0; u.dJ = 0;
    u.probe_timer = 0;
    u.tail_err = 0;
    u.ood_err = 0;
    u.flips_ema = 20;
}

static inline uint64_t physics_step(const Field64& u, uint64_t S_in, TopoMode tm) {
    if (tm == T_1D_1TICK || tm == T_1D_2TICK) {
        uint64_t L = rotl64(S_in, 1);
        uint64_t R = rotr64(S_in, 1);
        uint64_t C = S_in;
        uint64_t op_adv  = L;
        uint64_t op_xor  = L ^ R;
        uint64_t op_maj  = (L & C) | (C & R) | (L & R);
        uint64_t op_asym = (L & ~C) ^ R;
        return ((~u.K1 & ~u.K0 & op_adv)
              | (~u.K1 &  u.K0 & op_xor)
              | ( u.K1 & ~u.K0 & op_maj)
              | ( u.K1 &  u.K0 & op_asym)) ^ u.J;
    } else {
        uint64_t N, So, E, W;
        torus8x8_neighbors(S_in, N, So, E, W);
        uint64_t C = S_in;
        // 4 purely local 2D isotropic/wave operators:
        // 00: Directional wave flow (E)
        // 01: 2D Parity / Wave interference (N ^ S ^ E ^ W)
        // 10: 5-spin Majority (self + 4 neighbors -> droplet cohesion!)
        // 11: Nonlinear 2D Rule ((E ^ N) & ~W) ^ So
        uint64_t s1 = N ^ So ^ E;
        uint64_t c1 = (N & So) | (So & E) | (N & E);
        uint64_t s2 = W ^ C ^ s1;
        uint64_t c2 = (W & C) | (C & s1) | (W & s1);
        uint64_t maj5 = (c1 & c2) | ((c1 ^ c2) & s2);

        uint64_t op_adv  = E;
        uint64_t op_xor  = N ^ So ^ E ^ W;
        uint64_t op_maj  = maj5;
        uint64_t op_asym = ((E ^ N) & ~W) ^ So;
        return ((~u.K1 & ~u.K0 & op_adv)
              | (~u.K1 &  u.K0 & op_xor)
              | ( u.K1 & ~u.K0 & op_maj)
              | ( u.K1 &  u.K0 & op_asym)) ^ u.J;
    }
}

static inline uint32_t step_field(Field64& u, uint8_t target_bit, TopoMode tm, size_t t, bool in_tail, bool in_ood) {
    uint8_t pred = (uint8_t)(u.S & 1ULL);
    uint8_t err = pred ^ target_bit;
    if (in_tail) u.tail_err += err;
    if (in_ood)  u.ood_err += err;

    u.err_fast = ((u.err_fast * 63ULL) + (err ? 1024ULL : 0ULL)) >> 6;
    u.err_slow = ((u.err_slow * 511ULL) + (err ? 1024ULL : 0ULL)) >> 9;

    // External stimulus arrives at contact site 0
    uint64_t S_cur = (u.S & ~1ULL) | (uint64_t)(target_bit & 1U);
    int n_ticks = (tm == T_1D_2TICK || tm == T_2D_2TICK) ? 2 : 1;

    uint32_t flips = 0;
    for (int k = 0; k < n_ticks; ++k) {
        uint64_t S_nxt = physics_step(u, S_cur, tm);
        if (k == 0) flips = (uint32_t)__builtin_popcountll(S_nxt ^ S_cur);
        S_cur = S_nxt;
    }
    u.S = S_cur;
    u.flips_ema = (uint32_t)(((uint64_t)u.flips_ema * 15ULL + flips) >> 4);

    // Unified Physical Evolution (Autopoiesis + Criticality + Causal Probe Rollback)
    if (u.probe_timer > 0) {
        u.probe_timer--;
        if (u.probe_timer == 0 && (u.dK0 | u.dK1 | u.dJ)) {
            if (u.err_fast > u.snap_err + 4) {
                u.K0 ^= u.dK0; u.K1 ^= u.dK1; u.J ^= u.dJ;
            }
            u.dK0 = 0; u.dK1 = 0; u.dJ = 0;
        }
    } else if ((t & 15ULL) == 0ULL) {
        uint64_t r = splitmix64(u.rng);
        int site = (int)(r & 63ULL);
        uint64_t m = 1ULL << site;
        if (u.flips_ema < 8) {
            u.K1 &= ~m; u.K0 |= m; // Re-excite frozen tissue
        } else if (u.flips_ema > 32) {
            u.K1 |= m; u.K0 &= ~m; u.J &= ~m; // Ferromagnetic cohesion
        } else if (err && u.err_fast + 6 >= u.err_slow) {
            uint64_t sel = (r >> 8) & 3ULL;
            u.dK0 = (sel == 0 || sel == 3) ? m : 0ULL;
            u.dK1 = (sel == 1 || sel == 3) ? m : 0ULL;
            u.dJ  = (sel == 2) ? m : 0ULL;
            u.K0 ^= u.dK0; u.K1 ^= u.dK1; u.J ^= u.dJ;
            u.snap_err = u.err_fast;
            u.probe_timer = 64;
        }
    }
    return flips;
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
        if (reg & 1U) out[t >> 3] |= (1U << (t & 7));
        uint8_t fb = (reg ^ (reg >> 1)) & 1U;
        reg = (uint8_t)((reg >> 1) | (fb << 6));
    }
    return out;
}

static std::vector<uint8_t> gen_nl_lag(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    uint16_t st = 0xACE1U;
    for (size_t t = 0; t < n_bytes * 8; ++t) {
        if (st & 1U) out[t >> 3] |= (1U << (t & 7));
        uint8_t fb = (((st >> 2) & (st >> 4)) ^ (st >> 7) ^ (st >> 11)) & 1U;
        st = (uint16_t)((st >> 1) | ((uint16_t)fb << 15));
    }
    return out;
}

static std::vector<uint8_t> gen_rule110_with_ood(size_t n_bytes, size_t W = 64) {
    std::vector<uint8_t> out(n_bytes, 0);
    std::vector<uint8_t> row(W, 0), nxt(W, 0);
    row[W / 2] = 1; row[W / 3] = 1;
    size_t t = 0, tot = n_bytes * 8;
    size_t ood_start = (tot * 3) / 4;
    while (t < tot) {
        for (size_t x = 0; x < W && t < tot; ++x, ++t) {
            uint8_t bit = row[x] ^ (t >= ood_start ? 1U : 0U);
            if (bit) out[t >> 3] |= (1U << (t & 7));
        }
        for (size_t x = 0; x < W; ++x) {
            uint8_t pat = (row[(x + W - 1) % W] << 2) | (row[x] << 1) | row[(x + 1) % W];
            nxt[x] = (110 >> pat) & 1U;
        }
        row.swap(nxt);
    }
    return out;
}

struct BaselineStats { double o0, o1, o8; };
static BaselineStats compute_baselines(const std::vector<uint8_t>& data, size_t start, size_t end) {
    size_t c0[2]{}, c1[2][2]{}, c8[256][2]{};
    uint64_t ok0 = 0, ok1 = 0, ok8 = 0, n = 0;
    uint8_t prev = 0, h8 = 0;
    for (size_t t = 0; t < end; ++t) {
        uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
        uint8_t p0 = (c0[1] > c0[0]) ? 1U : 0U;
        uint8_t p1 = (c1[prev][1] > c1[prev][0]) ? 1U : 0U;
        uint8_t p8 = (c8[h8][1] > c8[h8][0]) ? 1U : 0U;
        if (t >= start) { n++; if (p0 == b) ok0++; if (p1 == b) ok1++; if (p8 == b) ok8++; }
        c0[b]++; c1[prev][b]++; c8[h8][b]++;
        prev = b; h8 = (uint8_t)((h8 << 1) | b);
    }
    return {100.0 * ok0 / n, 100.0 * ok1 / n, 100.0 * ok8 / n};
}

static void run_suite(const std::string& name, const std::vector<uint8_t>& data, size_t max_bits, bool has_ood = false) {
    size_t total_bits = std::min(data.size() * 8, max_bits);
    size_t tail_start = has_ood ? (total_bits / 2) : ((total_bits * 3) / 4);
    size_t tail_end   = has_ood ? ((total_bits * 3) / 4) : total_bits;
    size_t tail_bits  = tail_end - tail_start;
    size_t ood_start  = (total_bits * 3) / 4;
    size_t ood_bits   = total_bits - ood_start;
    BaselineStats bs  = compute_baselines(data, tail_start, tail_end);

    std::cout << "\n==================================================================================================\n";
    std::cout << "WORLD: " << name << " (" << total_bits << " bits)\n";
    std::cout << "BASELINES: Order-0=" << std::fixed << std::setprecision(2) << bs.o0
              << "% | Order-1=" << bs.o1 << "% | Order-8(256-ctx)=" << bs.o8 << "%\n";
    std::cout << "==================================================================================================\n";

    constexpr int POP = 512;
    const TopoMode modes[4] = {T_1D_1TICK, T_1D_2TICK, T_2D_1TICK, T_2D_2TICK};
    const char* names[4] = {"1D Ring (1 tick)", "1D Ring (2 tick)", "2D 8x8 Torus (1t)", "2D 8x8 Torus (2t)"};

    for (int m = 0; m < 4; ++m) {
        std::vector<Field64> pop(POP);
        for (int i = 0; i < POP; ++i) init_field(pop[i], 0x9E3779B97F4A7C15ULL * (i + 1) + (uint64_t)m * 7919ULL);

        uint64_t tourney_rng = 0xCAFEBABE12345678ULL + m;
        uint64_t sum_flips = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < total_bits; ++t) {
            uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
            bool in_tail = (t >= tail_start && t < tail_end);
            bool in_ood  = (has_ood && t >= ood_start);

            #pragma omp parallel for reduction(+:sum_flips) schedule(static)
            for (int i = 0; i < POP; ++i) {
                uint32_t f = step_field(pop[i], b, modes[m], t, in_tail, in_ood);
                if (in_tail) sum_flips += f;
            }

            if ((t & 511ULL) == 511ULL) {
                int best_i = 0, worst_i = 0;
                for (int i = 1; i < POP; ++i) {
                    if (pop[i].err_fast < pop[best_i].err_fast) best_i = i;
                    if (pop[i].err_fast > pop[worst_i].err_fast) worst_i = i;
                }
                if (best_i != worst_i && pop[best_i].err_fast + 12 < pop[worst_i].err_fast) {
                    uint64_t kt = pop[worst_i].tail_err, ko = pop[worst_i].ood_err, kr = pop[worst_i].rng;
                    pop[worst_i] = pop[best_i];
                    pop[worst_i].tail_err = kt;
                    pop[worst_i].ood_err  = ko;
                    pop[worst_i].rng      = kr;
                    uint64_t r = splitmix64(tourney_rng);
                    uint64_t bitm = 1ULL << (r & 63ULL);
                    if ((r >> 8) & 1ULL) pop[worst_i].K0 ^= bitm;
                    else pop[worst_i].K1 ^= bitm;
                }
            }
        }

        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        uint64_t min_err = pop[0].tail_err;
        double sum_err = 0.0, min_ood = 1e18;
        for (int i = 0; i < POP; ++i) {
            sum_err += (double)pop[i].tail_err;
            if (pop[i].tail_err < min_err) min_err = pop[i].tail_err;
            if ((double)pop[i].ood_err < min_ood) min_ood = (double)pop[i].ood_err;
        }
        double best_acc = 100.0 * (1.0 - (double)min_err / tail_bits);
        double avg_acc  = 100.0 * (1.0 - (sum_err / POP) / tail_bits);
        double avg_flips = (double)sum_flips / ((double)POP * tail_bits);
        double ood_acc = has_ood ? 100.0 * (1.0 - min_ood / ood_bits) : 0.0;

        std::cout << std::left << std::setw(18) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << (best_acc - bs.o0)
                  << ", O1:" << std::setw(6) << (best_acc - bs.o1) << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(4) << std::setprecision(1) << avg_flips << "/64";
        if (has_ood) std::cout << " | OOD=" << std::setprecision(2) << ood_acc << "%";
        std::cout << " | " << std::setprecision(2) << sec << "s\n";
    }
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1176g: 1D RING vs 2D 8x8 TORUS x RELAXATION TICKS (SINGLE uint64_t SUBSTRATE)\n";
    std::cout << " ZERO Layer-Cakes | ZERO Masks | ZERO Weighted Readout | Prediction = S & 1 at site 0\n";
    std::cout << "==================================================================================================\n";

    const size_t N_BYTES = 16384; // 131,072 bits
    run_suite("Control 0: SplitMix64 Crypto Noise", gen_splitmix(N_BYTES), N_BYTES * 8);
    run_suite("Law 1: LFSR-7 (50/50 balanced, zero bias)", gen_lfsr7(N_BYTES), N_BYTES * 8);
    run_suite("Law 2: Nonlinear Lag (b_{t-3}&b_{t-5})^b_{t-8}^b_{t-12}", gen_nl_lag(N_BYTES), N_BYTES * 8);
    run_suite("Law 3: Rule 110 + OOD Inversion at 75%", gen_rule110_with_ood(N_BYTES), N_BYTES * 8, true);

    std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
    if (fw) {
        std::vector<uint8_t> wiki(N_BYTES);
        fw.read((char*)wiki.data(), N_BYTES);
        wiki.resize((size_t)fw.gcount());
        run_suite("Real 4: Wikipedia Stream (1 bit/tick at site 0)", wiki, wiki.size() * 8);
    }
    return 0;
}
