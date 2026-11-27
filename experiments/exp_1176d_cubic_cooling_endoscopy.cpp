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
// KARYON-CORE EXP-1176d: DEEP ENDOSCOPY + UNIFIED COOLING / H1..H6 BENCHMARK
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// Key Lesson from 1176 -> 1176b -> 1176c:
//   In 1176c, mutating good organisms at 2*err_ema/256 (~55%) every 8 ticks
//   shredded every discovered causal ring before it could propagate or win!
//   In 1176, H5 cooled below ema <= 100 and jumped to 68.23% on Wikipedia!
//
// Here we inspect the EXACT evolved ring of 1176-style H5 + test a clean
// threshold-free **steep nonlinear cooling** (no magic constant 100!) AND
// compare H1..H6 + Unified Field on BOTH 131K-bit & OOD streams + honest baselines.
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

enum Hypothesis {
    H1_RANDOM_SURVIVAL = 0,
    H2_HIER_COMPOSITION = 1,
    H3_DISSIPATIVE_F0 = 2,
    H4_CRITICAL_EDGE = 3,
    H5_AUTOPOIESIS = 4,
    H6_CONJUGATE_WAVE = 5,
    H_UNIFIED_FIELD = 6
};

struct RingUniverse {
    uint64_t S;
    uint64_t P;
    uint64_t K0, K1, J;
    uint64_t H_stress;
    uint64_t Bond;

    uint64_t rng;
    uint64_t tail_err;
    uint64_t recent_err_ema; // [0..256], 128 = 50% error
    uint64_t min_err_ema;    // Lowest EMA achieved by this lineage (endogenous annealing!)
    uint64_t ood_err;
    uint32_t flips_ema;
};

static inline void init_universe(RingUniverse& u, uint64_t seed, Hypothesis h) {
    u.rng = seed;
    u.S = splitmix64(u.rng);
    u.P = splitmix64(u.rng);
    u.K0 = splitmix64(u.rng);
    u.K1 = splitmix64(u.rng);
    u.J  = (h == H4_CRITICAL_EDGE || h == H5_AUTOPOIESIS || h == H6_CONJUGATE_WAVE || h == H_UNIFIED_FIELD)
           ? 0ULL : splitmix64(u.rng);
    u.H_stress = ~0ULL;
    u.Bond = 0ULL;
    u.tail_err = 0;
    u.recent_err_ema = 128;
    u.min_err_ema = 128;
    u.ood_err = 0;
    u.flips_ema = 20;
}

static inline void flip_local_coupling(RingUniverse& u, int site, uint64_t r) {
    uint64_t m = 1ULL << (site & 63);
    uint64_t sel = (r >> 8) & 3ULL;
    if (sel == 0) u.K0 ^= m;
    else if (sel == 1) u.K1 ^= m;
    else if (sel == 2) u.J ^= m;
    else { u.K0 ^= m; u.K1 ^= m; }
}

static inline uint32_t step_universe(
    RingUniverse& u, uint8_t target_bit, Hypothesis h,
    size_t t, bool in_tail, bool in_ood)
{
    uint8_t pred = (uint8_t)(u.S & 1ULL);
    uint8_t err = pred ^ target_bit;

    if (in_tail) u.tail_err += err;
    if (in_ood)  u.ood_err += err;
    u.recent_err_ema = ((u.recent_err_ema * 63ULL) + (err ? 256ULL : 0ULL)) >> 6;
    if ((t & 511ULL) == 0ULL) {
        // Slowly relax baseline so OOD shift can re-heat if needed
        u.min_err_ema = (u.min_err_ema * 15ULL + u.recent_err_ema) >> 4;
    }
    if (u.recent_err_ema < u.min_err_ema) u.min_err_ema = u.recent_err_ema;

    uint64_t S_contact = (u.S & ~1ULL) | (uint64_t)(target_bit & 1U);

    uint64_t L = rotl64(S_contact, 1);
    uint64_t R = rotr64(S_contact, 1);
    uint64_t C = S_contact;

    uint64_t op_adv  = L;
    uint64_t op_xor  = L ^ R;
    uint64_t op_maj  = (L & C) | (C & R) | (L & R);
    uint64_t op_asym = (L & ~C) ^ R;

    uint64_t raw_next = (~u.K1 & ~u.K0 & op_adv)
                      | (~u.K1 &  u.K0 & op_xor)
                      | ( u.K1 & ~u.K0 & op_maj)
                      | ( u.K1 &  u.K0 & op_asym);
    raw_next ^= u.J;

    uint64_t next_S = raw_next;
    if (h == H6_CONJUGATE_WAVE || h == H_UNIFIED_FIELD) {
        // Wave-inertia sites (Adv/XOR with K1==0) carry reversible momentum;
        // nonlinear sites (Maj/Asym with K1==1) dissipate/mix.
        uint64_t wave_sites = (h == H6_CONJUGATE_WAVE) ? (~u.K1) : (~u.K1 & u.K0);
        next_S = (wave_sites & (raw_next ^ u.P)) | (~wave_sites & raw_next);
    }

    uint64_t diff = next_S ^ S_contact;
    uint32_t flips = (uint32_t)__builtin_popcountll(diff);
    u.flips_ema = (uint32_t)(((uint64_t)u.flips_ema * 15ULL + flips) >> 4);
    uint64_t bulk_flips = __builtin_popcountll((diff >> 8) & 0x0000FFFFFFFFFFFFULL);

    u.P = S_contact;
    u.S = next_S;

    // Continuous cubic cooling factor (zero magic threshold!):
    // If err_ema = 128 (50% chaos), (128/128)^3 = 1.0 (hot!).
    // If err_ema = 64  (75% acc),   (64/128)^3  = 0.125 (8x cooler!).
    // If err_ema = 32  (87% acc),   (32/128)^3  = 0.015 (64x cooler!).
    uint64_t e = (u.recent_err_ema > 128ULL) ? 128ULL : u.recent_err_ema;
    uint64_t temp256 = (e * e * e) >> 14; // 0..128 -> 0..128 cubic! Wait: 128^3 / 2^14 = 2^21 / 2^14 = 128!

    if (h == H1_RANDOM_SURVIVAL) {
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            flip_local_coupling(u, (int)(r & 63ULL), r);
        }
    } else if (h == H2_HIER_COMPOSITION) {
        if (err) {
            u.Bond &= ~(1ULL | 2ULL | (1ULL << 63));
            if ((t & 15ULL) == 0ULL) {
                uint64_t r = splitmix64(u.rng);
                if ((r & 127ULL) < temp256) {
                    uint64_t free_sites = ~u.Bond;
                    if (free_sites) {
                        int s = (__builtin_ctzll(rotl64(free_sites, (int)(r & 63))) - (int)(r & 63)) & 63;
                        flip_local_coupling(u, s, r);
                    }
                }
            }
        } else {
            u.Bond |= (diff & ~rotl64(diff, 1));
            if ((t & 127ULL) == 0ULL) u.Bond &= splitmix64(u.rng);
        }
    } else if (h == H3_DISSIPATIVE_F0) {
        if (err) u.H_stress |= 1ULL;
        else u.H_stress &= ~1ULL;
        uint64_t spread = rotl64(u.H_stress, 1) | rotr64(u.H_stress, 1);
        u.H_stress = err ? (u.H_stress | spread) : (u.H_stress & ~1ULL & spread);
        if ((t & 15ULL) == 0ULL && u.H_stress) {
            uint64_t r = splitmix64(u.rng);
            if (((r >> 16) & 127ULL) < temp256) {
                int s = (__builtin_ctzll(rotl64(u.H_stress, (int)(r & 63))) - (int)(r & 63)) & 63;
                flip_local_coupling(u, s, r);
            }
        }
    } else if (h == H4_CRITICAL_EDGE) {
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            int s = (int)(r & 63ULL);
            uint64_t m = 1ULL << s;
            if (u.flips_ema < 12) { u.K1 &= ~m; u.K0 |= m; }
            else if (u.flips_ema > 28) { u.K1 |= m; u.K0 &= ~m; u.J &= ~m; }
            else if (err && (((r >> 16) & 127ULL) < temp256)) {
                flip_local_coupling(u, s, r);
            }
        }
    } else if (h == H5_AUTOPOIESIS) {
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            int s = (int)(r & 63ULL);
            uint64_t m = 1ULL << s;
            if (bulk_flips < 6) {
                u.K1 &= ~m; u.K0 |= m; u.J ^= m;
            } else if (bulk_flips > 30) {
                u.K1 |= m; u.K0 &= ~m; // Keep exact EXP-1176 bulk damping
            } else if (err && (((r >> 16) & 127ULL) < temp256)) {
                flip_local_coupling(u, s, r);
            }
        }
    } else if (h == H6_CONJUGATE_WAVE) {
        if (err && (t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            if (((r >> 16) & 127ULL) < temp256) {
                flip_local_coupling(u, (int)((r >> 10) & 63ULL), r);
            }
        }
    } else if (h == H_UNIFIED_FIELD) {
        // Unified Field: cubic F->0 cooling + stress localization + critical bulk vitality + conjugate inertia
        if (err) u.H_stress |= 1ULL;
        else u.H_stress &= ~1ULL;
        uint64_t spread = rotl64(u.H_stress, 1) | rotr64(u.H_stress, 1);
        u.H_stress = err ? (u.H_stress | spread) : (u.H_stress & ~1ULL & spread);

        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            int s = (int)(r & 63ULL);
            uint64_t m = 1ULL << s;
            if (bulk_flips < 6) {
                u.K1 &= ~m; u.K0 |= m;
            } else if (bulk_flips > 30) {
                u.K1 |= m; u.K0 &= ~m; u.J &= ~m;
            } else if (err && (((r >> 16) & 127ULL) < temp256)) {
                uint64_t hot = u.H_stress ? u.H_stress : ~0ULL;
                int hs = (__builtin_ctzll(rotl64(hot, (int)(r & 63))) - (int)(r & 63)) & 63;
                flip_local_coupling(u, hs, r);
            }
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
        prev = b;
        h8 = (uint8_t)((h8 << 1) | b);
    }
    return {100.0 * ok0 / n, 100.0 * ok1 / n, 100.0 * ok8 / n};
}

static void print_ring_endoscopy(const char* label, const RingUniverse& u) {
    std::string ops(64, '.');
    int cnt[4]{}, j_cnt = 0;
    for (int i = 0; i < 64; ++i) {
        int k = ((u.K1 >> i) & 1ULL) * 2 + ((u.K0 >> i) & 1ULL);
        cnt[k]++;
        if ((u.J >> i) & 1ULL) j_cnt++;
        const char sym[4] = {'>', '^', 'M', 'A'};
        ops[i] = sym[k];
    }
    std::cout << "    [Endoscopy " << label << "] Ops[0..63]: " << ops
              << " | Adv=" << cnt[0] << " XOR=" << cnt[1] << " Maj=" << cnt[2] << " Asym=" << cnt[3]
              << " PolInv=" << j_cnt << "\n";
}

static void run_suite(const std::string& name, const std::vector<uint8_t>& data, size_t max_bits, bool has_ood = false) {
    size_t total_bits = std::min(data.size() * 8, max_bits);
    size_t tail_start = has_ood ? (total_bits / 2) : ((total_bits * 3) / 4);
    size_t tail_end   = has_ood ? ((total_bits * 3) / 4) : total_bits;
    size_t tail_bits  = tail_end - tail_start;
    size_t ood_start  = (total_bits * 3) / 4;
    size_t ood_bits   = total_bits - ood_start;

    BaselineStats bs = compute_baselines(data, tail_start, tail_end);

    std::cout << "\n==================================================================================================\n";
    std::cout << "WORLD: " << name << " (" << total_bits << " bits)\n";
    std::cout << "HONEST BASELINES: Order-0=" << std::fixed << std::setprecision(2) << bs.o0
              << "% | Order-1=" << bs.o1 << "% | Order-8(256-ctx)=" << bs.o8 << "%\n";
    std::cout << "==================================================================================================\n";

    constexpr int POP = 512;
    const Hypothesis hyps[7] = {
        H1_RANDOM_SURVIVAL, H2_HIER_COMPOSITION, H3_DISSIPATIVE_F0,
        H4_CRITICAL_EDGE, H5_AUTOPOIESIS, H6_CONJUGATE_WAVE, H_UNIFIED_FIELD
    };
    const char* names[7] = {
        "H1: Random+Survive", "H2: Hier.Dimers", "H3: F->0 Cooling",
        "H4: Edge-of-Chaos", "H5: Autopoiesis", "H6: ConjugateWave", "ALL: UnifiedField"
    };

    for (int m = 0; m < 7; ++m) {
        std::vector<RingUniverse> pop(POP);
        for (int i = 0; i < POP; ++i) init_universe(pop[i], 0x9E3779B97F4A7C15ULL * (i + 1) + (uint64_t)m * 7919ULL, hyps[m]);

        uint64_t tourney_rng = 0xCAFEBABE12345678ULL + m;
        uint64_t sum_flips = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < total_bits; ++t) {
            uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
            bool in_tail = (t >= tail_start && t < tail_end);
            bool in_ood  = (has_ood && t >= ood_start);

            #pragma omp parallel for reduction(+:sum_flips) schedule(static)
            for (int i = 0; i < POP; ++i) {
                uint32_t f = step_universe(pop[i], b, hyps[m], t, in_tail, in_ood);
                if (in_tail) sum_flips += f;
            }

            if ((t & 511ULL) == 511ULL) {
                int best_i = 0, worst_i = 0;
                for (int i = 1; i < POP; ++i) {
                    if (pop[i].recent_err_ema < pop[best_i].recent_err_ema) best_i = i;
                    if (pop[i].recent_err_ema > pop[worst_i].recent_err_ema) worst_i = i;
                }
                if (best_i != worst_i && pop[best_i].recent_err_ema + 3 < pop[worst_i].recent_err_ema) {
                    uint64_t kt = pop[worst_i].tail_err, ko = pop[worst_i].ood_err, kr = pop[worst_i].rng;
                    pop[worst_i] = pop[best_i];
                    pop[worst_i].tail_err = kt;
                    pop[worst_i].ood_err  = ko;
                    pop[worst_i].rng      = kr;
                    uint64_t r = splitmix64(tourney_rng);
                    flip_local_coupling(pop[worst_i], (int)(r & 63ULL), r);
                }
            }
        }

        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        uint64_t min_err = pop[0].tail_err;
        int best_idx = 0;
        double sum_err = 0.0, min_ood = 1e18;
        for (int i = 0; i < POP; ++i) {
            sum_err += (double)pop[i].tail_err;
            if (pop[i].tail_err < min_err) { min_err = pop[i].tail_err; best_idx = i; }
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
        if (m == 3 || m == 4 || m == 5 || m == 6) print_ring_endoscopy(names[m], pop[best_idx]);
    }
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1176d: CUBIC COOLING + HOMOGENEOUS RING ENDOSCOPY (H1..H6 + UNIFIED)\n";
    std::cout << " ZERO Layer-Cakes | ZERO Masks | ZERO Weighted Readout | 131,072 bits/world\n";
    std::cout << "==================================================================================================\n";

    const size_t N_BYTES = 16384; // 131,072 bits
    run_suite("Control 0: SplitMix64 Crypto Noise", gen_splitmix(N_BYTES), N_BYTES * 8);
    run_suite("Law 1: LFSR-7 (50/50 balanced, zero bias)", gen_lfsr7(N_BYTES), N_BYTES * 8);
    run_suite("Law 2: Rule 110 + OOD Inversion at 75%", gen_rule110_with_ood(N_BYTES), N_BYTES * 8, true);

    std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
    if (fw) {
        std::vector<uint8_t> wiki(N_BYTES);
        fw.read((char*)wiki.data(), N_BYTES);
        wiki.resize((size_t)fw.gcount());
        run_suite("Real 3: Wikipedia Stream (1 bit/tick at site 0)", wiki, wiki.size() * 8);
    }
    return 0;
}
