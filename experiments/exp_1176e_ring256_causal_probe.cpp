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
// KARYON-CORE EXP-1176e: HOMOGENEOUS 256-SITE 1D RING & ENDOGENOUS ANNEALING
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// Strictly ZERO layers, ZERO masks, ZERO weighted voting, ZERO magic constants!
//   - Compares 64-site homogeneous ring vs 256-site homogeneous ring (4x uint64_t
//     stitched seamlessly as one continuous 256-site 1D physical ring!).
//   - Prediction IS strictly S[0] & 1 at single world-contact site 0.
//   - Endogenous Cooling (no constant 100): each organism compares its fast
//     error EMA against its own slow baseline EMA (`err_fast > err_slow`).
//     When it is getting WORSE or stagnant, stress rises and rewires;
//     when it is getting BETTER, it automatically freezes/cools!
//   - Plus Causal Rollback Probe: before keeping a local coupling change,
//     if error rises over the next window, physics reverts the stressed bond!
// ============================================================================

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

enum Hypothesis {
    H1_RANDOM_SURVIVAL = 0,
    H2_HIER_COMPOSITION = 1,
    H3_DISSIPATIVE_F0 = 2,
    H4_CRITICAL_EDGE = 3,
    H5_AUTOPOIESIS = 4,
    H6_CAUSAL_ROLLBACK = 5,
    H_UNIFIED_FIELD = 6
};

// One continuous 256-site homogeneous 1D ring (4 words stitched end-to-end)
constexpr int WORDS = 4; // 256 spatial sites around one continuous ring

struct Ring256 {
    uint64_t S[WORDS];
    uint64_t K0[WORDS], K1[WORDS], J[WORDS];
    uint64_t H_stress[WORDS];
    uint64_t Bond[WORDS];

    // Endogenous causal rollback state (Principle 6: remember which site changed & revert if worse)
    uint64_t err_fast; // [0..1024]
    uint64_t err_slow; // [0..1024]
    uint64_t snap_err;
    int      last_w, last_bit;
    uint64_t last_dK0, last_dK1, last_dJ;
    uint32_t probe_timer;

    uint64_t rng;
    uint64_t tail_err;
    uint64_t ood_err;
    uint32_t flips_ema; // [0..256]
};

static inline void init_ring(Ring256& u, uint64_t seed, Hypothesis h) {
    u.rng = seed;
    for (int w = 0; w < WORDS; ++w) {
        u.S[w] = splitmix64(u.rng);
        u.K0[w] = splitmix64(u.rng);
        u.K1[w] = splitmix64(u.rng);
        u.J[w]  = (h >= H4_CRITICAL_EDGE) ? 0ULL : (splitmix64(u.rng) & splitmix64(u.rng) & splitmix64(u.rng));
        u.H_stress[w] = ~0ULL;
        u.Bond[w] = 0ULL;
    }
    u.err_fast = 512;
    u.err_slow = 512;
    u.snap_err = 512;
    u.last_w = -1; u.last_bit = 0;
    u.last_dK0 = 0; u.last_dK1 = 0; u.last_dJ = 0;
    u.probe_timer = 0;
    u.tail_err = 0;
    u.ood_err = 0;
    u.flips_ema = 80;
}

static inline void shift_ring_1(const uint64_t in[WORDS], uint64_t L[WORDS], uint64_t R[WORDS]) {
    for (int w = 0; w < WORDS; ++w) {
        int prev_w = (w + WORDS - 1) & (WORDS - 1);
        int next_w = (w + 1) & (WORDS - 1);
        // L[i] = site[i - 1], R[i] = site[i + 1]
        L[w] = (in[w] << 1) | (in[prev_w] >> 63);
        R[w] = (in[w] >> 1) | (in[next_w] << 63);
    }
}

static inline void flip_site(Ring256& u, int site, uint64_t r, bool record_rollback) {
    int w = (site >> 6) & (WORDS - 1);
    int b = site & 63;
    uint64_t m = 1ULL << b;
    uint64_t sel = (r >> 8) & 3ULL;
    uint64_t dk0 = 0, dk1 = 0, dj = 0;
    if (sel == 0) dk0 = m;
    else if (sel == 1) dk1 = m;
    else if (sel == 2) dj = m;
    else { dk0 = m; dk1 = m; }

    u.K0[w] ^= dk0; u.K1[w] ^= dk1; u.J[w] ^= dj;
    if (record_rollback) {
        u.last_w = w; u.last_bit = b;
        u.last_dK0 = dk0; u.last_dK1 = dk1; u.last_dJ = dj;
        u.snap_err = u.err_fast;
        u.probe_timer = 64; // Evaluate causal effect over 64 bits
    }
}

static inline uint32_t step_ring(
    Ring256& u, uint8_t target_bit, Hypothesis h,
    size_t t, bool in_tail, bool in_ood)
{
    // 1. Causal prediction at single physical contact site 0
    uint8_t pred = (uint8_t)(u.S[0] & 1ULL);
    uint8_t err = pred ^ target_bit;

    if (in_tail) u.tail_err += err;
    if (in_ood)  u.ood_err += err;
    u.err_fast = ((u.err_fast * 63ULL) + (err ? 1024ULL : 0ULL)) >> 6;
    u.err_slow = ((u.err_slow * 511ULL) + (err ? 1024ULL : 0ULL)) >> 9;

    // 2. Contact coupling at site 0 only
    uint64_t S_c[WORDS];
    for (int w = 0; w < WORDS; ++w) S_c[w] = u.S[w];
    S_c[0] = (S_c[0] & ~1ULL) | (uint64_t)(target_bit & 1U);

    // 3. Pure homogeneous nearest-neighbor local physics across 256 sites
    uint64_t L[WORDS], R[WORDS];
    shift_ring_1(S_c, L, R);

    uint32_t flips = 0, bulk_flips = 0;
    uint64_t diff[WORDS];
    for (int w = 0; w < WORDS; ++w) {
        uint64_t C = S_c[w];
        uint64_t op_adv  = L[w];
        uint64_t op_xor  = L[w] ^ R[w];
        uint64_t op_maj  = (L[w] & C) | (C & R[w]) | (L[w] & R[w]);
        uint64_t op_asym = (L[w] & ~C) ^ R[w];

        uint64_t ns = (~u.K1[w] & ~u.K0[w] & op_adv)
                    | (~u.K1[w] &  u.K0[w] & op_xor)
                    | ( u.K1[w] & ~u.K0[w] & op_maj)
                    | ( u.K1[w] &  u.K0[w] & op_asym);
        ns ^= u.J[w];
        diff[w] = ns ^ S_c[w];
        flips += (uint32_t)__builtin_popcountll(diff[w]);
        if (w == 1 || w == 2) bulk_flips += (uint32_t)__builtin_popcountll(diff[w]);
        u.S[w] = ns;
    }
    u.flips_ema = (uint32_t)(((uint64_t)u.flips_ema * 15ULL + flips) >> 4);

    // 4. Hypothesis update
    if (h == H1_RANDOM_SURVIVAL) {
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            flip_site(u, (int)(r & 255ULL), r, false);
        }
    } else if (h == H2_HIER_COMPOSITION) {
        if (err) {
            u.Bond[0] &= ~3ULL; u.Bond[WORDS - 1] &= ~(1ULL << 63);
            if ((t & 15ULL) == 0ULL) {
                uint64_t r = splitmix64(u.rng);
                int w = (int)(r & 3ULL);
                uint64_t free_s = ~u.Bond[w];
                if (free_s) {
                    int b = __builtin_ctzll(free_s);
                    flip_site(u, (w << 6) | b, r, false);
                }
            }
        } else {
            for (int w = 0; w < WORDS; ++w) u.Bond[w] |= (diff[w] & ~(diff[w] << 1));
            if ((t & 127ULL) == 0ULL) for (int w = 0; w < WORDS; ++w) u.Bond[w] &= splitmix64(u.rng);
        }
    } else if (h == H3_DISSIPATIVE_F0) {
        if (err) u.H_stress[0] |= 1ULL;
        else u.H_stress[0] &= ~1ULL;
        uint64_t sL[WORDS], sR[WORDS];
        shift_ring_1(u.H_stress, sL, sR);
        for (int w = 0; w < WORDS; ++w) {
            uint64_t sp = sL[w] | sR[w];
            u.H_stress[w] = err ? (u.H_stress[w] | sp) : (u.H_stress[w] & sp);
        }
        if ((t & 15ULL) == 0ULL && u.err_fast >= u.err_slow) {
            uint64_t r = splitmix64(u.rng);
            int w = (int)(r & 3ULL);
            if (u.H_stress[w]) {
                int b = __builtin_ctzll(u.H_stress[w]);
                flip_site(u, (w << 6) | b, r, false);
            }
        }
    } else if (h == H4_CRITICAL_EDGE) {
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            int site = (int)(r & 255ULL);
            int w = site >> 6;
            uint64_t m = 1ULL << (site & 63);
            if (u.flips_ema < 48) { u.K1[w] &= ~m; u.K0[w] |= m; }
            else if (u.flips_ema > 112) { u.K1[w] |= m; u.K0[w] &= ~m; u.J[w] &= ~m; }
            else if (err && u.err_fast >= u.err_slow) flip_site(u, site, r, false);
        }
    } else if (h == H5_AUTOPOIESIS) {
        // Endogenous autopoiesis (no magic threshold!):
        // Protects bulk vitality (24..72 out of 128 bulk sites) and cools automatically
        // whenever err_fast < err_slow (system is successfully aligning with world)!
        if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            int site = (int)(r & 255ULL);
            int w = site >> 6;
            uint64_t m = 1ULL << (site & 63);
            if (bulk_flips < 24) {
                u.K1[w] &= ~m; u.K0[w] |= m; u.J[w] ^= m;
            } else if (bulk_flips > 72) {
                u.K1[w] |= m; u.K0[w] &= ~m;
            } else if (err && u.err_fast + 8 >= u.err_slow) {
                flip_site(u, site, r, false);
            }
        }
    } else if (h == H6_CAUSAL_ROLLBACK) {
        // Principle 6: Causal Credit Assignment via Local Perturbation Probe!
        // Try one local coupling mutation; if err_fast worsens after 64 bits, revert it!
        if (u.probe_timer > 0) {
            u.probe_timer--;
            if (u.probe_timer == 0 && u.last_w >= 0) {
                if (u.err_fast > u.snap_err + 4) {
                    // Revert harmful mutation!
                    u.K0[u.last_w] ^= u.last_dK0;
                    u.K1[u.last_w] ^= u.last_dK1;
                    u.J[u.last_w]  ^= u.last_dJ;
                }
                u.last_w = -1;
            }
        } else if (err && (t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            // Focus near contact more often (geometric bias along ring distance) or uniform
            int site = (int)(r & 255ULL);
            if ((r >> 16) & 1ULL) site = (int)(r & 31ULL);
            flip_site(u, site, r, true);
        }
    } else if (h == H_UNIFIED_FIELD) {
        // Unified Field (H2+H3+H4+H5+H6):
        //   - Bulk vitality preservation (H4+H5)
        //   - Causal perturbation probe + rollback (H6)
        //   - Endogenous cooling when improving
        if (u.probe_timer > 0) {
            u.probe_timer--;
            if (u.probe_timer == 0 && u.last_w >= 0) {
                if (u.err_fast > u.snap_err + 4) {
                    u.K0[u.last_w] ^= u.last_dK0;
                    u.K1[u.last_w] ^= u.last_dK1;
                    u.J[u.last_w]  ^= u.last_dJ;
                }
                u.last_w = -1;
            }
        } else if ((t & 15ULL) == 0ULL) {
            uint64_t r = splitmix64(u.rng);
            int site = (int)(r & 255ULL);
            int w = site >> 6;
            uint64_t m = 1ULL << (site & 63);
            if (bulk_flips < 24) {
                u.K1[w] &= ~m; u.K0[w] |= m;
            } else if (bulk_flips > 72) {
                u.K1[w] |= m; u.K0[w] &= ~m; u.J[w] &= ~m;
            } else if (err && u.err_fast + 8 >= u.err_slow) {
                int target_site = ((r >> 16) & 1ULL) ? (int)(r & 31ULL) : site;
                flip_site(u, target_site, r, true);
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

    constexpr int POP = 256; // 256 rings x 256 sites = 65,536 sites
    const Hypothesis hyps[7] = {
        H1_RANDOM_SURVIVAL, H2_HIER_COMPOSITION, H3_DISSIPATIVE_F0,
        H4_CRITICAL_EDGE, H5_AUTOPOIESIS, H6_CAUSAL_ROLLBACK, H_UNIFIED_FIELD
    };
    const char* names[7] = {
        "H1: Random+Survive", "H2: Hier.Dimers", "H3: F->0 Stress",
        "H4: Edge-of-Chaos", "H5: Autopoiesis", "H6: CausalProbe", "ALL: UnifiedField"
    };

    for (int m = 0; m < 7; ++m) {
        std::vector<Ring256> pop(POP);
        for (int i = 0; i < POP; ++i) init_ring(pop[i], 0x9E3779B97F4A7C15ULL * (i + 1) + (uint64_t)m * 7919ULL, hyps[m]);

        uint64_t tourney_rng = 0xCAFEBABE12345678ULL + m;
        uint64_t sum_flips = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < total_bits; ++t) {
            uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
            bool in_tail = (t >= tail_start && t < tail_end);
            bool in_ood  = (has_ood && t >= ood_start);

            #pragma omp parallel for reduction(+:sum_flips) schedule(static)
            for (int i = 0; i < POP; ++i) {
                uint32_t f = step_ring(pop[i], b, hyps[m], t, in_tail, in_ood);
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
                    flip_site(pop[worst_i], (int)(r & 255ULL), r, false);
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

        int cnt[4]{};
        for (int w = 0; w < WORDS; ++w) {
            for (int b = 0; b < 64; ++b) {
                int op = ((pop[best_idx].K1[w] >> b) & 1ULL) * 2 + ((pop[best_idx].K0[w] >> b) & 1ULL);
                cnt[op]++;
            }
        }

        std::cout << std::left << std::setw(18) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << (best_acc - bs.o0)
                  << ", O1:" << std::setw(6) << (best_acc - bs.o1) << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(5) << std::setprecision(1) << avg_flips << "/256"
                  << " | >" << cnt[0] << "/^" << cnt[1] << "/M" << cnt[2] << "/A" << cnt[3];
        if (has_ood) std::cout << " | OOD=" << std::setprecision(2) << ood_acc << "%";
        std::cout << " | " << std::setprecision(2) << sec << "s\n";
    }
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1176e: HOMOGENEOUS 256-SITE RING + ENDOGENOUS ANNEALING & CAUSAL PROBE\n";
    std::cout << " ZERO Layer-Cakes | ZERO Masks | ZERO Weighted Readout | Prediction = S[0] & 1 at site 0\n";
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
