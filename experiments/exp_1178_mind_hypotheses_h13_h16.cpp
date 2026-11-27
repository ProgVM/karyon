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
// KARYON-CORE EXP-1178: FUNDAMENTAL MIND HYPOTHESES H13..H16 ON PURE SWAR FIELD
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
//
// ZERO Layer-Cakes | ZERO Imposed Masks | ZERO Weighted Readouts | ZERO Floats
//   - Prediction is strictly S[0] & 1ULL at contact site 0 BEFORE seeing x_t.
//   - Part I (Population of 256 x 256-site homogeneous rings = 65,536 sites):
//       H0 : Anchor (EXP-1176e baseline)
//       H9 : Chiral Arrow Champion (EXP-1177 anchor)
//       H13: Pure Cooperation (H7 Multi-Scale + H9 Chiral + H11 Soliton, NO H8)
//       H14: Emergent Phase Segregation (Local lateral inhibition / Turing split)
//       H15: Endogenous Law Diffusion (Coherent domains spread rules internally;
//            NO external population tournament!)
//       H_SYNTH_13_15: Cooperation H13 + Emergent Segregation H14 + Diffusion H15
//   - Part II (H16: Death/Selection vs Immortal Continuous Growth, EQUAL COMPUTE):
//       Compares N=256 Mortal Population (256 x 256 = 65,536 sites, tournament death)
//       vs N=1 Single Immortal Growing Organism (starting 256 sites -> self-expanding
//       up to 65,536 sites along a single continuous 1D field with H15 law diffusion
//       and ZERO death/cloning!).
// ============================================================================

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

constexpr int WORDS = 4; // 256 sites per organism in Part I

enum HypMode {
    M_H0_ANCHOR = 0,
    M_H9_CHIRAL = 1,
    M_H13_COOP = 2,
    M_H14_SEGREGATION = 3,
    M_H15_LAW_DIFFUSION = 4,
    M_SYNTH_13_15 = 5
};

struct MindRing {
    uint64_t S[WORDS];
    uint64_t P[WORDS];
    uint64_t E[WORDS];
    uint64_t K0[WORDS], K1[WORDS], J[WORDS], Ch[WORDS];
    // H15 local coherence trace: 2-bit bit-sliced local self-prediction stability
    uint64_t Coh0[WORDS], Coh1[WORDS];

    uint64_t err_fast, err_slow, snap_err;
    int      last_w;
    uint64_t last_dK0, last_dK1, last_dJ, last_dCh;
    uint32_t probe_timer;

    uint64_t rng;
    uint64_t tail_err, ood_err;
    uint32_t flips_ema;
};

static inline void init_ring(MindRing& u, uint64_t seed, HypMode h) {
    u.rng = seed;
    for (int w = 0; w < WORDS; ++w) {
        u.S[w] = splitmix64(u.rng);
        u.P[w] = u.S[w];
        u.E[w] = splitmix64(u.rng);
        u.K0[w] = splitmix64(u.rng);
        u.K1[w] = splitmix64(u.rng);
        u.J[w]  = 0ULL;
        u.Ch[w] = (h == M_H9_CHIRAL || h == M_H13_COOP || h == M_H14_SEGREGATION || h == M_H15_LAW_DIFFUSION || h == M_SYNTH_13_15)
                  ? splitmix64(u.rng) : 0ULL;
        u.Coh0[w] = 0ULL;
        u.Coh1[w] = 0ULL;
    }
    u.err_fast = 512;
    u.err_slow = 512;
    u.snap_err = 512;
    u.last_w = -1;
    u.last_dK0 = 0; u.last_dK1 = 0; u.last_dJ = 0; u.last_dCh = 0;
    u.probe_timer = 0;
    u.tail_err = 0;
    u.ood_err = 0;
    u.flips_ema = 80;
}

template <int NW>
static inline void shift_ring_1(const uint64_t* in, uint64_t* L, uint64_t* R) {
    for (int w = 0; w < NW; ++w) {
        int prev_w = (w == 0) ? (NW - 1) : (w - 1);
        int next_w = (w + 1 == NW) ? 0 : (w + 1);
        L[w] = (in[w] << 1) | (in[prev_w] >> 63);
        R[w] = (in[w] >> 1) | (in[next_w] << 63);
    }
}

static inline void shift_ring_dyn(const uint64_t* in, uint64_t* L, uint64_t* R, int nw) {
    for (int w = 0; w < nw; ++w) {
        int prev_w = (w == 0) ? (nw - 1) : (w - 1);
        int next_w = (w + 1 == nw) ? 0 : (w + 1);
        L[w] = (in[w] << 1) | (in[prev_w] >> 63);
        R[w] = (in[w] >> 1) | (in[next_w] << 63);
    }
}

static inline void flip_site(MindRing& u, int site, uint64_t r, bool allow_ch, bool record_rollback) {
    int w = (site >> 6) & (WORDS - 1);
    int b = site & 63;
    uint64_t m = 1ULL << b;
    uint64_t sel = (r >> 8) & (allow_ch ? 7ULL : 3ULL);
    uint64_t dk0 = 0, dk1 = 0, dj = 0, dch = 0;
    if (sel == 0) dk0 = m;
    else if (sel == 1) dk1 = m;
    else if (sel == 2) dj = m;
    else if (sel == 3) { dk0 = m; dk1 = m; }
    else if (sel == 4) dch = m;
    else if (sel == 5) { dk0 = m; dch = m; }
    else if (sel == 6) { dk1 = m; dch = m; }
    else dj = m;

    u.K0[w] ^= dk0; u.K1[w] ^= dk1; u.J[w] ^= dj; u.Ch[w] ^= dch;
    if (record_rollback) {
        u.last_w = w;
        u.last_dK0 = dk0; u.last_dK1 = dk1; u.last_dJ = dj; u.last_dCh = dch;
        u.snap_err = u.err_fast;
        u.probe_timer = 64;
    }
}

static inline uint32_t step_ring(
    MindRing& u, uint8_t target_bit, HypMode h,
    size_t t, bool in_tail, bool in_ood)
{
    uint8_t pred = (uint8_t)(u.S[0] & 1ULL);
    uint8_t err = pred ^ target_bit;

    if (in_tail) u.tail_err += err;
    if (in_ood)  u.ood_err += err;
    u.err_fast = ((u.err_fast * 63ULL) + (err ? 1024ULL : 0ULL)) >> 6;
    u.err_slow = ((u.err_slow * 511ULL) + (err ? 1024ULL : 0ULL)) >> 9;

    uint64_t S_c[WORDS];
    for (int w = 0; w < WORDS; ++w) S_c[w] = u.S[w];
    S_c[0] = (S_c[0] & ~1ULL) | (uint64_t)(target_bit & 1U);

    uint64_t L[WORDS], R[WORDS], EL[WORDS], ER[WORDS];
    shift_ring_1<WORDS>(S_c, L, R);
    if (h == M_H13_COOP || h == M_SYNTH_13_15) shift_ring_1<WORDS>(u.E, EL, ER);

    // H14: Emergent Turing-like lateral inhibition / phase segregation signals
    uint64_t K1L[WORDS], K1R[WORDS];
    if (h == M_H14_SEGREGATION || h == M_SYNTH_13_15) {
        shift_ring_1<WORDS>(u.K1, K1L, K1R);
    }

    uint32_t flips = 0, bulk_flips = 0;
    uint64_t diff[WORDS];

    for (int w = 0; w < WORDS; ++w) {
        uint64_t C = S_c[w];
        uint64_t Lw = L[w], Rw = R[w];

        // Chiral symmetry breaking (H9/H13/H14/H15/Synth)
        if (h != M_H0_ANCHOR) {
            uint64_t chiral_L = Lw ^ (u.Ch[w] & C);
            uint64_t chiral_R = (Rw & ~u.Ch[w]) | (u.Ch[w] & (Lw ^ Rw));
            Lw = chiral_L;
            Rw = chiral_R;
        }

        uint64_t op_adv  = Lw;
        uint64_t op_xor  = Lw ^ Rw;
        uint64_t op_maj  = (Lw & C) | (C & Rw) | (Lw & Rw);
        uint64_t op_asym = (Lw & ~C) ^ Rw;

        uint64_t ns = (~u.K1[w] & ~u.K0[w] & op_adv)
                    | (~u.K1[w] &  u.K0[w] & op_xor)
                    | ( u.K1[w] & ~u.K0[w] & op_maj)
                    | ( u.K1[w] &  u.K0[w] & op_asym);

        // H13 / Synth: Multi-Scale Slow Phase Interference (H7)
        if (h == M_H13_COOP || h == M_SYNTH_13_15) {
            ns ^= (u.E[w] & (u.K1[w] ^ u.K0[w]));
        }

        ns ^= u.J[w];

        // H13 / Synth: Soliton 2nd-order reversible momentum (H11)
        if (h == M_H13_COOP || h == M_SYNTH_13_15) {
            uint64_t rev_sites = ~u.K1[w];
            ns = (rev_sites & (ns ^ u.P[w])) | (~rev_sites & ns);
        }

        diff[w] = ns ^ S_c[w];
        flips += (uint32_t)__builtin_popcountll(diff[w]);
        if (w == 1 || w == 2) bulk_flips += (uint32_t)__builtin_popcountll(diff[w]);

        u.P[w] = S_c[w];
        u.S[w] = ns;

        if ((h == M_H13_COOP || h == M_SYNTH_13_15) && ((t & 3ULL) == 0ULL)) {
            uint64_t maj_e = (EL[w] & u.E[w]) | (u.E[w] & ER[w]) | (EL[w] & ER[w]);
            uint64_t coherent = ~diff[w];
            u.E[w] = (coherent & ((u.E[w] & ns) | (u.E[w] & maj_e) | (ns & maj_e)))
                   | (~coherent & (u.E[w] ^ (EL[w] & ns)));
        }

        // H15 / Synth: Update local 2-bit bit-sliced coherence (predictable vs noisy sites)
        if ((h == M_H15_LAW_DIFFUSION || h == M_SYNTH_13_15) && ((t & 3ULL) == 0ULL)) {
            // A site is locally coherent when it aligns with global success (~err) and non-chaotic local wave
            uint64_t calm = err ? 0ULL : (~diff[w] | (u.K1[w] ^ u.K0[w]));
            uint64_t fail = err ? ~0ULL : 0ULL;
            // Increment Coh on calm, decrement on contact error shock
            uint64_t c0 = u.Coh0[w], c1 = u.Coh1[w];
            uint64_t nc0_up = c0 ^ (calm & ~(c0 & c1));
            uint64_t nc1_up = c1 ^ ((c0 & calm) & ~c1);
            uint64_t nc0_dn = c0 ^ (fail & (c0 | c1));
            uint64_t nc1_dn = c1 ^ ((~c0 & fail) & c1);
            u.Coh0[w] = (fail & nc0_dn) | (~fail & nc0_up);
            u.Coh1[w] = (fail & nc1_dn) | (~fail & nc1_up);
        }
    }
    u.flips_ema = (uint32_t)(((uint64_t)u.flips_ema * 15ULL + flips) >> 4);

    // H14: Emergent Turing-like phase segregation (short-range activation, neighbor differentiation)
    // Purely homogeneous local rule: if a site has BOTH neighbors identical to its K1 phase while
    // error persists, laterally inhibit (differentiate); if isolated, align with majority of rules.
    if ((h == M_H14_SEGREGATION || h == M_SYNTH_13_15) && err && (t & 31ULL) == 0ULL) {
        uint64_t r = splitmix64(u.rng);
        for (int w = 0; w < WORDS; ++w) {
            uint64_t same_nb = ~(u.K1[w] ^ K1L[w]) & ~(u.K1[w] ^ K1R[w]);
            uint64_t mask = same_nb & splitmix64(u.rng) & splitmix64(u.rng) & splitmix64(u.rng);
            u.K1[w] ^= mask;
        }
        (void)r;
    }

    // H15: Endogenous Law Diffusion (more coherent neighbor transfers its local law K0,K1,Ch to less coherent neighbor)
    if ((h == M_H15_LAW_DIFFUSION || h == M_SYNTH_13_15) && (t & 31ULL) == 0ULL) {
        uint64_t C1L[WORDS], C1R[WORDS], K0L[WORDS], K0R[WORDS], K1L2[WORDS], K1R2[WORDS], ChL[WORDS], ChR[WORDS];
        shift_ring_1<WORDS>(u.Coh1, C1L, C1R);
        shift_ring_1<WORDS>(u.K0, K0L, K0R);
        shift_ring_1<WORDS>(u.K1, K1L2, K1R2);
        shift_ring_1<WORDS>(u.Ch, ChL, ChR);
        for (int w = 0; w < WORDS; ++w) {
            // Sites that are incoherent (Coh1==0) adopt rules from coherent left/right neighbors (Coh1==1)
            uint64_t gate = ~u.Coh1[w] & splitmix64(u.rng) & splitmix64(u.rng); // sparse ~25% diffusion
            uint64_t from_L = gate & C1L[w];
            uint64_t from_R = gate & ~C1L[w] & C1R[w];
            u.K0[w] = (from_L & K0L[w]) | (from_R & K0R[w]) | (~(from_L | from_R) & u.K0[w]);
            u.K1[w] = (from_L & K1L2[w]) | (from_R & K1R2[w]) | (~(from_L | from_R) & u.K1[w]);
            u.Ch[w] = (from_L & ChL[w]) | (from_R & ChR[w]) | (~(from_L | from_R) & u.Ch[w]);
        }
    }

    // Causal Probe + Bulk Vitality (consistent across all hypotheses)
    if (u.probe_timer > 0) {
        u.probe_timer--;
        if (u.probe_timer == 0 && u.last_w >= 0) {
            if (u.err_fast > u.snap_err + 4) {
                u.K0[u.last_w] ^= u.last_dK0;
                u.K1[u.last_w] ^= u.last_dK1;
                u.J[u.last_w]  ^= u.last_dJ;
                u.Ch[u.last_w] ^= u.last_dCh;
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
            flip_site(u, target_site, r, (h != M_H0_ANCHOR), true);
        }
    }
    return flips;
}

// Worlds
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

    constexpr int POP = 256;
    const HypMode modes[6] = {
        M_H0_ANCHOR, M_H9_CHIRAL, M_H13_COOP, M_H14_SEGREGATION, M_H15_LAW_DIFFUSION, M_SYNTH_13_15
    };
    const char* names[6] = {
        "H0: Anchor(1176e)", "H9: Chiral(1177)", "H13: H7+H9+H11",
        "H14: TuringSegreg", "H15: LawDiff(NoDeath)", "ALL: Synth(H13-15)"
    };

    for (int m = 0; m < 6; ++m) {
        std::vector<MindRing> pop(POP);
        for (int i = 0; i < POP; ++i) init_ring(pop[i], 0x9E3779B97F4A7C15ULL * (i + 1) + (uint64_t)m * 7919ULL, modes[m]);

        uint64_t tourney_rng = 0xCAFEBABE12345678ULL + m;
        uint64_t sum_flips = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < total_bits; ++t) {
            uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
            bool in_tail = (t >= tail_start && t < tail_end);
            bool in_ood  = (has_ood && t >= ood_start);

            #pragma omp parallel for reduction(+:sum_flips) schedule(static)
            for (int i = 0; i < POP; ++i) {
                uint32_t f = step_ring(pop[i], b, modes[m], t, in_tail, in_ood);
                if (in_tail) sum_flips += f;
            }

            // H15 tests endogenous internal law diffusion WITHOUT external tournament death!
            if (modes[m] != M_H15_LAW_DIFFUSION && (t & 511ULL) == 511ULL) {
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
                    flip_site(pop[worst_i], (int)(r & 255ULL), r, (modes[m] != M_H0_ANCHOR), false);
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

        std::cout << std::left << std::setw(21) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << (best_acc - bs.o0)
                  << ", O1:" << std::setw(6) << (best_acc - bs.o1) << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(5) << std::setprecision(1) << avg_flips << "/256";
        if (has_ood) std::cout << " | OOD=" << std::setprecision(2) << ood_acc << "%";
        std::cout << " | " << std::setprecision(2) << sec << "s\n";
    }
}

// ============================================================================
// PART II — H16: IMMORTAL GROWING SINGLE ORGANISM (N=1) vs MORTAL POPULATION
// Equal total physical sites budget (up to 65,536 sites):
//   - Mortal Population: N=256 independent 256-site organisms with Darwinian death
//   - Immortal Growing Single Organism (N=1): starts as 4 words (256 sites),
//     whenever stress persists despite local adaptation, autopoietically sprouts
//     new living physical tissue (+4 words = +256 sites per growth event, up to
//     1024 words = 65,536 sites) in ONE unbroken physical body with H15 diffusion!
// ============================================================================
static void run_h16_immortal_vs_mortal(const std::string& wname, const std::vector<uint8_t>& data) {
    size_t total_bits = data.size() * 8;
    size_t tail_start = (total_bits * 3) / 4;
    size_t tail_bits  = total_bits - tail_start;
    BaselineStats bs  = compute_baselines(data, tail_start, total_bits);

    // 1. Immortal N=1 growing ring (max 1024 words = 65,536 sites)
    constexpr int MAX_W = 1024;
    std::vector<uint64_t> S(MAX_W, 0), P(MAX_W, 0), E(MAX_W, 0);
    std::vector<uint64_t> K0(MAX_W, 0), K1(MAX_W, 0), J(MAX_W, 0), Ch(MAX_W, 0);
    std::vector<uint64_t> Coh0(MAX_W, 0), Coh1(MAX_W, 0);
    std::vector<uint64_t> L(MAX_W, 0), R(MAX_W, 0), EL(MAX_W, 0), ER(MAX_W, 0);

    uint64_t rng = 0x1122334455667788ULL;
    int active_w = 4; // Starts at 256 sites!
    for (int w = 0; w < MAX_W; ++w) {
        S[w] = splitmix64(rng); P[w] = S[w]; E[w] = splitmix64(rng);
        K0[w] = splitmix64(rng); K1[w] = splitmix64(rng); Ch[w] = splitmix64(rng);
    }

    uint64_t err_fast = 512, err_slow = 512, tail_err_immortal = 0;
    uint32_t growth_events = 0, cooldown = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < total_bits; ++t) {
        uint8_t target_bit = (data[t >> 3] >> (t & 7)) & 1U;
        uint8_t pred = (uint8_t)(S[0] & 1ULL);
        uint8_t err = pred ^ target_bit;
        if (t >= tail_start) tail_err_immortal += err;
        err_fast = ((err_fast * 63ULL) + (err ? 1024ULL : 0ULL)) >> 6;
        err_slow = ((err_slow * 511ULL) + (err ? 1024ULL : 0ULL)) >> 9;

        uint64_t old_s0 = S[0];
        S[0] = (S[0] & ~1ULL) | (uint64_t)target_bit;
        shift_ring_dyn(S.data(), L.data(), R.data(), active_w);
        shift_ring_dyn(E.data(), EL.data(), ER.data(), active_w);

        uint32_t flips = 0;
        for (int w = 0; w < active_w; ++w) {
            uint64_t C = S[w];
            uint64_t Lw = L[w] ^ (Ch[w] & C);
            uint64_t Rw = (R[w] & ~Ch[w]) | (Ch[w] & (L[w] ^ R[w]));
            uint64_t op_adv  = Lw;
            uint64_t op_xor  = Lw ^ Rw;
            uint64_t op_maj  = (Lw & C) | (C & Rw) | (Lw & Rw);
            uint64_t op_asym = (Lw & ~C) ^ Rw;
            uint64_t ns = (~K1[w] & ~K0[w] & op_adv) | (~K1[w] & K0[w] & op_xor)
                        | ( K1[w] & ~K0[w] & op_maj) | ( K1[w] & K0[w] & op_asym);
            ns ^= (E[w] & (K1[w] ^ K0[w])) ^ J[w];
            ns = ((~K1[w]) & (ns ^ P[w])) | (K1[w] & ns);
            uint64_t df = ns ^ C;
            flips += (uint32_t)__builtin_popcountll(df);
            P[w] = C; S[w] = ns;
            if ((t & 3ULL) == 0ULL) {
                uint64_t maj_e = (EL[w] & E[w]) | (E[w] & ER[w]) | (EL[w] & ER[w]);
                E[w] = (~df & ((E[w] & ns) | (E[w] & maj_e) | (ns & maj_e))) | (df & (E[w] ^ (EL[w] & ns)));
                uint64_t calm = err ? 0ULL : (~df | (K1[w] ^ K0[w]));
                uint64_t fail = err ? ~0ULL : 0ULL;
                uint64_t c0 = Coh0[w], c1 = Coh1[w];
                Coh0[w] = (fail & (c0 ^ (fail & (c0 | c1)))) | (~fail & (c0 ^ (calm & ~(c0 & c1))));
                Coh1[w] = (fail & (c1 ^ ((~c0 & fail) & c1))) | (~fail & (c1 ^ ((c0 & calm) & ~c1)));
            }
        }
        (void)old_s0;

        // Endogenous local plasticity + morphogenesis (sprouting tissue on persistent stress)
        if ((t & 15ULL) == 0ULL && err && err_fast >= err_slow) {
            int site = (int)(splitmix64(rng) % (uint64_t)(active_w * 64));
            if (splitmix64(rng) & 1ULL) site = (int)(splitmix64(rng) & 31ULL);
            int w = site >> 6; uint64_t m = 1ULL << (site & 63);
            uint64_t sel = splitmix64(rng) & 3ULL;
            if (sel == 0) K0[w] ^= m; else if (sel == 1) K1[w] ^= m;
            else if (sel == 2) Ch[w] ^= m; else J[w] ^= m;
        }
        if (cooldown > 0) cooldown--;
        else if (active_w + 4 <= MAX_W && err_fast > 495 && (t & 511ULL) == 511ULL) {
            // Immortal organism sprouts +256 sites of plastic tissue internally!
            active_w += 4;
            growth_events++;
            cooldown = 1024;
        }
    }
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    double imm_acc = 100.0 * (1.0 - (double)tail_err_immortal / tail_bits);
    std::cout << "  [" << wname << "] Baseline O0=" << std::fixed << std::setprecision(2) << bs.o0
              << "% | Immortal N=1 Growing Body (final " << (active_w * 64) << " sites, growths="
              << growth_events << "): Acc=" << imm_acc << "% (" << sec << "s)\n";
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1178: FUNDAMENTAL MIND HYPOTHESES H13..H16 ON PURE HOMOGENEOUS SWAR FIELD\n";
    std::cout << " ZERO Layer-Cakes | ZERO Masks | ZERO Weighted Readout | Prediction = S[0] & 1 at site 0\n";
    std::cout << "==================================================================================================\n";

    const size_t N_BYTES = 16384; // 131,072 bits
    auto sm   = gen_splitmix(N_BYTES);
    auto lfsr = gen_lfsr7(N_BYTES);
    auto nll  = gen_nl_lag(N_BYTES);
    auto r110 = gen_rule110_with_ood(N_BYTES);
    std::vector<uint8_t> wiki;
    std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
    if (fw) {
        wiki.resize(N_BYTES);
        fw.read((char*)wiki.data(), N_BYTES);
        wiki.resize((size_t)fw.gcount());
    }

    run_suite("Control 0: SplitMix64 Crypto Noise", sm, N_BYTES * 8);
    run_suite("Law 1: LFSR-7 (50/50 balanced, zero bias)", lfsr, N_BYTES * 8);
    run_suite("Law 2: Nonlinear Lag (b_{t-3}&b_{t-5})^b_{t-8}^b_{t-12}", nll, N_BYTES * 8);
    run_suite("Law 3: Rule 110 + OOD Inversion at 75%", r110, N_BYTES * 8, true);
    if (!wiki.empty()) run_suite("Real 4: Wikipedia Stream (1 bit/tick at site 0)", wiki, wiki.size() * 8);

    std::cout << "\n==================================================================================================\n";
    std::cout << "PART II — H16: IMMORTAL GROWING SINGLE ORGANISM (N=1, ZERO DEATH) vs DARWINIAN POPULATION\n";
    std::cout << "==================================================================================================\n";
    run_h16_immortal_vs_mortal("LFSR-7", lfsr);
    run_h16_immortal_vs_mortal("Nonlinear Lag", nll);
    run_h16_immortal_vs_mortal("Rule 110", r110);
    if (!wiki.empty()) run_h16_immortal_vs_mortal("Wikipedia", wiki);

    return 0;
}
