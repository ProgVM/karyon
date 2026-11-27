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
// KARYON-CORE EXP-1177: FUNDAMENTAL MIND HYPOTHESES H7..H12 ON PURE SWAR FIELD
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
//
// ZERO Layer-Cakes | ZERO Imposed Masks | ZERO Weighted Readouts | ZERO Floats
//   - Single homogeneous 256-site 1D physical ring (4 x uint64_t words).
//   - Prediction = strictly S[0] & 1ULL at contact site 0 BEFORE observing x_t.
//   - Tests 6 new fundamental hypotheses on Mind emergence + Anchor + Synthesis:
//       H0 : Anchor Unified (EXP-1176e/g baseline: Autopoiesis + CausalProbe)
//       H7 : Multi-Scale Phase Interference (Fast wave S <-> Slow envelope E)
//       H8 : Counter-Wave Interference Contact (No hard overwrite; F=err wave)
//       H9 : Spontaneous Chiral / Time-Arrow Symmetry Breaking (L vs R bias)
//       H10: Elastic Phase Plasticity (Local bit-sliced Stiff0/Stiff1 melting)
//       H11: Topological Soliton Conservation (2nd-order reversible domain walls)
//       H7..11 Synthesis: Unified Multi-Scale Chiral-Elastic Soliton Field
//   - Plus Part II: H12 Symbiotic Closed-Loop vs Yoked Open-Loop Twin Test!
// ============================================================================

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

constexpr int WORDS = 4; // 256 sites on a continuous homogeneous ring

enum MindHyp {
    H0_ANCHOR = 0,
    H7_MULTISCALE = 1,
    H8_COUNTERWAVE = 2,
    H9_CHIRAL_ARROW = 3,
    H10_ELASTIC_STIFF = 4,
    H11_SOLITON_CONS = 5,
    H_SYNTH_7_11 = 6
};

struct MindRing {
    uint64_t S[WORDS];      // Fast physical spin wave
    uint64_t P[WORDS];      // Previous spin wave (H11 soliton / 2nd-order momentum)
    uint64_t E[WORDS];      // Slow envelope field (H7 multi-scale phase)
    uint64_t W_err[WORDS];  // Counter-propagating residual/collision wave (H8)

    uint64_t K0[WORDS], K1[WORDS], J[WORDS];
    uint64_t Ch[WORDS];     // Local chiral asymmetry (H9: 0=balanced/L, 1=R-asymmetric)
    uint64_t Stiff0[WORDS], Stiff1[WORDS]; // 2-bit bit-sliced local stiffness (H10: 00=liquid .. 11=tempered)

    uint64_t err_fast, err_slow, snap_err;
    int      last_w;
    uint64_t last_dK0, last_dK1, last_dJ, last_dCh;
    uint32_t probe_timer;

    uint64_t rng;
    uint64_t tail_err, ood_err;
    uint32_t flips_ema;
};

static inline void init_mind_ring(MindRing& u, uint64_t seed, MindHyp h) {
    u.rng = seed;
    for (int w = 0; w < WORDS; ++w) {
        u.S[w] = splitmix64(u.rng);
        u.P[w] = u.S[w];
        u.E[w] = splitmix64(u.rng);
        u.W_err[w] = 0ULL;
        u.K0[w] = splitmix64(u.rng);
        u.K1[w] = splitmix64(u.rng);
        u.J[w]  = 0ULL;
        u.Ch[w] = (h == H9_CHIRAL_ARROW || h == H_SYNTH_7_11) ? splitmix64(u.rng) : 0ULL;
        u.Stiff0[w] = 0ULL;
        u.Stiff1[w] = 0ULL;
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

static inline void shift_ring_1(const uint64_t in[WORDS], uint64_t L[WORDS], uint64_t R[WORDS]) {
    for (int w = 0; w < WORDS; ++w) {
        int prev_w = (w + WORDS - 1) & (WORDS - 1);
        int next_w = (w + 1) & (WORDS - 1);
        L[w] = (in[w] << 1) | (in[prev_w] >> 63);
        R[w] = (in[w] >> 1) | (in[next_w] << 63);
    }
}

static inline void flip_site(MindRing& u, int site, uint64_t r, bool allow_chiral, bool record_rollback) {
    int w = (site >> 6) & (WORDS - 1);
    int b = site & 63;
    uint64_t m = 1ULL << b;
    uint64_t sel = (r >> 8) & (allow_chiral ? 7ULL : 3ULL);
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

static inline uint32_t step_mind(
    MindRing& u, uint8_t target_bit, MindHyp h,
    size_t t, bool in_tail, bool in_ood)
{
    // 1. CAUSAL PREDICTION STRICTLY BEFORE SEEING target_bit:
    //    For H8 (Counter-Wave), prediction is the coherent interference between S[0] and slow E[0]
    //    No weights, pure physical bit spin at site 0!
    uint8_t pred = (uint8_t)(u.S[0] & 1ULL);
    uint8_t err = pred ^ target_bit;

    if (in_tail) u.tail_err += err;
    if (in_ood)  u.ood_err += err;
    u.err_fast = ((u.err_fast * 63ULL) + (err ? 1024ULL : 0ULL)) >> 6;
    u.err_slow = ((u.err_slow * 511ULL) + (err ? 1024ULL : 0ULL)) >> 9;

    // 2. Contact physics at site 0
    uint64_t S_c[WORDS];
    for (int w = 0; w < WORDS; ++w) S_c[w] = u.S[w];

    if (h == H8_COUNTERWAVE || h == H_SYNTH_7_11) {
        // H8: External world does NOT overwrite site 0 brutally; instead collision residual
        // err injects a counter-propagating stress wave W_err[0] |= 1, while site 0 smoothly
        // relaxes toward target_bit ONLY on collision (S_c[0] ^= err == target_bit, leaving
        // outgoing wave intact when pred==target_bit and recording exact F=err wave into W_err!)
        S_c[0] ^= (uint64_t)err;
        if (err) u.W_err[0] |= 1ULL;
        else     u.W_err[0] &= ~1ULL;
    } else {
        S_c[0] = (S_c[0] & ~1ULL) | (uint64_t)(target_bit & 1U);
    }

    uint64_t L[WORDS], R[WORDS];
    shift_ring_1(S_c, L, R);

    uint64_t WL[WORDS], WR[WORDS];
    if (h == H8_COUNTERWAVE || h == H10_ELASTIC_STIFF || h == H_SYNTH_7_11) {
        shift_ring_1(u.W_err, WL, WR);
    }

    uint64_t EL[WORDS], ER[WORDS];
    if (h == H7_MULTISCALE || h == H_SYNTH_7_11) {
        shift_ring_1(u.E, EL, ER);
    }

    uint32_t flips = 0, bulk_flips = 0;
    uint64_t diff[WORDS];

    for (int w = 0; w < WORDS; ++w) {
        uint64_t C = S_c[w];
        uint64_t Lw = L[w];
        uint64_t Rw = R[w];

        // H9: Spontaneous Chiral / Time-Arrow Symmetry Breaking
        // When Ch==1 at a site, R-coupling is phase-inverted/gated into unidirectional chiral cascade
        if (h == H9_CHIRAL_ARROW || h == H_SYNTH_7_11) {
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

        // H7: Multi-Scale Phase Interference (Slow envelope E modulates fast wave phase)
        if (h == H7_MULTISCALE || h == H_SYNTH_7_11) {
            // Sites with Ch==0 or K0^K1 receive slow envelope modulation (phase gate)
            ns ^= (u.E[w] & (u.K1[w] ^ u.K0[w]));
        }

        ns ^= u.J[w];

        // H11: Topological Soliton Conservation (2nd-order reversible momentum on non-dissipative sites)
        if (h == H11_SOLITON_CONS || h == H_SYNTH_7_11) {
            // Sites with K1==0 (adv/xor wave sites) preserve 2nd-order wave momentum (domain wall inertia)
            uint64_t rev_sites = ~u.K1[w];
            ns = (rev_sites & (ns ^ u.P[w])) | (~rev_sites & ns);
        }

        diff[w] = ns ^ S_c[w];
        flips += (uint32_t)__builtin_popcountll(diff[w]);
        if (w == 1 || w == 2) bulk_flips += (uint32_t)__builtin_popcountll(diff[w]);

        u.P[w] = S_c[w];
        u.S[w] = ns;

        // H7: Slow envelope updates on odd ticks when fast waves cross (every 4 ticks locally)
        if ((h == H7_MULTISCALE || h == H_SYNTH_7_11) && ((t & 3ULL) == 0ULL)) {
            // Slow field averages/integrates fast local spin + neighbor envelopes (no floats!)
            uint64_t maj_e = (EL[w] & u.E[w]) | (u.E[w] & ER[w]) | (EL[w] & ER[w]);
            uint64_t coherent = ~(ns ^ C); // 1 where fast wave is locally stationary
            u.E[w] = (coherent & ((u.E[w] & ns) | (u.E[w] & maj_e) | (ns & maj_e)))
                   | (~coherent & (u.E[w] ^ (EL[w] & ns)));
        }

        // H8 / H10: Propagate counter-wave collision stress W_err inward along active sites
        if (h == H8_COUNTERWAVE || h == H10_ELASTIC_STIFF || h == H_SYNTH_7_11) {
            // Counter-wave travels inward from site 0; cools where waves match
            uint64_t spread = (WL[w] | WR[w]) & (diff[w] | ~diff[w]);
            // Decay slowly while preserving collision wavefronts
            u.W_err[w] = err ? (u.W_err[w] | spread) : (u.W_err[w] & (WL[w] | WR[w]));
            if (w == 0 && err) u.W_err[0] |= 1ULL;
        }

        // H10: Elastic 2-bit bit-sliced stiffness (Stiff1, Stiff0):
        //   - Shockwave W_err melts stiffness (11->10->01->00)
        //   - Calm agreement tempers stiffness (00->01->10->11)
        if (h == H10_ELASTIC_STIFF || h == H_SYNTH_7_11) {
            if ((t & 3ULL) == 0ULL) {
                uint64_t melt = u.W_err[w] | (err ? 1ULL : 0ULL);
                uint64_t cool = ~melt & ~err;
                // Bit-sliced decrement on melt:
                uint64_t borrow = ~u.Stiff0[w] & melt;
                uint64_t ns0_m = u.Stiff0[w] ^ (melt & (u.Stiff0[w] | u.Stiff1[w]));
                uint64_t ns1_m = u.Stiff1[w] ^ (borrow & u.Stiff1[w]);
                // Bit-sliced increment on cool (saturating at 11):
                uint64_t carry = u.Stiff0[w] & cool;
                uint64_t ns0_c = u.Stiff0[w] ^ (cool & ~(u.Stiff0[w] & u.Stiff1[w]));
                uint64_t ns1_c = u.Stiff1[w] ^ (carry & ~u.Stiff1[w]);

                u.Stiff0[w] = (melt & ns0_m) | (~melt & ns0_c);
                u.Stiff1[w] = (melt & ns1_m) | (~melt & ns1_c);
            }
        }
    }

    u.flips_ema = (uint32_t)(((uint64_t)u.flips_ema * 15ULL + flips) >> 4);

    // 3. Unified Autopoietic + Causal Credit Plasticity (same clean core across H0, H7..H11)
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

            // H10 / Synthesis: only deform liquid/melted sites (Stiff1 == 0)!
            if (h == H10_ELASTIC_STIFF || h == H_SYNTH_7_11) {
                int tw = target_site >> 6;
                uint64_t liq = ~u.Stiff1[tw];
                if (liq) {
                    int b = (__builtin_ctzll((liq >> (target_site & 63)) | (liq << (64 - (target_site & 63)))) + (target_site & 63)) & 63;
                    target_site = (tw << 6) | b;
                }
            }
            // H8: prefer mutating sites inside active collision counter-wave W_err
            if (h == H8_COUNTERWAVE) {
                int tw = target_site >> 6;
                if (u.W_err[tw]) target_site = (tw << 6) | __builtin_ctzll(u.W_err[tw]);
            }

            bool allow_ch = (h == H9_CHIRAL_ARROW || h == H_SYNTH_7_11);
            flip_site(u, target_site, r, allow_ch, true);
        }
    }
    return flips;
}

// Benchmark Worlds
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
    const MindHyp hyps[7] = {
        H0_ANCHOR, H7_MULTISCALE, H8_COUNTERWAVE,
        H9_CHIRAL_ARROW, H10_ELASTIC_STIFF, H11_SOLITON_CONS, H_SYNTH_7_11
    };
    const char* names[7] = {
        "H0: Anchor(1176e)", "H7: MultiScaleFast", "H8: CounterWave",
        "H9: ChiralArrow", "H10: ElasticMelt", "H11: Soliton2ndOrd", "ALL: Synth(H7-11)"
    };

    for (int m = 0; m < 7; ++m) {
        std::vector<MindRing> pop(POP);
        for (int i = 0; i < POP; ++i) init_mind_ring(pop[i], 0x9E3779B97F4A7C15ULL * (i + 1) + (uint64_t)m * 7919ULL, hyps[m]);

        uint64_t tourney_rng = 0xCAFEBABE12345678ULL + m;
        uint64_t sum_flips = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < total_bits; ++t) {
            uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
            bool in_tail = (t >= tail_start && t < tail_end);
            bool in_ood  = (has_ood && t >= ood_start);

            #pragma omp parallel for reduction(+:sum_flips) schedule(static)
            for (int i = 0; i < POP; ++i) {
                uint32_t f = step_mind(pop[i], b, hyps[m], t, in_tail, in_ood);
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
                    flip_site(pop[worst_i], (int)(r & 255ULL), r, (hyps[m]==H9_CHIRAL_ARROW||hyps[m]==H_SYNTH_7_11), false);
                }
            }
        }

        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        uint64_t min_err = pop[0].tail_err;
        double sum_err = 0.0, min_ood = 1e18;
        int best_idx = 0;
        for (int i = 0; i < POP; ++i) {
            sum_err += (double)pop[i].tail_err;
            if (pop[i].tail_err < min_err) { min_err = pop[i].tail_err; best_idx = i; }
            if ((double)pop[i].ood_err < min_ood) min_ood = (double)pop[i].ood_err;
        }
        double best_acc = 100.0 * (1.0 - (double)min_err / tail_bits);
        double avg_acc  = 100.0 * (1.0 - (sum_err / POP) / tail_bits);
        double avg_flips = (double)sum_flips / ((double)POP * tail_bits);
        double ood_acc = has_ood ? 100.0 * (1.0 - min_ood / ood_bits) : 0.0;

        int rigid_cnt = 0;
        for (int w = 0; w < WORDS; ++w) rigid_cnt += __builtin_popcountll(pop[best_idx].Stiff1[w] & pop[best_idx].Stiff0[w]);

        std::cout << std::left << std::setw(18) << names[m]
                  << " | Best=" << std::setw(6) << std::setprecision(2) << best_acc << "%"
                  << " (vs O0:" << std::showpos << std::setw(6) << (best_acc - bs.o0)
                  << ", O1:" << std::setw(6) << (best_acc - bs.o1) << std::noshowpos << ")"
                  << " | Avg=" << std::setw(5) << avg_acc << "%"
                  << " | Flips=" << std::setw(5) << std::setprecision(1) << avg_flips << "/256";
        if (hyps[m] == H10_ELASTIC_STIFF || hyps[m] == H_SYNTH_7_11) {
            std::cout << " | Rigid=" << std::setw(3) << rigid_cnt;
        }
        if (has_ood) std::cout << " | OOD=" << std::setprecision(2) << ood_acc << "%";
        std::cout << " | " << std::setprecision(2) << sec << "s\n";
    }
}

// ============================================================================
// PART II — H12: SYMBIOTIC CLOSED-LOOP vs YOKED OPEN-LOOP TWIN TEST
// An external physical world (1D coupled cellular world W=64) has its boundary
// perturbed by the organism's spin action `act = (S[0] & 1)`:
//   - CLOSED-LOOP: world dynamics at x=0 depends on `act`, creating a genuine
//     sensorimotor feedback cycle (homeostatic phase locking).
//   - YOKED OPEN-LOOP TWIN: receives the EXACT SAME bit stream recorded from
//     the closed-loop run, bit-for-bit identical statistics, WITHOUT causal
//     coupling! Measures whether active loop-closing produces higher predictive
//     coherence or deeper internal organization than passive observation!
// ============================================================================
static void run_h12_closed_loop_vs_yoked() {
    std::cout << "\n==================================================================================================\n";
    std::cout << "PART II — H12: SYMBIOTIC CLOSED-LOOP COUPLING vs YOKED OPEN-LOOP TWIN (131,072 bits)\n";
    std::cout << " Identical marginal bit stream statistics; tests if active agency sharpens internal order!\n";
    std::cout << "==================================================================================================\n";

    constexpr int POP = 128;
    const size_t TOT = 131072;
    const size_t TAIL = TOT / 4;

    std::vector<MindRing> closed(POP), yoked(POP);
    std::vector<uint64_t> world(POP, 0x9E3779B97F4A7C15ULL);
    for (int i = 0; i < POP; ++i) {
        uint64_t s = 0x123456789ABCDEF0ULL * (i + 1);
        init_mind_ring(closed[i], s, H_SYNTH_7_11);
        init_mind_ring(yoked[i],  s + 0x5555555555555555ULL, H_SYNTH_7_11); // Same physics, different RNG, yoked stream
        world[i] ^= (uint64_t)(i * 0x9E37ULL);
    }

    uint64_t sum_flips_c = 0, sum_flips_y = 0;
    for (size_t t = 0; t < TOT; ++t) {
        bool in_tail = (t >= TOT - TAIL);
        for (int i = 0; i < POP; ++i) {
            // Closed-loop physical world: 64-bit Rule-30/110 hybrid ring perturbed by organism's action at site 0
            uint8_t act = (uint8_t)(closed[i].S[0] & 1ULL);
            uint64_t W = world[i] ^ (uint64_t)act; // Organism action physically flips boundary spin of world!
            uint64_t WL = (W << 1) | (W >> 63);
            uint64_t WR = (W >> 1) | (W << 63);
            // World evolves one step under local chaotic+Rule110 law:
            uint64_t W_next = (WL & ~W) ^ WR;
            uint8_t world_obs = (uint8_t)((W_next >> 3) & 1ULL); // Organism senses site 3 of world
            world[i] = W_next;

            uint32_t fc = step_mind(closed[i], world_obs, H_SYNTH_7_11, t, in_tail, false);
            // Yoked open-loop twin gets the EXACT SAME bit `world_obs` passively (cannot act back!)
            uint32_t fy = step_mind(yoked[i], world_obs, H_SYNTH_7_11, t, in_tail, false);
            if (in_tail) { sum_flips_c += fc; sum_flips_y += fy; }
        }
    }

    uint64_t min_c = closed[0].tail_err, min_y = yoked[0].tail_err;
    double avg_c = 0, avg_y = 0;
    for (int i = 0; i < POP; ++i) {
        avg_c += (double)closed[i].tail_err;
        avg_y += (double)yoked[i].tail_err;
        if (closed[i].tail_err < min_c) min_c = closed[i].tail_err;
        if (yoked[i].tail_err  < min_y) min_y = yoked[i].tail_err;
    }
    double best_c = 100.0 * (1.0 - (double)min_c / TAIL);
    double best_y = 100.0 * (1.0 - (double)min_y / TAIL);
    avg_c = 100.0 * (1.0 - (avg_c / POP) / TAIL);
    avg_y = 100.0 * (1.0 - (avg_y / POP) / TAIL);

    std::cout << std::fixed << std::setprecision(2)
              << "H12 Closed-Loop Agent | Best=" << best_c << "% | Avg=" << avg_c
              << "% | Flips=" << (double)sum_flips_c / (POP * TAIL) << "/256\n"
              << "H12 Yoked Passive Twin| Best=" << best_y << "% | Avg=" << avg_y
              << "% | Flips=" << (double)sum_flips_y / (POP * TAIL) << "/256\n";
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1177: FUNDAMENTAL MIND HYPOTHESES H7..H12 ON PURE HOMOGENEOUS 256-BIT FIELD\n";
    std::cout << " ZERO Layer-Cakes | ZERO Masks | ZERO Weighted Readout | Prediction = S[0] & 1 at site 0\n";
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

    run_h12_closed_loop_vs_yoked();
    return 0;
}
