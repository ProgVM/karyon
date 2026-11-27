#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1179: PURE UNREMARKABLE MEDIUM + STRUCTURED NOISE + WAIT
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | ZERO goals, ZERO fitness, ZERO mutation, ZERO cloning.
//
// L1-Resident Bit-Packed SWAR (256x256 = 65,536 sites, 8 KB per grid!):
//   - Vertical axis y (256 sites) packed into 4 x uint64_t words per column x.
//   - Structured noise (vs pure random white noise control) enters at x=0 and
//     dissipates at x=W-1.
//   - Zero outer optimization, zero error functions, zero artificial triggers.
//   - We simply WAIT 500,000 continuous ticks per medium and passively observe
//     what (if anything) condenses out of 4 completely unremarkable physics:
//       A: Isotropic Spin Majority + faint thermal noise (Ising-like)
//       B: Isotropic B3/S23 + faint thermal noise
//       C: Isotropic B36/S23 + faint thermal noise
//       D: Conserved Reversible Spin-Gas (Margolus 2x2 Billiard/Critters-style)
// ============================================================================

constexpr int W = 256;
constexpr int WY = 4; // 4 * 64 = 256 vertical sites
constexpr int H = WY * 64;

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Branchless bit-sliced adder: a + b + c -> (sum, carry) across 64 sites at once
static inline void full_add(uint64_t a, uint64_t b, uint64_t c, uint64_t& s, uint64_t& cout) {
    uint64_t p = a ^ b;
    s = p ^ c;
    cout = (a & b) | (p & c);
}

static inline void half_add(uint64_t a, uint64_t b, uint64_t& s, uint64_t& cout) {
    s = a ^ b;
    cout = a & b;
}

// Step 256x256 medium using pure SWAR bit-sliced arithmetic
static void step_swar(
    const uint64_t* cur, uint64_t* nxt,
    const uint64_t* stream_words, size_t n_stream_words,
    uint64_t tick, int phys_type, uint64_t& rng)
{
    // Boundary x=0: continuous flux enters (4 words = 256 bits per tick)
    size_t spos = (size_t)((tick * WY) % n_stream_words);
    for (int wy = 0; wy < WY; ++wy) {
        nxt[wy] = stream_words[(spos + wy) % n_stream_words];
    }

    // Absorbing sink at x = W - 1
    for (int wy = 0; wy < WY; ++wy) {
        uint64_t r = splitmix64(rng);
        uint64_t prev_col = cur[(W - 2) * WY + wy];
        uint64_t mask = splitmix64(rng);
        nxt[(W - 1) * WY + wy] = (mask & r) | (~mask & prev_col);
    }

    if (phys_type == 3) {
        // Medium D: Pure Conserved Reversible Margolus 2x2 Physics (Critters / BBM-like)
        // Strictly conserves matter in the interior (even/odd staggered 2x2 blocks)!
        // Copy interior first, then transform staggered 2x2 blocks
        for (int i = WY; i < (W - 1) * WY; ++i) nxt[i] = cur[i];
        int phase = (int)(tick & 1ULL);
        for (int x = 1 + phase; x < W - 2; x += 2) {
            for (int wy = 0; wy < WY; ++wy) {
                // In y, phase=0 pairs (0,1),(2,3)...; phase=1 pairs (1,2),(3,4)...
                uint64_t a0 = cur[x * WY + wy];
                uint64_t b0 = cur[(x + 1) * WY + wy];
                uint64_t a1, b1;
                if (phase == 0) {
                    // Pair bit 2k with 2k+1
                    uint64_t even_m = 0x5555555555555555ULL;
                    uint64_t A = a0 & even_m;
                    uint64_t B = b0 & even_m;
                    uint64_t C = (a0 >> 1) & even_m;
                    uint64_t D = (b0 >> 1) & even_m;
                    // Pure reversible billiard-like collision (conserves particle count exactly!):
                    // 0 particles -> 0; 1 -> rotates CW; 2 diagonal -> swaps; 2 adj -> stays; 3 -> rotates CCW; 4 -> 4
                    uint64_t sum = A + B + C + D; // max 4 in even bits? Wait: 2-bit slots can overflow 1 bit! Use bit logic!
                    uint64_t nA = A, nB = B, nC = C, nD = D;
                    uint64_t one = (A ^ B ^ C ^ D) & ~( (A&B)|(A&C)|(A&D)|(B&C)|(B&D)|(C&D) );
                    uint64_t three = ( (A&B&C&~D)|(A&B&~C&D)|(A&~B&C&D)|(~A&B&C&D) );
                    uint64_t diag1 = (A & D & ~B & ~C);
                    uint64_t diag2 = (~A & ~D & B & C);
                    // Rotate CW on 1 particle: A->B->D->C->A; CCW on 3; swap on diag
                    uint64_t rot_cw_A = C, rot_cw_B = A, rot_cw_C = D, rot_cw_D = B;
                    uint64_t rot_ccw_A = B, rot_ccw_B = D, rot_ccw_C = A, rot_ccw_D = C;
                    nA = (one & rot_cw_A) | (three & rot_ccw_A) | (diag1 & B) | (diag2 & A) | (~(one|three|diag1|diag2) & A);
                    nB = (one & rot_cw_B) | (three & rot_ccw_B) | (diag1 & A) | (diag2 & D) | (~(one|three|diag1|diag2) & B);
                    nC = (one & rot_cw_C) | (three & rot_ccw_C) | (diag1 & D) | (diag2 & C) | (~(one|three|diag1|diag2) & C);
                    nD = (one & rot_cw_D) | (three & rot_ccw_D) | (diag1 & C) | (diag2 & B) | (~(one|three|diag1|diag2) & D);
                    (void)sum;
                    nxt[x * WY + wy]       = (nA & even_m) | ((nC & even_m) << 1);
                    nxt[(x + 1) * WY + wy] = (nB & even_m) | ((nD & even_m) << 1);
                } else {
                    // Staggered y: shift down by 1 across word boundary, pair even bits, shift back
                    int next_wy = (wy + 1) & (WY - 1);
                    uint64_t a_sh = (a0 >> 1) | (cur[x * WY + next_wy] << 63);
                    uint64_t b_sh = (b0 >> 1) | (cur[(x + 1) * WY + next_wy] << 63);
                    uint64_t even_m = 0x5555555555555555ULL;
                    uint64_t A = a_sh & even_m;
                    uint64_t B = b_sh & even_m;
                    uint64_t C = (a_sh >> 1) & even_m;
                    uint64_t D = (b_sh >> 1) & even_m;
                    uint64_t one = (A ^ B ^ C ^ D) & ~( (A&B)|(A&C)|(A&D)|(B&C)|(B&D)|(C&D) );
                    uint64_t three = ( (A&B&C&~D)|(A&B&~C&D)|(A&~B&C&D)|(~A&B&C&D) );
                    uint64_t diag1 = (A & D & ~B & ~C);
                    uint64_t diag2 = (~A & ~D & B & C);
                    uint64_t nA = (one & C) | (three & B) | (diag1 & B) | (diag2 & A) | (~(one|three|diag1|diag2) & A);
                    uint64_t nB = (one & A) | (three & D) | (diag1 & A) | (diag2 & D) | (~(one|three|diag1|diag2) & B);
                    uint64_t nC = (one & D) | (three & A) | (diag1 & D) | (diag2 & C) | (~(one|three|diag1|diag2) & C);
                    uint64_t nD = (one & B) | (three & C) | (diag1 & C) | (diag2 & B) | (~(one|three|diag1|diag2) & D);
                    uint64_t out_a_sh = (nA & even_m) | ((nC & even_m) << 1);
                    uint64_t out_b_sh = (nB & even_m) | ((nD & even_m) << 1);
                    // Put back unshifted
                    nxt[x * WY + wy] = (nxt[x * WY + wy] & 1ULL) | (out_a_sh << 1);
                    nxt[x * WY + next_wy] = (nxt[x * WY + next_wy] & ~1ULL) | (out_a_sh >> 63);
                    nxt[(x + 1) * WY + wy] = (nxt[(x + 1) * WY + wy] & 1ULL) | (out_b_sh << 1);
                    nxt[(x + 1) * WY + next_wy] = (nxt[(x + 1) * WY + next_wy] & ~1ULL) | (out_b_sh >> 63);
                }
            }
        }
        return;
    }

    // Standard isotropic local 2D neighborhood via parallel bit-sliced SWAR
    for (int x = 1; x < W - 1; ++x) {
        const uint64_t* colL = cur + (x - 1) * WY;
        const uint64_t* colC = cur + x * WY;
        const uint64_t* colR = cur + (x + 1) * WY;
        uint64_t* colN = nxt + x * WY;

        for (int wy = 0; wy < WY; ++wy) {
            int wy_m = (wy == 0) ? (WY - 1) : (wy - 1);
            int wy_p = (wy + 1 == WY) ? 0 : (wy + 1);

            auto up   = [&](const uint64_t* col) { return (col[wy] << 1) | (col[wy_m] >> 63); };
            auto down = [&](const uint64_t* col) { return (col[wy] >> 1) | (col[wy_p] << 63); };

            uint64_t C  = colC[wy];
            uint64_t L  = colL[wy],  R  = colR[wy];
            uint64_t U  = up(colC),  D  = down(colC);
            uint64_t UL = up(colL),  UR = up(colR);
            uint64_t DL = down(colL),DR = down(colR);

            // Faint uniform background thermal noise
            uint64_t r1 = splitmix64(rng), r2 = splitmix64(rng), r3 = splitmix64(rng);

            if (phys_type == 0) {
                // Medium A: Spin Majority of 4 neighbors (sum4 > 2 -> 1, sum4 < 2 -> 0, sum4 == 2 -> C) + ~0.4% thermal flip
                uint64_t s0, c0, s1, c1, sum_b0, sum_b1, sum_b2;
                half_add(L, R, s0, c0);
                half_add(U, D, s1, c1);
                half_add(s0, s1, sum_b0, sum_b1);
                uint64_t carry_hi;
                full_add(c0, c1, sum_b1, sum_b1, carry_hi);
                sum_b2 = carry_hi;
                // sum >= 3 iff (sum_b2) or (sum_b1 & sum_b0); sum == 2 iff (~sum_b2 & sum_b1 & ~sum_b0)
                uint64_t gt2 = sum_b2 | (sum_b1 & sum_b0);
                uint64_t eq2 = ~sum_b2 & sum_b1 & ~sum_b0;
                uint64_t out = gt2 | (eq2 & C);
                uint64_t thermal = r1 & r2 & r3 & splitmix64(rng) & splitmix64(rng) & splitmix64(rng) & splitmix64(rng) & splitmix64(rng); // ~1/256 (~0.39%)
                colN[wy] = out ^ thermal;
            } else {
                // Medium B (B3/S23) or Medium C (B36/S23): 8-neighbor bit-sliced sum
                uint64_t s0, c0, s1, c1, s2, c2, s3, c3;
                full_add(L, R, U, s0, c0);
                full_add(D, UL, UR, s1, c1);
                half_add(DL, DR, s2, c2);

                uint64_t b0, cb0, b0_final, cb0_2;
                full_add(s0, s1, s2, b0, cb0);
                // Now sum carries: c0, c1, c2, cb0 (4 bits -> bit 1 and bit 2,3)
                uint64_t t0, ct0, t1, ct1;
                full_add(c0, c1, c2, t0, ct0);
                half_add(t0, cb0, b0_final, cb0_2); // wait: b0_final is bit 1!
                uint64_t b1 = b0_final;
                uint64_t b2, b3;
                half_add(ct0, cb0_2, b2, b3);

                // Bit masks for sum8 == 2, 3, 6:
                // 2 = 0010, 3 = 0011, 6 = 0110
                uint64_t is_2 = ~b3 & ~b2 &  b1 & ~b0;
                uint64_t is_3 = ~b3 & ~b2 &  b1 &  b0;
                uint64_t is_6 = ~b3 &  b2 &  b1 & ~b0;

                uint64_t out;
                if (phys_type == 1) {
                    out = is_3 | (C & is_2); // Pure B3/S23
                } else {
                    out = is_3 | (~C & is_6) | (C & is_2); // Pure B36/S23
                }
                // Faint ~0.024% thermal noise (1 in 4096 = 12 ANDed bits)
                uint64_t thermal = r1 & r2 & r3 & (r1 >> 1) & (r2 >> 1) & (r3 >> 1)
                                 & (r1 >> 2) & (r2 >> 2) & (r3 >> 2)
                                 & (r1 >> 3) & (r2 >> 3) & (r3 >> 3);
                colN[wy] = out ^ thermal;
            }
        }
    }
}

static std::vector<uint64_t> pack_bytes_to_words(const std::vector<uint8_t>& bytes) {
    size_t nw = bytes.size() / 8;
    std::vector<uint64_t> words(nw);
    for (size_t i = 0; i < nw; ++i) {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b) v |= ((uint64_t)bytes[i * 8 + b] << (b * 8));
        words[i] = v;
    }
    return words;
}

static std::vector<uint8_t> gen_white_noise(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes);
    uint64_t x = 0x123456789ABCDEF0ULL;
    for (size_t i = 0; i < n_bytes; ++i) out[i] = (uint8_t)(splitmix64(x) & 0xFF);
    return out;
}

static std::vector<uint8_t> gen_rule110_stream(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    const int LW = 256;
    std::vector<uint8_t> row(LW, 0), nxt(LW, 0);
    uint64_t s = 0xCAFEBABE12345678ULL;
    for (int i = 0; i < LW; ++i) row[i] = (uint8_t)((splitmix64(s) >> 63) & 1ULL);
    size_t t = 0, tot = n_bytes * 8;
    while (t < tot) {
        for (int i = 0; i < LW && t < tot; ++i, ++t) {
            if (row[i]) out[t >> 3] |= (1U << (t & 7));
        }
        for (int i = 0; i < LW; ++i) {
            uint8_t pat = (row[(i + LW - 1) % LW] << 2) | (row[i] << 1) | row[(i + 1) % LW];
            nxt[i] = (110 >> pat) & 1U;
        }
        row.swap(nxt);
    }
    return out;
}

struct Stats {
    double dens;
    double near_dens, mid_dens, far_dens; // x=1..63, x=64..191, x=192..254
    double H4;
    size_t uniq_4x4;
    double top_motif_pct;
    double persist_100t;
};

static Stats measure_stats(const std::vector<uint64_t>& cur, const std::vector<uint64_t>& prev) {
    uint64_t near1 = 0, mid1 = 0, far1 = 0, same = 0;
    for (int x = 1; x < W - 1; ++x) {
        for (int wy = 0; wy < WY; ++wy) {
            uint64_t v = cur[x * WY + wy];
            uint64_t c = (uint64_t)__builtin_popcountll(v);
            if (x < 64) near1 += c;
            else if (x < 192) mid1 += c;
            else far1 += c;
            same += (uint64_t)__builtin_popcountll(~(v ^ prev[x * WY + wy]));
        }
    }
    double tot = (double)(W - 2) * H;
    double dens = (double)(near1 + mid1 + far1) / tot;

    std::unordered_map<uint16_t, uint32_t> hist;
    hist.reserve(4096);
    uint64_t tiles = 0;
    for (int x = 8; x < W - 8; x += 4) {
        for (int y = 0; y < H; y += 4) {
            int wy = y >> 6;
            int bit = y & 63;
            uint16_t pat = 0;
            for (int dx = 0; dx < 4; ++dx) {
                uint64_t nib = (cur[(x + dx) * WY + wy] >> bit) & 0xFULL;
                pat |= (uint16_t)(nib << (dx * 4));
            }
            hist[pat]++;
            tiles++;
        }
    }

    double H4 = 0.0;
    uint32_t max_nt = 0;
    for (auto& kv : hist) {
        double p = (double)kv.second / (double)tiles;
        H4 -= p * std::log2(p);
        if (kv.first != 0 && kv.first != 0xFFFF && kv.second > max_nt) max_nt = kv.second;
    }

    return {
        dens,
        (double)near1 / (63.0 * H),
        (double)mid1  / (128.0 * H),
        (double)far1  / (63.0 * H),
        H4,
        hist.size(),
        100.0 * (double)max_nt / (double)tiles,
        100.0 * (double)same / tot
    };
}

static void run_medium(const char* title, int phys_type, const std::vector<uint64_t>& stream, uint64_t ticks) {
    std::vector<uint64_t> A(W * WY), B(W * WY), snap_prev(W * WY);
    uint64_t rng = 0x9E3779B97F4A7C15ULL + (uint64_t)phys_type * 1337ULL;
    for (int i = 0; i < W * WY; ++i) A[i] = splitmix64(rng);

    std::cout << "\n--- " << title << " ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t* cur = A.data();
    uint64_t* nxt = B.data();

    for (uint64_t t = 0; t <= ticks; ++t) {
        step_swar(cur, nxt, stream.data(), stream.size(), t, phys_type, rng);
        std::swap(cur, nxt);

        if (t == 1000 || t == 10000 || t == 50000 || t == 200000 || t == ticks) {
            std::copy(cur, cur + W * WY, snap_prev.begin());
            for (int k = 1; k <= 100; ++k) {
                step_swar(cur, nxt, stream.data(), stream.size(), t + k, phys_type, rng);
                std::swap(cur, nxt);
            }
            std::vector<uint64_t> cur_vec(cur, cur + W * WY);
            Stats st = measure_stats(cur_vec, snap_prev);
            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  t=" << std::setw(6) << t
                      << " | dens=" << std::fixed << std::setprecision(3) << st.dens
                      << " (near=" << st.near_dens << ", mid=" << st.mid_dens << ", far=" << st.far_dens << ")"
                      << " | H_4x4=" << std::setw(5) << std::setprecision(2) << st.H4 << "/16"
                      << " | uniq4x4=" << std::setw(5) << st.uniq_4x4
                      << " | topMotif=" << std::setw(5) << st.top_motif_pct << "%"
                      << " | persist100t=" << std::setw(6) << st.persist_100t << "%"
                      << " | " << sec << "s\n";
            t += 100;
        }
    }
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1179: PURE UNREMARKABLE MEDIUM + STRUCTURED NOISE + WAIT (SWAR 256x256)\n";
    std::cout << " 65,536 sites | ZERO goals, ZERO fitness, ZERO mutation, ZERO cloning, ZERO imposed mechanisms.\n";
    std::cout << "==================================================================================================\n";

    const size_t BYTES = 131072; // 1M bits
    auto white_w = pack_bytes_to_words(gen_white_noise(BYTES));
    auto r110_w  = pack_bytes_to_words(gen_rule110_stream(BYTES));

    std::vector<uint8_t> wiki_b;
    std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
    if (fw) {
        wiki_b.resize(BYTES);
        fw.read((char*)wiki_b.data(), BYTES);
        wiki_b.resize((size_t)(fw.gcount() & ~7LL));
    }
    auto wiki_w = pack_bytes_to_words(wiki_b);

    const uint64_t TICKS = 200000;

    run_medium("Medium A (Spin Majority + Faint Thermal) | Control: White Noise", 0, white_w, TICKS);
    run_medium("Medium A (Spin Majority + Faint Thermal) | Structured: Rule 110", 0, r110_w, TICKS);
    if (!wiki_w.empty()) run_medium("Medium A (Spin Majority + Faint Thermal) | Structured: Wikipedia", 0, wiki_w, TICKS);

    run_medium("Medium B (Isotropic B3/S23 + Faint Thermal) | Control: White Noise", 1, white_w, TICKS);
    run_medium("Medium B (Isotropic B3/S23 + Faint Thermal) | Structured: Rule 110", 1, r110_w, TICKS);
    if (!wiki_w.empty()) run_medium("Medium B (Isotropic B3/S23 + Faint Thermal) | Structured: Wikipedia", 1, wiki_w, TICKS);

    run_medium("Medium C (Isotropic B36/S23 + Faint Thermal) | Control: White Noise", 2, white_w, TICKS);
    run_medium("Medium C (Isotropic B36/S23 + Faint Thermal) | Structured: Rule 110", 2, r110_w, TICKS);
    if (!wiki_w.empty()) run_medium("Medium C (Isotropic B36/S23 + Faint Thermal) | Structured: Wikipedia", 2, wiki_w, TICKS);

    run_medium("Medium D (Conserved Reversible Spin-Gas) | Control: White Noise", 3, white_w, TICKS);
    run_medium("Medium D (Conserved Reversible Spin-Gas) | Structured: Rule 110", 3, r110_w, TICKS);
    if (!wiki_w.empty()) run_medium("Medium D (Conserved Reversible Spin-Gas) | Structured: Wikipedia", 3, wiki_w, TICKS);

    return 0;
}
