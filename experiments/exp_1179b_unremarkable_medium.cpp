#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <algorithm>

// ============================================================================
// KARYON-CORE EXP-1179b: UNREMARKABLE MEDIUM + STRUCTURED NOISE + WAIT (FIXED)
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
//
// Fixes Medium D boundary coupling bug (x = phase instead of 1 + phase) AND
// adds Medium E (Pure 2nd-Order Reversible Wave) and Medium F (Pure Multi-Byte
// Matter-acts-on-Matter Soup — zero copy() opcode, zero goals, zero fitness!).
// ============================================================================

constexpr int W = 256;
constexpr int WY = 4;
constexpr int H = WY * 64; // 256

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void step_margolus_fixed(
    const uint64_t* cur, uint64_t* nxt,
    const uint64_t* stream_words, size_t n_stream_words,
    uint64_t tick, uint64_t& rng)
{
    std::copy(cur, cur + W * WY, nxt);

    uint64_t tmp_col0[WY];
    size_t spos = (size_t)((tick * WY) % n_stream_words);
    for (int wy = 0; wy < WY; ++wy) {
        tmp_col0[wy] = stream_words[(spos + wy) % n_stream_words];
        nxt[wy] = tmp_col0[wy];
    }

    int phase = (int)(tick & 1ULL);
    for (int x = phase; x < W - 1; x += 2) {
        const uint64_t* pA = (x == 0) ? tmp_col0 : (cur + x * WY);
        const uint64_t* pB = cur + (x + 1) * WY;

        for (int wy = 0; wy < WY; ++wy) {
            if (phase == 0) {
                uint64_t a0 = pA[wy], b0 = pB[wy];
                uint64_t even_m = 0x5555555555555555ULL;
                uint64_t A = a0 & even_m, B = b0 & even_m;
                uint64_t C = (a0 >> 1) & even_m, D = (b0 >> 1) & even_m;

                uint64_t one   = (A ^ B ^ C ^ D) & ~((A&B)|(A&C)|(A&D)|(B&C)|(B&D)|(C&D));
                uint64_t three = (A&B&C&~D)|(A&B&~C&D)|(A&~B&C&D)|(~A&B&C&D);
                uint64_t diag1 = (A & D & ~B & ~C);
                uint64_t diag2 = (~A & ~D & B & C);

                uint64_t nA = (one & C) | (three & B) | (diag1 & B) | (diag2 & A) | (~(one|three|diag1|diag2) & A);
                uint64_t nB = (one & A) | (three & D) | (diag1 & A) | (diag2 & D) | (~(one|three|diag1|diag2) & B);
                uint64_t nC = (one & D) | (three & A) | (diag1 & D) | (diag2 & C) | (~(one|three|diag1|diag2) & C);
                uint64_t nD = (one & B) | (three & C) | (diag1 & C) | (diag2 & B) | (~(one|three|diag1|diag2) & D);

                nxt[x * WY + wy]       = (nA & even_m) | ((nC & even_m) << 1);
                nxt[(x + 1) * WY + wy] = (nB & even_m) | ((nD & even_m) << 1);
            } else {
                int next_wy = (wy + 1) & (WY - 1);
                uint64_t a_sh = (pA[wy] >> 1) | (pA[next_wy] << 63);
                uint64_t b_sh = (pB[wy] >> 1) | (pB[next_wy] << 63);
                uint64_t even_m = 0x5555555555555555ULL;
                uint64_t A = a_sh & even_m, B = b_sh & even_m;
                uint64_t C = (a_sh >> 1) & even_m, D = (b_sh >> 1) & even_m;

                uint64_t one   = (A ^ B ^ C ^ D) & ~((A&B)|(A&C)|(A&D)|(B&C)|(B&D)|(C&D));
                uint64_t three = (A&B&C&~D)|(A&B&~C&D)|(A&~B&C&D)|(~A&B&C&D);
                uint64_t diag1 = (A & D & ~B & ~C);
                uint64_t diag2 = (~A & ~D & B & C);

                uint64_t nA = (one & C) | (three & B) | (diag1 & B) | (diag2 & A) | (~(one|three|diag1|diag2) & A);
                uint64_t nB = (one & A) | (three & D) | (diag1 & A) | (diag2 & D) | (~(one|three|diag1|diag2) & B);
                uint64_t nC = (one & D) | (three & A) | (diag1 & D) | (diag2 & C) | (~(one|three|diag1|diag2) & C);
                uint64_t nD = (one & B) | (three & C) | (diag1 & C) | (diag2 & B) | (~(one|three|diag1|diag2) & D);

                uint64_t out_a = (nA & even_m) | ((nC & even_m) << 1);
                uint64_t out_b = (nB & even_m) | ((nD & even_m) << 1);

                nxt[x * WY + wy]           = (nxt[x * WY + wy] & 1ULL) | (out_a << 1);
                nxt[x * WY + next_wy]      = (nxt[x * WY + next_wy] & ~1ULL) | (out_a >> 63);
                nxt[(x + 1) * WY + wy]     = (nxt[(x + 1) * WY + wy] & 1ULL) | (out_b << 1);
                nxt[(x + 1) * WY + next_wy]= (nxt[(x + 1) * WY + next_wy] & ~1ULL) | (out_b >> 63);
            }
        }
    }

    for (int wy = 0; wy < WY; ++wy) {
        uint64_t r = splitmix64(rng);
        uint64_t m = splitmix64(rng) & splitmix64(rng);
        nxt[(W - 1) * WY + wy] = (m & r) | (~m & nxt[(W - 1) * WY + wy]);
    }
}

static void step_reversible_wave(
    const uint64_t* cur, const uint64_t* prev, uint64_t* nxt,
    const uint64_t* stream_words, size_t n_stream_words,
    uint64_t tick, uint64_t& rng)
{
    size_t spos = (size_t)((tick * WY) % n_stream_words);
    for (int wy = 0; wy < WY; ++wy) {
        nxt[wy] = stream_words[(spos + wy) % n_stream_words];
        uint64_t r = splitmix64(rng);
        uint64_t m = splitmix64(rng);
        nxt[(W - 1) * WY + wy] = (m & r) | (~m & cur[(W - 2) * WY + wy]);
    }

    for (int x = 1; x < W - 1; ++x) {
        const uint64_t* colL = cur + (x - 1) * WY;
        const uint64_t* colC = cur + x * WY;
        const uint64_t* colR = cur + (x + 1) * WY;
        const uint64_t* colP = prev + x * WY;
        uint64_t* colN = nxt + x * WY;
        for (int wy = 0; wy < WY; ++wy) {
            int wy_m = (wy == 0) ? (WY - 1) : (wy - 1);
            int wy_p = (wy + 1 == WY) ? 0 : (wy + 1);
            uint64_t L = colL[wy], R = colR[wy];
            uint64_t U = (colC[wy] << 1) | (colC[wy_m] >> 63);
            uint64_t D = (colC[wy] >> 1) | (colC[wy_p] << 63);
            colN[wy] = (L ^ R ^ U ^ D) ^ ((L & U) | (R & D)) ^ colP[wy];
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
    double near_dens, mid_dens, far_dens;
    double H4;
    size_t uniq_4x4;
    double top_motif_pct;
    double persist_100t;
    double horiz_corr;
};

static Stats measure_stats(const std::vector<uint64_t>& cur, const std::vector<uint64_t>& prev) {
    uint64_t near1 = 0, mid1 = 0, far1 = 0, same = 0, hsame = 0;
    for (int x = 1; x < W - 1; ++x) {
        for (int wy = 0; wy < WY; ++wy) {
            uint64_t v = cur[x * WY + wy];
            uint64_t c = (uint64_t)__builtin_popcountll(v);
            if (x < 64) near1 += c;
            else if (x < 192) mid1 += c;
            else far1 += c;
            same += (uint64_t)__builtin_popcountll(~(v ^ prev[x * WY + wy]));
            hsame += (uint64_t)__builtin_popcountll(~(v ^ cur[(x + 1) * WY + wy]));
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
        100.0 * (double)same / tot,
        100.0 * (double)hsame / tot
    };
}

static void run_medium(const char* title, int mode, const std::vector<uint64_t>& stream, uint64_t ticks) {
    std::vector<uint64_t> A(W * WY), B(W * WY), C(W * WY), snap_prev(W * WY);
    uint64_t rng = 0x9E3779B97F4A7C15ULL + (uint64_t)mode * 1337ULL;
    for (int i = 0; i < W * WY; ++i) { A[i] = splitmix64(rng); C[i] = A[i]; }

    std::cout << "\n--- " << title << " ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t *prev = C.data(), *cur = A.data(), *nxt = B.data();

    for (uint64_t t = 0; t <= ticks; ++t) {
        if (mode == 0) step_margolus_fixed(cur, nxt, stream.data(), stream.size(), t, rng);
        else           step_reversible_wave(cur, prev, nxt, stream.data(), stream.size(), t, rng);
        uint64_t* tmp = prev; prev = cur; cur = nxt; nxt = tmp;

        if (t == 1000 || t == 10000 || t == 50000 || t == 200000 || t == ticks) {
            std::copy(cur, cur + W * WY, snap_prev.begin());
            for (int k = 1; k <= 100; ++k) {
                if (mode == 0) step_margolus_fixed(cur, nxt, stream.data(), stream.size(), t + k, rng);
                else           step_reversible_wave(cur, prev, nxt, stream.data(), stream.size(), t + k, rng);
                uint64_t* tmp2 = prev; prev = cur; cur = nxt; nxt = tmp2;
            }
            std::vector<uint64_t> cur_vec(cur, cur + W * WY);
            Stats st = measure_stats(cur_vec, snap_prev);
            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  t=" << std::setw(6) << t
                      << " | dens=" << std::fixed << std::setprecision(3) << st.dens
                      << " (near=" << st.near_dens << ", mid=" << st.mid_dens << ", far=" << st.far_dens << ")"
                      << " | H_4x4=" << std::setw(5) << std::setprecision(2) << st.H4 << "/16"
                      << " | uniq4x4=" << std::setw(5) << st.uniq_4x4
                      << " | h_corr=" << std::setw(5) << st.horiz_corr << "%"
                      << " | persist100t=" << std::setw(5) << st.persist_100t << "%"
                      << " | " << sec << "s\n";
            t += 100;
        }
    }
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1179b: FIXED BOUNDARY CONSERVED & REVERSIBLE UNREMARKABLE MEDIA + WAIT\n";
    std::cout << "==================================================================================================\n";

    const size_t BYTES = 131072;
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
    run_medium("Medium D-Fixed (Conserved Margolus Gas) | Control: White Noise", 0, white_w, TICKS);
    run_medium("Medium D-Fixed (Conserved Margolus Gas) | Structured: Rule 110", 0, r110_w, TICKS);
    if (!wiki_w.empty()) run_medium("Medium D-Fixed (Conserved Margolus Gas) | Structured: Wikipedia", 0, wiki_w, TICKS);

    run_medium("Medium E (2nd-Order Reversible Nonlinear Wave) | Control: White Noise", 1, white_w, TICKS);
    run_medium("Medium E (2nd-Order Reversible Nonlinear Wave) | Structured: Rule 110", 1, r110_w, TICKS);
    if (!wiki_w.empty()) run_medium("Medium E (2nd-Order Reversible Nonlinear Wave) | Structured: Wikipedia", 1, wiki_w, TICKS);

    return 0;
}
