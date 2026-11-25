#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <string>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1173b: ENDOGENOUS REGIME-GATED SWAR SUBSTRATE + FUTURE ROLLOUT
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
// ============================================================================
// Improvements over EXP-1173:
//   1. Endogenous V=2 Regime-Gated Weight Field W_reg[64][S] (4 KB L1-resident):
//      Regime index derived strictly from endogenous dynamics (recent 3 error
//      bits E0 & 7, consecutive same-bit run length bucket, and last bit H0 & 1)
//      with zero byte-boundary hardcoding. Modulates long-range vs short-range
//      field trust dynamically per bit without floats.
//   2. True Non-Duplicate Morphogenesis & Faster Maturity Cycle:
//      Uniqueness enforced on (mask0, mask1) pairs, eliminating duplicate masks
//      and excessive 1 MB memset stalls while discovering compact Rule 110 taps.
//   3. Strict Detached Autonomous Future Rollout (H=+64, H=+256 bits):
//      Measures pure predictive continuation without teacher ground-truth input.
// ============================================================================

static std::vector<uint8_t> load_bytes(const char* path, size_t max_bytes = 0) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    size_t sz = (size_t)ifs.tellg();
    if (max_bytes > 0 && sz > max_bytes) sz = max_bytes;
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    if (sz > 0) ifs.read((char*)buf.data(), sz);
    return buf;
}

static inline uint64_t xorshift64(uint64_t& s) {
    uint64_t x = s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double bit_nll(double p1, uint8_t actual) {
    double p = actual ? p1 : (1.0 - p1);
    if (p < 1e-9) p = 1e-9;
    return -std::log2(p);
}

struct ExactLUT64K {
    int16_t logit[65536];
    uint8_t sig_lut[4097];
    double  prob_lut[4097];

    ExactLUT64K() {
        for (int c1 = 0; c1 < 256; ++c1) {
            for (int c0 = 0; c0 < 256; ++c0) {
                int s = (c1 << 8) | c0;
                double p = (c1 + 0.35) / (c0 + c1 + 0.70);
                logit[s] = (int16_t)std::round(std::log(p / (1.0 - p)) * 128.0);
            }
        }
        for (int i = 0; i <= 4096; ++i) {
            double x = (double)(i - 2048) / 128.0;
            double p = 1.0 / (1.0 + std::exp(-x));
            prob_lut[i] = p;
            int q = (int)std::round(p * 256.0);
            if (q < 1) q = 1; if (q > 255) q = 255;
            sig_lut[i] = (uint8_t)q;
        }
    }
    inline int32_t clamp_idx(int32_t x) const {
        return (x < -2048) ? 0 : ((x > 2048) ? 4096 : (x + 2048));
    }
};
static const ExactLUT64K g_lut;

struct OnlineF2Basis128 {
    uint64_t piv0[128]{}, piv1[128]{};
    uint8_t  pivy[128]{}, has[128]{};
    uint32_t hits = 0, misses = 0, lock_streak = 0, resets = 0;

    inline bool predict_if_spanned(uint64_t x0, uint64_t x1, uint8_t& out_y) const {
        uint8_t y = 0;
        while (x0) {
            int p = __builtin_ctzll(x0);
            if (!has[p]) return false;
            x0 ^= piv0[p]; x1 ^= piv1[p]; y ^= pivy[p];
        }
        while (x1) {
            int p = 64 + __builtin_ctzll(x1);
            if (!has[p]) return false;
            x0 ^= piv0[p]; x1 ^= piv1[p]; y ^= pivy[p];
        }
        out_y = y;
        return true;
    }

    inline void update(uint64_t x0, uint64_t x1, uint8_t actual_y) {
        uint8_t y = actual_y;
        while (x0) {
            int p = __builtin_ctzll(x0);
            if (!has[p]) { has[p] = 1; piv0[p] = x0; piv1[p] = x1; pivy[p] = y; return; }
            x0 ^= piv0[p]; x1 ^= piv1[p]; y ^= pivy[p];
        }
        while (x1) {
            int p = 64 + __builtin_ctzll(x1);
            if (!has[p]) { has[p] = 1; piv0[p] = x0; piv1[p] = x1; pivy[p] = y; return; }
            x0 ^= piv0[p]; x1 ^= piv1[p]; y ^= pivy[p];
        }
        if (y == 0) {
            hits++; lock_streak++;
        } else {
            misses++; lock_streak = 0;
            if (misses > 12) {
                std::fill(std::begin(has), std::end(has), 0);
                hits = 0; misses = 0; lock_streak = 0; resets++;
            }
        }
    }
};

static std::vector<uint8_t> make_lfsr96(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    uint64_t hi = 0xCAFEBABEDEADBEEFULL; uint32_t lo = 0x12345678U;
    for (size_t t = 0; t < n_bytes * 8; ++t) {
        uint8_t b = (uint8_t)(lo & 1U);
        if (b) out[t >> 3] |= (1U << (t & 7));
        uint8_t fb = b ^ ((lo >> 6) & 1U) ^ ((lo >> 9) & 1U) ^ ((lo >> 10) & 1U);
        lo = (lo >> 1) | ((uint32_t)(hi & 1ULL) << 31);
        hi = (hi >> 1) | ((uint64_t)fb << 63);
    }
    return out;
}

static std::vector<uint8_t> make_rule110(size_t n_bytes, size_t W = 64) {
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

struct StreamResult {
    std::string name;
    size_t bytes = 0;
    double karyon_bpb = 0.0;
    double karyon_tail_bpb = 0.0;
    double karyon_acc = 0.0;
    double karyon_tail_acc = 0.0;
    double karyon_mb_s = 0.0;
    double rollout_64_acc = 0.0;
    double rollout_256_acc = 0.0;
    uint32_t f2_streak = 0;
    uint64_t muts = 0;
    uint64_t top_m0[3]{}, top_m1[3]{};
    int32_t  top_w[3]{};
};

static StreamResult eval_karyon_1173b(const std::string& name, const std::vector<uint8_t>& data) {
    StreamResult res;
    res.name = name;
    res.bytes = data.size();

    size_t total_bits = data.size() * 8;
    size_t train_bits = total_bits - 256; // Reserve last 256 bits strictly for detached rollout
    auto t0 = std::chrono::high_resolution_clock::now();

    constexpr int S = 16;
    constexpr int REGIMES = 64; // (E0 & 7) << 3 | (run_b << 1) | (H0 & 1)
    constexpr uint32_t HASH_BITS = 19; // 512K * 2B * 16 = 16 MB
    constexpr uint32_t B_MASK = (1U << HASH_BITS) - 1U;
    std::vector<uint16_t> cells((size_t)S << HASH_BITS, 0);

    uint64_t mask0[S], mask1[S], maskE[S];
    int32_t  w_glob[S];
    int32_t  w_reg[REGIMES][S]{};
    uint32_t age[S];
    uint64_t rng = 0x9E3779B97F4A7C15ULL;

    for (int s = 0; s < S; ++s) {
        w_glob[s] = 128; age[s] = 0; maskE[s] = 0ULL;
        if (s < 10) {
            int len = 1 + s * 4;
            mask0[s] = (len >= 64) ? ~0ULL : ((1ULL << len) - 1ULL);
            mask1[s] = 0ULL;
        } else {
            int base = (s - 10) * 19;
            mask0[s] = 0ULL; mask1[s] = 0ULL;
            for (int q = 0; q < 3; ++q) {
                int pos = (base + q) & 127;
                if (pos < 64) mask0[s] |= (1ULL << pos); else mask1[s] |= (1ULL << (pos - 64));
            }
        }
    }

    OnlineF2Basis128 f2;
    uint64_t H0 = 0, H1 = 0, E0 = 0;
    double nll = 0.0, tail_nll = 0.0;
    uint64_t total_err = 0, tail_err = 0, muts = 0;
    size_t tail_start = (train_bits * 3) / 4;
    uint32_t idxs[S]; int32_t logits[S]; uint16_t st_cache[S];
    uint32_t same_run = 0;
    uint8_t prev_b = 0;

    for (size_t t = 0; t < train_bits; ++t) {
        uint8_t f2_y = 0;
        bool f2_active = (f2.resets <= 4 && f2.lock_streak >= 32) && f2.predict_if_spanned(H0, H1, f2_y);

        uint32_t run_b = (same_run >= 6) ? 3U : (same_run >> 1);
        uint32_t reg = ((uint32_t)(E0 & 7ULL) << 3) | (run_b << 1) | (uint32_t)(H0 & 1ULL);

        int32_t net_logit = 0;
        if (f2_active) {
            net_logit = f2_y ? 2048 : -2048;
        } else {
            for (int s = 0; s < S; ++s) {
                uint64_t key = (H0 & mask0[s]) ^ ((H1 & mask1[s]) * 0x9E3779B97F4A7C15ULL) ^ ((E0 & maskE[s]) * 0xBF58476D1CE4E5B9ULL);
                key ^= key >> 30; key *= 0xBF58476D1CE4E5B9ULL;
                uint32_t idx = (uint32_t)((key ^ (key >> 27)) & B_MASK);
                idxs[s] = idx;
                __builtin_prefetch(&cells[((size_t)s << HASH_BITS) + idx], 1, 3);
            }
            int64_t mix_logit = 0;
            for (int s = 0; s < S; ++s) {
                uint16_t st = cells[((size_t)s << HASH_BITS) + idxs[s]];
                st_cache[s] = st;
                logits[s] = g_lut.logit[st];
                int32_t eff_w = w_glob[s] + w_reg[reg][s];
                if (eff_w < 0) eff_w = 0;
                mix_logit += (int64_t)eff_w * logits[s];
            }
            net_logit = (int32_t)(mix_logit >> 10);
        }

        double p1 = g_lut.prob_lut[g_lut.clamp_idx(net_logit)];
        uint8_t b = (data[t >> 3] >> (t & 7)) & 1U;
        double bit_loss = bit_nll(p1, b);
        nll += bit_loss;
        if (t >= tail_start) tail_nll += bit_loss;

        uint8_t pred = (net_logit >= 0) ? 1U : 0U;
        uint8_t e = pred ^ b;
        if (e) { total_err++; if (t >= tail_start) tail_err++; }

        if (f2.resets <= 4 && t < 50000ULL) f2.update(H0, H1, b);
        else if (f2_active && e) f2.update(H0, H1, b);

        if (!f2_active) {
            int32_t p_mix = g_lut.sig_lut[g_lut.clamp_idx(net_logit)];
            int32_t err_q8 = (b ? 256 : 0) - p_mix;
            for (int s = 0; s < S; ++s) {
                int32_t dw = (err_q8 * logits[s]) >> 12;
                w_glob[s] += dw;
                if (w_glob[s] < 0) w_glob[s] = 0; if (w_glob[s] > 4096) w_glob[s] = 4096;
                w_reg[reg][s] += (dw >> 1);
                if (w_reg[reg][s] < -512) w_reg[reg][s] = -512;
                if (w_reg[reg][s] > 1024) w_reg[reg][s] = 1024;
                age[s]++;

                uint16_t st = st_cache[s];
                uint32_t c0 = st & 0xFFU, c1 = st >> 8;
                if (b) { c1++; if (c1 == 255U) { c0 >>= 1; c1 >>= 1; } }
                else   { c0++; if (c0 == 255U) { c0 >>= 1; c1 >>= 1; } }
                cells[((size_t)s << HASH_BITS) + idxs[s]] = (uint16_t)(c0 | (c1 << 8));
            }

            if ((t & 4095ULL) == 4095ULL) {
                int rank_order[S];
                for (int s = 0; s < S; ++s) rank_order[s] = s;
                std::sort(rank_order, rank_order + S, [&](int a, int b_i){ return w_glob[a] > w_glob[b_i]; });
                int worst_s = -1, min_w = 240;
                for (int r_i = 4; r_i < S; ++r_i) {
                    int s = rank_order[r_i];
                    if (age[s] >= 12288U && w_glob[s] < min_w) { min_w = w_glob[s]; worst_s = s; }
                }
                if (worst_s >= 0) {
                    uint64_t new_m0 = 0ULL, new_m1 = 0ULL, new_mE = 0ULL;
                    uint64_t r = xorshift64(rng);
                    if ((r % 100ULL) < 80ULL) {
                        int par = rank_order[(r >> 8) % 4ULL];
                        new_m0 = mask0[par]; new_m1 = mask1[par]; new_mE = maskE[par];
                        uint64_t r2 = xorshift64(rng);
                        int op = (int)(r2 % 100ULL);
                        if (op < 45 && (__builtin_popcountll(new_m0) + __builtin_popcountll(new_m1)) > 2) {
                            int np = 1 + (int)((r2 >> 8) & 1ULL);
                            for (int q = 0; q < np; ++q) {
                                int bpos = (int)(xorshift64(rng) % 64ULL);
                                if (new_m0 && (!new_m1 || (xorshift64(rng) & 1ULL))) {
                                    uint64_t rot = (new_m0 >> bpos) | (bpos ? (new_m0 << (64 - bpos)) : 0ULL);
                                    if (rot) new_m0 &= ~(1ULL << ((bpos + __builtin_ctzll(rot)) & 63));
                                } else if (new_m1) {
                                    uint64_t rot = (new_m1 >> bpos) | (bpos ? (new_m1 << (64 - bpos)) : 0ULL);
                                    if (rot) new_m1 &= ~(1ULL << ((bpos + __builtin_ctzll(rot)) & 63));
                                }
                            }
                        } else if (op < 80) {
                            uint64_t missing = ~new_m0;
                            if (missing) new_m0 |= (1ULL << __builtin_ctzll(missing));
                            else if (~new_m1) new_m1 |= (1ULL << __builtin_ctzll(~new_m1));
                        } else if (op < 94) {
                            int sh = (int)((r2 >> 16) % 128ULL);
                            if (sh < 64) new_m0 ^= (1ULL << sh); else new_m1 ^= (1ULL << (sh - 64));
                        } else {
                            new_mE ^= (1ULL << ((r2 >> 16) % 16ULL));
                        }
                    } else {
                        int base = (int)((r >> 8) % 122ULL);
                        int width = 2 + (int)((r >> 16) % 4ULL);
                        for (int q = 0; q < width; ++q) {
                            int pos = (base + q) & 127;
                            if (pos < 64) new_m0 |= (1ULL << pos); else new_m1 |= (1ULL << (pos - 64));
                        }
                    }
                    bool dup = false;
                    for (int k = 0; k < S; ++k) {
                        if (k != worst_s && mask0[k] == new_m0 && mask1[k] == new_m1) { dup = true; break; }
                    }
                    if (!dup && (new_m0 || new_m1)) {
                        mask0[worst_s] = new_m0; mask1[worst_s] = new_m1; maskE[worst_s] = new_mE;
                        w_glob[worst_s] = 128; age[worst_s] = 0;
                        for (int rg = 0; rg < REGIMES; ++rg) w_reg[rg][worst_s] = 0;
                        std::fill(cells.begin() + ((size_t)worst_s << HASH_BITS),
                                  cells.begin() + ((size_t)(worst_s + 1) << HASH_BITS), 0);
                        muts++;
                    }
                }
            }
        }

        if (b == prev_b) same_run++; else { prev_b = b; same_run = 1; }
        H1 = (H1 << 1) | (H0 >> 63);
        H0 = (H0 << 1) | b;
        E0 = (E0 << 1) | e;
    }

    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    double train_bytes = (double)train_bits / 8.0;
    double tail_bytes = (double)(train_bits - tail_start) / 8.0;
    res.karyon_bpb = nll / train_bytes;
    res.karyon_tail_bpb = tail_nll / tail_bytes;
    res.karyon_acc = 100.0 * (1.0 - (double)total_err / train_bits);
    res.karyon_tail_acc = 100.0 * (1.0 - (double)tail_err / (train_bits - tail_start));
    res.karyon_mb_s = (train_bytes / 1e6) / sec;
    res.f2_streak = f2.lock_streak;
    res.muts = muts;

    // Detached Autonomous Future Rollout (H=64 and H=256 unseen bits, zero teacher input!)
    uint64_t sim_H0 = H0, sim_H1 = H1, sim_E0 = E0;
    uint32_t sim_run = same_run;
    uint8_t sim_prev_b = prev_b;
    uint64_t ok64 = 0, ok256 = 0;
    for (size_t k = 0; k < 256; ++k) {
        size_t t = train_bits + k;
        uint8_t f2_y = 0;
        bool f2_active = (f2.resets <= 4 && f2.lock_streak >= 32) && f2.predict_if_spanned(sim_H0, sim_H1, f2_y);
        int32_t net_logit = 0;
        if (f2_active) {
            net_logit = f2_y ? 2048 : -2048;
        } else {
            uint32_t run_b = (sim_run >= 6) ? 3U : (sim_run >> 1);
            uint32_t reg = ((uint32_t)(sim_E0 & 7ULL) << 3) | (run_b << 1) | (uint32_t)(sim_H0 & 1ULL);
            int64_t mix_logit = 0;
            for (int s = 0; s < S; ++s) {
                uint64_t key = (sim_H0 & mask0[s]) ^ ((sim_H1 & mask1[s]) * 0x9E3779B97F4A7C15ULL) ^ ((sim_E0 & maskE[s]) * 0xBF58476D1CE4E5B9ULL);
                key ^= key >> 30; key *= 0xBF58476D1CE4E5B9ULL;
                uint32_t idx = (uint32_t)((key ^ (key >> 27)) & B_MASK);
                int32_t eff_w = w_glob[s] + w_reg[reg][s];
                if (eff_w < 0) eff_w = 0;
                mix_logit += (int64_t)eff_w * g_lut.logit[cells[((size_t)s << HASH_BITS) + idx]];
            }
            net_logit = (int32_t)(mix_logit >> 10);
        }
        uint8_t pred = (net_logit >= 0) ? 1U : 0U;
        uint8_t actual = (data[t >> 3] >> (t & 7)) & 1U;
        if (pred == actual) { ok256++; if (k < 64) ok64++; }
        if (pred == sim_prev_b) sim_run++; else { sim_prev_b = pred; sim_run = 1; }
        sim_H1 = (sim_H1 << 1) | (sim_H0 >> 63);
        sim_H0 = (sim_H0 << 1) | pred;
        sim_E0 = (sim_E0 << 1); // autonomous imagination assumes 0 surprise
    }
    res.rollout_64_acc = 100.0 * (double)ok64 / 64.0;
    res.rollout_256_acc = 100.0 * (double)ok256 / 256.0;

    int ord[S]; for (int s = 0; s < S; ++s) ord[s] = s;
    std::sort(ord, ord + S, [&](int a, int b_i){ return w_glob[a] > w_glob[b_i]; });
    for (int i = 0; i < 3; ++i) {
        int s = ord[i];
        res.top_m0[i] = mask0[s];
        res.top_m1[i] = mask1[s];
        res.top_w[i] = w_glob[s];
    }
    return res;
}

int main() {
    std::cout << "========================================================================================\n"
              << " KARYON-CORE EXP-1173b: ENDOGENOUS REGIME-GATED SWAR SUBSTRATE (V = 2, ZERO FLOAT STATE)\n"
              << " Strict Online bits/byte | Endogenous 64-Regime L1 Gating | Detached H=256 Rollout\n"
              << "========================================================================================\n\n";

    struct Job { std::string name; std::vector<uint8_t> data; };
    std::vector<Job> jobs;
    jobs.push_back({"LFSR-96 (100KB)", make_lfsr96(100000)});
    jobs.push_back({"Rule 110 CA (100KB)", make_rule110(100000)});
    auto kep = load_bytes("KEP.md");
    if (!kep.empty()) jobs.push_back({"Spec KEP.md", std::move(kep)});
    auto wiki = load_bytes("data/real_wikipedia_stream.txt");
    if (!wiki.empty()) jobs.push_back({"FULL Wikipedia (2.94MB)", std::move(wiki)});
    auto web = load_bytes("data/real_web_stream.txt");
    if (!web.empty()) jobs.push_back({"FULL Web Stream (4.19MB)", std::move(web)});

    std::vector<StreamResult> results(jobs.size());
    auto wall_t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < jobs.size(); ++i) {
        results[i] = eval_karyon_1173b(jobs[i].name, jobs[i].data);
    }

    double wall_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_t0).count();
    for (const auto& r : results) {
        double tail_ppl = std::pow(2.0, r.karyon_tail_bpb);
        std::cout << "[" << std::left << std::setw(24) << r.name << std::right << " | "
                  << std::setw(7) << r.bytes << " B]\n"
                  << "  Online BPB : Full = " << std::fixed << std::setprecision(4) << r.karyon_bpb
                  << " bpb | Tail(25%) = " << r.karyon_tail_bpb
                  << " bpb (Byte PPL = " << std::setprecision(2) << tail_ppl << ")\n"
                  << "  Bit Acc    : Full = " << r.karyon_acc << "% | Tail = " << r.karyon_tail_acc
                  << "% | Speed = " << r.karyon_mb_s << " MB/s | muts=" << r.muts
                  << " | f2_streak=" << r.f2_streak << "\n"
                  << "  Rollout    : H=+64 bits = " << r.rollout_64_acc
                  << "% | H=+256 bits = " << r.rollout_256_acc << "% (zero teacher feedback!)\n";
        if (r.f2_streak == 0) {
            std::cout << "  Top Evolved: ";
            for (int k = 0; k < 3; ++k) {
                std::cout << "[w=" << r.top_w[k] << " m0=0x" << std::hex << r.top_m0[k]
                          << " m1=0x" << r.top_m1[k] << std::dec << "] ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "Total Multi-Core Wall Time: " << std::fixed << std::setprecision(2) << wall_s << " s\n";
    return 0;
}
