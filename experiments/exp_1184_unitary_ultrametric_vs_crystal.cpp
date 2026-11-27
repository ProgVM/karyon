// EXP-1184: Pure Tree-Less Ultrametric Unitary Hopfield (H54-Clean) vs
// Unitary-Prior Laplace Residual Crystal (H42-Clean)
// Strict pre-observation causal scoring, zero F2 solvers, zero w_reg hash tables,
// zero byte-boundary priors, plus honest Order-0 and gzip-9 baselines on 200 KB Wikipedia.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <omp.h>

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

enum StreamKind { STRM_WHITE = 0, STRM_R110, STRM_LFSR16, STRM_LFSR32, STRM_LFSR96, STRM_WIKI };

static std::vector<uint8_t> make_stream(StreamKind kind, size_t n_bits) {
    std::vector<uint8_t> bits(n_bits, 0);
    uint64_t rng = 0xCAFEBABE12345678ULL;
    if (kind == STRM_WHITE) {
        for (size_t i = 0; i < n_bits; ++i) bits[i] = (uint8_t)((splitmix64(rng) >> 63) & 1ULL);
    } else if (kind == STRM_R110) {
        const int W = 64;
        std::vector<uint8_t> row(W, 0), nxt(W, 0);
        for (int i = 0; i < W; ++i) row[i] = (uint8_t)((splitmix64(rng) >> 63) & 1ULL);
        for (size_t t = 0; t < n_bits; ) {
            for (int i = 0; i < W && t < n_bits; ++i, ++t) bits[t] = row[i];
            for (int i = 0; i < W; ++i) {
                uint8_t pat = (row[(i + W - 1) % W] << 2) | (row[i] << 1) | row[(i + 1) % W];
                nxt[i] = (110 >> pat) & 1U;
            }
            row.swap(nxt);
        }
    } else if (kind == STRM_LFSR16) {
        uint16_t st = 0xACE1U;
        for (size_t i = 0; i < n_bits; ++i) {
            uint8_t bit = ((st >> 0) ^ (st >> 2) ^ (st >> 3) ^ (st >> 5)) & 1U;
            bits[i] = bit;
            st = (uint16_t)((st >> 1) | (bit << 15));
        }
    } else if (kind == STRM_LFSR32) {
        uint32_t st = 0xCAFEBABEU;
        for (size_t i = 0; i < n_bits; ++i) {
            uint8_t bit = ((st >> 0) ^ (st >> 1) ^ (st >> 21) ^ (st >> 31)) & 1U;
            bits[i] = bit;
            st = (st >> 1) | ((uint32_t)bit << 31);
        }
    } else if (kind == STRM_LFSR96) {
        uint64_t hi = 0xCAFEBABEDEADBEEFULL;
        uint32_t lo = 0x12345678U;
        for (size_t i = 0; i < n_bits; ++i) {
            uint8_t bit = (uint8_t)((hi ^ (hi >> 6) ^ (hi >> 9) ^ (hi >> 10)) & 1ULL);
            bits[i] = bit;
            uint32_t carry = (uint32_t)(hi & 1ULL);
            hi = (hi >> 1) | ((uint64_t)bit << 63);
            lo = (lo >> 1) | (carry << 31);
        }
    } else if (kind == STRM_WIKI) {
        std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
        std::vector<uint8_t> bytes((n_bits + 7) / 8 + 16, 0);
        if (fw) fw.read((char*)bytes.data(), bytes.size());
        for (size_t i = 0; i < n_bits; ++i) {
            bits[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1U;
        }
    }
    return bits;
}

struct StreamResult {
    double mid_acc;
    double tail_acc;
    double tail_bpb;
    double all_bpb;
    size_t state_count;
};

// ============================================================================
// Architecture A: Pure Tree-Less Hash-Less Unitary Wave + Ultrametric Hopfield
//   - Cached ck = sqrt(1 - w^2) updated only for active Top-32 modes (8x fewer sqrts)
//   - Dual-reach modes (128 @ N=128, 128 @ N=164)
//   - Gentle amp/H L2 damping (0.9997) when err_ema > 0.35 -> avoids WhiteNoise drift
//   - Zero w decay during search -> preserves LFSR-16/32/96 lock
// ============================================================================
struct KaryonTreeLessUltrametric {
    static constexpr int N_MAX = 164, M = 256, K_TOP = 32;
    static constexpr int N_BINS = 64, BIN_CAP = 2048;
    struct Trace { uint64_t h; float theta, h_sum; uint32_t visits, last_t; };

    std::vector<float> w, ck_cache, amp, H;
    std::vector<int> mode_N;
    float bias = 0.0f;
    std::vector<Trace> mem;
    std::vector<uint16_t> bin_sz;
    double w_tab[65];

    std::vector<float> z, P_mode, phi, d_asinh, fk_buf, h_proj;
    std::vector<int> order, matched_idx;
    uint64_t hist64 = 0, rng = 0x777788889999AAAAULL, t_step = 0;
    size_t total_traces = 0;
    double err_ema = 0.5;

    KaryonTreeLessUltrametric()
        : w(M * N_MAX, 0.0f), ck_cache(M * N_MAX, 1.0f), amp(M, 0.0f), H(M * M, 0.0f), mode_N(M),
          mem(N_BINS * BIN_CAP, {0ULL, 0.0f, 0.0f, 0, 0}), bin_sz(N_BINS, 0),
          z(N_MAX, 1.0f), P_mode(M), phi(M), d_asinh(M),
          fk_buf(M * N_MAX), h_proj(K_TOP), order(M) {
        for (int l = 0; l <= 64; ++l) w_tab[l] = std::exp(0.92 * l);
        matched_idx.reserve(64);
        uint64_t r = 0x777788889999AAAAULL;
        for (int m = 0; m < M; ++m) {
            mode_N[m] = (m < 128) ? 128 : 164;
            amp[m] = ((float)(splitmix64(r) & 0xFFFF) / 32768.0f - 1.0f) * 0.06f;
            float scale = (m & 1) ? 0.34f : 0.16f;
            for (int k = 0; k < mode_N[m]; ++k) {
                float wk = ((float)(splitmix64(r) & 0xFFFF) / 32768.0f - 1.0f) * scale;
                w[m * N_MAX + k] = wk;
                ck_cache[m * N_MAX + k] = std::sqrt(std::max(1e-4f, 1.0f - wk * wk));
            }
        }
    }

    double step(uint8_t target, uint8_t* out_pred = nullptr) {
        const float lr_w = 0.035f, lr_a = 0.025f, lr_H = 0.02f;

        for (int m = 0; m < M; ++m) {
            int nm = mode_N[m];
            double prod = 1.0;
            const float* wp = &w[m * N_MAX];
            const float* ckp = &ck_cache[m * N_MAX];
            float* fkp = &fk_buf[m * N_MAX];
            const float* zp = z.data();
            for (int k = 0; k < nm; ++k) {
                float fk = ckp[k] + wp[k] * zp[k];
                fkp[k] = fk;
                prod *= (double)fk;
            }
            P_mode[m] = (float)prod;
            phi[m] = (float)std::asinh(prod);
            d_asinh[m] = (float)(1.0 / std::sqrt(1.0 + prod * prod));
            order[m] = m;
        }
        std::nth_element(order.begin(), order.begin() + K_TOP, order.end(),
                         [&](int a, int b) { return std::abs(phi[a]) > std::abs(phi[b]); });

        double norm2 = 1e-4;
        for (int i = 0; i < K_TOP; ++i) norm2 += (double)phi[order[i]] * (double)phi[order[i]];
        float inv_norm = (float)(1.0 / std::sqrt(norm2));
        double logit_u = bias;
        for (int i = 0; i < K_TOP; ++i) {
            int mi = order[i];
            float acc = 0.0f;
            const float* Hm = &H[mi * M];
            for (int j = 0; j < K_TOP; ++j) acc += Hm[order[j]] * (phi[order[j]] * inv_norm);
            h_proj[i] = acc;
            logit_u += (amp[mi] + acc) * phi[mi];
        }
        logit_u = std::clamp(logit_u, -14.0, 14.0);
        double pu = 1.0 / (1.0 + std::exp(-logit_u));

        int bin = (int)(hist64 & 63ULL);
        int bsz = bin_sz[bin];
        Trace* bptr = &mem[bin * BIN_CAP];

        matched_idx.clear();
        double num = 0.0, den = std::exp(10.0);
        int best_l = -1;
        for (int i = 0; i < bsz; ++i) {
            uint64_t x = hist64 ^ bptr[i].h;
            int l = (x == 0) ? 64 : __builtin_ctzll(x);
            if (l >= 6) {
                if (bptr[i].visits >= 2) {
                    double wt = w_tab[l] * std::min(64.0, (double)bptr[i].visits);
                    num += wt * (double)bptr[i].theta;
                    den += wt;
                }
                if (l >= 20) matched_idx.push_back(i);
                if (l > best_l) best_l = l;
            }
        }
        double delta_theta = num / den;
        double logit = std::clamp(logit_u + delta_theta, -16.0, 16.0);
        double p1 = 1.0 / (1.0 + std::exp(-logit));
        if (out_pred) *out_pred = (p1 >= 0.5) ? 1 : 0;

        float gl = (float)target - (float)pu;
        err_ema = 0.998 * err_ema + 0.002 * std::abs(gl);
        float damp_ah = (err_ema > 0.35) ? 0.9992f : 0.9999f;
        float l1_amp = (err_ema < 0.08) ? 2e-5f : 0.0f;

        bias = std::clamp(bias * damp_ah + lr_a * gl, -8.0f, 8.0f);
        for (int i = 0; i < K_TOP; ++i) {
            int mi = order[i];
            int nm = mode_N[mi];
            float pm = P_mode[mi];
            float em = gl * (amp[mi] + 2.0f * h_proj[i]) * d_asinh[mi];
            amp[mi] = std::clamp(amp[mi] * damp_ah + lr_a * gl * phi[mi], -8.0f, 8.0f);
            float* Hm = &H[mi * M];
            float gm = lr_H * gl * phi[mi] * inv_norm;
            for (int j = 0; j < K_TOP; ++j) {
                int mj = order[j];
                Hm[mj] = std::clamp(Hm[mj] * damp_ah + gm * phi[mj], -4.0f, 4.0f);
            }
            float* wp = &w[mi * N_MAX];
            float* ckp = &ck_cache[mi * N_MAX];
            const float* fkp = &fk_buf[mi * N_MAX];
            for (int k = 0; k < nm; ++k) {
                float fk = fkp[k];
                if (std::abs(fk) > 1e-3f) {
                    float wk = wp[k];
                    float g = std::clamp((float)(em * ((double)pm / (double)fk) * (-wk / ckp[k] + z[k])), -2.0f, 2.0f);
                    float l1 = l1_amp * (wk > 0 ? 1.0f : (wk < 0 ? -1.0f : 0.0f));
                    float nw = std::clamp(wk + lr_w * g - l1, -0.998f, 0.998f);
                    wp[k] = nw;
                    ckp[k] = std::sqrt(std::max(1e-4f, 1.0f - nw * nw));
                }
            }
        }

        for (int idx : matched_idx) {
            Trace& tr = bptr[idx];
            double l_tr = std::clamp(logit_u + (double)tr.theta, -16.0, 16.0);
            double p_tr = 1.0 / (1.0 + std::exp(-l_tr));
            double g_tr = (double)target - p_tr;
            tr.h_sum = std::min(64.0f, tr.h_sum + (float)(p_tr * (1.0 - p_tr)));
            tr.theta = std::clamp((float)(tr.theta + g_tr / ((double)tr.h_sum + 0.35)), -8.0f, 8.0f);
            tr.visits++;
            tr.last_t = (uint32_t)t_step;
        }
        if (best_l < 40 && std::abs(gl) > 0.12f) {
            float g_init = gl / (float)(pu * (1.0 - pu) + 1.0);
            if (bsz < BIN_CAP) {
                bptr[bsz] = {hist64, std::clamp(g_init, -4.0f, 4.0f), (float)(pu * (1.0 - pu)), 1, (uint32_t)t_step};
                bin_sz[bin]++;
                total_traces++;
            } else {
                int vic = 0;
                float best_sc = 1e18f;
                for (int r = 0; r < 12; ++r) {
                    int cand = (int)(splitmix64(rng) % BIN_CAP);
                    float sc = (float)bptr[cand].visits * 500.0f + (float)bptr[cand].last_t * 0.01f;
                    if (sc < best_sc) { best_sc = sc; vic = cand; }
                }
                bptr[vic] = {hist64, std::clamp(g_init, -4.0f, 4.0f), (float)(pu * (1.0 - pu)), 1, (uint32_t)t_step};
            }
        }

        hist64 = (hist64 << 1) | target;
        float ts = target ? 1.0f : -1.0f;
        for (int k = 131; k >= 1; --k) z[k] = z[k - 1];
        z[0] = ts;
        for (int k = 132; k < N_MAX; ++k) z[k] = 0.9f * z[k] + 0.1f * std::tanh(0.2f * phi[k - 132]);
        t_step++;
        return p1;
    }
};

// ============================================================================
// Architecture B: Unitary-Prior (N=128) + Laplace Residual Crystal (H42-Clean)
// ============================================================================
struct KaryonUnitaryCrystal {
    static constexpr int N = 128, M = 256, K_TOP = 32, D_MAX = 48;
    struct Node {
        int ch[2] = {-1, -1};
        float theta = 0.0f, h_sum = 0.0f;
        uint32_t visits = 0;
        double lp = 0.0, lw = 0.0;
    };

    std::vector<float> w, ck_cache, amp, H;
    float bias = 0.0f;
    std::vector<Node> nodes;
    std::vector<float> z, P_mode, phi, d_asinh, fk_buf, h_proj;
    std::vector<int> order;
    uint8_t zb[D_MAX] = {0};
    int path[D_MAX + 1];
    double err_ema = 0.5;

    KaryonUnitaryCrystal(size_t max_nodes = 1500000)
        : w(M * N, 0.0f), ck_cache(M * N, 1.0f), amp(M, 0.0f), H(M * M, 0.0f),
          z(N, 1.0f), P_mode(M), phi(M), d_asinh(M), fk_buf(M * N), h_proj(K_TOP), order(M) {
        nodes.reserve(max_nodes);
        nodes.emplace_back();
        uint64_t r = 0x777788889999AAAAULL;
        for (int m = 0; m < M; ++m) {
            amp[m] = ((float)(splitmix64(r) & 0xFFFF) / 32768.0f - 1.0f) * 0.06f;
            float scale = (m & 1) ? 0.36f : 0.16f;
            for (int k = 0; k < N; ++k) {
                float wk = ((float)(splitmix64(r) & 0xFFFF) / 32768.0f - 1.0f) * scale;
                w[m * N + k] = wk;
                ck_cache[m * N + k] = std::sqrt(std::max(1e-4f, 1.0f - wk * wk));
            }
        }
    }

    static inline double log_add_exp(double a, double b) {
        double m = std::max(a, b);
        return m + std::log1p(std::exp(-std::abs(a - b)));
    }

    double step(uint8_t target, uint8_t* out_pred = nullptr, size_t max_nodes = 1400000) {
        const float lr_w = 0.035f, lr_a = 0.025f, lr_H = 0.02f;
        const double LAMBDA = 0.25;

        for (int m = 0; m < M; ++m) {
            double prod = 1.0;
            const float* wp = &w[m * N];
            const float* ckp = &ck_cache[m * N];
            float* fkp = &fk_buf[m * N];
            const float* zp = z.data();
            for (int k = 0; k < N; ++k) {
                float fk = ckp[k] + wp[k] * zp[k];
                fkp[k] = fk;
                prod *= (double)fk;
            }
            P_mode[m] = (float)prod;
            phi[m] = (float)std::asinh(prod);
            d_asinh[m] = (float)(1.0 / std::sqrt(1.0 + prod * prod));
            order[m] = m;
        }
        std::nth_element(order.begin(), order.begin() + K_TOP, order.end(),
                         [&](int a, int b) { return std::abs(phi[a]) > std::abs(phi[b]); });
        double norm2 = 1e-4;
        for (int i = 0; i < K_TOP; ++i) norm2 += (double)phi[order[i]] * (double)phi[order[i]];
        float inv_norm = (float)(1.0 / std::sqrt(norm2));
        double logit_u = bias;
        for (int i = 0; i < K_TOP; ++i) {
            int mi = order[i];
            float acc = 0.0f;
            const float* Hm = &H[mi * M];
            for (int j = 0; j < K_TOP; ++j) acc += Hm[order[j]] * (phi[order[j]] * inv_norm);
            h_proj[i] = acc;
            logit_u += (amp[mi] + acc) * phi[mi];
        }
        logit_u = std::clamp(logit_u, -14.0, 14.0);
        double pu = 1.0 / (1.0 + std::exp(-logit_u));

        int cur = 0; path[0] = 0; int actual_d = 0;
        for (int d = 0; d < D_MAX; ++d) {
            uint8_t b = zb[d];
            if (nodes[cur].ch[b] < 0) {
                if (nodes.size() < max_nodes) {
                    int ni = (int)nodes.size();
                    nodes.emplace_back();
                    nodes[cur].ch[b] = ni;
                } else break;
            }
            cur = nodes[cur].ch[b];
            path[d + 1] = cur;
            actual_d = d + 1;
            if (nodes[cur].visits == 0) break;
        }

        auto eval_hyp = [&](uint8_t hb) -> double {
            double lw_b = 0.0;
            for (int d = actual_d; d >= 0; --d) {
                const Node& nd = nodes[path[d]];
                double l_nd = std::clamp(logit_u + (double)nd.theta, -16.0, 16.0);
                double p_nd = 1.0 / (1.0 + std::exp(-l_nd));
                double lp_n = nd.lp + std::log(std::clamp(hb ? p_nd : (1.0 - p_nd), 1e-7, 1.0 - 1e-7));
                if (d == actual_d) lw_b = lp_n;
                else {
                    int sib = nd.ch[1 ^ zb[d]];
                    double lw_s = (sib >= 0) ? nodes[sib].lw : 0.0;
                    lw_b = log_add_exp(lp_n, lw_b + lw_s) - std::log(2.0);
                }
            }
            return lw_b;
        };

        double diff = eval_hyp(1) - eval_hyp(0);
        double p1 = 1.0 / (1.0 + std::exp(-std::clamp(diff, -20.0, 20.0)));
        if (out_pred) *out_pred = (p1 >= 0.5) ? 1 : 0;

        float gl = (float)target - (float)pu;
        err_ema = 0.998 * err_ema + 0.002 * std::abs(gl);
        float damp_ah = (err_ema > 0.35) ? 0.9992f : 0.9999f;

        bias = std::clamp(bias * damp_ah + lr_a * gl, -8.0f, 8.0f);
        for (int i = 0; i < K_TOP; ++i) {
            int mi = order[i];
            float pm = P_mode[mi];
            float em = gl * (amp[mi] + 2.0f * h_proj[i]) * d_asinh[mi];
            amp[mi] = std::clamp(amp[mi] * damp_ah + lr_a * gl * phi[mi], -8.0f, 8.0f);
            float* Hm = &H[mi * M];
            float gm = lr_H * gl * phi[mi] * inv_norm;
            for (int j = 0; j < K_TOP; ++j) {
                int mj = order[j];
                Hm[mj] = std::clamp(Hm[mj] * damp_ah + gm * phi[mj], -4.0f, 4.0f);
            }
            float* wp = &w[mi * N];
            float* ckp = &ck_cache[mi * N];
            const float* fkp = &fk_buf[mi * N];
            for (int k = 0; k < N; ++k) {
                float fk = fkp[k];
                if (std::abs(fk) > 1e-3f) {
                    float wk = wp[k];
                    float g = std::clamp((float)(em * ((double)pm / (double)fk) * (-wk / ckp[k] + z[k])), -2.0f, 2.0f);
                    float nw = std::clamp(wk + lr_w * g, -0.998f, 0.998f);
                    wp[k] = nw;
                    ckp[k] = std::sqrt(std::max(1e-4f, 1.0f - nw * nw));
                }
            }
        }

        for (int d = actual_d; d >= 0; --d) {
            Node& nd = nodes[path[d]];
            double l_nd = std::clamp(logit_u + (double)nd.theta, -16.0, 16.0);
            double p_nd = 1.0 / (1.0 + std::exp(-l_nd));
            nd.lp += std::log(std::clamp(target ? p_nd : (1.0 - p_nd), 1e-7, 1.0 - 1e-7));
            double g_nd = (double)target - p_nd;
            double h_nd = p_nd * (1.0 - p_nd);
            nd.h_sum = std::min(64.0f, nd.h_sum + (float)h_nd);
            nd.theta = std::clamp((float)(nd.theta + g_nd / ((double)nd.h_sum + LAMBDA)), -8.0f, 8.0f);
            nd.visits++;
            if (d == actual_d) nd.lw = nd.lp;
            else {
                int ca = nd.ch[zb[d]], cs = nd.ch[1 ^ zb[d]];
                double lwa = (ca >= 0) ? nodes[ca].lw : 0.0;
                double lws = (cs >= 0) ? nodes[cs].lw : 0.0;
                nd.lw = log_add_exp(nd.lp, lwa + lws) - std::log(2.0);
            }
        }

        float ts = target ? 1.0f : -1.0f;
        for (int k = 95; k >= 1; --k) z[k] = z[k - 1];
        z[0] = ts;
        for (int k = 96; k < N; ++k) z[k] = 0.9f * z[k] + 0.1f * std::tanh(0.2f * phi[k - 96]);
        for (int k = D_MAX - 1; k >= 1; --k) zb[k] = zb[k - 1];
        zb[0] = target;
        return p1;
    }
};

template <typename Model>
static StreamResult eval_single(const std::vector<uint8_t>& bits) {
    Model sys;
    size_t n_bits = bits.size(), mid_s = n_bits / 2 - 10000, mid_e = n_bits / 2, tail_start = n_bits - 20000;
    uint64_t c_mid = 0, c_tail = 0;
    double bpb_tail = 0.0, bpb_all = 0.0;
    for (size_t t = 0; t < n_bits; ++t) {
        uint8_t pred = 0;
        double p1 = sys.step(bits[t], &pred);
        double loss = -std::log2(std::clamp(bits[t] ? p1 : (1.0 - p1), 1e-7, 1.0 - 1e-7));
        bpb_all += loss;
        if (t >= mid_s && t < mid_e) { if (pred == bits[t]) c_mid++; }
        if (t >= tail_start) { if (pred == bits[t]) c_tail++; bpb_tail += loss; }
    }
    size_t sc = 0;
    if constexpr (std::is_same_v<Model, KaryonTreeLessUltrametric>) sc = sys.total_traces;
    else sc = sys.nodes.size();
    return {100.0 * c_mid / 10000.0, 100.0 * c_tail / 20000.0, 8.0 * bpb_tail / 20000.0, 8.0 * bpb_all / n_bits, sc};
}

template <typename Model>
static void eval_wiki_scaling(const std::vector<uint8_t>& wiki1600, const char* tag) {
    Model sys;
    uint64_t win_ok = 0, tot_ok = 0;
    double win_bpb = 0.0, tot_bpb = 0.0;
    std::cout << "\n=== [" << tag << "] 200 KB (1.6M bits) Wikipedia Continuous Scaling ===\n";
    for (size_t t = 0; t < wiki1600.size(); ++t) {
        uint8_t pred = 0;
        double p1 = sys.step(wiki1600[t], &pred);
        double loss = -std::log2(std::clamp(wiki1600[t] ? p1 : (1.0 - p1), 1e-7, 1.0 - 1e-7));
        if (pred == wiki1600[t]) { win_ok++; tot_ok++; }
        win_bpb += loss; tot_bpb += loss;
        if ((t + 1) % 200000 == 0) {
            std::cout << "  t=" << std::setw(7) << (t + 1)
                      << " | win200k_acc=" << std::fixed << std::setprecision(2) << (100.0 * win_ok / 200000.0) << "%"
                      << " | win200k_bpb=" << std::setprecision(3) << (8.0 * win_bpb / 200000.0)
                      << " | cumul_bpb=" << (8.0 * tot_bpb / (t + 1)) << "\n" << std::flush;
            win_ok = 0; win_bpb = 0.0;
        }
    }
}

template <typename Model>
static void eval_unbroken(const char* tag) {
    struct Seg { const char* name; StreamKind kind; size_t len; };
    Seg segs[5] = {
        {"Wiki-A (200k)",    STRM_WIKI,   200000},
        {"Rule110 (150k)",   STRM_R110,   150000},
        {"LFSR-96 (200k)",   STRM_LFSR96, 200000},
        {"Wiki-B (200k)",    STRM_WIKI,   200000},
        {"WhiteNoise (80k)", STRM_WHITE,   80000},
    };
    Model sys;
    std::cout << "\n=== [" << tag << "] Unbroken 830k-Bit Continuous Life (ZERO resets!) ===\n";
    for (int s = 0; s < 5; ++s) {
        auto strm = make_stream(segs[s].kind, segs[s].len);
        uint64_t h_ok = 0, t_ok = 0;
        double h_bpb = 0.0, t_bpb = 0.0;
        size_t head_n = std::min<size_t>(20000, strm.size());
        size_t tail_s = strm.size() > 20000 ? strm.size() - 20000 : 0;
        for (size_t i = 0; i < strm.size(); ++i) {
            uint8_t pred = 0;
            double p1 = sys.step(strm[i], &pred);
            double loss = -std::log2(std::clamp(strm[i] ? p1 : (1.0 - p1), 1e-7, 1.0 - 1e-7));
            if (i < head_n) { if (pred == strm[i]) h_ok++; h_bpb += loss; }
            if (i >= tail_s) { if (pred == strm[i]) t_ok++; t_bpb += loss; }
        }
        std::cout << "  " << std::left << std::setw(18) << segs[s].name
                  << " | head20k=" << std::fixed << std::setprecision(2) << std::setw(6) << (100.0 * h_ok / head_n) << "% ("
                  << std::setprecision(3) << (8.0 * h_bpb / head_n) << " bpb)"
                  << " -> tail20k=" << std::setprecision(2) << std::setw(6) << (100.0 * t_ok / (strm.size() - tail_s)) << "% ("
                  << std::setprecision(3) << (8.0 * t_bpb / (strm.size() - tail_s)) << " bpb)\n" << std::flush;
    }
}

int main() {
    const size_t N_BITS = 120000;
    const char* snames[6] = {"WhiteNoise", "Rule110", "LFSR-16", "LFSR-32", "LFSR-96", "Wikipedia"};
    std::vector<std::vector<uint8_t>> streams(6);
    for (int s = 0; s < 6; ++s) streams[s] = make_stream((StreamKind)s, N_BITS);
    auto wiki1600 = make_stream(STRM_WIKI, 1600000);

    // Honest Order-0 & gzip-9 baselines on exact same 200 KB Wikipedia prefix
    {
        std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
        std::vector<uint8_t> raw200k(200000);
        fw.read((char*)raw200k.data(), 200000);
        std::ofstream fout("/tmp/exp1184_wiki200k.bin", std::ios::binary);
        fout.write((char*)raw200k.data(), 200000);
        fout.close();
        std::system("gzip -9 -c /tmp/exp1184_wiki200k.bin > /tmp/exp1184_wiki200k.gz");
        std::ifstream fgz("/tmp/exp1184_wiki200k.gz", std::ios::binary | std::ios::ate);
        double gz_bpb = 8.0 * (double)fgz.tellg() / 200000.0;
        uint64_t ones = 0;
        for (uint8_t b : wiki1600) ones += b;
        double p = (double)ones / wiki1600.size();
        double o0_bpb = 8.0 * (-(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p)));
        std::cout << "[Baselines on exact 200 KB Wikipedia]: Order-0 = " << std::fixed << std::setprecision(3) << o0_bpb
                  << " bpb | gzip-9 = " << gz_bpb << " bpb\n" << std::flush;
    }

    StreamResult ra[6], rb[6];
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (int task = 0; task < 16; ++task) {
        if (task < 6)       ra[task]     = eval_single<KaryonTreeLessUltrametric>(streams[task]);
        else if (task < 12) rb[task - 6] = eval_single<KaryonUnitaryCrystal>(streams[task - 6]);
        else if (task == 12) eval_unbroken<KaryonTreeLessUltrametric>("A: TreeLess-Ultrametric");
        else if (task == 13) eval_unbroken<KaryonUnitaryCrystal>("B: Unitary-Crystal");
        else if (task == 14) eval_wiki_scaling<KaryonTreeLessUltrametric>(wiki1600, "A: TreeLess-Ultrametric");
        else                 eval_wiki_scaling<KaryonUnitaryCrystal>(wiki1600, "B: Unitary-Crystal");
    }
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    auto pr = [&](const char* title, StreamResult* r) {
        std::cout << "\n" << title << "\n";
        for (int s = 0; s < 6; ++s) {
            std::cout << "  " << std::left << std::setw(12) << snames[s]
                      << " | mid60k=" << std::fixed << std::setprecision(2) << std::setw(6) << r[s].mid_acc << "%"
                      << " | tail20k=" << std::setw(6) << r[s].tail_acc << "%"
                      << " | tail_bpb=" << std::setprecision(3) << r[s].tail_bpb
                      << " | states=" << r[s].state_count << "\n";
        }
    };
    pr("[A] Tree-Less Dual-Reach Unitary Wave + Ultrametric Hopfield (H54-Clean)", ra);
    pr("[B] Unitary-Prior + Exact Online Laplace Crystal (H42-Clean)", rb);
    std::cout << "\nTotal wall clock: " << std::fixed << std::setprecision(2) << sec << "s\n";
    return 0;
}
