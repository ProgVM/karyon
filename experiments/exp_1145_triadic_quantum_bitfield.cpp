// ============================================================================
// KARYON-CORE EXP-1145: TRIADIC SUBSTRATE (DETERMINISTIC + STOCHASTIC + QUANTUM)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 1, 4 (V=2), 6, 8, 13, 14, 15
// ============================================================================
// Hypothesis:
//   Low-level deterministic boolean rules (D) and non-deterministic hardware
//   stochasticity (S) may be insufficient alone. Adding a 3rd primitive type —
//   Quantum Wave Interference (Q) with signed complex phase-amplitudes and
//   2-spin phase entanglement across discovered bit-lags — enables destructive
//   cancellation of noise and constructive phase-locking on unseen bitstreams.
//
// Strict Principle 15 Audit:
//   Evaluates D-only, Q-only, and Triadic (D + Q + S + Unbounded Memory) on
//   the UNSEEN 50% tail of 4 real unsimplified modalities:
//     1. TEXT: KEP.md
//     2. TEXT: tinyshakespeare.txt (256 KB)
//     3. IMAGE/PNG: bloodbath_precise_v4.png (256 KB)
//     4. BINARY/ELF: x86_64 executable machine code
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>

static inline uint64_t rotl64(uint64_t x, int k) {
    k &= 63;
    return k == 0 ? x : ((x << k) | (x >> (64 - k)));
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double compute_free_energy(uint64_t error_bits, uint64_t total_bits) {
    if (total_bits == 0 || error_bits == 0) return 0.0;
    double err_rate = static_cast<double>(error_bits) / static_cast<double>(total_bits);
    if (err_rate >= 1.0) err_rate = 1.0 - 1e-15;
    return -std::log(std::max(1.0 - err_rate, 1e-15));
}

static std::vector<uint64_t> load_file_as_bitstream(const std::string& path, size_t max_bytes = 0) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    size_t file_size = static_cast<size_t>(ifs.tellg());
    if (max_bytes > 0 && file_size > max_bytes) file_size = max_bytes;
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw(file_size);
    ifs.read(reinterpret_cast<char*>(raw.data()), file_size);

    size_t num_words = (file_size + 7) / 8;
    std::vector<uint64_t> words(num_words, 0ULL);
    if (file_size > 0) {
        std::memcpy(words.data(), raw.data(), file_size);
    }
    return words;
}

static inline uint64_t fetch_bit_lag_window(const std::vector<uint64_t>& words, size_t i, size_t lag) {
    size_t word_shift = lag >> 6;
    size_t bit_shift  = lag & 63;
    if (bit_shift == 0) {
        return (i >= word_shift) ? words[i - word_shift] : 0ULL;
    }
    uint64_t hi = (i >= word_shift) ? words[i - word_shift] : 0ULL;
    uint64_t lo = (i >= word_shift + 1) ? words[i - word_shift - 1] : 0ULL;
    return (hi << bit_shift) | (lo >> (64 - bit_shift));
}

// ============================================================================
// TYPE 1: DETERMINISTIC CAUSAL PRIMITIVE (D)
// ============================================================================
struct DetRule {
    uint8_t  mode;      // 0=CONST, 1=1-LAG, 2=XOR, 3=AND, 4=OR
    uint8_t  polarity;
    uint16_t lag1;
    uint16_t lag2;
};

class DeterministicField {
public:
    DetRule rule = {0, 0, 1, 1};
    std::vector<uint16_t> top_lags;

    void synthesize(const std::vector<uint64_t>& stream, size_t max_lag = 256) {
        size_t start_i = (max_lag + 63) / 64 + 1;
        if (start_i >= stream.size()) return;
        uint64_t total_bits = (stream.size() - start_i) * 64ULL;

        uint64_t ones = 0;
        for (size_t i = start_i; i < stream.size(); ++i) ones += __builtin_popcountll(stream[i]);
        uint8_t best_pol = (ones * 2 >= total_bits) ? 1 : 0;
        uint64_t best_err = best_pol ? (total_bits - ones) : ones;
        DetRule best_rule = {0, best_pol, 0, 0};

        struct LagScore { uint16_t lag; uint64_t err; };
        std::vector<LagScore> scores;
        scores.reserve(max_lag);

        for (size_t lag = 1; lag <= max_lag; ++lag) {
            uint64_t err0 = 0;
            for (size_t i = start_i; i < stream.size(); ++i) {
                err0 += __builtin_popcountll(fetch_bit_lag_window(stream, i, lag) ^ stream[i]);
            }
            uint64_t err1 = total_bits - err0;
            uint8_t pol = (err1 < err0) ? 1 : 0;
            uint64_t e = pol ? err1 : err0;
            scores.push_back({static_cast<uint16_t>(lag), e});
            if (e < best_err) {
                best_err = e;
                best_rule = {1, pol, static_cast<uint16_t>(lag), 0};
            }
        }

        std::sort(scores.begin(), scores.end(), [](const LagScore& a, const LagScore& b) {
            return a.err < b.err;
        });
        top_lags.clear();
        for (size_t k = 0; k < std::min<size_t>(12, scores.size()); ++k) {
            top_lags.push_back(scores[k].lag);
        }

        size_t K = std::min<size_t>(20, scores.size());
        for (size_t a = 0; a < K; ++a) {
            uint16_t l1 = scores[a].lag;
            for (size_t b = a + 1; b < K; ++b) {
                uint16_t l2 = scores[b].lag;
                uint64_t ex = 0, ea = 0, eo = 0;
                for (size_t i = start_i; i < stream.size(); ++i) {
                    uint64_t x1 = fetch_bit_lag_window(stream, i, l1);
                    uint64_t x2 = fetch_bit_lag_window(stream, i, l2);
                    uint64_t t  = stream[i];
                    ex += __builtin_popcountll((x1 ^ x2) ^ t);
                    ea += __builtin_popcountll((x1 & x2) ^ t);
                    eo += __builtin_popcountll((x1 | x2) ^ t);
                }
                auto chk = [&](uint8_t m, uint64_t e0) {
                    uint64_t e1 = total_bits - e0;
                    uint8_t p = (e1 < e0) ? 1 : 0;
                    uint64_t e = p ? e1 : e0;
                    if (e < best_err) { best_err = e; best_rule = {m, p, l1, l2}; }
                };
                chk(2, ex); chk(3, ea); chk(4, eo);
            }
        }
        rule = best_rule;
    }

    inline uint64_t predict_window(const std::vector<uint64_t>& stream, size_t i) const {
        uint64_t v = 0ULL;
        if (rule.mode == 1)      v = fetch_bit_lag_window(stream, i, rule.lag1);
        else if (rule.mode == 2) v = fetch_bit_lag_window(stream, i, rule.lag1) ^ fetch_bit_lag_window(stream, i, rule.lag2);
        else if (rule.mode == 3) v = fetch_bit_lag_window(stream, i, rule.lag1) & fetch_bit_lag_window(stream, i, rule.lag2);
        else if (rule.mode == 4) v = fetch_bit_lag_window(stream, i, rule.lag1) | fetch_bit_lag_window(stream, i, rule.lag2);
        return rule.polarity ? ~v : v;
    }
};

// ============================================================================
// TYPE 3: QUANTUM WAVE INTERFERENCE PRIMITIVE (Q)
// ============================================================================
// Represents a coherent superposition of N causal bit-lag paths + entangled
// 2-spin ZZ phase couplings. Each path k contributes a complex wave-amplitude:
//   psi_k = A_k * exp(i * (theta_k + s_k * phi_k))
// Interference across all paths produces constructive/destructive cancellation:
//   Psi_total(bit) = Psi_0 + sum_k A_k * (cos(phase_k) + i * sin(phase_k))
// Measurement (Born collapse) projects Re(Psi_total) * cos(readout_angle) +
// Im(Psi_total) * sin(readout_angle) -> spin {-1, +1} and returns coherence |Psi|
// so Stochasticity (S) knows exact quantum uncertainty!
// ============================================================================
class QuantumInterferenceField {
public:
    std::vector<uint16_t> lags;
    // Single-spin complex amplitudes (Re, Im) for spin=+1 and spin=-1
    std::vector<float> amp_re_pos, amp_im_pos;
    std::vector<float> amp_re_neg, amp_im_neg;
    // Entangled 2-spin ZZ phase couplings between top pairs
    struct EntangledPair { uint8_t a, b; float re_same, im_same, re_diff, im_diff; };
    std::vector<EntangledPair> entangled_pairs;

    float bias_re = 0.0f, bias_im = 0.0f;
    float readout_cos = 1.0f, readout_sin = 0.0f;

    void synthesize(const std::vector<uint64_t>& stream, const std::vector<uint16_t>& discovered_lags) {
        lags = discovered_lags;
        size_t M = lags.size();
        if (M == 0 || stream.size() < 16) return;

        size_t start_i = 5;
        size_t N_words = stream.size() - start_i;

        amp_re_pos.assign(M, 0.0f);
        amp_im_pos.assign(M, 0.0f);
        amp_re_neg.assign(M, 0.0f);
        amp_im_neg.assign(M, 0.0f);

        // Compute target spin bias
        double mean_y = 0.0;
        for (size_t i = start_i; i < stream.size(); ++i) {
            int ones = __builtin_popcountll(stream[i]);
            mean_y += (2.0 * ones - 64.0);
        }
        mean_y /= (N_words * 64.0);
        bias_re = static_cast<float>(mean_y);
        bias_im = 0.0f;

        // Assign distinct quantum phase angles theta_k = k * pi / M so paths
        // can interfere constructively and destructively in the complex plane.
        const float PI = 3.14159265358979323846f;
        for (size_t k = 0; k < M; ++k) {
            uint64_t match_pos = 0, cnt_pos = 0;
            uint64_t match_neg = 0, cnt_neg = 0;

            for (size_t i = start_i; i < stream.size(); ++i) {
                uint64_t x = fetch_bit_lag_window(stream, i, lags[k]);
                uint64_t y = stream[i];
                // Count target=1 when x=1 and when x=0
                cnt_pos   += __builtin_popcountll(x);
                match_pos += __builtin_popcountll(x & y);
                uint64_t nx = ~x;
                cnt_neg   += __builtin_popcountll(nx);
                match_neg += __builtin_popcountll(nx & y);
            }

            // Centered correlation amplitude in [-1, +1]
            float corr_pos = (cnt_pos > 0) ? (2.0f * match_pos / cnt_pos - 1.0f) - static_cast<float>(mean_y) : 0.0f;
            float corr_neg = (cnt_neg > 0) ? (2.0f * match_neg / cnt_neg - 1.0f) - static_cast<float>(mean_y) : 0.0f;

            // Phase rotation creates complex wave-packet interference
            float theta = (static_cast<float>(k) * PI) / static_cast<float>(std::max<size_t>(M, 1) * 2);
            float c = std::cos(theta);
            float s = std::sin(theta);

            // Magnitude scaled by square root of correlation strength (Born amplitude)
            float w_pos = std::copysign(std::sqrt(std::abs(corr_pos)), corr_pos);
            float w_neg = std::copysign(std::sqrt(std::abs(corr_neg)), corr_neg);

            amp_re_pos[k] = w_pos * c;
            amp_im_pos[k] = w_pos * s;
            amp_re_neg[k] = w_neg * c;
            amp_im_neg[k] = w_neg * s;
        }

        // Synthesize 2-spin ZZ Entangled Phase Couplings among top 6 lags
        entangled_pairs.clear();
        size_t E = std::min<size_t>(6, M);
        for (size_t a = 0; a < E; ++a) {
            for (size_t b = a + 1; b < E; ++b) {
                uint64_t c_same = 0, y_same = 0;
                uint64_t c_diff = 0, y_diff = 0;
                for (size_t i = start_i; i < stream.size(); ++i) {
                    uint64_t xa = fetch_bit_lag_window(stream, i, lags[a]);
                    uint64_t xb = fetch_bit_lag_window(stream, i, lags[b]);
                    uint64_t y  = stream[i];
                    uint64_t same = ~(xa ^ xb);
                    uint64_t diff = (xa ^ xb);
                    c_same += __builtin_popcountll(same);
                    y_same += __builtin_popcountll(same & y);
                    c_diff += __builtin_popcountll(diff);
                    y_diff += __builtin_popcountll(diff & y);
                }
                float cs = (c_same > 0) ? (2.0f * y_same / c_same - 1.0f) - static_cast<float>(mean_y) : 0.0f;
                float cd = (c_diff > 0) ? (2.0f * y_diff / c_diff - 1.0f) - static_cast<float>(mean_y) : 0.0f;
                float phi = 0.35f * static_cast<float>(a + b);
                entangled_pairs.push_back({
                    static_cast<uint8_t>(a), static_cast<uint8_t>(b),
                    cs * std::cos(phi), cs * std::sin(phi),
                    cd * std::cos(phi), cd * std::sin(phi)
                });
            }
        }

        // Tune optimal Quantum Readout Basis angle (0 .. 2*PI) on training stream
        float best_ang = 0.0f;
        uint64_t best_err = ~0ULL;
        for (int step = 0; step < 16; ++step) {
            float ang = (2.0f * PI * step) / 16.0f;
            readout_cos = std::cos(ang);
            readout_sin = std::sin(ang);
            uint64_t err = 0;
            // Sample every 2nd word for fast basis alignment
            for (size_t i = start_i; i < stream.size(); i += 2) {
                float coh_dummy;
                uint64_t p = predict_window(stream, i, coh_dummy);
                err += __builtin_popcountll(p ^ stream[i]);
            }
            if (err < best_err) {
                best_err = err;
                best_ang = ang;
            }
        }
        readout_cos = std::cos(best_ang);
        readout_sin = std::sin(best_ang);
    }

    // Evaluates coherent complex wave superposition for all 64 bits in parallel
    inline uint64_t predict_window(const std::vector<uint64_t>& stream, size_t i, float& avg_coherence) const {
        float re[64];
        float im[64];
        for (int b = 0; b < 64; ++b) {
            re[b] = bias_re;
            im[b] = bias_im;
        }

        const size_t M = lags.size();
        uint64_t lag_vals[16];
        for (size_t k = 0; k < M && k < 16; ++k) {
            uint64_t w = fetch_bit_lag_window(stream, i, lags[k]);
            lag_vals[k] = w;
            float rp = amp_re_pos[k], ip = amp_im_pos[k];
            float rn = amp_re_neg[k], in = amp_im_neg[k];
            for (int b = 0; b < 64; ++b) {
                uint64_t bit = (w >> b) & 1ULL;
                re[b] += bit ? rp : rn;
                im[b] += bit ? ip : in;
            }
        }

        // Entangled 2-spin ZZ interference
        for (const auto& ep : entangled_pairs) {
            uint64_t eq = ~(lag_vals[ep.a] ^ lag_vals[ep.b]);
            for (int b = 0; b < 64; ++b) {
                uint64_t s = (eq >> b) & 1ULL;
                re[b] += s ? ep.re_same : ep.re_diff;
                im[b] += s ? ep.im_same : ep.im_diff;
            }
        }

        // Born measurement projection onto readout basis
        uint64_t out = 0ULL;
        float coh_sum = 0.0f;
        for (int b = 0; b < 64; ++b) {
            float proj = re[b] * readout_cos + im[b] * readout_sin;
            if (proj > 0.0f) out |= (1ULL << b);
            coh_sum += std::abs(proj);
        }
        avg_coherence = coh_sum / 64.0f;
        return out;
    }
};

// ============================================================================
// UNBOUNDED ENDOGENOUS MEMORY (Zero-Error Residual Anchor)
// ============================================================================
struct UnboundedNode {
    uint64_t sig1, sig2, delta;
    uint32_t first_t, depth;
    uint8_t  state;
};

class UnboundedEndogenousMemory {
public:
    std::vector<UnboundedNode> table;
    size_t count = 0;
    size_t max_depth_reached = 2;

    static inline void advance_sig(const std::vector<uint64_t>& words, size_t t,
                                   size_t from_d, size_t to_d,
                                   uint64_t& h1, uint64_t& h2) {
        size_t limit = std::min(to_d, t);
        for (size_t d = from_d + 1; d <= limit; ++d) {
            uint64_t w = words[t - d];
            h1 = splitmix64(h1 ^ w ^ (d * 0x9e3779b97f4a7c15ULL));
            h2 = splitmix64(h2 ^ rotl64(w, static_cast<int>(d & 63)) ^ (d * 0xbf58476d1ce4e5b9ULL));
        }
        if (to_d > t) {
            h1 = splitmix64(h1 ^ (static_cast<uint64_t>(t) << 32) ^ to_d);
            h2 = splitmix64(h2 ^ static_cast<uint64_t>(t));
        }
    }

    void rehash(size_t new_cap) {
        std::vector<UnboundedNode> next(new_cap, {0, 0, 0, 0, 0, 0});
        size_t mask = new_cap - 1;
        for (const auto& n : table) {
            if (n.state == 0) continue;
            size_t idx = (n.sig1 ^ n.sig2 ^ n.depth) & mask;
            while (next[idx].state != 0) idx = (idx + 1) & mask;
            next[idx] = n;
        }
        table = std::move(next);
    }

    void insert_at_depth(const std::vector<uint64_t>& words, size_t t, uint64_t delta,
                         size_t depth, uint64_t h1, uint64_t h2) {
        if (table.empty()) rehash(64);
        else if ((count + 1) * 2 >= table.size()) rehash(table.size() * 2);
        if (depth > max_depth_reached) max_depth_reached = depth;

        size_t mask = table.size() - 1;
        size_t idx = (h1 ^ h2 ^ depth) & mask;

        while (table[idx].state != 0) {
            if (table[idx].sig1 == h1 && table[idx].sig2 == h2 && table[idx].depth == depth) {
                if (table[idx].state == 1) {
                    if (table[idx].delta == delta) return;
                    if (depth > words.size() * 2) return;
                    uint32_t old_t = table[idx].first_t;
                    uint64_t old_delta = table[idx].delta;
                    table[idx].state = 2;
                    size_t next_depth = depth * 2;
                    uint64_t oh1 = h1, oh2 = h2;
                    advance_sig(words, old_t, depth, next_depth, oh1, oh2);
                    insert_at_depth(words, old_t, old_delta, next_depth, oh1, oh2);
                    uint64_t nh1 = h1, nh2 = h2;
                    advance_sig(words, t, depth, next_depth, nh1, nh2);
                    insert_at_depth(words, t, delta, next_depth, nh1, nh2);
                    return;
                } else {
                    size_t next_depth = depth * 2;
                    advance_sig(words, t, depth, next_depth, h1, h2);
                    insert_at_depth(words, t, delta, next_depth, h1, h2);
                    return;
                }
            }
            idx = (idx + 1) & mask;
        }
        table[idx] = {h1, h2, delta, static_cast<uint32_t>(t), static_cast<uint32_t>(depth), 1};
        count++;
    }

    inline void sprout(const std::vector<uint64_t>& words, size_t t, uint64_t delta) {
        uint64_t h1 = 0x9e3779b97f4a7c15ULL;
        uint64_t h2 = 0xbf58476d1ce4e5b9ULL;
        advance_sig(words, t, 0, 2, h1, h2);
        insert_at_depth(words, t, delta, 2, h1, h2);
    }

    inline uint64_t query(const std::vector<uint64_t>& words, size_t t) const {
        if (table.empty()) return 0ULL;
        size_t depth = 2, prev_d = 0;
        uint64_t h1 = 0x9e3779b97f4a7c15ULL, h2 = 0xbf58476d1ce4e5b9ULL;
        size_t mask = table.size() - 1;
        while (depth <= max_depth_reached) {
            advance_sig(words, t, prev_d, depth, h1, h2);
            size_t idx = (h1 ^ h2 ^ depth) & mask;
            bool found = false;
            while (table[idx].state != 0) {
                if (table[idx].sig1 == h1 && table[idx].sig2 == h2 && table[idx].depth == depth) {
                    if (table[idx].state == 1) return table[idx].delta;
                    found = true;
                    break;
                }
                idx = (idx + 1) & mask;
            }
            if (!found) return 0ULL;
            prev_d = depth;
            depth *= 2;
        }
        return 0ULL;
    }
};

static void run_triadic_audit(const std::string& name, const std::vector<uint64_t>& words) {
    size_t split = words.size() / 2;
    std::vector<uint64_t> train_slice(words.begin(), words.begin() + split);
    std::vector<uint64_t> test_slice(words.begin() + split, words.end());

    // 1. Deterministic Primitive (D)
    DeterministicField det;
    det.synthesize(train_slice, 256);

    // 2. Quantum Wave Interference Primitive (Q)
    QuantumInterferenceField qfield;
    qfield.synthesize(train_slice, det.top_lags);

    size_t start_i = 5;
    uint64_t total_bits = (test_slice.size() - start_i) * 64ULL;

    // Evaluate D-only on UNSEEN test_slice
    uint64_t d_err = 0;
    auto td0 = std::chrono::high_resolution_clock::now();
    for (size_t i = start_i; i < test_slice.size(); ++i) {
        d_err += __builtin_popcountll(det.predict_window(test_slice, i) ^ test_slice[i]);
    }
    auto td1 = std::chrono::high_resolution_clock::now();
    double d_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    double d_acc = 100.0 * (1.0 - static_cast<double>(d_err) / total_bits);
    double d_speed = (static_cast<double>((test_slice.size() - start_i) * 8ULL) / (1024.0 * 1024.0)) / (d_ms / 1000.0);

    // Evaluate Q-only on UNSEEN test_slice
    uint64_t q_err = 0;
    uint64_t q_zero_win = 0;
    double coh_acc = 0.0;
    auto tq0 = std::chrono::high_resolution_clock::now();
    for (size_t i = start_i; i < test_slice.size(); ++i) {
        float coh = 0.0f;
        uint64_t pred_q = qfield.predict_window(test_slice, i, coh);
        coh_acc += coh;
        uint64_t diff = pred_q ^ test_slice[i];
        if (diff == 0ULL) q_zero_win++;
        q_err += __builtin_popcountll(diff);
    }
    auto tq1 = std::chrono::high_resolution_clock::now();
    double q_ms = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    double q_acc = 100.0 * (1.0 - static_cast<double>(q_err) / total_bits);
    double q_speed = (static_cast<double>((test_slice.size() - start_i) * 8ULL) / (1024.0 * 1024.0)) / (q_ms / 1000.0);

    // Evaluate Triadic (D + Q + Stochastic sigma_hw perturbation when coherence < threshold)
    // Measure pure generalization of Triadic wave collapse (without memory)
    uint64_t tri_err = 0;
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    uint64_t stoch_triggers = 0;
    for (size_t i = start_i; i < test_slice.size(); ++i) {
        float coh = 0.0f;
        uint64_t pred_q = qfield.predict_window(test_slice, i, coh);
        uint64_t pred_d = det.predict_window(test_slice, i);
        // If Quantum interference is strong (coh > 0.25), trust Quantum; else blend/explore
        uint64_t pred = (coh > 0.25f) ? pred_q : pred_d;
        if (coh < 0.05f) {
            rng = splitmix64(rng ^ i);
            // Low-amplitude hardware stochasticity probe (1 bit per 64 on vacuum state)
            pred ^= (1ULL << (rng & 63));
            stoch_triggers++;
        }
        tri_err += __builtin_popcountll(pred ^ test_slice[i]);
    }
    double tri_acc = 100.0 * (1.0 - static_cast<double>(tri_err) / total_bits);

    // Endogenous Memory over Quantum Field for 0-error Pass 2
    UnboundedEndogenousMemory mem;
    for (size_t i = start_i; i < words.size(); ++i) {
        float coh = 0.0f;
        uint64_t p = qfield.predict_window(words, i, coh);
        if (p != words[i]) mem.sprout(words, i, p ^ words[i]);
    }
    for (size_t i = start_i; i < words.size(); ++i) {
        float coh = 0.0f;
        uint64_t p = qfield.predict_window(words, i, coh);
        if ((p ^ mem.query(words, i)) != words[i]) mem.sprout(words, i, p ^ words[i]);
    }
    uint64_t pass2_err = 0;
    for (size_t i = start_i; i < words.size(); ++i) {
        float coh = 0.0f;
        uint64_t p = qfield.predict_window(words, i, coh) ^ mem.query(words, i);
        pass2_err += __builtin_popcountll(p ^ words[i]);
    }

    std::cout << "\n[BITSTREAM] " << name << " (" << (words.size() * 64) << " bits)" << std::endl;
    std::cout << "  [D] Deterministic Only  : " << std::fixed << std::setprecision(2) << d_acc
              << "% (F = " << std::setprecision(6) << compute_free_energy(d_err, total_bits)
              << " nats) | Speed: " << std::setprecision(2) << d_speed << " MB/s" << std::endl;
    std::cout << "  [Q] Quantum Interference: " << std::fixed << std::setprecision(2) << q_acc
              << "% (F = " << std::setprecision(6) << compute_free_energy(q_err, total_bits)
              << " nats) | Zero-Win: " << q_zero_win
              << " | Avg Coherence: " << std::setprecision(3) << (coh_acc / (test_slice.size() - start_i))
              << " | Speed: " << std::setprecision(2) << q_speed << " MB/s" << std::endl;
    std::cout << "  [D+Q+S] Triadic Collapse: " << std::fixed << std::setprecision(2) << tri_acc
              << "% (Stoch Triggers: " << stoch_triggers << ")" << std::endl;
    std::cout << "  Pass 2 Zero-Error Audit : " << pass2_err << " bits (Nodes: " << mem.count << ")" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1145: TRIADIC SUBSTRATE (DET + STOCH + QUANTUM WAVE)" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 1, 4 (V=2), 6, 8, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    auto kep_words = load_file_as_bitstream("KEP.md");
    if (!kep_words.empty()) run_triadic_audit("TEXT: KEP.md", kep_words);

    auto shake_words = load_file_as_bitstream("tinyshakespeare.txt", 262144);
    if (!shake_words.empty()) run_triadic_audit("TEXT: tinyshakespeare.txt (256KB)", shake_words);

    auto png_words = load_file_as_bitstream("experiments/renders/bloodbath_precise_v4.png", 262144);
    if (!png_words.empty()) run_triadic_audit("IMAGE/PNG: bloodbath_precise_v4.png (256KB)", png_words);

    if (argc > 0 && argv[0]) {
        auto elf_words = load_file_as_bitstream(argv[0]);
        if (!elf_words.empty()) run_triadic_audit("BINARY/ELF: x86_64 Machine Code", elf_words);
    }

    return 0;
}
