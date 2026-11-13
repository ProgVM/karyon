// ============================================================================
// KARYON-CORE EXP-1143: UNSHACKLED BIT-LAG CAUSAL FIELD (ZERO BYTE/WORD RIGIDITY)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 1, 4 (V=2), 6, 8, 13, 14, 15
// ============================================================================
// Zero Human Structural Rigidity:
//   - No hardcoded 8-bit bytes, no hardcoded 64-bit word boundaries in the law.
//   - The substrate treats every modality as a raw continuous bitstream V=2.
//   - Physical 64-bit CPU registers are used purely for SWAR parallel speed:
//     arbitrary bit-lag shifts (delta = 1, 2, 3, ..., MAX_LAG bits) are
//     evaluated 64 bits per clock via unaligned bit-stream shifting.
//   - Endogenously discovers the dominant physical bit-lags and boolean
//     interactions (1-tap, 2-tap XOR, 2-tap AND, polarity) purely from
//     bit-level Free Energy minimization, then sprouts unbounded recursive
//     memory ONLY on remaining residual bit friction.
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

// ============================================================================
// CONTINUOUS UNALIGNED BIT-LAG ACCESSOR (Zero Word/Byte Boundary Rigidity)
// ============================================================================
// Returns the 64-bit window of bits ending at the same bit positions as word[i],
// shifted back by an ARBITRARY bit lag `lag` (e.g., lag = 1, 7, 8, 13, 24, ...).
// ============================================================================
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
// 1. UNSHACKLED BIT-LAG CAUSAL FIELD (Endogenous Discovery of Lags & Logic)
// ============================================================================
struct BitLagRule {
    uint8_t  mode;      // 0 = const, 1 = 1-lag, 2 = 2-lag XOR, 3 = 2-lag AND, 4 = 2-lag OR
    uint8_t  polarity;  // 0 or 1 (invert output)
    uint16_t lag1;      // arbitrary bit-lag in [1..max_lag]
    uint16_t lag2;      // arbitrary bit-lag in [1..max_lag]
};

class UnshackledBitLagField {
public:
    BitLagRule rule = {0, 0, 1, 1};
    std::vector<uint16_t> discovered_top_lags;

    // Endogenously scans arbitrary continuous bit-lags [1..max_lag] (no byte bias!)
    // using 64-bit SWAR popcountll across the training bitstream.
    void synthesize(const std::vector<uint64_t>& stream, size_t max_lag = 256) {
        if (stream.size() < 16) return;
        size_t start_i = (max_lag + 63) / 64 + 1;
        if (start_i >= stream.size()) return;

        uint64_t total_bits = (stream.size() - start_i) * 64ULL;

        // Baseline constant polarity
        uint64_t total_ones = 0;
        for (size_t i = start_i; i < stream.size(); ++i) {
            total_ones += __builtin_popcountll(stream[i]);
        }
        uint8_t best_pol = (total_ones * 2 >= total_bits) ? 1 : 0;
        uint64_t best_err = best_pol ? (total_bits - total_ones) : total_ones;
        BitLagRule best_rule = {0, best_pol, 0, 0};

        // Stage 1: Scan all continuous bit lags 1..max_lag (64 bits per clock!)
        struct LagScore {
            uint16_t lag;
            uint8_t  pol;
            uint64_t err;
        };
        std::vector<LagScore> lag_scores;
        lag_scores.reserve(max_lag);

        for (size_t lag = 1; lag <= max_lag; ++lag) {
            uint64_t err0 = 0;
            for (size_t i = start_i; i < stream.size(); ++i) {
                uint64_t shifted = fetch_bit_lag_window(stream, i, lag);
                err0 += __builtin_popcountll(shifted ^ stream[i]);
            }
            uint64_t err1 = total_bits - err0;
            uint8_t pol = (err1 < err0) ? 1 : 0;
            uint64_t err = pol ? err1 : err0;
            lag_scores.push_back({static_cast<uint16_t>(lag), pol, err});

            if (err < best_err) {
                best_err = err;
                best_rule = {1, pol, static_cast<uint16_t>(lag), 0};
            }
        }

        // Sort lags by causal friction reduction (top 16 strongest physical lags)
        std::sort(lag_scores.begin(), lag_scores.end(), [](const LagScore& a, const LagScore& b) {
            return a.err < b.err;
        });

        discovered_top_lags.clear();
        for (size_t k = 0; k < std::min<size_t>(8, lag_scores.size()); ++k) {
            discovered_top_lags.push_back(lag_scores[k].lag);
        }

        // Stage 2: Non-linear 2-lag Boolean interactions (XOR, AND, OR) among top 24 lags
        size_t K = std::min<size_t>(24, lag_scores.size());
        for (size_t a = 0; a < K; ++a) {
            uint16_t l1 = lag_scores[a].lag;
            for (size_t b = a + 1; b < K; ++b) {
                uint16_t l2 = lag_scores[b].lag;
                uint64_t err_xor = 0, err_and = 0, err_or = 0;

                for (size_t i = start_i; i < stream.size(); ++i) {
                    uint64_t x1 = fetch_bit_lag_window(stream, i, l1);
                    uint64_t x2 = fetch_bit_lag_window(stream, i, l2);
                    uint64_t t  = stream[i];
                    err_xor += __builtin_popcountll((x1 ^ x2) ^ t);
                    err_and += __builtin_popcountll((x1 & x2) ^ t);
                    err_or  += __builtin_popcountll((x1 | x2) ^ t);
                }

                auto check_op = [&](uint8_t mode, uint64_t e0) {
                    uint64_t e1 = total_bits - e0;
                    uint8_t pol = (e1 < e0) ? 1 : 0;
                    uint64_t e = pol ? e1 : e0;
                    if (e < best_err) {
                        best_err = e;
                        best_rule = {mode, pol, l1, l2};
                    }
                };
                check_op(2, err_xor);
                check_op(3, err_and);
                check_op(4, err_or);
            }
        }

        rule = best_rule;
    }

    inline uint64_t predict_window(const std::vector<uint64_t>& stream, size_t i) const {
        uint64_t out = 0ULL;
        if (rule.mode == 0) {
            out = 0ULL;
        } else if (rule.mode == 1) {
            out = fetch_bit_lag_window(stream, i, rule.lag1);
        } else if (rule.mode == 2) {
            out = fetch_bit_lag_window(stream, i, rule.lag1) ^ fetch_bit_lag_window(stream, i, rule.lag2);
        } else if (rule.mode == 3) {
            out = fetch_bit_lag_window(stream, i, rule.lag1) & fetch_bit_lag_window(stream, i, rule.lag2);
        } else {
            out = fetch_bit_lag_window(stream, i, rule.lag1) | fetch_bit_lag_window(stream, i, rule.lag2);
        }
        return rule.polarity ? ~out : out;
    }
};

// ============================================================================
// 2. UNBOUNDED ENDOGENOUS CONTEXT MEMORY (Principle 8 & Principle 13)
// ============================================================================
struct UnboundedNode {
    uint64_t sig1;
    uint64_t sig2;
    uint64_t delta;
    uint32_t first_t;
    uint32_t depth;
    uint8_t  state;
};

class UnboundedEndogenousMemory {
public:
    std::vector<UnboundedNode> table;
    size_t count = 0;
    size_t branch_promotions = 0;
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
                    branch_promotions++;

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
        size_t depth = 2;
        size_t prev_d = 0;
        uint64_t h1 = 0x9e3779b97f4a7c15ULL;
        uint64_t h2 = 0xbf58476d1ce4e5b9ULL;
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

struct ModalityAudit {
    std::string name;
    size_t total_words;
    BitLagRule rule;
    std::vector<uint16_t> top_lags;
    uint64_t unseen_err_bits;
    uint64_t unseen_total_bits;
    double unseen_F;
    double unseen_accuracy_pct;
    double law_speed_mb_s;
    size_t zero_friction_segments;
    size_t sprouted_nodes;
    uint64_t pass2_err_bits;
    double pass2_F;
    double hybrid_speed_mb_s;
};

static ModalityAudit audit_bitstream(const std::string& name, const std::vector<uint64_t>& words) {
    ModalityAudit res;
    res.name = name;
    res.total_words = words.size();

    size_t split = words.size() / 2;
    std::vector<uint64_t> train_slice(words.begin(), words.begin() + split);
    std::vector<uint64_t> test_slice(words.begin() + split, words.end());

    // 1. Endogenously synthesize Bit-Lag Causal Field on train_slice ONLY
    UnshackledBitLagField field;
    field.synthesize(train_slice, 256);
    res.rule = field.rule;
    res.top_lags = field.discovered_top_lags;

    // 2. Evaluate Law-Only generalization & SWAR speed on UNSEEN test_slice
    size_t start_i = 5; // >= 256 bits lookback
    uint64_t err_bits = 0;
    uint64_t total_bits = (test_slice.size() - start_i) * 64ULL;

    auto tl0 = std::chrono::high_resolution_clock::now();
    for (size_t i = start_i; i < test_slice.size(); ++i) {
        uint64_t pred = field.predict_window(test_slice, i);
        err_bits += __builtin_popcountll(pred ^ test_slice[i]);
    }
    auto tl1 = std::chrono::high_resolution_clock::now();
    double law_ms = std::chrono::duration<double, std::milli>(tl1 - tl0).count();

    res.unseen_err_bits = err_bits;
    res.unseen_total_bits = total_bits;
    res.unseen_F = compute_free_energy(err_bits, total_bits);
    res.unseen_accuracy_pct = 100.0 * (1.0 - static_cast<double>(err_bits) / static_cast<double>(total_bits));
    res.law_speed_mb_s = (static_cast<double>((test_slice.size() - start_i) * 8ULL) / (1024.0 * 1024.0)) / (law_ms / 1000.0);

    // 3. Pass 1: Endogenous Memory sprouts ONLY where bit-friction != 0
    UnboundedEndogenousMemory mem;
    size_t zero_segs = 0;
    for (size_t i = start_i; i < words.size(); ++i) {
        uint64_t pred = field.predict_window(words, i);
        uint64_t delta = pred ^ words[i];
        if (delta == 0ULL) {
            zero_segs++;
        } else {
            mem.sprout(words, i, delta);
        }
    }
    for (size_t i = start_i; i < words.size(); ++i) {
        uint64_t pred = field.predict_window(words, i);
        if ((pred ^ mem.query(words, i)) != words[i]) {
            mem.sprout(words, i, pred ^ words[i]);
        }
    }
    res.zero_friction_segments = zero_segs;
    res.sprouted_nodes = mem.count;

    // 4. Pass 2: Full-speed zero-error verification
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t pass2_err = 0;
    for (size_t i = start_i; i < words.size(); ++i) {
        uint64_t pred = field.predict_window(words, i) ^ mem.query(words, i);
        pass2_err += __builtin_popcountll(pred ^ words[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    res.pass2_err_bits = pass2_err;
    res.pass2_F = compute_free_energy(pass2_err, (words.size() - start_i) * 64ULL);
    res.hybrid_speed_mb_s = (static_cast<double>((words.size() - start_i) * 8ULL) / (1024.0 * 1024.0)) / (ms / 1000.0);

    return res;
}

static const char* mode_name(uint8_t m) {
    switch (m) {
        case 0: return "CONST";
        case 1: return "1-LAG";
        case 2: return "XOR";
        case 3: return "AND";
        case 4: return "OR";
        default: return "?";
    }
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1143: UNSHACKLED BIT-LAG CAUSAL FIELD (V=2)" << std::endl;
    std::cout << " Standard: KEP v51.0 | Zero Byte/Word Rigidity | Principles 1,4,8,13,15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<std::pair<std::string, std::vector<uint64_t>>> datasets;

    auto kep_words = load_file_as_bitstream("KEP.md");
    if (!kep_words.empty()) datasets.push_back({"TEXT: KEP.md", kep_words});

    auto shake_words = load_file_as_bitstream("tinyshakespeare.txt", 262144);
    if (!shake_words.empty()) datasets.push_back({"TEXT: tinyshakespeare.txt (256KB)", shake_words});

    auto png_words = load_file_as_bitstream("experiments/renders/bloodbath_precise_v4.png", 262144);
    if (!png_words.empty()) datasets.push_back({"IMAGE/PNG: bloodbath_precise_v4.png (256KB)", png_words});

    if (argc > 0 && argv[0]) {
        auto elf_words = load_file_as_bitstream(argv[0]);
        if (!elf_words.empty()) datasets.push_back({"BINARY/ELF: x86_64 Machine Code", elf_words});
    }

    uint64_t total_pass2_err = 0;
    for (const auto& ds : datasets) {
        auto r = audit_bitstream(ds.first, ds.second);
        total_pass2_err += r.pass2_err_bits;

        std::cout << "\n[BITSTREAM] " << r.name << " (" << (r.total_words * 64) << " bits)" << std::endl;
        std::cout << "  Discovered Top Bit-Lags : [";
        for (size_t k = 0; k < r.top_lags.size(); ++k) {
            std::cout << r.top_lags[k] << (k + 1 < r.top_lags.size() ? ", " : "");
        }
        std::cout << "] bits" << std::endl;
        std::cout << "  Synthesized Law         : " << mode_name(r.rule.mode)
                  << " (lag1=" << r.rule.lag1 << ", lag2=" << r.rule.lag2
                  << ", inv=" << (int)r.rule.polarity << ")" << std::endl;
        std::cout << "  Unseen Bit Accuracy     : " << std::fixed << std::setprecision(2)
                  << r.unseen_accuracy_pct << "% (F = " << std::setprecision(6) << r.unseen_F
                  << " nats) | Law Speed: " << std::setprecision(2) << r.law_speed_mb_s << " MB/s" << std::endl;
        std::cout << "  Zero-Friction Windows   : " << r.zero_friction_segments << " / " << (r.total_words - 5)
                  << " | Sprouted Nodes: " << r.sprouted_nodes << std::endl;
        std::cout << "  Pass 2 Zero-Error Audit : " << r.pass2_err_bits << " bits (F = "
                  << std::setprecision(8) << r.pass2_F << " nats) | Hybrid Speed: "
                  << std::setprecision(2) << r.hybrid_speed_mb_s << " MB/s" << std::endl;
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << " TOTAL PASS 2 BIT ERRORS ACROSS ALL MODALITIES: " << total_pass2_err << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
