// ============================================================================
// KARYON-CORE EXP-1142: UNIVERSAL LAW CRYSTALLIZATION & MEMORY DISSOLUTION
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 1, 4 (V=2 Bitstream), 6, 8, 13, 14, 15
// ============================================================================
// Modality-Agnostic Autopoietic Loop (Text, PNG Image, x86_64 ELF Machine Code):
//   1. Global Vectorized Causal Law (O(1) hardware word-plane ALU ops).
//   2. Endogenous Friction Memory (sprouts ONLY on residual error, unbounded
//      recursive depth on collision).
//   3. Endogenous Law Crystallization & Memory Dissolution:
//      The substrate inspects its own sprouted friction memory, discovers
//      conditional bit-plane & delta regularities directly from the residual
//      friction, crystallizes them into compact O(1) conditional word-ops,
//      and DISSOLVES all memory nodes whose residual error drops to 0.
//   4. Strict Principle 15 Audit:
//      Measured on UNSEEN 50% tail of each modality + exact node dissolution.
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
#include <unordered_map>

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

static std::vector<uint64_t> load_file_as_words(const std::string& path, size_t max_bytes = 0) {
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
// 1. VECTORIZED WORD-PLANE CAUSAL LAW (Base + Crystallized Conditional Rules)
// ============================================================================
struct BitCausalRule {
    uint8_t mode;       // 0 = const, 1 = 1-tap, 2 = 2-tap XOR, 3 = 2-tap AND
    uint8_t polarity;   // 0 or 1
    uint8_t src1;       // 0..3
    uint8_t bit1;       // 0..63
    uint8_t src2;       // 0..3
    uint8_t bit2;       // 0..63
};

struct WordPlaneOp {
    uint8_t mode;
    uint8_t src1;
    uint8_t rot1;
    uint8_t src2;
    uint8_t rot2;
    uint64_t target_mask;
};

// Crystallized Conditional Causal Rule (born endogenously from Memory friction):
// Matches a physical bit-pattern on (prev1 & mask) == pattern -> applies
// compact ALU correction (base_shift/xor/add_delta) without storing a memory node!
struct CrystallizedMotif {
    uint64_t ctx_mask;
    uint64_t ctx_pattern;
    uint64_t xor_correction;
    uint64_t support_count;
};

class AutopoieticLaw {
public:
    BitCausalRule rules[64];
    uint64_t polarity_mask = 0ULL;
    std::vector<WordPlaneOp> word_ops;
    std::vector<CrystallizedMotif> crystallized_motifs;

    AutopoieticLaw() {
        for (int b = 0; b < 64; ++b) rules[b] = {0, 0, 0, 0, 0, 0};
    }

    static inline void build_history_regs(uint64_t prev1, uint64_t prev2, uint64_t h[4]) {
        h[0] = prev1;
        h[1] = prev2;
        h[2] = prev1 ^ prev2;
        h[3] = prev1 - prev2;
    }

    void compile_to_word_ops() {
        word_ops.clear();
        polarity_mask = 0ULL;
        for (int b = 0; b < 64; ++b) {
            const auto& r = rules[b];
            if (r.polarity) polarity_mask |= (1ULL << b);
            if (r.mode == 0) continue;

            uint8_t rot1 = static_cast<uint8_t>((b - r.bit1) & 63);
            uint8_t rot2 = (r.mode >= 2) ? static_cast<uint8_t>((b - r.bit2) & 63) : 0;

            bool merged = false;
            for (auto& op : word_ops) {
                if (op.mode == r.mode && op.src1 == r.src1 && op.rot1 == rot1 &&
                    (r.mode == 1 || (op.src2 == r.src2 && op.rot2 == rot2))) {
                    op.target_mask |= (1ULL << b);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                word_ops.push_back({r.mode, r.src1, rot1, r.src2, rot2, (1ULL << b)});
            }
        }
    }

    void synthesize_base(const std::vector<uint64_t>& stream) {
        if (stream.size() < 4) return;
        const size_t N = stream.size() - 2;

        std::vector<uint64_t> h0(N), h1(N), h2(N), h3(N), targets(N);
        for (size_t t = 2; t < stream.size(); ++t) {
            uint64_t h[4];
            build_history_regs(stream[t - 1], stream[t - 2], h);
            size_t idx = t - 2;
            h0[idx] = h[0];
            h1[idx] = h[1];
            h2[idx] = h[2];
            h3[idx] = h[3];
            targets[idx] = stream[t];
        }
        const std::vector<uint64_t>* H[4] = {&h0, &h1, &h2, &h3};

        for (int b = 0; b < 64; ++b) {
            uint64_t ones_count = 0;
            for (size_t i = 0; i < N; ++i) ones_count += (targets[i] >> b) & 1ULL;

            uint8_t best_pol = (ones_count * 2 >= N) ? 1 : 0;
            uint64_t best_err = (best_pol == 1) ? (N - ones_count) : ones_count;
            BitCausalRule best_rule = {0, best_pol, 0, 0, 0, 0};

            for (uint8_t s = 0; s < 4; ++s) {
                const auto& reg_vec = *H[s];
                for (uint8_t sb = 0; sb < 64; ++sb) {
                    uint64_t match_same = 0;
                    for (size_t i = 0; i < N; ++i) {
                        uint64_t x = (reg_vec[i] >> sb) & 1ULL;
                        uint64_t y = (targets[i] >> b) & 1ULL;
                        match_same += (x == y);
                    }
                    if (N - match_same < best_err) { best_err = N - match_same; best_rule = {1, 0, s, sb, 0, 0}; }
                    if (match_same < best_err)     { best_err = match_same;     best_rule = {1, 1, s, sb, 0, 0}; }
                }
            }

            if (best_err > 0 && best_rule.mode == 1) {
                uint8_t anchor_s = best_rule.src1;
                uint8_t anchor_b = best_rule.bit1;
                const auto& anchor_vec = *H[anchor_s];

                for (uint8_t s2 = 0; s2 < 4; ++s2) {
                    const auto& vec2 = *H[s2];
                    for (uint8_t b2 = 0; b2 < 64; ++b2) {
                        if (s2 == anchor_s && b2 == anchor_b) continue;
                        uint64_t match_xor = 0, match_and = 0;
                        for (size_t i = 0; i < N; ++i) {
                            uint64_t x1 = (anchor_vec[i] >> anchor_b) & 1ULL;
                            uint64_t x2 = (vec2[i] >> b2) & 1ULL;
                            uint64_t y  = (targets[i] >> b) & 1ULL;
                            match_xor += ((x1 ^ x2) == y);
                            match_and += ((x1 & x2) == y);
                        }
                        if (N - match_xor < best_err) { best_err = N - match_xor; best_rule = {2, 0, anchor_s, anchor_b, s2, b2}; }
                        if (match_xor < best_err)     { best_err = match_xor;     best_rule = {2, 1, anchor_s, anchor_b, s2, b2}; }
                        if (N - match_and < best_err) { best_err = N - match_and; best_rule = {3, 0, anchor_s, anchor_b, s2, b2}; }
                        if (match_and < best_err)     { best_err = match_and;     best_rule = {3, 1, anchor_s, anchor_b, s2, b2}; }
                    }
                }
            }
            rules[b] = best_rule;
        }
        compile_to_word_ops();
    }

    inline uint64_t predict_base(uint64_t prev1, uint64_t prev2) const {
        uint64_t h[4];
        build_history_regs(prev1, prev2, h);
        uint64_t out = polarity_mask;
        for (const auto& op : word_ops) {
            uint64_t v = rotl64(h[op.src1], op.rot1);
            if (op.mode == 2) {
                v ^= rotl64(h[op.src2], op.rot2);
            } else if (op.mode == 3) {
                v &= rotl64(h[op.src2], op.rot2);
            }
            out ^= (v & op.target_mask);
        }
        return out;
    }

    inline uint64_t predict(uint64_t prev1, uint64_t prev2) const {
        uint64_t out = predict_base(prev1, prev2);
        // Apply crystallized conditional motifs (ordered by specificity/support)
        for (const auto& m : crystallized_motifs) {
            if ((prev1 & m.ctx_mask) == m.ctx_pattern) {
                out ^= m.xor_correction;
                break; // First matching crystallized physical regime
            }
        }
        return out;
    }

    // Endogenous Law Crystallization:
    // Inspects residual friction (prev1, residual_delta) across the training
    // stream, discovers high-frequency physical invariant motifs, and
    // crystallizes them into O(1) rules ONLY if they strictly reduce total
    // bit error on the training stream.
    void crystallize_from_friction(const std::vector<uint64_t>& train_slice, size_t max_motifs = 32) {
        crystallized_motifs.clear();
        if (train_slice.size() < 8) return;

        // Collect residual friction pairs (prev1, delta)
        struct PairHash {
            size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
                return static_cast<size_t>(splitmix64(p.first ^ rotl64(p.second, 19)));
            }
        };

        // 1. Exact 64-bit regime invariant: full prev1 -> dominant delta
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, size_t>> exact_hist;
        // 2. Half-word / 32-bit physical invariant: (prev1 & 0xFFFFFFFF00000000ULL) -> dominant delta
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, size_t>> hi32_hist;
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, size_t>> lo32_hist;

        for (size_t t = 2; t < train_slice.size(); ++t) {
            uint64_t p1 = train_slice[t - 1];
            uint64_t p2 = train_slice[t - 2];
            uint64_t delta = predict_base(p1, p2) ^ train_slice[t];
            if (delta != 0ULL) {
                exact_hist[p1][delta]++;
                hi32_hist[p1 & 0xFFFFFFFF00000000ULL][delta]++;
                lo32_hist[p1 & 0x00000000FFFFFFFFULL][delta]++;
            }
        }

        // Evaluate candidate motifs by net bit-error reduction across the ENTIRE training stream
        struct Candidate {
            uint64_t mask;
            uint64_t pattern;
            uint64_t delta;
            int64_t  net_bit_gain;
            size_t   hits;
        };
        std::vector<Candidate> candidates;

        auto evaluate_bucket = [&](uint64_t mask, const auto& hist) {
            for (const auto& [pat, delta_counts] : hist) {
                uint64_t best_delta = 0;
                size_t best_cnt = 0;
                for (const auto& [d, cnt] : delta_counts) {
                    if (cnt > best_cnt) {
                        best_cnt = cnt;
                        best_delta = d;
                    }
                }
                if (best_cnt < 3 || best_delta == 0ULL) continue;

                // Measure TRUE net bit-error gain across train_slice
                int64_t gain = 0;
                for (size_t t = 2; t < train_slice.size(); ++t) {
                    if ((train_slice[t - 1] & mask) == pat) {
                        uint64_t base_p = predict_base(train_slice[t - 1], train_slice[t - 2]);
                        uint64_t err_before = __builtin_popcountll(base_p ^ train_slice[t]);
                        uint64_t err_after  = __builtin_popcountll((base_p ^ best_delta) ^ train_slice[t]);
                        gain += static_cast<int64_t>(err_before) - static_cast<int64_t>(err_after);
                    }
                }
                if (gain > 0) {
                    candidates.push_back({mask, pat, best_delta, gain, best_cnt});
                }
            }
        };

        evaluate_bucket(0xFFFFFFFFFFFFFFFFULL, exact_hist);
        evaluate_bucket(0xFFFFFFFF00000000ULL, hi32_hist);
        evaluate_bucket(0x00000000FFFFFFFFULL, lo32_hist);

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.net_bit_gain != b.net_bit_gain) return a.net_bit_gain > b.net_bit_gain;
            return a.mask > b.mask;
        });

        // Greedily accept non-overlapping crystallized motifs that strictly reduce train error
        for (const auto& c : candidates) {
            if (crystallized_motifs.size() >= max_motifs) break;
            bool overlaps = false;
            for (const auto& m : crystallized_motifs) {
                if ((c.pattern & m.ctx_mask) == m.ctx_pattern || (m.ctx_pattern & c.mask) == c.pattern) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                crystallized_motifs.push_back({c.mask, c.pattern, c.delta, c.hits});
            }
        }
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
    uint8_t  state; // 0 = EMPTY, 1 = LEAF, 2 = BRANCH
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
    size_t word_ops_count;
    size_t crystallized_count;
    uint64_t base_law_unseen_err;
    uint64_t cryst_law_unseen_err;
    uint64_t total_test_bits;
    double base_reduction_pct;
    double cryst_reduction_pct;
    size_t nodes_before_cryst;
    size_t nodes_after_cryst;
    double node_dissolution_pct;
    uint64_t pass2_err_bits;
    double pass2_F;
    double law_speed_mb_s;
    double hybrid_speed_mb_s;
};

static ModalityAudit audit_modality(const std::string& name, const std::vector<uint64_t>& words) {
    ModalityAudit res;
    res.name = name;
    res.total_words = words.size();

    size_t split = words.size() / 2;
    std::vector<uint64_t> train_slice(words.begin(), words.begin() + split);
    std::vector<uint64_t> test_slice(words.begin() + split, words.end());

    // 1. Synthesize Base Causal Law on train_slice ONLY
    AutopoieticLaw law;
    law.synthesize_base(train_slice);
    res.word_ops_count = law.word_ops.size();

    // Measure Base Law vs Persistence Baseline on UNSEEN test_slice
    uint64_t persist_err = 0, base_law_err = 0;
    uint64_t total_bits = (test_slice.size() - 2) * 64ULL;
    for (size_t t = 2; t < test_slice.size(); ++t) {
        persist_err += __builtin_popcountll(test_slice[t - 1] ^ test_slice[t]);
        base_law_err += __builtin_popcountll(law.predict_base(test_slice[t - 1], test_slice[t - 2]) ^ test_slice[t]);
    }
    res.base_law_unseen_err = base_law_err;
    res.total_test_bits = total_bits;
    res.base_reduction_pct = 100.0 * (1.0 - static_cast<double>(base_law_err) / static_cast<double>(persist_err));

    // 2. Endogenous Memory BEFORE Crystallization
    UnboundedEndogenousMemory mem_before;
    for (size_t t = 2; t < words.size(); ++t) {
        uint64_t delta = law.predict_base(words[t - 1], words[t - 2]) ^ words[t];
        if (delta != 0ULL) mem_before.sprout(words, t, delta);
    }
    res.nodes_before_cryst = mem_before.count;

    // 3. Endogenous Law Crystallization from training friction ONLY
    law.crystallize_from_friction(train_slice, 32);
    res.crystallized_count = law.crystallized_motifs.size();

    // Measure Crystallized Law on UNSEEN test_slice + Law inference speed
    uint64_t cryst_law_err = 0;
    auto tl0 = std::chrono::high_resolution_clock::now();
    for (size_t t = 2; t < test_slice.size(); ++t) {
        uint64_t pred = law.predict(test_slice[t - 1], test_slice[t - 2]);
        cryst_law_err += __builtin_popcountll(pred ^ test_slice[t]);
    }
    auto tl1 = std::chrono::high_resolution_clock::now();
    double law_ms = std::chrono::duration<double, std::milli>(tl1 - tl0).count();
    res.cryst_law_unseen_err = cryst_law_err;
    res.cryst_reduction_pct = 100.0 * (1.0 - static_cast<double>(cryst_law_err) / static_cast<double>(persist_err));
    res.law_speed_mb_s = (static_cast<double>((test_slice.size() - 2) * 8ULL) / (1024.0 * 1024.0)) / (law_ms / 1000.0);

    // 4. Memory Dissolution: Rebuild Endogenous Memory AFTER Crystallization
    UnboundedEndogenousMemory mem_after;
    for (size_t t = 2; t < words.size(); ++t) {
        uint64_t delta = law.predict(words[t - 1], words[t - 2]) ^ words[t];
        if (delta != 0ULL) mem_after.sprout(words, t, delta);
    }
    for (size_t t = 2; t < words.size(); ++t) {
        uint64_t p = law.predict(words[t - 1], words[t - 2]);
        if ((p ^ mem_after.query(words, t)) != words[t]) {
            mem_after.sprout(words, t, p ^ words[t]);
        }
    }
    res.nodes_after_cryst = mem_after.count;
    res.node_dissolution_pct = 100.0 * (1.0 - static_cast<double>(res.nodes_after_cryst) /
                                        std::max<size_t>(1, res.nodes_before_cryst));

    // 5. Pass 2 Zero-Error Verification & Hybrid Speed
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t pass2_err = 0;
    for (size_t t = 2; t < words.size(); ++t) {
        uint64_t pred = law.predict(words[t - 1], words[t - 2]) ^ mem_after.query(words, t);
        pass2_err += __builtin_popcountll(pred ^ words[t]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    res.pass2_err_bits = pass2_err;
    res.pass2_F = compute_free_energy(pass2_err, (words.size() - 2) * 64ULL);
    res.hybrid_speed_mb_s = (static_cast<double>((words.size() - 2) * 8ULL) / (1024.0 * 1024.0)) / (ms / 1000.0);

    return res;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1142: UNIVERSAL LAW CRYSTALLIZATION (MULTI-MODALITY)" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 1, 4 (V=2), 6, 8, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<std::pair<std::string, std::vector<uint64_t>>> datasets;

    auto kep_words = load_file_as_words("KEP.md");
    if (!kep_words.empty()) datasets.push_back({"TEXT: KEP.md", kep_words});

    auto shake_words = load_file_as_words("tinyshakespeare.txt", 262144);
    if (!shake_words.empty()) datasets.push_back({"TEXT: tinyshakespeare.txt (256KB)", shake_words});

    auto png_words = load_file_as_words("experiments/renders/bloodbath_precise_v4.png", 262144);
    if (!png_words.empty()) datasets.push_back({"IMAGE/PNG: bloodbath_precise_v4.png (256KB)", png_words});

    if (argc > 0 && argv[0]) {
        auto elf_words = load_file_as_words(argv[0]);
        if (!elf_words.empty()) datasets.push_back({"BINARY/ELF: x86_64 Machine Code", elf_words});
    }

    uint64_t total_pass2_err = 0;
    for (const auto& ds : datasets) {
        auto r = audit_modality(ds.first, ds.second);
        total_pass2_err += r.pass2_err_bits;

        std::cout << "\n[MODALITY] " << r.name << " (" << r.total_words << " words / "
                  << (r.total_words * 8) << " bytes)" << std::endl;
        std::cout << "  Base Word-Ops / Motifs  : " << r.word_ops_count << " ops + "
                  << r.crystallized_count << " crystallized motifs" << std::endl;
        std::cout << "  Unseen Base Law Reduct. : " << std::fixed << std::setprecision(2)
                  << r.base_reduction_pct << "% (" << r.base_law_unseen_err << " err bits)" << std::endl;
        std::cout << "  Unseen Cryst Law Reduct.: " << std::fixed << std::setprecision(2)
                  << r.cryst_reduction_pct << "% (" << r.cryst_law_unseen_err << " err bits) | Law Speed: "
                  << r.law_speed_mb_s << " MB/s" << std::endl;
        std::cout << "  Memory Dissolution      : " << r.nodes_before_cryst << " -> " << r.nodes_after_cryst
                  << " nodes (Dissolved: " << std::setprecision(2) << r.node_dissolution_pct << "%)" << std::endl;
        std::cout << "  Pass 2 Zero-Error Audit : " << r.pass2_err_bits << " bits (F = "
                  << std::setprecision(8) << r.pass2_F << " nats) | Hybrid Speed: "
                  << std::setprecision(2) << r.hybrid_speed_mb_s << " MB/s" << std::endl;
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << " TOTAL PASS 2 BIT ERRORS ACROSS ALL MODALITIES: " << total_pass2_err << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
