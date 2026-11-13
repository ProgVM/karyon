// ============================================================================
// KARYON-CORE EXP-1156: PURE SINGLE-PASS AUTOPOIETIC V=2 SUBSTRATE
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
// ============================================================================
// Addressing Bazilevs' Directive:
//   "Epochs bother me — we have SINGLE-PASS learning, remember?!"
//
// EXP-1156 Strict Single-Pass Architecture (ZERO Epochs, ZERO Multi-Pass):
//   - Every bit b_t (t = 0 .. N-1) is visited EXACTLY ONCE.
//   - Predict b_t BEFORE observing b_t. Record strict causal online accuracy.
//   - Fixes EXP-1155's greedy tie-breaking bug: uses (min_err, max_balance)
//     so ties on homogeneous regions bisect cleanly instead of picking d=1.
//   - Immediate Child Inheritance: when a leaf sprouts on tap_lag, its
//     recent causal buffer (recent_t, recent_y) is partitioned directly into
//     the two new children so no causal experience is lost.
//   - ZERO external GA, ZERO separate hash table, ZERO epochs, ZERO 64-bit boxes.
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
#include <string>
#include <limits>

static inline double compute_free_energy(uint64_t error_bits, uint64_t total_bits) {
    if (total_bits == 0 || error_bits == 0) return 0.0;
    double err_rate = static_cast<double>(error_bits) / static_cast<double>(total_bits);
    if (err_rate >= 1.0) err_rate = 1.0 - 1e-15;
    return -std::log(std::max(1.0 - err_rate, 1e-15));
}

struct BitStream {
    std::vector<uint8_t> bits;
    std::vector<uint64_t> words;
};

static BitStream load_file_bits(const std::string& path, size_t max_bytes = 0) {
    BitStream bs;
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return bs;
    size_t file_size = static_cast<size_t>(ifs.tellg());
    if (max_bytes > 0 && file_size > max_bytes) file_size = max_bytes;
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw(file_size);
    ifs.read(reinterpret_cast<char*>(raw.data()), file_size);

    bs.bits.resize(file_size * 8);
    for (size_t i = 0; i < file_size; ++i) {
        uint8_t b = raw[i];
        for (int k = 0; k < 8; ++k) {
            bs.bits[i * 8 + k] = (b >> k) & 1U;
        }
    }
    size_t num_words = (file_size + 7) / 8;
    bs.words.assign(num_words, 0ULL);
    if (file_size > 0) std::memcpy(bs.words.data(), raw.data(), file_size);
    return bs;
}

static BitStream pack_bits_to_stream(std::vector<uint8_t>&& bits) {
    BitStream bs;
    bs.bits = std::move(bits);
    size_t nw = (bs.bits.size() + 63) / 64;
    bs.words.assign(nw, 0ULL);
    for (size_t i = 0; i < bs.bits.size(); ++i) {
        if (bs.bits[i]) bs.words[i >> 6] |= (1ULL << (i & 63));
    }
    return bs;
}

// ============================================================================
// SINGLE-PASS AUTOPOIETIC V=2 SUBSTRATE
// ============================================================================
static constexpr size_t LEAF_BUF_CAP = 16;
static constexpr uint32_t UNSET_T = std::numeric_limits<uint32_t>::max();

struct AutopoieticNode {
    int32_t  child[2]   = {-1, -1};
    uint32_t tap_lag    = 0;
    uint32_t count[2]   = {0, 0};
    uint32_t first_t[2] = {UNSET_T, UNSET_T};
    uint32_t depth      = 0;

    uint32_t recent_t[LEAF_BUF_CAP];
    uint8_t  recent_y[LEAF_BUF_CAP];
    uint8_t  buf_len    = 0;
    uint8_t  buf_pos    = 0;
    uint16_t visits_since_split = 0;

    inline uint8_t predict() const {
        return (count[1] > count[0]) ? 1U : 0U;
    }

    inline bool is_leaf() const {
        return child[0] < 0;
    }
};

class SinglePassAutopoieticSubstrate {
public:
    std::vector<AutopoieticNode> nodes;
    size_t max_depth_reached = 0;
    size_t max_lag_discovered = 0;
    size_t total_splits = 0;

    SinglePassAutopoieticSubstrate() {
        reset();
    }

    void reset() {
        nodes.clear();
        nodes.emplace_back(); // Tabula Rasa: 1 empty root leaf
        max_depth_reached = 0;
        max_lag_discovered = 0;
        total_splits = 0;
    }

    static inline uint8_t tap_bit(const BitStream& bs, size_t t, size_t lag) {
        if (lag <= t) return bs.bits[t - lag];
        if (lag == t + 1) return 1U;
        return 0U;
    }

    static inline uint64_t fetch_causal_word_64(const BitStream& bs, size_t t, size_t lag_start) {
        if (lag_start + 63 <= t) {
            size_t end_bit = t - lag_start;
            size_t start_bit = end_bit - 63;
            size_t wi = start_bit >> 6;
            size_t bo = start_bit & 63;
            if (bo == 0) return bs.words[wi];
            return (bs.words[wi] >> bo) | (bs.words[wi + 1] << (64 - bo));
        }
        uint64_t w = 0ULL;
        for (int j = 0; j < 64; ++j) {
            if (tap_bit(bs, t, lag_start + 63 - j)) w |= (1ULL << j);
        }
        return w;
    }

    inline uint32_t traverse_leaf(const BitStream& bs, size_t t) const {
        uint32_t cur = 0;
        while (!nodes[cur].is_leaf()) {
            uint8_t bit = tap_bit(bs, t, nodes[cur].tap_lag);
            cur = static_cast<uint32_t>(nodes[cur].child[bit]);
        }
        return cur;
    }

    inline uint8_t step_single_pass(const BitStream& bs, size_t t, size_t split_threshold = 4) {
        uint32_t leaf = traverse_leaf(bs, t);
        uint8_t pred = nodes[leaf].predict();
        uint8_t actual = bs.bits[t];

        AutopoieticNode& n = nodes[leaf];
        n.count[actual]++;
        if (n.first_t[actual] == UNSET_T) {
            n.first_t[actual] = static_cast<uint32_t>(t);
        }
        n.recent_t[n.buf_pos] = static_cast<uint32_t>(t);
        n.recent_y[n.buf_pos] = actual;
        n.buf_pos = (n.buf_pos + 1) % LEAF_BUF_CAP;
        if (n.buf_len < LEAF_BUF_CAP) n.buf_len++;
        n.visits_since_split++;

        if (pred != actual && n.first_t[0] != UNSET_T && n.first_t[1] != UNSET_T &&
            n.visits_since_split >= split_threshold) {
            sprout_split(leaf, bs, t);
        }
        return (pred == actual) ? 1U : 0U;
    }

private:
    inline void eval_lag_candidate(const BitStream& bs, uint8_t len, const AutopoieticNode& n,
                                   size_t d, uint32_t& best_lag, uint32_t& best_err,
                                   uint32_t& best_bal) {
        uint32_t c[2][2] = {{0, 0}, {0, 0}};
        for (uint8_t k = 0; k < len; ++k) {
            uint8_t x = tap_bit(bs, n.recent_t[k], d);
            uint8_t y = n.recent_y[k];
            c[x][y]++;
        }
        uint32_t n0 = c[0][0] + c[0][1];
        uint32_t n1 = c[1][0] + c[1][1];
        if (n0 == 0 || n1 == 0) return;

        uint32_t err = std::min(c[0][0], c[0][1]) + std::min(c[1][0], c[1][1]);
        uint32_t bal = std::min(n0, n1);

        // Minimize error; on ties, maximize branch balance (bisects homogeneous plateaus!)
        if (err < best_err || (err == best_err && bal > best_bal)) {
            best_err = err;
            best_bal = bal;
            best_lag = static_cast<uint32_t>(d);
        }
    }

    void sprout_split(uint32_t leaf_idx, const BitStream& bs, size_t t) {
        const AutopoieticNode& n = nodes[leaf_idx];
        uint8_t len = n.buf_len;

        uint32_t best_lag = 0;
        uint32_t best_err = len + 1;
        uint32_t best_bal = 0;

        // Phase 1: Local causal horizon (1..128) + Dyadic multiscale taps (192..t+1)
        if (len >= 2) {
            size_t local_cap = std::min<size_t>(128, t + 1);
            for (size_t d = 1; d <= local_cap; ++d) {
                eval_lag_candidate(bs, len, n, d, best_lag, best_err, best_bal);
                if (best_err == 0 && best_bal >= len / 3) break;
            }
            if (best_err > 0) {
                for (size_t d = 192; d <= t + 1; d = (d * 3) / 2) {
                    eval_lag_candidate(bs, len, n, d, best_lag, best_err, best_bal);
                    if (best_err == 0 && best_bal >= len / 3) break;
                }
            }
        }

        // Phase 2: Native 64-bit Hardware Bit-Scan (XOR + __builtin_clzll)
        if (best_lag == 0) {
            uint32_t t0 = n.first_t[0];
            uint32_t t1 = n.first_t[1];
            if (t0 != UNSET_T && t1 != UNSET_T && t0 != t1) {
                size_t max_d = std::max(t0, t1) + 1;
                for (size_t d_base = 1; d_base <= max_d; d_base += 64) {
                    uint64_t w0 = fetch_causal_word_64(bs, t0, d_base);
                    uint64_t w1 = fetch_causal_word_64(bs, t1, d_base);
                    uint64_t diff = w0 ^ w1;
                    if (diff != 0ULL) {
                        int j = 63 - __builtin_clzll(diff);
                        size_t d = d_base + static_cast<size_t>(63 - j);
                        if (d <= max_d) {
                            best_lag = static_cast<uint32_t>(d);
                            break;
                        }
                    }
                }
            }
        }

        if (best_lag == 0) {
            nodes[leaf_idx].visits_since_split = 0;
            return;
        }

        if (best_lag > max_lag_discovered) max_lag_discovered = best_lag;

        uint32_t left_idx = static_cast<uint32_t>(nodes.size());
        uint32_t right_idx = left_idx + 1;
        uint32_t next_depth = nodes[leaf_idx].depth + 1;
        if (next_depth > max_depth_reached) max_depth_reached = next_depth;

        // Snapshot parent's recent buffer and witnesses before vector reallocation
        uint32_t witness_t0 = nodes[leaf_idx].first_t[0];
        uint32_t witness_t1 = nodes[leaf_idx].first_t[1];
        uint32_t buf_t[LEAF_BUF_CAP];
        uint8_t  buf_y[LEAF_BUF_CAP];
        for (uint8_t k = 0; k < len; ++k) {
            buf_t[k] = nodes[leaf_idx].recent_t[k];
            buf_y[k] = nodes[leaf_idx].recent_y[k];
        }

        nodes.emplace_back();
        nodes.emplace_back();

        nodes[leaf_idx].tap_lag = best_lag;
        nodes[leaf_idx].child[0] = static_cast<int32_t>(left_idx);
        nodes[leaf_idx].child[1] = static_cast<int32_t>(right_idx);

        nodes[left_idx].depth = next_depth;
        nodes[right_idx].depth = next_depth;

        // Immediate Child Inheritance: route parent's causal buffer into children
        for (uint8_t k = 0; k < len; ++k) {
            uint8_t side = tap_bit(bs, buf_t[k], best_lag);
            AutopoieticNode& ch = nodes[side == 0 ? left_idx : right_idx];
            ch.count[buf_y[k]]++;
            if (ch.first_t[buf_y[k]] == UNSET_T) ch.first_t[buf_y[k]] = buf_t[k];
            ch.recent_t[ch.buf_pos] = buf_t[k];
            ch.recent_y[ch.buf_pos] = buf_y[k];
            ch.buf_pos = (ch.buf_pos + 1) % LEAF_BUF_CAP;
            if (ch.buf_len < LEAF_BUF_CAP) ch.buf_len++;
        }

        if (witness_t0 != UNSET_T) {
            uint8_t side = tap_bit(bs, witness_t0, best_lag);
            AutopoieticNode& ch = nodes[side == 0 ? left_idx : right_idx];
            if (ch.first_t[0] == UNSET_T) ch.first_t[0] = witness_t0;
            if (ch.count[0] == 0) ch.count[0] = 1;
        }
        if (witness_t1 != UNSET_T) {
            uint8_t side = tap_bit(bs, witness_t1, best_lag);
            AutopoieticNode& ch = nodes[side == 0 ? left_idx : right_idx];
            if (ch.first_t[1] == UNSET_T) ch.first_t[1] = witness_t1;
            if (ch.count[1] == 0) ch.count[1] = 1;
        }

        total_splits++;
    }
};

// ============================================================================
// DETERMINISTIC PHYSICAL BENCHMARK GENERATORS
// ============================================================================
static BitStream generate_lfsr16_stream(size_t num_bits) {
    std::vector<uint8_t> bits(num_bits);
    uint16_t state = 0xACE1U;
    for (size_t i = 0; i < num_bits; ++i) {
        uint8_t out = state & 1U;
        bits[i] = out;
        state >>= 1;
        if (out) state ^= 0xB400U;
    }
    return pack_bits_to_stream(std::move(bits));
}

static BitStream generate_rule110_stream(size_t num_bits, size_t width = 64) {
    std::vector<uint8_t> bits(num_bits);
    std::vector<uint8_t> row(width, 0), next_row(width, 0);
    row[width / 2] = 1;
    row[width / 3] = 1;
    size_t idx = 0;
    while (idx < num_bits) {
        for (size_t x = 0; x < width && idx < num_bits; ++x) bits[idx++] = row[x];
        for (size_t x = 0; x < width; ++x) {
            uint8_t l = row[(x + width - 1) % width];
            uint8_t c = row[x];
            uint8_t r = row[(x + 1) % width];
            uint8_t pat = (l << 2) | (c << 1) | r;
            next_row[x] = (110 >> pat) & 1U;
        }
        row.swap(next_row);
    }
    return pack_bits_to_stream(std::move(bits));
}

// ============================================================================
// STRICT SINGLE-PASS RUNNER (ZERO EPOCHS)
// ============================================================================
static void run_single_pass(const std::string& name, const BitStream& bs, size_t split_threshold) {
    if (bs.bits.size() < 1024) return;
    SinglePassAutopoieticSubstrate sub;
    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t total_err = 0;
    uint64_t half_err = 0;
    uint64_t tail_err = 0;
    size_t half = bs.bits.size() / 2;
    size_t tail = (bs.bits.size() * 3) / 4;

    size_t zero_error_start = bs.bits.size();
    uint64_t consecutive_correct = 0;

    for (size_t t = 0; t < bs.bits.size(); ++t) {
        uint8_t ok = sub.step_single_pass(bs, t, split_threshold);
        if (!ok) {
            total_err++;
            if (t >= half) half_err++;
            if (t >= tail) tail_err++;
            consecutive_correct = 0;
            zero_error_start = bs.bits.size();
        } else {
            if (consecutive_correct == 0) zero_error_start = t;
            consecutive_correct++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb_s = (bs.bits.size() / 8.0 / (1024.0 * 1024.0)) / (ms / 1000.0);

    double full_acc = 100.0 * (1.0 - static_cast<double>(total_err) / bs.bits.size());
    double half_acc = 100.0 * (1.0 - static_cast<double>(half_err) / (bs.bits.size() - half));
    double tail_acc = 100.0 * (1.0 - static_cast<double>(tail_err) / (bs.bits.size() - tail));
    double tail_F = compute_free_energy(tail_err, bs.bits.size() - tail);

    std::cout << "\n[SINGLE-PASS ONLINE] " << name << " (" << bs.bits.size() << " bits)" << std::endl;
    std::cout << "  Full 1-Pass Acc   : " << std::fixed << std::setprecision(2) << full_acc
              << "% | 2nd Half: " << half_acc << "% | Tail (last 25%): "
              << std::setprecision(4) << tail_acc << "% (F = " << std::setprecision(8)
              << tail_F << " nats, tail err = " << tail_err << ")" << std::endl;
    if (consecutive_correct > 1000) {
        std::cout << "  Zero-Error Lock   : bit t = " << zero_error_start
                  << " (" << consecutive_correct << " consecutive 0-error bits!)" << std::endl;
    }
    std::cout << "  Living Topology   : " << sub.nodes.size() << " nodes (" << sub.total_splits
              << " splits, max lag " << sub.max_lag_discovered << ", max depth "
              << sub.max_depth_reached << ") | " << std::setprecision(2) << ms
              << " ms (" << mb_s << " MB/s)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1156: PURE SINGLE-PASS AUTOPOIETIC V=2 SUBSTRATE" << std::endl;
    std::cout << " ZERO Epochs | ZERO Multi-Pass | ZERO GA | ZERO Separate Hash Table" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 3, 4, 8, 9, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    run_single_pass("DETERMINISTIC: 16-bit Galois LFSR", generate_lfsr16_stream(200000), 4);
    run_single_pass("DETERMINISTIC: Turing-Complete Rule 110 CA", generate_rule110_stream(200000, 64), 4);

    auto kep_bs = load_file_bits("KEP.md");
    if (!kep_bs.bits.empty()) run_single_pass("TEXT: KEP.md", kep_bs, 6);

    auto shake_bs = load_file_bits("tinyshakespeare.txt", 262144);
    if (!shake_bs.bits.empty()) run_single_pass("LITERATURE: tinyshakespeare.txt (256KB)", shake_bs, 6);

    auto web_bs = load_file_bits("data/real_web_stream.txt", 262144);
    if (!web_bs.bits.empty()) run_single_pass("WEB STREAM: real_web_stream.txt (256KB)", web_bs, 6);

    auto png_bs = load_file_bits("experiments/renders/bloodbath_precise_v4.png", 131072);
    if (!png_bs.bits.empty()) run_single_pass("IMAGE/PNG: bloodbath_precise_v4.png (128KB)", png_bs, 6);

    if (argc > 0 && argv[0]) {
        auto elf_bs = load_file_bits(argv[0], 131072);
        if (!elf_bs.bits.empty()) run_single_pass("BINARY/ELF: x86_64 Machine Code (128KB)", elf_bs, 6);
    }

    return 0;
}
