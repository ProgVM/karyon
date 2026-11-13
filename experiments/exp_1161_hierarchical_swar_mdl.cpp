// ============================================================================
// KARYON-CORE EXP-1161: HIERARCHICAL KT-PATH + ZERO-COPY 64-BIT SWAR + F2
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
// ============================================================================
// Evolution over EXP-1160:
//   1. Hierarchical Path-Backed KT Prediction: Every node along the causal
//      path tracks its own (count[0], count[1]). When a leaf splits, its
//      children no longer suffer cold-start amnesia (0/0): prediction blends
//      the leaf's local KT posterior with its ancestors' posterior along the
//      exact traversal path.
//   2. Early Dyadic Trigger (N >= 4): Restores immediate early morphogenesis
//      on small/structured streams (KEP.md, Rule 110) without any cooldowns.
//   3. Zero-Copy LeafHistory Probing: Eliminates per-probe vector copies.
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

static inline double compute_free_energy(uint64_t error_bits, uint64_t total_bits) {
    if (total_bits == 0 || error_bits == 0) return 0.0;
    double err_rate = static_cast<double>(error_bits) / static_cast<double>(total_bits);
    if (err_rate >= 1.0) err_rate = 1.0 - 1e-15;
    return -std::log(std::max(1.0 - err_rate, 1e-15));
}

struct KTTable {
    static constexpr size_t MAX_N = 1024;
    double lgamma_half[MAX_N + 1];
    double lgamma_one[MAX_N + 1];

    KTTable() {
        lgamma_half[0] = 0.0;
        lgamma_one[0]  = 0.0;
        for (size_t i = 0; i < MAX_N; ++i) {
            lgamma_half[i + 1] = lgamma_half[i] + std::log2(static_cast<double>(i) + 0.5);
            lgamma_one[i + 1]  = lgamma_one[i]  + std::log2(static_cast<double>(i) + 1.0);
        }
    }

    inline double codelength(uint32_t c0, uint32_t c1) const {
        uint32_t n = c0 + c1;
        if (n == 0) return 0.0;
        if (n <= MAX_N) {
            return lgamma_one[n] - lgamma_half[c0] - lgamma_half[c1];
        }
        return (std::lgamma(n + 1.0) - std::lgamma(c0 + 0.5) - std::lgamma(c1 + 0.5)
                + std::log(M_PI)) / std::log(2.0);
    }
};

static const KTTable g_kt;

struct BitStream {
    std::vector<uint64_t> words;
    size_t num_bits = 0;

    inline uint8_t get_bit(size_t idx) const {
        return static_cast<uint8_t>((words[idx >> 6] >> (idx & 63)) & 1ULL);
    }

    inline uint64_t get_lag_word64(size_t t, size_t lag_start) const {
        if (t >= lag_start + 63) {
            size_t lo_bit = t - (lag_start + 63);
            size_t wi = lo_bit >> 6;
            size_t bo = lo_bit & 63;
            return (bo == 0) ? words[wi] : ((words[wi] >> bo) | (words[wi + 1] << (64 - bo)));
        }
        uint64_t w = 0ULL;
        for (size_t j = 0; j < 64; ++j) {
            size_t lag = lag_start + 63 - j;
            uint8_t b = 0;
            if (lag <= t) b = get_bit(t - lag);
            else if (lag == t + 1) b = 1U;
            if (b) w |= (1ULL << j);
        }
        return w;
    }
};

static BitStream load_file_bits(const std::string& path, size_t max_bytes = 0) {
    BitStream bs;
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return bs;
    size_t file_size = static_cast<size_t>(ifs.tellg());
    if (max_bytes > 0 && file_size > max_bytes) file_size = max_bytes;
    ifs.seekg(0, std::ios::beg);

    size_t num_words = (file_size + 7) / 8 + 2;
    bs.words.assign(num_words, 0ULL);
    if (file_size > 0) {
        ifs.read(reinterpret_cast<char*>(bs.words.data()), file_size);
    }
    bs.num_bits = file_size * 8;
    return bs;
}

static inline void transpose_64x64(uint64_t A[64]) {
    int j, k;
    uint64_t m, t;
    m = 0x00000000FFFFFFFFULL;
    for (j = 32; j != 0; j >>= 1, m = m ^ (m << j)) {
        for (k = 0; k < 64; k = (k + j + 1) & ~j) {
            t = (A[k] ^ (A[k + j] >> j)) & m;
            A[k] = A[k] ^ t;
            A[k + j] = A[k + j] ^ (t << j);
        }
    }
}

struct CausalNode {
    int32_t  child[2]     = {-1, -1};
    int32_t  parent       = -1;
    uint32_t tap_lag      = 0;
    uint32_t count[2]     = {0, 0};
    uint64_t parity_mask  = 0ULL;
    int32_t  leaf_buf_idx = -1;

    inline bool is_leaf() const { return child[0] < 0; }
};

struct LeafHistory {
    std::vector<uint32_t> t;
    std::vector<uint8_t>  y;

    inline void clear() {
        std::vector<uint32_t>().swap(t);
        std::vector<uint8_t>().swap(y);
    }
    inline void push(uint32_t ts, uint8_t val) {
        if (t.size() >= 256) {
            std::memmove(t.data(), t.data() + 128, 128 * sizeof(uint32_t));
            std::memmove(y.data(), y.data() + 128, 128 * sizeof(uint8_t));
            t.resize(128);
            y.resize(128);
        }
        t.push_back(ts);
        y.push_back(val);
    }
    inline size_t size() const { return t.size(); }
};

class HierarchicalSWARMdlSubstrate {
public:
    std::vector<CausalNode>  nodes;
    std::vector<LeafHistory> leaf_bufs;
    std::vector<int32_t>     free_buf_ids;

    size_t max_depth_reached = 0;
    size_t max_lag_discovered = 0;
    size_t total_splits = 0;
    size_t total_f2_locks = 0;

    HierarchicalSWARMdlSubstrate() {
        reset();
    }

    void reset() {
        nodes.clear();
        leaf_bufs.clear();
        free_buf_ids.clear();

        nodes.emplace_back();
        nodes[0].leaf_buf_idx = alloc_buf();
        max_depth_reached = 0;
        max_lag_discovered = 0;
        total_splits = 0;
        total_f2_locks = 0;
    }

    static inline uint8_t tap_bit(const BitStream& bs, size_t t, size_t lag) {
        if (lag <= t) return bs.get_bit(t - lag);
        if (lag == t + 1) return 1U;
        return 0U;
    }

    static inline uint64_t fetch_window_64(const BitStream& bs, size_t t) {
        return bs.get_lag_word64(t, 1);
    }

    inline uint8_t step_single_pass(const BitStream& bs, size_t t) {
        // Traverse from root to leaf, recording path for hierarchical KT update
        uint32_t cur = 0;
        uint32_t depth = 0;
        uint8_t actual = bs.get_bit(t);

        while (!nodes[cur].is_leaf()) {
            nodes[cur].count[actual]++;
            uint8_t bit = tap_bit(bs, t, nodes[cur].tap_lag);
            cur = static_cast<uint32_t>(nodes[cur].child[bit]);
            depth++;
        }

        CausalNode& n = nodes[cur];
        uint8_t f2_pred = 0;
        if (n.parity_mask != 0ULL) {
            uint64_t w = fetch_window_64(bs, t);
            f2_pred = static_cast<uint8_t>(__builtin_popcountll(w & n.parity_mask) & 1);
        }

        // Hierarchical KT-backed prediction: if leaf has a strict majority, use it;
        // if tied (0/0 cold-start or equal counts), inherit from nearest decisive ancestor!
        uint8_t bias = 0;
        if (n.parity_mask != 0ULL) {
            bias = (n.count[1] > n.count[0]) ? 1U : 0U;
        } else {
            int32_t walker = static_cast<int32_t>(cur);
            while (walker >= 0) {
                if (nodes[walker].count[1] != nodes[walker].count[0]) {
                    bias = (nodes[walker].count[1] > nodes[walker].count[0]) ? 1U : 0U;
                    break;
                }
                walker = nodes[walker].parent;
            }
        }

        uint8_t pred = f2_pred ^ bias;
        uint8_t residual = actual ^ f2_pred;

        n.count[residual]++;
        uint32_t n_tot = n.count[0] + n.count[1];

        if (n.parity_mask != 0ULL && pred != actual && n.count[0] > 0 && n.count[1] > 0) {
            n.parity_mask = 0ULL;
            n.count[0] = 0;
            n.count[1] = 0;
            if (n.leaf_buf_idx < 0) n.leaf_buf_idx = alloc_buf();
        }

        int32_t bidx = n.leaf_buf_idx;
        if (bidx >= 0) {
            leaf_bufs[bidx].push(static_cast<uint32_t>(t), actual);

            // Dyadic information trigger starting at N >= 4
            if (n.count[0] > 0 && n.count[1] > 0 && n_tot >= 4 && (n_tot & (n_tot - 1)) == 0) {
                sprout_if_mdl_positive(cur, depth, bs, t);
            }
        }
        return (pred == actual) ? 1U : 0U;
    }

private:
    int32_t alloc_buf() {
        if (!free_buf_ids.empty()) {
            int32_t id = free_buf_ids.back();
            free_buf_ids.pop_back();
            leaf_bufs[id].clear();
            return id;
        }
        leaf_bufs.emplace_back();
        return static_cast<int32_t>(leaf_bufs.size() - 1);
    }

    void release_buf(int32_t id) {
        if (id >= 0) {
            leaf_bufs[id].clear();
            free_buf_ids.push_back(id);
        }
    }

    uint64_t solve_f2_parity(const BitStream& bs, const LeafHistory& lb) {
        size_t start_idx = 0;
        while (start_idx < lb.size() && lb.t[start_idx] < 64 && (lb.size() - start_idx) > 64) {
            start_idx++;
        }
        size_t len = lb.size() - start_idx;
        if (len < 64) return 0ULL;

        uint32_t tot0 = 0, tot1 = 0;
        for (size_t i = 0; i < len; ++i) {
            if (lb.y[start_idx + i]) tot1++; else tot0++;
        }
        double parent_cost = g_kt.codelength(tot0, tot1);

        size_t num_chunks = (len + 63) / 64;
        std::vector<uint64_t> cols(64 * num_chunks, 0ULL);
        std::vector<uint64_t> rhs(num_chunks, 0ULL);

        for (size_t i = 0; i < len; ++i) {
            size_t ch = i >> 6;
            size_t bit = i & 63;
            uint64_t w = fetch_window_64(bs, lb.t[start_idx + i]);
            for (int j = 0; j < 64; ++j) {
                if ((w >> j) & 1ULL) cols[j * num_chunks + ch] |= (1ULL << bit);
            }
            if (lb.y[start_idx + i]) rhs[ch] |= (1ULL << bit);
        }

        std::vector<std::vector<uint64_t>> basis_vec(len, std::vector<uint64_t>(num_chunks, 0ULL));
        std::vector<uint64_t> basis_mask(len, 0ULL);
        std::vector<uint8_t> has_basis(len, 0);

        std::vector<uint64_t> v(num_chunks);
        for (int j = 0; j < 64; ++j) {
            std::memcpy(v.data(), &cols[j * num_chunks], num_chunks * sizeof(uint64_t));
            uint64_t m = (1ULL << j);
            for (size_t ch = 0; ch < num_chunks; ++ch) {
                while (v[ch] != 0ULL) {
                    size_t piv = ch * 64 + static_cast<size_t>(__builtin_ctzll(v[ch]));
                    if (!has_basis[piv]) {
                        has_basis[piv] = 1;
                        basis_vec[piv] = v;
                        basis_mask[piv] = m;
                        ch = num_chunks;
                        break;
                    }
                    for (size_t c2 = ch; c2 < num_chunks; ++c2) v[c2] ^= basis_vec[piv][c2];
                    m ^= basis_mask[piv];
                }
            }
        }

        for (int flip = 0; flip < 2; ++flip) {
            std::vector<uint64_t> target = rhs;
            if (flip) {
                for (size_t i = 0; i < len; ++i) target[i >> 6] ^= (1ULL << (i & 63));
            }
            uint64_t sol_mask = 0ULL;
            bool solvable = true;
            for (size_t ch = 0; ch < num_chunks && solvable; ++ch) {
                while (target[ch] != 0ULL) {
                    size_t piv = ch * 64 + static_cast<size_t>(__builtin_ctzll(target[ch]));
                    if (!has_basis[piv]) {
                        solvable = false;
                        break;
                    }
                    for (size_t c2 = ch; c2 < num_chunks; ++c2) target[c2] ^= basis_vec[piv][c2];
                    sol_mask ^= basis_mask[piv];
                }
            }
            if (solvable && sol_mask != 0ULL) {
                int k_taps = __builtin_popcountll(sol_mask);
                double f2_cost = g_kt.codelength(static_cast<uint32_t>(len), 0) +
                                 static_cast<double>(k_taps) * 6.0 + 1.0;
                if (f2_cost < parent_cost) {
                    return sol_mask;
                }
            }
        }
        return 0ULL;
    }

    void sprout_if_mdl_positive(uint32_t leaf_idx, uint32_t depth, const BitStream& bs, size_t t) {
        int32_t bidx = nodes[leaf_idx].leaf_buf_idx;
        const LeafHistory& lb = leaf_bufs[bidx];
        size_t len = lb.size();

        uint32_t tot0 = 0, tot1 = 0;
        for (size_t i = 0; i < len; ++i) {
            if (lb.y[i]) tot1++; else tot0++;
        }
        if (tot0 == 0 || tot1 == 0) return;

        double parent_cost = g_kt.codelength(tot0, tot1);
        uint32_t best_lag = 0;
        double   best_gain = 0.0;

        size_t num_chunks = (len + 63) / 64;
        std::vector<uint64_t> y_masks(num_chunks, 0ULL);
        std::vector<uint64_t> y0_masks(num_chunks, 0ULL);
        for (size_t i = 0; i < len; ++i) {
            if (lb.y[i]) y_masks[i >> 6] |= (1ULL << (i & 63));
            else y0_masks[i >> 6] |= (1ULL << (i & 63));
        }

        size_t max_blk = (t + 1) / 64;
        for (size_t blk = 0; blk <= max_blk; ) {
            size_t lag_start = blk * 64 + 1;
            uint32_t c11_acc[64] = {0};
            uint32_t c10_acc[64] = {0};

            for (size_t ch = 0; ch < num_chunks; ++ch) {
                uint64_t mat[64] = {0ULL};
                size_t base = ch * 64;
                size_t chunk_len = std::min<size_t>(64, len - base);
                for (size_t k = 0; k < chunk_len; ++k) {
                    mat[63 - k] = bs.get_lag_word64(lb.t[base + k], lag_start);
                }
                transpose_64x64(mat);

                uint64_t ym = y_masks[ch];
                uint64_t y0m = y0_masks[ch];
                for (size_t b = 0; b < 64; ++b) {
                    uint64_t x_col = mat[63 - b];
                    c11_acc[b] += static_cast<uint32_t>(__builtin_popcountll(x_col & ym));
                    c10_acc[b] += static_cast<uint32_t>(__builtin_popcountll(x_col & y0m));
                }
            }

            for (size_t b = 0; b < 64; ++b) {
                size_t d = lag_start + 63 - b;
                if (d > t + 1) continue;

                uint32_t c11 = c11_acc[b];
                uint32_t c10 = c10_acc[b];
                uint32_t n1 = c11 + c10;
                uint32_t n0 = static_cast<uint32_t>(len) - n1;
                if (n0 == 0 || n1 == 0) continue;

                uint32_t c01 = tot1 - c11;
                uint32_t c00 = tot0 - c10;

                double split_cost = g_kt.codelength(c00, c01) +
                                    g_kt.codelength(c10, c11) +
                                    (std::log2(static_cast<double>(d)) + 1.0);
                double gain = parent_cost - split_cost;
                if (gain > best_gain || (gain == best_gain && best_gain > 0.0 && d < best_lag)) {
                    best_gain = gain;
                    best_lag = static_cast<uint32_t>(d);
                }
            }

            if (blk < 4) blk++;
            else blk *= 2;
        }

        if (best_lag == 0) {
            uint64_t f2_mask = solve_f2_parity(bs, lb);
            if (f2_mask != 0ULL) {
                nodes[leaf_idx].parity_mask = f2_mask;
                nodes[leaf_idx].count[0] = 0;
                nodes[leaf_idx].count[1] = 0;
                for (size_t i = 0; i < len; ++i) {
                    uint64_t w = fetch_window_64(bs, lb.t[i]);
                    uint8_t p = static_cast<uint8_t>(__builtin_popcountll(w & f2_mask) & 1);
                    nodes[leaf_idx].count[lb.y[i] ^ p]++;
                }
                release_buf(bidx);
                nodes[leaf_idx].leaf_buf_idx = -1;
                total_f2_locks++;
                return;
            }
            return;
        }

        if (best_lag > max_lag_discovered) max_lag_discovered = best_lag;

        uint32_t left_idx = static_cast<uint32_t>(nodes.size());
        uint32_t right_idx = left_idx + 1;
        uint32_t next_depth = depth + 1;
        if (next_depth > max_depth_reached) max_depth_reached = next_depth;

        // Allocate children buffers BEFORE touching lb (zero-copy until split!)
        int32_t left_buf = alloc_buf();
        int32_t right_buf = alloc_buf();

        nodes.emplace_back();
        nodes.emplace_back();

        nodes[leaf_idx].tap_lag = best_lag;
        nodes[leaf_idx].child[0] = static_cast<int32_t>(left_idx);
        nodes[leaf_idx].child[1] = static_cast<int32_t>(right_idx);
        nodes[leaf_idx].leaf_buf_idx = -1;

        nodes[left_idx].parent = static_cast<int32_t>(leaf_idx);
        nodes[right_idx].parent = static_cast<int32_t>(leaf_idx);
        nodes[left_idx].leaf_buf_idx = left_buf;
        nodes[right_idx].leaf_buf_idx = right_buf;

        const LeafHistory& lb_ref = leaf_bufs[bidx];
        for (size_t i = 0; i < len; ++i) {
            uint8_t side = tap_bit(bs, lb_ref.t[i], best_lag);
            uint32_t ch_idx = (side == 0) ? left_idx : right_idx;
            int32_t ch_bidx = (side == 0) ? left_buf : right_buf;
            nodes[ch_idx].count[lb_ref.y[i]]++;
            leaf_bufs[ch_bidx].push(lb_ref.t[i], lb_ref.y[i]);
        }

        release_buf(bidx);
        total_splits++;
    }
};

static BitStream pack_raw_bits(const std::vector<uint8_t>& bits) {
    BitStream bs;
    bs.num_bits = bits.size();
    bs.words.assign((bits.size() + 63) / 64 + 2, 0ULL);
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) bs.words[i >> 6] |= (1ULL << (i & 63));
    }
    return bs;
}

static BitStream generate_lfsr16_stream(size_t num_bits) {
    std::vector<uint8_t> bits(num_bits);
    uint16_t state = 0xACE1U;
    for (size_t i = 0; i < num_bits; ++i) {
        uint8_t out = state & 1U;
        bits[i] = out;
        state >>= 1;
        if (out) state ^= 0xB400U;
    }
    return pack_raw_bits(bits);
}

static BitStream generate_lfsr48_stream(size_t num_bits) {
    std::vector<uint8_t> bits(num_bits);
    uint64_t state = 0xDEADBEEFCAFEULL;
    for (size_t i = 0; i < num_bits; ++i) {
        uint8_t out = state & 1ULL;
        bits[i] = out;
        state >>= 1;
        if (out) state ^= 0xE00000000001ULL;
    }
    return pack_raw_bits(bits);
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
    return pack_raw_bits(bits);
}

static void run_single_pass(const std::string& name, const BitStream& bs) {
    if (bs.num_bits < 1024) return;
    HierarchicalSWARMdlSubstrate sub;
    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t total_err = 0;
    uint64_t half_err = 0;
    uint64_t tail_err = 0;
    size_t half = bs.num_bits / 2;
    size_t tail = (bs.num_bits * 3) / 4;

    size_t zero_error_start = bs.num_bits;
    uint64_t consecutive_correct = 0;

    for (size_t t = 0; t < bs.num_bits; ++t) {
        uint8_t ok = sub.step_single_pass(bs, t);
        if (!ok) {
            total_err++;
            if (t >= half) half_err++;
            if (t >= tail) tail_err++;
            consecutive_correct = 0;
            zero_error_start = bs.num_bits;
        } else {
            if (consecutive_correct == 0) zero_error_start = t;
            consecutive_correct++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb_s = (bs.num_bits / 8.0 / (1024.0 * 1024.0)) / (ms / 1000.0);

    double full_acc = 100.0 * (1.0 - static_cast<double>(total_err) / bs.num_bits);
    double half_acc = 100.0 * (1.0 - static_cast<double>(half_err) / (bs.num_bits - half));
    double tail_acc = 100.0 * (1.0 - static_cast<double>(tail_err) / (bs.num_bits - tail));
    double tail_F = compute_free_energy(tail_err, bs.num_bits - tail);

    std::cout << "\n[HIER-SWAR 1-PASS] " << name << " (" << bs.num_bits << " bits)" << std::endl;
    std::cout << "  Full 1-Pass Acc   : " << std::fixed << std::setprecision(2) << full_acc
              << "% | 2nd Half: " << half_acc << "% | Tail (last 25%): "
              << std::setprecision(4) << tail_acc << "% (F = " << std::setprecision(8)
              << tail_F << " nats, tail err = " << tail_err << ")" << std::endl;
    if (consecutive_correct > 1000) {
        std::cout << "  Zero-Error Lock   : bit t = " << zero_error_start
                  << " (" << consecutive_correct << " consecutive 0-error bits!)" << std::endl;
    }
    std::cout << "  Living Topology   : " << sub.nodes.size() << " nodes (" << sub.total_splits
              << " splits, " << sub.total_f2_locks << " F2 locks, max lag "
              << sub.max_lag_discovered << ", max depth " << sub.max_depth_reached << ") | "
              << std::setprecision(2) << ms << " ms (" << mb_s << " MB/s)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1161: HIERARCHICAL KT-PATH + ZERO-COPY SWAR + F2" << std::endl;
    std::cout << " Zero Cooldowns | Zero Cold-Start Amnesia | Early N>=4 Dyadic Probes" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 3, 4, 8, 9, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    run_single_pass("DETERMINISTIC: 16-bit Galois LFSR", generate_lfsr16_stream(200000));
    run_single_pass("DETERMINISTIC: 48-bit LFSR (beyond 32-bit!)", generate_lfsr48_stream(200000));
    run_single_pass("DETERMINISTIC: Turing-Complete Rule 110 CA", generate_rule110_stream(200000, 64));

    auto img_bs = load_file_bits("data/bloodbath_center_pixels.bin");
    if (img_bs.num_bits > 0) run_single_pass("RAW IMAGE: 512x128 RGBA Pixels (256KB)", img_bs);

    auto kep_bs = load_file_bits("KEP.md");
    if (kep_bs.num_bits > 0) run_single_pass("TEXT: KEP.md", kep_bs);

    auto shake_bs = load_file_bits("tinyshakespeare.txt");
    if (shake_bs.num_bits > 0) run_single_pass("LITERATURE: FULL tinyshakespeare.txt (1.1MB)", shake_bs);

    auto web_bs = load_file_bits("data/real_web_stream.txt");
    if (web_bs.num_bits > 0) run_single_pass("WEB STREAM: FULL real_web_stream.txt (4.0MB)", web_bs);

    if (argc > 0 && argv[0]) {
        auto elf_bs = load_file_bits(argv[0], 131072);
        if (elf_bs.num_bits > 0) run_single_pass("BINARY/ELF: x86_64 Machine Code (128KB)", elf_bs);
    }

    return 0;
}
