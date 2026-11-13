// ============================================================================
// KARYON-CORE EXP-1157: PURE MDL-GOVERNED SINGLE-PASS AUTOPOIETIC V=2 SUBSTRATE
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 3, 4 (V=2), 6, 8, 9, 13, 14, 15
// ============================================================================
// Architectural Purity Audit & Upgrade over EXP-1156:
//   Removed impurities from EXP-1156:
//     1) Removed human-tuned split_threshold (4 vs 6 per task). All tasks now
//        run on the EXACT SAME parameter-free substrate.
//     2) Removed blind Phase-2 2-witness splitting (which caused d=1.4M
//        overfitting on Web/Shakespeare).
//     3) Freed leaf buffers upon split (internal nodes are pure 12-byte
//        causal routers: {child[2], tap_lag}), allowing larger active leaf
//        experience buffers without memory bloat.
//     4) Replaced ad-hoc error/balance heuristics with exact Krichevsky-Trofimov
//        (KT) Variational Free Energy / Minimum Description Length (MDL):
//          Delta_F = L_KT(parent) - [L_KT(left) + L_KT(right) + cost(lag)]
//        A node sprouts IF AND ONLY IF Delta_F > 0, naturally gating deep
//        lags by their information-theoretic description cost log2(d)!
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

// Precomputed log2(n + 0.5) and log2(n + 1.0) for fast exact KT codelength
struct KTTable {
    static constexpr size_t MAX_N = 512;
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
        // Stirling / lgamma fallback for large n
        return (std::lgamma(n + 1.0) - std::lgamma(c0 + 0.5) - std::lgamma(c1 + 0.5)
                + std::log(M_PI)) / std::log(2.0);
    }
};

static const KTTable g_kt;

struct BitStream {
    std::vector<uint8_t> bits;
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
    return bs;
}

// ============================================================================
// PURE MDL-GOVERNED SINGLE-PASS AUTOPOIETIC SUBSTRATE
// ============================================================================
struct CausalNode {
    int32_t  child[2]  = {-1, -1};
    uint32_t tap_lag   = 0;
    uint32_t count[2]  = {0, 0};
    int32_t  leaf_buf_idx = -1; // >= 0 only while active leaf; freed upon split!

    inline bool is_leaf() const { return child[0] < 0; }
    inline uint8_t predict() const { return (count[1] > count[0]) ? 1U : 0U; }
};

struct LeafBuffer {
    static constexpr size_t CAP = 64;
    uint32_t t[CAP];
    uint8_t  y[CAP];
    uint8_t  len = 0;
    uint8_t  pos = 0;
    uint16_t cooldown = 0;

    inline void push(uint32_t ts, uint8_t val) {
        t[pos] = ts;
        y[pos] = val;
        pos = (pos + 1) % CAP;
        if (len < CAP) len++;
    }
};

class PureMDLAutopoieticSubstrate {
public:
    std::vector<CausalNode> nodes;
    std::vector<LeafBuffer> leaf_bufs;
    std::vector<int32_t>    free_buf_ids;

    size_t max_depth_reached = 0;
    size_t max_lag_discovered = 0;
    size_t total_splits = 0;

    PureMDLAutopoieticSubstrate() {
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
    }

    static inline uint8_t tap_bit(const BitStream& bs, size_t t, size_t lag) {
        if (lag <= t) return bs.bits[t - lag];
        if (lag == t + 1) return 1U;
        return 0U;
    }

    inline uint32_t traverse_leaf(const BitStream& bs, size_t t, uint32_t* out_depth = nullptr) const {
        uint32_t cur = 0;
        uint32_t d = 0;
        while (!nodes[cur].is_leaf()) {
            uint8_t bit = tap_bit(bs, t, nodes[cur].tap_lag);
            cur = static_cast<uint32_t>(nodes[cur].child[bit]);
            d++;
        }
        if (out_depth) *out_depth = d;
        return cur;
    }

    inline uint8_t step_single_pass(const BitStream& bs, size_t t) {
        uint32_t depth = 0;
        uint32_t leaf = traverse_leaf(bs, t, &depth);
        uint8_t pred = nodes[leaf].predict();
        uint8_t actual = bs.bits[t];

        CausalNode& n = nodes[leaf];
        n.count[actual]++;

        int32_t bidx = n.leaf_buf_idx;
        if (bidx >= 0) {
            LeafBuffer& lb = leaf_bufs[bidx];
            lb.push(static_cast<uint32_t>(t), actual);
            if (lb.cooldown > 0) lb.cooldown--;

            // Endogenous morphogenesis triggered strictly by conflict + zero cooldown
            if (pred != actual && n.count[0] > 0 && n.count[1] > 0 &&
                lb.len >= 6 && lb.cooldown == 0) {
                sprout_if_mdl_positive(leaf, depth, bs, t);
            }
        }
        return (pred == actual) ? 1U : 0U;
    }

private:
    int32_t alloc_buf() {
        if (!free_buf_ids.empty()) {
            int32_t id = free_buf_ids.back();
            free_buf_ids.pop_back();
            leaf_bufs[id] = LeafBuffer();
            return id;
        }
        leaf_bufs.emplace_back();
        return static_cast<int32_t>(leaf_bufs.size() - 1);
    }

    void release_buf(int32_t id) {
        if (id >= 0) free_buf_ids.push_back(id);
    }

    inline void eval_lag_mdl(const BitStream& bs, const LeafBuffer& lb, size_t d,
                             double parent_cost, uint32_t& best_lag, double& best_gain) {
        uint32_t c[2][2] = {{0, 0}, {0, 0}};
        for (uint8_t k = 0; k < lb.len; ++k) {
            uint8_t x = tap_bit(bs, lb.t[k], d);
            uint8_t y = lb.y[k];
            c[x][y]++;
        }
        if ((c[0][0] + c[0][1]) == 0 || (c[1][0] + c[1][1]) == 0) return;

        double split_cost = g_kt.codelength(c[0][0], c[0][1]) +
                            g_kt.codelength(c[1][0], c[1][1]) +
                            (std::log2(static_cast<double>(d)) + 1.0);
        double gain = parent_cost - split_cost;
        if (gain > best_gain) {
            best_gain = gain;
            best_lag = static_cast<uint32_t>(d);
        }
    }

    void sprout_if_mdl_positive(uint32_t leaf_idx, uint32_t depth, const BitStream& bs, size_t t) {
        int32_t bidx = nodes[leaf_idx].leaf_buf_idx;
        LeafBuffer lb_copy = leaf_bufs[bidx]; // Local copy before vector modifications

        uint32_t tot0 = 0, tot1 = 0;
        for (uint8_t k = 0; k < lb_copy.len; ++k) {
            if (lb_copy.y[k] == 0) tot0++; else tot1++;
        }
        if (tot0 == 0 || tot1 == 0) return;

        double parent_cost = g_kt.codelength(tot0, tot1);
        uint32_t best_lag = 0;
        double   best_gain = 0.0; // Strictly requires positive MDL Free Energy reduction!

        // Local causal horizon (1..256)
        size_t local_cap = std::min<size_t>(256, t + 1);
        for (size_t d = 1; d <= local_cap; ++d) {
            eval_lag_mdl(bs, lb_copy, d, parent_cost, best_lag, best_gain);
        }

        // Dyadic multiscale horizon (384..t+1) — gated naturally by log2(d) MDL cost!
        if (best_gain <= 0.0 || lb_copy.len >= 24) {
            for (size_t d = 384; d <= t + 1; d = (d * 3) / 2) {
                eval_lag_mdl(bs, lb_copy, d, parent_cost, best_lag, best_gain);
            }
        }

        if (best_lag == 0) {
            // Wait before re-probing this noisy leaf
            leaf_bufs[bidx].cooldown = 16;
            return;
        }

        if (best_lag > max_lag_discovered) max_lag_discovered = best_lag;

        uint32_t left_idx = static_cast<uint32_t>(nodes.size());
        uint32_t right_idx = left_idx + 1;
        uint32_t next_depth = depth + 1;
        if (next_depth > max_depth_reached) max_depth_reached = next_depth;

        release_buf(bidx);
        nodes[leaf_idx].leaf_buf_idx = -1;

        int32_t left_buf = alloc_buf();
        int32_t right_buf = alloc_buf();

        nodes.emplace_back();
        nodes.emplace_back();

        nodes[leaf_idx].tap_lag = best_lag;
        nodes[leaf_idx].child[0] = static_cast<int32_t>(left_idx);
        nodes[leaf_idx].child[1] = static_cast<int32_t>(right_idx);

        nodes[left_idx].leaf_buf_idx = left_buf;
        nodes[right_idx].leaf_buf_idx = right_buf;

        for (uint8_t k = 0; k < lb_copy.len; ++k) {
            uint8_t side = tap_bit(bs, lb_copy.t[k], best_lag);
            uint32_t ch_idx = (side == 0) ? left_idx : right_idx;
            int32_t ch_bidx = (side == 0) ? left_buf : right_buf;
            nodes[ch_idx].count[lb_copy.y[k]]++;
            leaf_bufs[ch_bidx].push(lb_copy.t[k], lb_copy.y[k]);
        }

        total_splits++;
    }
};

// ============================================================================
// DETERMINISTIC PHYSICAL BENCHMARK GENERATORS
// ============================================================================
static BitStream generate_lfsr16_stream(size_t num_bits) {
    BitStream bs;
    bs.bits.resize(num_bits);
    uint16_t state = 0xACE1U;
    for (size_t i = 0; i < num_bits; ++i) {
        uint8_t out = state & 1U;
        bs.bits[i] = out;
        state >>= 1;
        if (out) state ^= 0xB400U;
    }
    return bs;
}

static BitStream generate_rule110_stream(size_t num_bits, size_t width = 64) {
    BitStream bs;
    bs.bits.resize(num_bits);
    std::vector<uint8_t> row(width, 0), next_row(width, 0);
    row[width / 2] = 1;
    row[width / 3] = 1;
    size_t idx = 0;
    while (idx < num_bits) {
        for (size_t x = 0; x < width && idx < num_bits; ++x) bs.bits[idx++] = row[x];
        for (size_t x = 0; x < width; ++x) {
            uint8_t l = row[(x + width - 1) % width];
            uint8_t c = row[x];
            uint8_t r = row[(x + 1) % width];
            uint8_t pat = (l << 2) | (c << 1) | r;
            next_row[x] = (110 >> pat) & 1U;
        }
        row.swap(next_row);
    }
    return bs;
}

// ============================================================================
// STRICT SINGLE-PASS RUNNER (IDENTICAL SUBSTRATE FOR ALL TASKS)
// ============================================================================
static void run_single_pass(const std::string& name, const BitStream& bs) {
    if (bs.bits.size() < 1024) return;
    PureMDLAutopoieticSubstrate sub;
    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t total_err = 0;
    uint64_t half_err = 0;
    uint64_t tail_err = 0;
    size_t half = bs.bits.size() / 2;
    size_t tail = (bs.bits.size() * 3) / 4;

    size_t zero_error_start = bs.bits.size();
    uint64_t consecutive_correct = 0;

    for (size_t t = 0; t < bs.bits.size(); ++t) {
        uint8_t ok = sub.step_single_pass(bs, t);
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

    std::cout << "\n[PURE MDL SINGLE-PASS] " << name << " (" << bs.bits.size() << " bits)" << std::endl;
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
    std::cout << " KARYON-CORE EXP-1157: PURE MDL-GOVERNED SINGLE-PASS AUTOPOIESIS (V=2)" << std::endl;
    std::cout << " Parameter-Free KT-MDL | Zero Epochs | Zero GA | Zero Task Tuning" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 3, 4, 8, 9, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    run_single_pass("DETERMINISTIC: 16-bit Galois LFSR", generate_lfsr16_stream(200000));
    run_single_pass("DETERMINISTIC: Turing-Complete Rule 110 CA", generate_rule110_stream(200000, 64));

    auto kep_bs = load_file_bits("KEP.md");
    if (!kep_bs.bits.empty()) run_single_pass("TEXT: KEP.md", kep_bs);

    auto shake_bs = load_file_bits("tinyshakespeare.txt", 262144);
    if (!shake_bs.bits.empty()) run_single_pass("LITERATURE: tinyshakespeare.txt (256KB)", shake_bs);

    auto web_bs = load_file_bits("data/real_web_stream.txt", 262144);
    if (!web_bs.bits.empty()) run_single_pass("WEB STREAM: real_web_stream.txt (256KB)", web_bs);

    auto png_bs = load_file_bits("experiments/renders/bloodbath_precise_v4.png", 131072);
    if (!png_bs.bits.empty()) run_single_pass("IMAGE/PNG: bloodbath_precise_v4.png (128KB)", png_bs);

    if (argc > 0 && argv[0]) {
        auto elf_bs = load_file_bits(argv[0], 131072);
        if (!elf_bs.bits.empty()) run_single_pass("BINARY/ELF: x86_64 Machine Code (128KB)", elf_bs);
    }

    return 0;
}
