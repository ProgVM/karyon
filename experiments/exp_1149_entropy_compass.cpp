// ============================================================================
// KARYON-CORE EXP-1149: ENTROPY COMPASS — INFORMATION-THEORETIC DIAGNOSTIC
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 1, 4 (V=2), 6, 8, 13, 14, 15
// ============================================================================
// Purpose (Bazilevs' Directive):
//   Use Shannon Entropy H(Y | X) as a physical compass to discover WHERE the
//   uncaptured predictability in real bitstreams actually lives, and WHY our
//   current word-parallel Q_SWAR + D + S substrate hits a ~69-70% ceiling.
//
// Measures exact conditional entropy H(b_t | Context) on UNSEEN 50% tail across:
//   Axis 1: Marginal Bit Entropy H(b_t)
//   Axis 2: Best Single-Lag Linear MI: min_k H(b_t | b_{t-k})
//   Axis 3: Bit-Phase / Position Entropy: H(b_t | t mod 8) & H(b_t | b_{t-k}, t mod 8)
//           -> Tests if mixing all 8 bit-positions in 64-bit SWAR destroys signal!
//   Axis 4: Non-Linear Joint Context Entropy: H(b_t | last K bits) for K=4,8,12,16
//           -> Quantifies information locked in higher-order conjunctions vs linear waves.
//   Axis 5: Long-Range Horizon Entropy: best lag in [257..8192] bits.
//   Axis 6: Autoregressive Error Entropy H(e_t | ...) of Q_SWAR residuals.
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <array>

static inline double binary_H(double p1) {
    if (p1 <= 0.0 || p1 >= 1.0) return 0.0;
    return -p1 * std::log2(p1) - (1.0 - p1) * std::log2(1.0 - p1);
}

static std::vector<uint8_t> load_bytes(const std::string& path, size_t max_bytes = 0) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    size_t sz = static_cast<size_t>(ifs.tellg());
    if (max_bytes > 0 && sz > max_bytes) sz = max_bytes;
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    ifs.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static inline uint8_t get_bit(const std::vector<uint8_t>& bytes, size_t bit_idx) {
    return (bytes[bit_idx >> 3] >> (bit_idx & 7)) & 1U;
}

struct EntropyResult {
    double H_cond;      // Conditional Shannon entropy in bits/bit [0..1]
    double bayes_acc;   // Bayes-optimal accuracy (%) achievable from this context on unseen half
};

// Evaluates H(b_t | context) and unseen Bayes-optimal accuracy using train (first 50%) -> test (second 50%)
static EntropyResult eval_context_table(const std::vector<uint8_t>& bytes,
                                        size_t start_bit, size_t split_bit, size_t total_bits,
                                        size_t num_states,
                                        const std::vector<uint32_t>& ctx_train,
                                        const std::vector<uint32_t>& ctx_test) {
    std::vector<uint32_t> c0(num_states, 0), c1(num_states, 0);
    for (size_t idx = 0; idx < ctx_train.size(); ++idx) {
        size_t t = start_bit + idx;
        uint8_t b = get_bit(bytes, t);
        if (b) c1[ctx_train[idx]]++;
        else   c0[ctx_train[idx]]++;
    }

    // Measure unseen test accuracy + test conditional entropy
    std::vector<uint32_t> t0(num_states, 0), t1(num_states, 0);
    uint64_t correct = 0;
    uint64_t count = ctx_test.size();

    for (size_t idx = 0; idx < ctx_test.size(); ++idx) {
        size_t t = split_bit + idx;
        uint8_t b = get_bit(bytes, t);
        uint32_t s = ctx_test[idx];
        uint8_t pred = (c1[s] >= c0[s]) ? 1 : 0;
        if (pred == b) correct++;
        if (b) t1[s]++;
        else   t0[s]++;
    }

    double H = 0.0;
    for (size_t s = 0; s < num_states; ++s) {
        uint64_t n = t0[s] + t1[s];
        if (n == 0) continue;
        double p_s = static_cast<double>(n) / count;
        double p1  = static_cast<double>(t1[s]) / n;
        H += p_s * binary_H(p1);
    }

    return {H, 100.0 * static_cast<double>(correct) / count};
}

static void run_entropy_compass(const std::string& name, const std::vector<uint8_t>& bytes) {
    size_t total_bits = bytes.size() * 8;
    size_t start_bit = 8192;
    if (total_bits <= start_bit + 2048) return;
    size_t split_bit = total_bits / 2;
    if (split_bit <= start_bit) return;

    size_t n_train = split_bit - start_bit;
    size_t n_test  = total_bits - split_bit;

    // 1. Marginal Entropy H(b_t)
    uint64_t tr_ones = 0, te_ones = 0;
    for (size_t t = start_bit; t < split_bit; ++t) tr_ones += get_bit(bytes, t);
    for (size_t t = split_bit; t < total_bits; ++t) te_ones += get_bit(bytes, t);
    double H_marg = binary_H(static_cast<double>(te_ones) / n_test);
    uint8_t marg_pred = (tr_ones * 2 >= n_train) ? 1 : 0;
    double marg_acc = 100.0 * (marg_pred ? te_ones : (n_test - te_ones)) / n_test;

    // 2. Best Single-Lag Entropy (Shared across all bit positions) in [1..256]
    double best_lag_H = 1.0;
    double best_lag_acc = 50.0;
    size_t best_lag = 1;
    for (size_t lag = 1; lag <= 256; ++lag) {
        uint64_t c00 = 0, c01 = 0, c10 = 0, c11 = 0;
        for (size_t t = start_bit; t < split_bit; ++t) {
            uint8_t x = get_bit(bytes, t - lag);
            uint8_t y = get_bit(bytes, t);
            if (x == 0 && y == 0) c00++;
            else if (x == 0 && y == 1) c01++;
            else if (x == 1 && y == 0) c10++;
            else c11++;
        }
        uint8_t p_x0 = (c01 >= c00) ? 1 : 0;
        uint8_t p_x1 = (c11 >= c10) ? 1 : 0;

        uint64_t t00 = 0, t01 = 0, t10 = 0, t11 = 0, corr = 0;
        for (size_t t = split_bit; t < total_bits; ++t) {
            uint8_t x = get_bit(bytes, t - lag);
            uint8_t y = get_bit(bytes, t);
            if (x == 0) { if (y == p_x0) corr++; if (y) t01++; else t00++; }
            else        { if (y == p_x1) corr++; if (y) t11++; else t10++; }
        }
        double n0 = t00 + t01, n1 = t10 + t11;
        double H = (n0 / n_test) * binary_H(n0 > 0 ? t01 / n0 : 0.5)
                 + (n1 / n_test) * binary_H(n1 > 0 ? t11 / n1 : 0.5);
        if (H < best_lag_H) {
            best_lag_H = H;
            best_lag_acc = 100.0 * corr / n_test;
            best_lag = lag;
        }
    }

    // 3. Bit-Position (t mod 8) Entropy — Does phase within byte break symmetry?
    std::vector<uint32_t> tr_pos(n_train), te_pos(n_test);
    for (size_t i = 0; i < n_train; ++i) tr_pos[i] = (start_bit + i) & 7U;
    for (size_t i = 0; i < n_test; ++i)  te_pos[i] = (split_bit + i) & 7U;
    auto res_pos = eval_context_table(bytes, start_bit, split_bit, total_bits, 8, tr_pos, te_pos);

    // 3b. Bit-Position + Best Lag (t mod 8, b_{t - best_lag})
    std::vector<uint32_t> tr_pos_lag(n_train), te_pos_lag(n_test);
    for (size_t i = 0; i < n_train; ++i) {
        size_t t = start_bit + i;
        tr_pos_lag[i] = ((t & 7U) << 1) | get_bit(bytes, t - best_lag);
    }
    for (size_t i = 0; i < n_test; ++i) {
        size_t t = split_bit + i;
        te_pos_lag[i] = ((t & 7U) << 1) | get_bit(bytes, t - best_lag);
    }
    auto res_pos_lag = eval_context_table(bytes, start_bit, split_bit, total_bits, 16, tr_pos_lag, te_pos_lag);

    // 4. Non-Linear Joint Bit-Context Entropy: last K contiguous bits (K = 4, 8, 12, 16)
    auto eval_joint_k = [&](int K) {
        size_t states = 1ULL << K;
        uint32_t mask = static_cast<uint32_t>(states - 1);
        std::vector<uint32_t> tr(n_train), te(n_test);
        uint32_t ctx = 0;
        for (size_t t = start_bit - K; t < start_bit; ++t) ctx = ((ctx << 1) | get_bit(bytes, t)) & mask;
        for (size_t i = 0; i < n_train; ++i) {
            tr[i] = ctx;
            ctx = ((ctx << 1) | get_bit(bytes, start_bit + i)) & mask;
        }
        for (size_t i = 0; i < n_test; ++i) {
            te[i] = ctx;
            ctx = ((ctx << 1) | get_bit(bytes, split_bit + i)) & mask;
        }
        return eval_context_table(bytes, start_bit, split_bit, total_bits, states, tr, te);
    };
    auto res_k4  = eval_joint_k(4);
    auto res_k8  = eval_joint_k(8);
    auto res_k12 = eval_joint_k(12);
    auto res_k16 = eval_joint_k(16);

    // 4b. Byte-Aligned + Joint Context (t mod 8 + previous 1 byte = 11 bits -> 2048 states)
    std::vector<uint32_t> tr_pb(n_train), te_pb(n_test);
    for (size_t i = 0; i < n_train; ++i) {
        size_t t = start_bit + i;
        uint32_t prev_byte = bytes[(t >> 3) - 1];
        uint32_t intra_bits = bytes[t >> 3] & ((1U << (t & 7U)) - 1U);
        tr_pb[i] = ((t & 7U) << 8) | prev_byte;
        (void)intra_bits;
    }
    for (size_t i = 0; i < n_test; ++i) {
        size_t t = split_bit + i;
        uint32_t prev_byte = bytes[(t >> 3) - 1];
        te_pb[i] = ((t & 7U) << 8) | prev_byte;
    }
    auto res_pos_prevbyte = eval_context_table(bytes, start_bit, split_bit, total_bits, 2048, tr_pb, te_pb);

    // 5. Long-Range Horizon Scan (sampled lags in [257..8192])
    double best_long_H = 1.0;
    double best_long_acc = 50.0;
    size_t best_long_lag = 0;
    for (size_t lag = 264; lag <= 8192; lag += 8) {
        uint64_t c00 = 0, c01 = 0, c10 = 0, c11 = 0;
        for (size_t t = start_bit; t < split_bit; t += 4) {
            uint8_t x = get_bit(bytes, t - lag);
            uint8_t y = get_bit(bytes, t);
            if (x == 0 && y == 0) c00++;
            else if (x == 0 && y == 1) c01++;
            else if (x == 1 && y == 0) c10++;
            else c11++;
        }
        uint8_t p_x0 = (c01 >= c00) ? 1 : 0;
        uint8_t p_x1 = (c11 >= c10) ? 1 : 0;
        uint64_t t00 = 0, t01 = 0, t10 = 0, t11 = 0, corr = 0, cnt = 0;
        for (size_t t = split_bit; t < total_bits; t += 4) {
            uint8_t x = get_bit(bytes, t - lag);
            uint8_t y = get_bit(bytes, t);
            cnt++;
            if (x == 0) { if (y == p_x0) corr++; if (y) t01++; else t00++; }
            else        { if (y == p_x1) corr++; if (y) t11++; else t10++; }
        }
        double n0 = t00 + t01, n1 = t10 + t11;
        double H = (n0 / cnt) * binary_H(n0 > 0 ? t01 / n0 : 0.5)
                 + (n1 / cnt) * binary_H(n1 > 0 ? t11 / n1 : 0.5);
        if (H < best_long_H) {
            best_long_H = H;
            best_long_acc = 100.0 * corr / cnt;
            best_long_lag = lag;
        }
    }

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "[ENTROPY COMPASS] " << name << " (" << total_bits << " bits)" << std::endl;
    std::cout << "  1. Marginal H(b_t)                   : H = " << std::fixed << std::setprecision(4) << H_marg
              << " bits | Unseen Acc = " << std::setprecision(2) << marg_acc << "%" << std::endl;
    std::cout << "  2. Best Single-Lag [1..256] (lag=" << std::setw(3) << best_lag << "): H = " << std::setprecision(4) << best_lag_H
              << " bits | Unseen Acc = " << std::setprecision(2) << best_lag_acc << "%" << std::endl;
    std::cout << "  3. Bit-Position ONLY (t mod 8)       : H = " << std::setprecision(4) << res_pos.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_pos.bayes_acc << "%" << std::endl;
    std::cout << "  3b. Bit-Pos + Best Lag (t%8, lag=" << std::setw(3) << best_lag << "): H = " << std::setprecision(4) << res_pos_lag.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_pos_lag.bayes_acc << "%" << std::endl;
    std::cout << "  4a. Joint Context K=4  bits          : H = " << std::setprecision(4) << res_k4.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_k4.bayes_acc << "%" << std::endl;
    std::cout << "  4b. Joint Context K=8  bits          : H = " << std::setprecision(4) << res_k8.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_k8.bayes_acc << "%" << std::endl;
    std::cout << "  4c. Joint Context K=12 bits          : H = " << std::setprecision(4) << res_k12.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_k12.bayes_acc << "%" << std::endl;
    std::cout << "  4d. Joint Context K=16 bits          : H = " << std::setprecision(4) << res_k16.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_k16.bayes_acc << "%" << std::endl;
    std::cout << "  4e. Bit-Pos + Prev Byte (11b ctx)    : H = " << std::setprecision(4) << res_pos_prevbyte.H_cond
              << " bits | Unseen Acc = " << std::setprecision(2) << res_pos_prevbyte.bayes_acc << "%" << std::endl;
    std::cout << "  5. Best Long-Range [264..8192] (lag=" << best_long_lag << "): H = " << std::setprecision(4) << best_long_H
              << " bits | Unseen Acc = " << std::setprecision(2) << best_long_acc << "%" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1149: ENTROPY COMPASS (DIRECTIVE DISCOVERY ENGINE)" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 4 (V=2), 6, 8, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    auto kep = load_bytes("KEP.md");
    if (!kep.empty()) run_entropy_compass("TEXT: KEP.md", kep);

    auto shake = load_bytes("tinyshakespeare.txt", 262144);
    if (!shake.empty()) run_entropy_compass("TEXT: tinyshakespeare.txt (256KB)", shake);

    auto png = load_bytes("experiments/renders/bloodbath_precise_v4.png", 262144);
    if (!png.empty()) run_entropy_compass("IMAGE/PNG: bloodbath_precise_v4.png (256KB)", png);

    if (argc > 0 && argv[0]) {
        auto elf = load_bytes(argv[0]);
        if (!elf.empty()) run_entropy_compass("BINARY/ELF: x86_64 Machine Code", elf);
    }

    return 0;
}
