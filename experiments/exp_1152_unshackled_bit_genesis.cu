// ============================================================================
// KARYON-CORE EXP-1152: UNSHACKLED BIT-LEVEL GENESIS (ZERO 64-BIT BOUNDARIES)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 3, 4 (V=2), 8 (Unshackled Flow), 13, 14, 15
// ============================================================================
// Why EXP-1152?
//   In EXP-1151, 64-bit word-level operators (ROTL, ADD, SUB) introduced an
//   artificial 64-bit frame: either leaking future bits within a 64-bit word
//   when lag < 64, or forcing a blind 64-bit gap (words[i-1]) that blocked
//   access to immediate local history b_{t-1} .. b_{t-63}.
//
// Pure Unshackled V=2 Substrate:
//   - Every bit b_t (t = 0 .. N-1) is predicted strictly causally from its true
//     continuous bit history b_{t-1} .. b_{t-16384} with ZERO 64-bit boundaries.
//   - All ALU primitives are strictly lane-wise (bit-independent) so 64 bits are
//     evaluated in parallel on Tesla T4 without any cross-lane word artifacts:
//       LAG, NOT, XOR, AND, OR, MAJ3, CARRY_BIT, MUX, XNOR, NAND
//   - Also includes continuous phase oscillator source PHASE(period, offset)
//     for arbitrary period P in [2..128] (no hardcoded 8-bit or 64-bit boxes).
//   - Scale: 2,048 organisms x 80 generations = 163,840 DAGs per modality
//     (up to 12 instructions deep) across 5 multi-MB real-world streams.
// ============================================================================

#include <cuda_runtime.h>
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

static inline uint64_t rotl64_host(uint64_t x, int k) {
    k &= 63;
    return k == 0 ? x : ((x << k) | (x >> (64 - k)));
}

static inline uint64_t splitmix64_host(uint64_t x) {
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
// STRICTLY LANE-WISE BIT-LEVEL DAG GENOME (UP TO 12 INSTRUCTIONS)
// ============================================================================
enum BitOp : uint8_t {
    BOP_LAG   = 0,  // a
    BOP_NOT   = 1,  // ~a
    BOP_XOR   = 2,  // a ^ b
    BOP_AND   = 3,  // a & b
    BOP_OR    = 4,  // a | b
    BOP_NAND  = 5,  // ~(a & b)
    BOP_XNOR  = 6,  // ~(a ^ b)
    BOP_MAJ3  = 7,  // (a & b) | (a & c) | (b & c)
    BOP_CARRY = 8,  // (a & b) | ((a ^ b) & c)
    BOP_MUX   = 9,  // (a & b) | (~a & c)
    BOP_COUNT = 10
};

struct SrcSpec {
    uint8_t  mode;    // 0 = bit-lag [1..16384], 1 = register R[id], 2 = phase oscillator(period, offset)
    uint8_t  reg_id;
    uint16_t val;     // lag in [1..16384] or packed (period, offset)
};

struct Instruction {
    BitOp   op;
    SrcSpec s0, s1, s2;
};

static constexpr int NUM_REGS = 12;
static constexpr int MAX_LAG_BITS = 16384;

struct Organism {
    Instruction inst[NUM_REGS];
    uint8_t active_len; // 1 .. NUM_REGS
    uint8_t out_pol;    // 0 or 1
};

__host__ __device__ static inline uint64_t fetch_causal_bit_lag(const uint64_t* __restrict__ words, size_t i, uint16_t lag) {
    // lag >= 1 strictly!
    // Bit k in output word corresponds to stream bit t = 64*i + k, fetching bit (t - lag).
    // Zero 64-bit word boundary artifacts; strictly causal for all lag >= 1.
    if (lag < 1) lag = 1;
    size_t ws = lag >> 6;
    size_t bs = lag & 63;
    if (bs == 0) return words[i - ws];
    return (words[i - ws] << bs) | (words[i - ws - 1] >> (64 - bs));
}

__host__ __device__ static inline uint64_t eval_phase_oscillator(size_t i, uint16_t code) {
    uint32_t period = (code & 127U) + 2U;     // 2..129
    uint32_t offset = (code >> 7) & 127U;     // 0..127
    uint32_t mode   = (code >> 14) & 1U;      // 0: impulse (t+off)%P==0, 1: square wave
    uint64_t base_t = static_cast<uint64_t>(i) * 64ULL;
    uint64_t m = 0ULL;
    if (mode == 0) {
        uint32_t rem = static_cast<uint32_t>((base_t + offset) % period);
        uint32_t first = (rem == 0) ? 0 : (period - rem);
        for (uint32_t b = first; b < 64U; b += period) m |= (1ULL << b);
    } else {
        uint32_t half = period >> 1;
        for (uint32_t b = 0; b < 64U; ++b) {
            if (((base_t + b + offset) % period) < half) m |= (1ULL << b);
        }
    }
    return m;
}

__host__ __device__ static inline uint64_t eval_src(const SrcSpec& s, int k, const uint64_t regs[NUM_REGS],
                                                    const uint64_t* __restrict__ words, size_t i) {
    if (s.mode == 1 && k > 0) return regs[s.reg_id % k];
    if (s.mode == 2) return eval_phase_oscillator(i, s.val);
    return fetch_causal_bit_lag(words, i, s.val);
}

__host__ __device__ static inline uint64_t eval_organism(const Organism& org, const uint64_t* __restrict__ words, size_t i) {
    uint64_t regs[NUM_REGS];
    int len = org.active_len;
    if (len < 1) len = 1;
    if (len > NUM_REGS) len = NUM_REGS;

    for (int k = 0; k < len; ++k) {
        const Instruction& in = org.inst[k];
        uint64_t a = eval_src(in.s0, k, regs, words, i);
        uint64_t b = 0ULL, c = 0ULL;
        if (in.op >= BOP_XOR)  b = eval_src(in.s1, k, regs, words, i);
        if (in.op >= BOP_MAJ3) c = eval_src(in.s2, k, regs, words, i);

        uint64_t r = a;
        switch (in.op) {
            case BOP_LAG:   r = a; break;
            case BOP_NOT:   r = ~a; break;
            case BOP_XOR:   r = a ^ b; break;
            case BOP_AND:   r = a & b; break;
            case BOP_OR:    r = a | b; break;
            case BOP_NAND:  r = ~(a & b); break;
            case BOP_XNOR:  r = ~(a ^ b); break;
            case BOP_MAJ3:  r = (a & b) | (a & c) | (b & c); break;
            case BOP_CARRY: r = (a & b) | ((a ^ b) & c); break;
            case BOP_MUX:   r = (a & b) | ((~a) & c); break;
            default:        r = a; break;
        }
        regs[k] = r;
    }
    uint64_t out = regs[len - 1];
    return org.out_pol ? ~out : out;
}

__global__ void eval_population_kernel(const uint64_t* __restrict__ words,
                                       size_t start_i, size_t end_i, size_t stride,
                                       const Organism* __restrict__ pop,
                                       uint64_t* __restrict__ out_errors) {
    __shared__ Organism s_org;
    __shared__ uint64_t s_err[128];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) s_org = pop[bid];
    __syncthreads();

    uint64_t local_err = 0ULL;
    for (size_t i = start_i + tid * stride; i < end_i; i += blockDim.x * stride) {
        uint64_t pred = eval_organism(s_org, words, i);
        local_err += __popcll(pred ^ words[i]);
    }
    s_err[tid] = local_err;
    __syncthreads();

    for (int offset = 64; offset > 0; offset >>= 1) {
        if (tid < offset) s_err[tid] += s_err[tid + offset];
        __syncthreads();
    }
    if (tid == 0) out_errors[bid] = s_err[0];
}

// ============================================================================
// HOST FORMULA PRINTER & EVOLUTION OPERATORS
// ============================================================================
static std::string src_to_str(const SrcSpec& s, int k) {
    if (s.mode == 1 && k > 0) return "R" + std::to_string(s.reg_id % k);
    if (s.mode == 2) {
        uint32_t p = (s.val & 127U) + 2U;
        uint32_t o = (s.val >> 7) & 127U;
        uint32_t m = (s.val >> 14) & 1U;
        return (m ? "SQ(" : "OSC(") + std::to_string(p) + "," + std::to_string(o) + ")";
    }
    return "b_{t-" + std::to_string(s.val) + "}";
}

static std::string org_to_str(const Organism& org) {
    std::string out = "";
    int len = std::max(1, std::min<int>(org.active_len, NUM_REGS));
    for (int k = 0; k < len; ++k) {
        if (k > 0) out += "; ";
        const auto& in = org.inst[k];
        std::string a = src_to_str(in.s0, k);
        std::string b = src_to_str(in.s1, k);
        std::string c = src_to_str(in.s2, k);
        out += "R" + std::to_string(k) + "=";
        switch (in.op) {
            case BOP_LAG:   out += a; break;
            case BOP_NOT:   out += "~" + a; break;
            case BOP_XOR:   out += "(" + a + "^" + b + ")"; break;
            case BOP_AND:   out += "(" + a + "&" + b + ")"; break;
            case BOP_OR:    out += "(" + a + "|" + b + ")"; break;
            case BOP_NAND:  out += "~(" + a + "&" + b + ")"; break;
            case BOP_XNOR:  out += "~(" + a + "^" + b + ")"; break;
            case BOP_MAJ3:  out += "MAJ(" + a + "," + b + "," + c + ")"; break;
            case BOP_CARRY: out += "CARRY(" + a + "," + b + "," + c + ")"; break;
            case BOP_MUX:   out += "MUX(" + a + "?" + b + ":" + c + ")"; break;
            default: break;
        }
    }
    if (org.out_pol) out += " [~OUT]";
    return out;
}

static void random_src(SrcSpec& s, int k, uint64_t& rng) {
    rng = splitmix64_host(rng);
    uint64_t r = rng % 100;
    if (k > 0 && r < 40) {
        s.mode = 1;
        s.reg_id = static_cast<uint8_t>((rng >> 8) % k);
        s.val = 1;
    } else if (r < 52) {
        s.mode = 2;
        s.reg_id = 0;
        s.val = static_cast<uint16_t>((rng >> 12) & 0x7FFFU);
    } else {
        s.mode = 0;
        s.reg_id = 0;
        uint64_t r2 = splitmix64_host(rng ^ 0xA5A5A5A5ULL);
        uint64_t bucket = r2 % 3;
        if (bucket == 0) s.val = static_cast<uint16_t>(1 + (r2 >> 4) % 32);
        else if (bucket == 1) s.val = static_cast<uint16_t>(1 + (r2 >> 4) % 512);
        else s.val = static_cast<uint16_t>(1 + (r2 >> 4) % MAX_LAG_BITS);
    }
}

static Organism make_random_organism(uint64_t& rng, int max_len) {
    Organism org{};
    rng = splitmix64_host(rng);
    org.active_len = static_cast<uint8_t>(1 + (rng % max_len));
    org.out_pol = (rng >> 8) & 1;
    for (int k = 0; k < NUM_REGS; ++k) {
        rng = splitmix64_host(rng);
        org.inst[k].op = static_cast<BitOp>(rng % BOP_COUNT);
        random_src(org.inst[k].s0, k, rng);
        random_src(org.inst[k].s1, k, rng);
        random_src(org.inst[k].s2, k, rng);
    }
    return org;
}

static void mutate_organism(Organism& org, uint64_t& rng) {
    rng = splitmix64_host(rng);
    int num_edits = 1 + (rng & 1ULL);
    for (int e = 0; e < num_edits; ++e) {
        rng = splitmix64_host(rng);
        uint64_t choice = rng % 10;
        int k = static_cast<int>((rng >> 8) % std::max<int>(1, org.active_len));
        if (choice == 0) {
            org.active_len = static_cast<uint8_t>(1 + ((rng >> 16) % NUM_REGS));
        } else if (choice == 1) {
            org.out_pol ^= 1;
        } else if (choice <= 3) {
            org.inst[k].op = static_cast<BitOp>((rng >> 16) % BOP_COUNT);
        } else {
            int slot = static_cast<int>((rng >> 16) % 3);
            SrcSpec& s = (slot == 0) ? org.inst[k].s0 : ((slot == 1) ? org.inst[k].s1 : org.inst[k].s2);
            if (s.mode == 0 && (rng & 32ULL) == 0) {
                int delta = static_cast<int>((rng >> 20) % 17) - 8;
                int nl = static_cast<int>(s.val) + delta;
                if (nl < 1) nl = 1;
                if (nl > MAX_LAG_BITS) nl = MAX_LAG_BITS;
                s.val = static_cast<uint16_t>(nl);
            } else {
                random_src(s, k, rng);
            }
        }
    }
}

// ============================================================================
// UNBOUNDED ENDOGENOUS MEMORY (Pass 2 Zero-Error Verification)
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
            h1 = splitmix64_host(h1 ^ w ^ (d * 0x9e3779b97f4a7c15ULL));
            h2 = splitmix64_host(h2 ^ rotl64_host(w, static_cast<int>(d & 63)) ^ (d * 0xbf58476d1ce4e5b9ULL));
        }
        if (to_d > t) {
            h1 = splitmix64_host(h1 ^ (static_cast<uint64_t>(t) << 32) ^ to_d);
            h2 = splitmix64_host(h2 ^ static_cast<uint64_t>(t));
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

// ============================================================================
// UNSHACKLED GPU EXPEDITION RUNNER
// ============================================================================
static void run_unshackled_expedition(const std::string& name, const std::vector<uint64_t>& words) {
    const size_t start_i = (MAX_LAG_BITS + 63) / 64 + 2;
    if (words.size() <= start_i + 1024) return;
    size_t split = words.size() / 2;

    const int POP_SIZE = 2048;
    const int GENS = 80;
    uint64_t rng = 0x517cc1b727220a95ULL ^ words.size();

    uint64_t* d_words = nullptr;
    Organism* d_pop = nullptr;
    uint64_t* d_errs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_words, words.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_words, words.data(), words.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pop, POP_SIZE * sizeof(Organism)));
    CUDA_CHECK(cudaMalloc(&d_errs, POP_SIZE * sizeof(uint64_t)));

    std::vector<Organism> pop(POP_SIZE);
    std::vector<uint64_t> errs(POP_SIZE);

    for (int p = 0; p < POP_SIZE; ++p) {
        int max_l = (p < POP_SIZE / 4) ? 2 : ((p < POP_SIZE / 2) ? 6 : NUM_REGS);
        pop[p] = make_random_organism(rng, max_l);
    }

    size_t train_len = split - start_i;
    size_t evo_stride = std::max<size_t>(1, train_len / 65536);

    auto t0 = std::chrono::high_resolution_clock::now();

    Organism global_best{};
    uint64_t global_best_train_err = ~0ULL;

    for (int g = 0; g < GENS; ++g) {
        CUDA_CHECK(cudaMemcpy(d_pop, pop.data(), POP_SIZE * sizeof(Organism), cudaMemcpyHostToDevice));
        eval_population_kernel<<<POP_SIZE, 128>>>(d_words, start_i, split, evo_stride, d_pop, d_errs);
        CUDA_CHECK(cudaMemcpy(errs.data(), d_errs, POP_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        std::vector<int> order(POP_SIZE);
        for (int p = 0; p < POP_SIZE; ++p) order[p] = p;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return errs[a] < errs[b]; });

        if (errs[order[0]] < global_best_train_err) {
            global_best_train_err = errs[order[0]];
            global_best = pop[order[0]];
        }

        if (g == GENS - 1) break;

        std::vector<Organism> next_pop(POP_SIZE);
        const int ELITE = 64;
        for (int e = 0; e < ELITE; ++e) next_pop[e] = pop[order[e]];
        for (int p = ELITE; p < POP_SIZE - 128; ++p) {
            Organism child = pop[order[p % ELITE]];
            mutate_organism(child, rng);
            next_pop[p] = child;
        }
        for (int p = POP_SIZE - 128; p < POP_SIZE; ++p) {
            next_pop[p] = make_random_organism(rng, NUM_REGS);
        }
        pop = std::move(next_pop);
    }

    // Strict Causal Full-Stream Evaluation on UNSEEN 50% Test Tail
    CUDA_CHECK(cudaMemcpy(d_pop, &global_best, sizeof(Organism), cudaMemcpyHostToDevice));
    eval_population_kernel<<<1, 128>>>(d_words, split, words.size(), 1, d_pop, d_errs);
    uint64_t unseen_test_err = 0;
    CUDA_CHECK(cudaMemcpy(&unseen_test_err, d_errs, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    auto t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaFree(d_words));
    CUDA_CHECK(cudaFree(d_pop));
    CUDA_CHECK(cudaFree(d_errs));

    uint64_t unseen_bits = (words.size() - split) * 64ULL;
    double test_acc = 100.0 * (1.0 - static_cast<double>(unseen_test_err) / unseen_bits);
    double F_test = compute_free_energy(unseen_test_err, unseen_bits);

    size_t audit_words = std::min<size_t>(words.size(), 131072);
    UnboundedEndogenousMemory mem;
    auto tp0 = std::chrono::high_resolution_clock::now();
    for (size_t i = start_i; i < audit_words; ++i) {
        uint64_t p = eval_organism(global_best, words.data(), i);
        if (p != words[i]) mem.sprout(words, i, p ^ words[i]);
    }
    for (size_t i = start_i; i < audit_words; ++i) {
        uint64_t p = eval_organism(global_best, words.data(), i);
        if ((p ^ mem.query(words, i)) != words[i]) mem.sprout(words, i, p ^ words[i]);
    }
    uint64_t pass2_err = 0;
    for (size_t i = start_i; i < audit_words; ++i) {
        uint64_t p = eval_organism(global_best, words.data(), i) ^ mem.query(words, i);
        pass2_err += __builtin_popcountll(p ^ words[i]);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    double pass2_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    double total_eval_gb = (static_cast<double>(POP_SIZE) * GENS * (train_len / evo_stride) * 8.0) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "[UNSHACKLED BIT-GENESIS] " << name << " (" << std::fixed << std::setprecision(2)
              << (words.size() * 8.0 / (1024.0 * 1024.0)) << " MB | " << (words.size() * 64ULL) << " bits)" << std::endl;
    std::cout << "  Search Scale    : " << (POP_SIZE * GENS) << " DAGs | " << total_eval_gb
              << " GB scanned in " << std::setprecision(1) << gpu_ms << " ms" << std::endl;
    std::cout << "  Evolved Law DAG : " << org_to_str(global_best) << std::endl;
    std::cout << "  Unseen 50% Acc  : " << std::setprecision(2) << test_acc
              << "% (F = " << std::setprecision(6) << F_test << " nats)" << std::endl;
    std::cout << "  Pass 2 Audit    : " << pass2_err << " bit errors (" << mem.count
              << " sprouted nodes, " << std::setprecision(2) << pass2_ms << " ms)" << std::endl;
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    cudaDeviceProp prop{};
    if (dev_count > 0) cudaGetDeviceProperties(&prop, 0);

    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1152: UNSHACKLED BIT-LEVEL GENESIS (ZERO 64-BIT FRAME)" << std::endl;
    std::cout << " Hardware: " << dev_count << "x " << prop.name << " | Standard: KEP v51.0" << std::endl;
    std::cout << "======================================================================" << std::endl;

    auto web_words = load_file_as_bitstream("data/real_web_stream.txt");
    if (!web_words.empty()) run_unshackled_expedition("WEB STREAM: real_web_stream.txt", web_words);

    auto wiki_words = load_file_as_bitstream("data/real_wikipedia_stream.txt");
    if (!wiki_words.empty()) run_unshackled_expedition("WIKIPEDIA: real_wikipedia_stream.txt", wiki_words);

    auto shake_words = load_file_as_bitstream("tinyshakespeare.txt");
    if (!shake_words.empty()) run_unshackled_expedition("LITERATURE: tinyshakespeare.txt (Full)", shake_words);

    auto png_words = load_file_as_bitstream("experiments/renders/bloodbath_precise_v4.png");
    if (!png_words.empty()) run_unshackled_expedition("IMAGE/PNG: bloodbath_precise_v4.png (Full)", png_words);

    auto elf_words = load_file_as_bitstream("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", 4 * 1024 * 1024);
    if (!elf_words.empty()) run_unshackled_expedition("BINARY/ELF: libstdc++.so.6 (4MB)", elf_words);

    return 0;
}
