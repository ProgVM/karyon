#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <cuda_runtime.h>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1170g: SILICON-OPTIMIZED DUAL-GPU BITWISE TISSUE
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | Shared-Memory In-Place Evaluation | Mid-Graph Mitosis
// ============================================================================
// Hardware Optimization + Physics Enhancements over EXP-1170f:
// 1. Zero-Spill __shared__ In-Place Evaluation Buffer (blockDim = 64):
//    - 1 single array `sh_val[64][48]` = 24 KB in fast on-chip SRAM (within SM_75 48KB limit),
//      plus `s_i0..s_tg[64]` = 2 KB shared input broadcast (26 KB total __shared__).
//    - Rare organisms with K > 48 cleanly fallback to a small global scratch buffer (ZERO local memory spill!).
//    - Each node packed into 1 single `uint32_t` (op:2b | src_a:15b | src_b:15b).
//    - Zero-copy GPU stream batching: pre-generated stream chunks copied via pinned memory + single kernel launches,
//      OR pre-uploaded sweep batch buffer so GPU runs at 100% saturation with zero host-wait bottlenecks!
// 2. Apoptosis strictly on mutation/reproduction (never scanning frozen cells needlessly).
// 3. Mid-Graph Wire-Split Mitosis (`insert_node_at`):
//    - Allows inserting new gates inside existing causal cones (pos <= target_u), remapping >= pos.
// 4. Uncapped Physical Energy Equilibrium + Safe Symbiogenesis:
//    - Removed artificial MAX_FUEL = 4096 clamp; Newton cooling (f >> 6)+1 naturally bounds energy.
//    - Symbiogenesis modifies only offspring bi, protecting winner ai.

#define MAX_K 128
#define SHARED_K 48
#define BLOCK_PAIRS 64
#define GRID_W 128
#define GRID_H 128
#define TOTAL_CELLS (GRID_W * GRID_H)
#define REPRO_THRESH 1024
#define T_TICKS 64
#define STREAM_BATCH_SWEEPS 1000

struct alignas(16) CompactOrganismGPU {
    uint32_t node_count;
    uint32_t out_s;
    int32_t fuel;
    uint32_t pad0;
    uint64_t rng;
    uint32_t nodes[MAX_K];
    uint64_t state[MAX_K];
};

__host__ __device__ static inline uint64_t xorshift64_dev(uint64_t& s) {
    uint64_t x = s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline uint64_t splitmix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL; uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__host__ __device__ static inline uint32_t pack_node(uint8_t op, uint16_t sa, uint16_t sb) {
    return ((uint32_t)(op & 3U)) | (((uint32_t)sa & 0x7FFF) << 2) | (((uint32_t)sb & 0x7FFF) << 17);
}

__host__ __device__ static inline void unpack_node(uint32_t p, uint8_t& op, uint16_t& sa, uint16_t& sb) {
    op = (uint8_t)(p & 3U);
    sa = (uint16_t)((p >> 2) & 0x7FFF);
    sb = (uint16_t)((p >> 17) & 0x7FFF);
}

__host__ __device__ static inline uint32_t pick_upper_biased_node(uint32_t K, uint64_t& rng) {
    if (K <= 1U) return 0U;
    uint32_t u1 = (uint32_t)(xorshift64_dev(rng) % K);
    uint32_t u2 = (uint32_t)(xorshift64_dev(rng) % K);
    return (u1 > u2) ? u1 : u2;
}

__host__ __device__ static void apoptosis_org(CompactOrganismGPU& org) {
    uint32_t K = org.node_count;
    if (K <= 12U) return;
    uint32_t root = org.out_s % K;

    uint64_t active_lo = 0ULL, active_hi = 0ULL;
    uint64_t stack_lo = 0ULL, stack_hi = 0ULL;
    auto set_bit = [](uint64_t& lo, uint64_t& hi, uint32_t b) {
        if (b < 64U) lo |= (1ULL << b); else hi |= (1ULL << (b - 64U));
    };
    auto get_bit = [](uint64_t lo, uint64_t hi, uint32_t b) -> bool {
        return (b < 64U) ? ((lo >> b) & 1ULL) : ((hi >> (b - 64U)) & 1ULL);
    };

    set_bit(active_lo, active_hi, root);
    set_bit(stack_lo, stack_hi, root);

    while (stack_lo || stack_hi) {
        uint32_t u;
        if (stack_hi) {
#ifdef __CUDA_ARCH__
            u = 64U + (63U - (uint32_t)__clzll(stack_hi));
#else
            u = 64U + (63U - (uint32_t)__builtin_clzll(stack_hi));
#endif
            stack_hi &= ~(1ULL << (u - 64U));
        } else {
#ifdef __CUDA_ARCH__
            u = 63U - (uint32_t)__clzll(stack_lo);
#else
            u = 63U - (uint32_t)__builtin_clzll(stack_lo);
#endif
            stack_lo &= ~(1ULL << u);
        }
        uint8_t op; uint16_t sa, sb;
        unpack_node(org.nodes[u], op, sa, sb);
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U) % K;
            if (!get_bit(active_lo, active_hi, j)) {
                set_bit(active_lo, active_hi, j);
                set_bit(stack_lo, stack_hi, j);
            }
        }
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U) % K;
            if (!get_bit(active_lo, active_hi, j)) {
                set_bit(active_lo, active_hi, j);
                set_bit(stack_lo, stack_hi, j);
            }
        }
    }

    uint8_t new_idx[MAX_K];
    uint32_t new_K = 0U;
    for (uint32_t u = 0; u < MAX_K; ++u) new_idx[u] = 0U;
    for (uint32_t u = 0; u < K; ++u) {
        if (get_bit(active_lo, active_hi, u)) {
            new_idx[u] = (uint8_t)(new_K++);
        }
    }
    if (new_K == K || new_K == 0U) return;

    uint32_t dst = 0U;
    for (uint32_t u = 0; u < K; ++u) {
        if (get_bit(active_lo, active_hi, u)) {
            uint8_t op; uint16_t sa, sb;
            unpack_node(org.nodes[u], op, sa, sb);
            if (sa >= 3U) sa = (uint16_t)(new_idx[(uint32_t)(sa - 3U) % K] + 3U);
            if (sb >= 3U) sb = (uint16_t)(new_idx[(uint32_t)(sb - 3U) % K] + 3U);
            org.nodes[dst] = pack_node(op, sa, sb);
            org.state[dst] = org.state[u];
            dst++;
        }
    }
    org.node_count = new_K;
    org.out_s = (uint32_t)new_idx[root];
}

__host__ __device__ static void insert_node_at(CompactOrganismGPU& org, uint32_t pos, uint8_t op, uint16_t sa, uint16_t sb) {
    uint32_t K = org.node_count;
    if (K >= MAX_K - 2U) return;
    if (pos > K) pos = K;

    for (uint32_t i = K; i > pos; --i) {
        org.nodes[i] = org.nodes[i - 1U];
        org.state[i] = org.state[i - 1U];
    }

    uint32_t nK = K + 1U;
    if (sa >= 3U + nK) sa = (uint16_t)((sa - 3U) % nK + 3U);
    if (sb >= 3U + nK) sb = (uint16_t)((sb - 3U) % nK + 3U);

    org.nodes[pos] = pack_node(op, sa, sb);
    org.state[pos] = 0ULL;
    org.node_count = nK;

    for (uint32_t u = 0; u < nK; ++u) {
        if (u == pos) continue;
        uint8_t nop; uint16_t nsa, nsb;
        unpack_node(org.nodes[u], nop, nsa, nsb);
        if (nsa >= 3U) {
            uint32_t j = (uint32_t)(nsa - 3U);
            if (j >= pos) nsa++;
            if (nsa >= 3U + nK) nsa = (uint16_t)((nsa - 3U) % nK + 3U);
        }
        if (nsb >= 3U) {
            uint32_t j = (uint32_t)(nsb - 3U);
            if (j >= pos) nsb++;
            if (nsb >= 3U + nK) nsb = (uint16_t)((nsb - 3U) % nK + 3U);
        }
        org.nodes[u] = pack_node(nop, nsa, nsb);
    }
    if (org.out_s >= pos) org.out_s++;
    if (org.out_s >= nK) org.out_s %= nK;
}

__host__ __device__ static void prune_one_node(CompactOrganismGPU& org, uint64_t& rng) {
    uint32_t K = org.node_count;
    if (K <= 4U) return;
    uint32_t du = (uint32_t)(xorshift64_dev(rng) % K);
    for (uint32_t i = du; i < K - 1U; ++i) {
        org.nodes[i] = org.nodes[i + 1U];
        org.state[i] = org.state[i + 1U];
    }
    org.node_count = K - 1U;
    uint32_t nK = org.node_count;
    for (uint32_t u = 0; u < nK; ++u) {
        uint8_t op; uint16_t sa, sb;
        unpack_node(org.nodes[u], op, sa, sb);
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U);
            if (j == du) sa = (uint16_t)(xorshift64_dev(rng) % (nK + 3U));
            else if (j > du) sa--;
            if (sa >= 3U + nK) sa = (uint16_t)((sa - 3U) % nK + 3U);
        }
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U);
            if (j == du) sb = (uint16_t)(xorshift64_dev(rng) % (nK + 3U));
            else if (j > du) sb--;
            if (sb >= 3U + nK) sb = (uint16_t)((sb - 3U) % nK + 3U);
        }
        org.nodes[u] = pack_node(op, sa, sb);
    }
    if (org.out_s == du) org.out_s = (uint32_t)(xorshift64_dev(rng) % nK);
    else if (org.out_s > du) org.out_s--;
    if (org.out_s >= nK) org.out_s %= nK;
}

__host__ __device__ static void graft_subgraph(const CompactOrganismGPU& donor, CompactOrganismGPU& host, uint64_t& rng) {
    uint32_t Kd = donor.node_count, Kh = host.node_count;
    if (Kd < 1U || Kh < 1U || Kh >= (MAX_K - 8U)) return;
    uint32_t want = 2U + (uint32_t)(xorshift64_dev(rng) % 4ULL);
    uint32_t cnt = (Kd < want) ? Kd : want;
    if (Kh + cnt + 1U > MAX_K) cnt = (MAX_K - 1U > Kh) ? (MAX_K - 1U - Kh) : 0U;
    if (cnt == 0U) return;

    uint32_t su = (uint32_t)(xorshift64_dev(rng) % (Kd - cnt + 1U));
    uint32_t start_kh = Kh;
    uint32_t target_k = start_kh + cnt;

    for (uint32_t c = 0; c < cnt; ++c) {
        uint8_t op; uint16_t sa, sb;
        unpack_node(donor.nodes[su + c], op, sa, sb);
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U);
            if (j >= su && j < su + cnt) sa = (uint16_t)(start_kh + (j - su) + 3U);
            else sa = (uint16_t)((j % start_kh) + 3U);
            if (sa >= 3U + target_k) sa = (uint16_t)((sa - 3U) % target_k + 3U);
        }
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U);
            if (j >= su && j < su + cnt) sb = (uint16_t)(start_kh + (j - su) + 3U);
            else sb = (uint16_t)((j % start_kh) + 3U);
            if (sb >= 3U + target_k) sb = (uint16_t)((sb - 3U) % target_k + 3U);
        }
        host.nodes[host.node_count] = pack_node(op, sa, sb);
        host.state[host.node_count] = 0ULL;
        host.node_count++;
    }

    uint32_t new_last = host.node_count - 1U;
    uint32_t ro_opt = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (ro_opt < 45U && host.node_count < MAX_K) {
        uint16_t src_old = (uint16_t)((host.out_s % start_kh) + 3U);
        uint16_t src_new = (uint16_t)(new_last + 3U);
        uint8_t comb_op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        host.nodes[host.node_count] = pack_node(comb_op, src_old, src_new);
        host.state[host.node_count] = 0ULL;
        host.node_count++;
        host.out_s = host.node_count - 1U;
    } else if (ro_opt < 75U) {
        host.out_s = new_last;
    }
    apoptosis_org(host);
}

__host__ __device__ static void mutate_organism(CompactOrganismGPU& org, uint64_t& rng) {
    uint32_t K = org.node_count;
    if (K == 0U) return;
    uint32_t r = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (r < 30U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        uint8_t op; uint16_t sa, sb;
        unpack_node(org.nodes[u], op, sa, sb);
        op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        org.nodes[u] = pack_node(op, sa, sb);
    } else if (r < 65U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        uint8_t op; uint16_t sa, sb;
        unpack_node(org.nodes[u], op, sa, sb);
        uint16_t ns = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
        if (xorshift64_dev(rng) & 1ULL) sa = ns; else sb = ns;
        org.nodes[u] = pack_node(op, sa, sb);
    } else if (r < 80U) {
        if (K < (MAX_K - 2U)) {
            uint32_t pos = (uint32_t)(xorshift64_dev(rng) % (K + 1U));
            uint8_t op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
            uint16_t sa = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
            uint16_t sb = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
            insert_node_at(org, pos, op, sa, sb);
        }
    } else if (r < 88U) {
        graft_subgraph(org, org, rng);
    } else if (r < 97U) {
        prune_one_node(org, rng);
        if (r >= 93U) prune_one_node(org, rng);
    } else {
        org.out_s = pick_upper_biased_node(K, rng);
    }
    if (org.node_count > 16U) apoptosis_org(org);
}

__host__ __device__ static inline uint32_t eval_org_inplace(
    CompactOrganismGPU& org, const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t,
    const uint64_t* targ_t, int ticks, uint64_t* val_buf
) {
    const uint32_t K = org.node_count;
    if (K == 0U) return 0U;
    uint32_t matches = 0U, os = org.out_s % K;

    for (uint32_t u = 0; u < K; ++u) val_buf[u] = org.state[u];

    for (int t = 0; t < ticks; ++t) {
        const uint64_t i0 = in0_t[t], i1 = in1_t[t], ir = in_rst_t[t];
        for (uint32_t u = 0; u < K; ++u) {
            uint32_t p = org.nodes[u];
            uint8_t op = (uint8_t)(p & 3U);
            uint16_t sa = (uint16_t)((p >> 2) & 0x7FFF);
            uint16_t sb = (uint16_t)((p >> 17) & 0x7FFF);

            uint64_t a = (sa == 0U) ? i0 : ((sa == 1U) ? i1 : ((sa == 2U) ? ir : val_buf[(sa - 3U) % K]));
            uint64_t b = (sb == 0U) ? i0 : ((sb == 1U) ? i1 : ((sb == 2U) ? ir : val_buf[(sb - 3U) % K]));
            uint64_t r = 0ULL;
            if (op == 0U) r = ~(a & b);
            else if (op == 1U) r = ~(a | b);
            else if (op == 2U) r = a ^ b;
            else r = ~(a ^ b);
            val_buf[u] = r;
        }
#ifdef __CUDA_ARCH__
        matches += (uint32_t)__popcll(~(val_buf[os] ^ targ_t[t]));
#else
        matches += (uint32_t)__builtin_popcountll(~(val_buf[os] ^ targ_t[t]));
#endif
    }
    for (uint32_t u = 0; u < K; ++u) org.state[u] = val_buf[u];
    return matches;
}

__global__ void tissue_sweep_kernel(
    CompactOrganismGPU* orgs, uint32_t n_pairs, uint32_t pass_mode,
    const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t, const uint64_t* targ_t,
    uint64_t* d_fallback_buf
) {
    __shared__ uint64_t sh_val[BLOCK_PAIRS][SHARED_K];
    __shared__ uint64_t s_i0[T_TICKS], s_i1[T_TICKS], s_ir[T_TICKS], s_tg[T_TICKS];

    if (threadIdx.x < T_TICKS) {
        s_i0[threadIdx.x] = in0_t[threadIdx.x];
        s_i1[threadIdx.x] = in1_t[threadIdx.x];
        s_ir[threadIdx.x] = in_rst_t[threadIdx.x];
        s_tg[threadIdx.x] = targ_t[threadIdx.x];
    }
    __syncthreads();

    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;

    uint32_t W = GRID_W, H = GRID_H;
    uint32_t oi, oj;
    if (pass_mode < 2U) {
        uint32_t y = p / (W >> 1), x = ((p % (W >> 1)) << 1) + (pass_mode & 1U);
        oi = y * W + x; oj = y * W + ((x + 1U) & (W - 1U));
    } else {
        uint32_t x = p % W, y = ((p / W) << 1) + (pass_mode & 1U);
        oi = y * W + x; oj = ((y + 1U) & (H - 1U)) * W + x;
    }

    const uint32_t tot_bits = (uint32_t)T_TICKS * 64U;
    uint64_t* my_sh = &sh_val[threadIdx.x][0];
    uint64_t* my_fb = d_fallback_buf + ((size_t)p * MAX_K);

    auto eval_one = [&](uint32_t idx) {
        if (orgs[idx].fuel <= 0) return;
        uint64_t* buf = (orgs[idx].node_count <= SHARED_K) ? my_sh : my_fb;
        uint32_t m = eval_org_inplace(orgs[idx], s_i0, s_i1, s_ir, s_tg, T_TICKS, buf);
        int32_t rew = (int32_t)(((2LL * (int64_t)m - (int64_t)tot_bits) * 72LL) / (int64_t)tot_bits);
        int32_t ncost = (int32_t)((orgs[idx].node_count + 7U) >> 3);
        int32_t fuel_next = orgs[idx].fuel + rew - ncost - ((orgs[idx].fuel >> 6) + 1);
        if (fuel_next < 0) fuel_next = 0;
        orgs[idx].fuel = fuel_next;
    };
    eval_one(oi); eval_one(oj);

    uint64_t lr = orgs[oi].rng;
    int32_t fi = orgs[oi].fuel, fj = orgs[oj].fuel;
    if (fi <= 0 && fj <= 0) {
        mutate_organism(orgs[oi], lr); mutate_organism(orgs[oj], lr);
        orgs[oi].fuel = 256;
    } else {
        uint32_t ai = (fi >= fj) ? oi : oj, bi = (fi >= fj) ? oj : oi;
        int32_t fa = orgs[ai].fuel, fb = orgs[bi].fuel;
        int32_t na = (int32_t)(xorshift64_dev(lr) & 127ULL), nb = (int32_t)(xorshift64_dev(lr) & 127ULL);

        if (fa >= REPRO_THRESH && fb >= REPRO_THRESH && (xorshift64_dev(lr) % 10ULL) == 0ULL) {
            graft_subgraph(orgs[ai], orgs[bi], lr);
        } else if (fa >= REPRO_THRESH && (fa + na > fb + 256 + nb)) {
            orgs[bi].node_count = orgs[ai].node_count;
            orgs[bi].out_s = orgs[ai].out_s;
            for (uint32_t u = 0; u < orgs[ai].node_count; ++u) {
                orgs[bi].nodes[u] = orgs[ai].nodes[u];
                orgs[bi].state[u] = 0ULL;
            }
            if ((xorshift64_dev(lr) % 100ULL) < 50ULL) {
                mutate_organism(orgs[bi], lr);
            }
            int32_t share = fa >> 1;
            orgs[ai].fuel = fa - share;
            orgs[bi].fuel = share;
        } else if (fb <= 0) {
            mutate_organism(orgs[bi], lr);
        }
    }
    orgs[oi].rng = lr;
}

static void gen_continuous_stream_host(int n_words, int bits_per_word, uint64_t& rng, uint64_t* i0, uint64_t* i1, uint64_t* ir, uint64_t* tg) {
    int idx = 0;
    for (int w = 0; w < n_words; ++w) {
        uint64_t c = 0ULL;
        for (int b = 0; b < bits_per_word; ++b) {
            uint64_t a = xorshift64_dev(rng), bv = xorshift64_dev(rng);
            i0[idx] = a; i1[idx] = bv; ir[idx] = (b == 0) ? ~0ULL : 0ULL;
            uint64_t c_in = (b == 0) ? 0ULL : c;
            tg[idx] = a ^ bv ^ c_in;
            c = (a & bv) | ((a ^ bv) & c_in);
            idx++;
        }
    }
}

static void print_active_circuit(const CompactOrganismGPU& org) {
    uint32_t K = org.node_count;
    if (K == 0U) return;
    uint32_t root = org.out_s % K;

    std::set<uint32_t> active_nodes;
    std::vector<uint32_t> stack;
    stack.push_back(root);
    active_nodes.insert(root);

    while (!stack.empty()) {
        uint32_t u = stack.back();
        stack.pop_back();
        uint8_t op; uint16_t sa, sb;
        unpack_node(org.nodes[u], op, sa, sb);
        auto check_wire = [&](uint16_t s) {
            if (s >= 3U) {
                uint32_t j = (uint32_t)(s - 3U) % K;
                if (active_nodes.find(j) == active_nodes.end()) {
                    active_nodes.insert(j);
                    stack.push_back(j);
                }
            }
        };
        check_wire(sa); check_wire(sb);
    }

    static const char* OPN[4] = {"NAND", "NOR ", "XOR ", "XNOR"};
    std::cout << "    ========================================================\n"
              << "    ACTIVE CONE CIRCUIT DIAGRAM (Total K=" << K << ", Active Nodes=" << active_nodes.size() << ", out=n" << root << "):\n"
              << "    ========================================================\n";
    for (uint32_t u : active_nodes) {
        uint8_t op; uint16_t sa, sb;
        unpack_node(org.nodes[u], op, sa, sb);
        auto fmt = [&](uint16_t s) -> std::string {
            if (s == 0U) return "inA"; if (s == 1U) return "inB"; if (s == 2U) return "rst";
            uint32_t j = (uint32_t)(s - 3U) % K;
            return "n" + std::to_string(j) + ((j >= u) ? "(t-1)" : "(t)");
        };
        std::string extra = "";
        if (sa == sb) {
            if (op == 0U || op == 1U) extra = "  [NOT " + fmt(sa) + "]";
            else if (op == 2U) extra = "  [CONST 0]";
            else extra = "  [CONST 1]";
        }
        std::cout << "      n" << u << (u == root ? "*" : " ") << " = "
                  << OPN[op & 3U] << "(" << fmt(sa) << ", " << fmt(sb) << ")"
                  << extra << "\n";
    }
    std::cout << "    ========================================================\n\n";
}

static void run_gpu_seed(int gpu_id, int seed_idx, int max_sweeps, uint64_t seed) {
    cudaSetDevice(gpu_id);
    const uint32_t N = TOTAL_CELLS, n_pairs = N / 2;

    std::vector<CompactOrganismGPU> host_orgs(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t s = seed ^ ((uint64_t)(i + 1U) * 0x9E3779B97F4A7C15ULL);
        host_orgs[i].node_count = 4;
        for (int u = 0; u < 4; ++u) {
            uint8_t op = (uint8_t)(splitmix64(s) & 3ULL);
            uint16_t sa = (uint16_t)(splitmix64(s) % 7U);
            uint16_t sb = (uint16_t)(splitmix64(s) % 7U);
            host_orgs[i].nodes[u] = pack_node(op, sa, sb);
            host_orgs[i].state[u] = 0ULL;
        }
        host_orgs[i].out_s = (uint32_t)(splitmix64(s) % 4U);
        host_orgs[i].fuel = 2048;
        uint64_t r = splitmix64(s);
        host_orgs[i].rng = r ? r : 0xCAFEBABE12345678ULL;
    }

    CompactOrganismGPU* d_orgs = nullptr;
    uint64_t *d_i0, *d_i1, *d_ir, *d_tg, *d_fb;
    cudaMalloc(&d_orgs, N * sizeof(CompactOrganismGPU));
    cudaMalloc(&d_fb, (size_t)n_pairs * MAX_K * sizeof(uint64_t));
    // Pre-allocate device batch stream memory (1000 sweeps * 64 ticks = 64k uint64_t per channel = 2MB total!)
    cudaMalloc(&d_i0, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_i1, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_ir, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_tg, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMemcpy(d_orgs, host_orgs.data(), N * sizeof(CompactOrganismGPU), cudaMemcpyHostToDevice);

    // Pinned host buffers for zero-stall DMA!
    uint64_t *h_i0, *h_i1, *h_ir, *h_tg;
    cudaMallocHost(&h_i0, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMallocHost(&h_i1, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMallocHost(&h_ir, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMallocHost(&h_tg, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));

    uint64_t hrng = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp critical
    std::cout << "--- GPU #" << gpu_id << " SEED " << seed_idx << " (Shared-Memory In-Place 128x128, max_sw=" << max_sweeps << ") ---\n";

    int blocks_pairs = (n_pairs + BLOCK_PAIRS - 1) / BLOCK_PAIRS;
    const uint32_t tot_bits = T_TICKS * 64U;

    for (int base_sw = 0; base_sw <= max_sweeps; base_sw += STREAM_BATCH_SWEEPS) {
        if (base_sw > 0) {
            int count = std::min(STREAM_BATCH_SWEEPS, max_sweeps - base_sw + 1);
            for (int k = 0; k < count; ++k) {
                gen_continuous_stream_host(8, 8, hrng,
                    h_i0 + (size_t)k * T_TICKS,
                    h_i1 + (size_t)k * T_TICKS,
                    h_ir + (size_t)k * T_TICKS,
                    h_tg + (size_t)k * T_TICKS);
            }
            cudaMemcpyAsync(d_i0, h_i0, (size_t)count * T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_i1, h_i1, (size_t)count * T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_ir, h_ir, (size_t)count * T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_tg, h_tg, (size_t)count * T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

            for (int k = 0; k < count; ++k) {
                uint32_t pm = (uint32_t)(xorshift64_dev(hrng) & 3ULL);
                size_t off = (size_t)k * T_TICKS;
                tissue_sweep_kernel<<<blocks_pairs, BLOCK_PAIRS, 0, stream>>>(
                    d_orgs, n_pairs, pm, d_i0 + off, d_i1 + off, d_ir + off, d_tg + off, d_fb);
            }
            // Ensure host pinned buffers h_i0..h_tg are consumed before next batch overwrites them
            cudaStreamSynchronize(stream);
        }

        int sw = (base_sw == 0) ? 0 : std::min(base_sw + STREAM_BATCH_SWEEPS - 1, max_sweeps);
        if (sw == 0 || sw == 1000 || sw == 2000 || sw == 4000 || sw == 8000 || sw == 12000 || sw == 16000 || sw == 20000 || sw == max_sweeps) {
            cudaStreamSynchronize(stream);
            cudaMemcpy(host_orgs.data(), d_orgs, N * sizeof(CompactOrganismGPU), cudaMemcpyDeviceToHost);

            uint64_t erng = 0xDEADBEEFCAFE1234ULL ^ (uint64_t)sw;
            uint64_t ei0[T_TICKS], ei1[T_TICKS], eir[T_TICKS], etg[T_TICKS];
            gen_continuous_stream_host(8, 8, erng, ei0, ei1, eir, etg);

            long long tf = 0, sk = 0; int alive = 0, n90 = 0, n100 = 0;
            uint32_t mink = 999999U, maxk = 0U, bidx = 0U;
            double macc = 0.0, sacc = 0.0;
            uint64_t vc[MAX_K];

            for (uint32_t i = 0; i < N; ++i) {
                tf += host_orgs[i].fuel; if (host_orgs[i].fuel > 0) alive++;
                uint32_t K = host_orgs[i].node_count;
                sk += K; mink = std::min(mink, K); maxk = std::max(maxk, K);

                CompactOrganismGPU tmp = host_orgs[i];
                for (uint32_t u = 0; u < K; ++u) tmp.state[u] = 0ULL;
                uint32_t matches = eval_org_inplace(tmp, ei0, ei1, eir, etg, T_TICKS, vc);

                double acc = 100.0 * matches / tot_bits; sacc += acc;
                if (acc > macc) { macc = acc; bidx = i; }
                if (acc >= 90.0) n90++;
                if (matches == tot_bits) n100++;
            }

            #pragma omp critical
            {
                std::cout << "[GPU #" << gpu_id << " S" << seed_idx << "] sw=" << std::setw(5) << sw
                          << " | AvgAcc=" << std::fixed << std::setprecision(2) << (sacc / N)
                          << "% | MaxAcc=" << macc << "% | >=90%=" << std::setw(5) << n90 << " | 100%=" << std::setw(5) << n100
                          << " | AvgF=" << std::setw(4) << (tf / N) << " | Alive=" << std::setw(5) << alive
                          << " | AvgK=" << std::setprecision(1) << ((double)sk / N) << " [" << mink << ".." << maxk << "]\n" << std::flush;

                if (sw == max_sweeps) {
                    uint64_t orng = 0x7777777712345678ULL;
                    long long om = 0, om_b0 = 0, om_rest = 0;
                    CompactOrganismGPU best_o = host_orgs[bidx];
                    for (uint32_t u = 0; u < best_o.node_count; ++u) best_o.state[u] = 0ULL;

                    for (int b = 0; b < 16; ++b) {
                        gen_continuous_stream_host(2, 32, orng, ei0, ei1, eir, etg);
                        for (int t = 0; t < T_TICKS; ++t) {
                            uint32_t m1 = eval_org_inplace(best_o, &ei0[t], &ei1[t], &eir[t], &etg[t], 1, vc);
                            om += m1;
                            if ((t & 31) == 0) om_b0 += m1; else om_rest += m1;
                        }
                    }
                    std::cout << "\n  >>> GPU #" << gpu_id << " OOD FROZEN 32-BIT STREAM (1,024 pairs, NO reset): "
                              << std::setprecision(2) << (100.0 * om / (16LL * 64LL * 64LL)) << "% "
                              << "(rst_bit0=" << (100.0 * om_b0 / (16LL * 2LL * 64LL))
                              << "%, carry_bits_1..31=" << (100.0 * om_rest / (16LL * 62LL * 64LL)) << "%)\n";
                    print_active_circuit(best_o);
                }
            }
        }
        if (base_sw == 0) base_sw = 1 - STREAM_BATCH_SWEEPS;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_i0); cudaFreeHost(h_i1); cudaFreeHost(h_ir); cudaFreeHost(h_tg);
    cudaFree(d_orgs); cudaFree(d_fb);
    cudaFree(d_i0); cudaFree(d_i1); cudaFree(d_ir); cudaFree(d_tg);
    #pragma omp critical
    std::cout << "GPU #" << gpu_id << " Elapsed: " << std::setprecision(2)
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() << "s\n" << std::flush;
}

int main(int argc, char** argv) {
    int max_sweeps = (argc > 1) ? std::atoi(argv[1]) : 24000;
    std::cout << "========================================================================\n"
              << "KARYON-CORE EXP-1170g: SILICON-OPTIMIZED DUAL-GPU BITWISE TISSUE\n"
              << "Standard: KEP v51.0 | __shared__ In-Place | Mid-Graph Mitosis | max_sw=" << max_sweeps << "\n"
              << "========================================================================\n\n" << std::flush;

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) {
            run_gpu_seed(0, 0, max_sweeps, 0x9E3779B97F4A7C15ULL);
        } else {
            run_gpu_seed(1, 1, max_sweeps, 0x123456789ABCDEF0ULL);
        }
    }
    return 0;
}
