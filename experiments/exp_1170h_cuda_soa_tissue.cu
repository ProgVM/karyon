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
// KARYON-CORE EXP-1170h: PURE SoA COALESCED DUAL-GPU BITWISE TISSUE + WIRE-SPLIT MITOSIS
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | 100% Coalesced SoA | Shared Nodes+Val Cache | Wire-Split Mitosis
// ============================================================================
// Hardware Optimization + Physics Enhancements over EXP-1170g:
// 1. 100% Coalesced Structure-of-Arrays (SoA) Memory Layout:
//    - Eliminates the 1568-byte AoS stride (32x memory bandwidth waste!).
//    - `d_nodes[u * N + i]` and `d_state[u * N + i]` give stride-1 warp coalescing.
// 2. Split Evaluation & Interaction Kernels (2x Thread Parallelism = 16,384 threads/GPU):
//    - `eval_all_kernel<<<16384/64, 64>>>`: 1 thread per organism!
//      Loads `nodes[0..K-1]` and `state[0..K-1]` ONCE into `__shared__` SRAM (`sh_nodes[64][32]` + `sh_val[64][32]`),
//      executes all 64 ticks entirely in on-chip SRAM + registers (64x fewer global DRAM reads!).
//    - `pair_interact_kernel<<<8192/64, 64>>>`: checkerboard thermodynamic pairing & mitosis.
// 3. True Wire-Split Mitosis (Splicing into Existing Causal Cones):
//    - Instead of dropping disconnected dead nodes, `wire_split_mitosis` inserts a new gate at `pos <= target_u`,
//      reads `target_src`, and splices the new gate directly into an existing node's input wire!
//      (Also preserves tip-growth onto `out_s`!).
// 4. Pure Natural Fuel Equilibrium (No MAX_FUEL clamp, linear reward, safe symbiogenesis on `bi`).

#define MAX_K 64
#define SHARED_K 32
#define EVAL_BLOCK 64
#define PAIR_BLOCK 64
#define GRID_W 128
#define GRID_H 128
#define TOTAL_CELLS (GRID_W * GRID_H)
#define REPRO_THRESH 1024
#define T_TICKS 64
#define STREAM_BATCH_SWEEPS 1000

struct TissueSoA {
    uint32_t* node_count; // [N]
    uint32_t* out_s;      // [N]
    int32_t*  fuel;       // [N]
    uint64_t* rng;        // [N]
    uint32_t* nodes;      // [MAX_K * N]
    uint64_t* state;      // [MAX_K * N]
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

__device__ static void apoptosis_soa(TissueSoA& t, uint32_t idx) {
    uint32_t K = t.node_count[idx];
    if (K <= 12U) return;
    uint32_t root = t.out_s[idx] % K;

    uint64_t active = 0ULL, stack = 0ULL;
    active |= (1ULL << root);
    stack |= (1ULL << root);

    while (stack) {
        uint32_t u = 63U - (uint32_t)__clzll(stack);
        stack &= ~(1ULL << u);
        uint8_t op; uint16_t sa, sb;
        unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U) % K;
            if (!((active >> j) & 1ULL)) { active |= (1ULL << j); stack |= (1ULL << j); }
        }
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U) % K;
            if (!((active >> j) & 1ULL)) { active |= (1ULL << j); stack |= (1ULL << j); }
        }
    }

    uint8_t new_idx[MAX_K];
    for (uint32_t u = 0; u < MAX_K; ++u) new_idx[u] = 0U;
    uint32_t new_K = 0U;
    for (uint32_t u = 0; u < K; ++u) {
        if ((active >> u) & 1ULL) new_idx[u] = (uint8_t)(new_K++);
    }
    if (new_K == K || new_K == 0U) return;

    uint32_t dst = 0U;
    for (uint32_t u = 0; u < K; ++u) {
        if ((active >> u) & 1ULL) {
            uint8_t op; uint16_t sa, sb;
            unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
            if (sa >= 3U) sa = (uint16_t)(new_idx[(uint32_t)(sa - 3U) % K] + 3U);
            if (sb >= 3U) sb = (uint16_t)(new_idx[(uint32_t)(sb - 3U) % K] + 3U);
            t.nodes[(size_t)dst * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
            t.state[(size_t)dst * TOTAL_CELLS + idx] = t.state[(size_t)u * TOTAL_CELLS + idx];
            dst++;
        }
    }
    t.node_count[idx] = new_K;
    t.out_s[idx] = (uint32_t)new_idx[root];
}

// True Wire-Split Mitosis: Splits a wire reading into target node `v` (or `out_s`), inserting new gate `pos <= v`!
__device__ static void wire_split_mitosis_soa(TissueSoA& t, uint32_t idx, uint64_t& rng) {
    uint32_t K = t.node_count[idx];
    if (K >= MAX_K - 2U || K == 0U) return;

    uint32_t rtype = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (rtype < 65U) {
        // Interior wire-split: pick target node `v`, insert new gate at `pos` right BEFORE `v` (or <= v)
        uint32_t v = (uint32_t)(xorshift64_dev(rng) % K);
        uint32_t pos = v; // inserting at `v` shifts old `v` to `v + 1`, so `pos < v+1` is same-tick combinatorial!
        uint8_t op_v; uint16_t sa_v, sb_v;
        unpack_node(t.nodes[(size_t)v * TOTAL_CELLS + idx], op_v, sa_v, sb_v);

        bool use_a = (xorshift64_dev(rng) & 1ULL) != 0ULL;
        uint16_t orig_wire = use_a ? sa_v : sb_v;
        uint16_t second_wire = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
        uint8_t new_op = (uint8_t)(xorshift64_dev(rng) & 3ULL);

        // Shift nodes >= pos upward by 1
        for (uint32_t i = K; i > pos; --i) {
            t.nodes[(size_t)i * TOTAL_CELLS + idx] = t.nodes[(size_t)(i - 1U) * TOTAL_CELLS + idx];
            t.state[(size_t)i * TOTAL_CELLS + idx] = t.state[(size_t)(i - 1U) * TOTAL_CELLS + idx];
        }

        uint32_t nK = K + 1U;
        t.node_count[idx] = nK;

        auto shift_wire = [&](uint16_t w) -> uint16_t {
            if (w >= 3U) {
                uint32_t j = (uint32_t)(w - 3U);
                if (j >= pos) w = (uint16_t)(j + 1U + 3U);
            }
            if (w >= 3U + nK) w = (uint16_t)((w - 3U) % nK + 3U);
            return w;
        };

        // Remap all shifted nodes (except pos)
        for (uint32_t u = 0; u < nK; ++u) {
            if (u == pos) continue;
            uint8_t nop; uint16_t nsa, nsb;
            unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], nop, nsa, nsb);
            nsa = shift_wire(nsa);
            nsb = shift_wire(nsb);
            t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(nop, nsa, nsb);
        }
        if (t.out_s[idx] >= pos) t.out_s[idx]++;
        if (t.out_s[idx] >= nK) t.out_s[idx] %= nK;

        // Place new gate at `pos`, feeding from shifted `orig_wire` and `second_wire`
        uint16_t new_sa = shift_wire(orig_wire);
        uint16_t new_sb = shift_wire(second_wire);
        t.nodes[(size_t)pos * TOTAL_CELLS + idx] = pack_node(new_op, new_sa, new_sb);
        t.state[(size_t)pos * TOTAL_CELLS + idx] = 0ULL;

        // Splice `v` (now at `pos + 1`) to read `pos` instead of `orig_wire`
        uint32_t new_v = pos + 1U;
        uint8_t vop; uint16_t vsa, vsb;
        unpack_node(t.nodes[(size_t)new_v * TOTAL_CELLS + idx], vop, vsa, vsb);
        if (use_a) vsa = (uint16_t)(pos + 3U);
        else vsb = (uint16_t)(pos + 3U);
        t.nodes[(size_t)new_v * TOTAL_CELLS + idx] = pack_node(vop, vsa, vsb);
    } else {
        // Tip growth at end K + optional readout splice onto out_s
        uint8_t op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        uint16_t sa = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
        uint16_t sb = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
        t.nodes[(size_t)K * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
        t.state[(size_t)K * TOTAL_CELLS + idx] = 0ULL;
        t.node_count[idx] = K + 1U;
        uint32_t new_u = K;
        uint32_t ro = (uint32_t)(xorshift64_dev(rng) % 100ULL);
        if (ro < 50U && t.node_count[idx] < MAX_K) {
            uint16_t s_old = (uint16_t)((t.out_s[idx] % K) + 3U);
            uint16_t s_new = (uint16_t)(new_u + 3U);
            uint8_t cop = (uint8_t)(xorshift64_dev(rng) & 3ULL);
            uint32_t comb_u = t.node_count[idx];
            t.nodes[(size_t)comb_u * TOTAL_CELLS + idx] = pack_node(cop, s_old, s_new);
            t.state[(size_t)comb_u * TOTAL_CELLS + idx] = 0ULL;
            t.node_count[idx]++;
            t.out_s[idx] = comb_u;
        } else if (ro < 80U) {
            t.out_s[idx] = new_u;
        }
    }
}

__device__ static void prune_one_node_soa(TissueSoA& t, uint32_t idx, uint64_t& rng) {
    uint32_t K = t.node_count[idx];
    if (K <= 4U) return;
    uint32_t du = (uint32_t)(xorshift64_dev(rng) % K);
    for (uint32_t i = du; i < K - 1U; ++i) {
        t.nodes[(size_t)i * TOTAL_CELLS + idx] = t.nodes[(size_t)(i + 1U) * TOTAL_CELLS + idx];
        t.state[(size_t)i * TOTAL_CELLS + idx] = t.state[(size_t)(i + 1U) * TOTAL_CELLS + idx];
    }
    uint32_t nK = K - 1U;
    t.node_count[idx] = nK;
    for (uint32_t u = 0; u < nK; ++u) {
        uint8_t op; uint16_t sa, sb;
        unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
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
        t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
    }
    if (t.out_s[idx] == du) t.out_s[idx] = (uint32_t)(xorshift64_dev(rng) % nK);
    else if (t.out_s[idx] > du) t.out_s[idx]--;
    if (t.out_s[idx] >= nK) t.out_s[idx] %= nK;
}

__device__ static void graft_subgraph_soa(TissueSoA& t, uint32_t donor_idx, uint32_t host_idx, uint64_t& rng) {
    uint32_t Kd = t.node_count[donor_idx], Kh = t.node_count[host_idx];
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
        unpack_node(t.nodes[(size_t)(su + c) * TOTAL_CELLS + donor_idx], op, sa, sb);
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
        uint32_t dst_u = t.node_count[host_idx];
        t.nodes[(size_t)dst_u * TOTAL_CELLS + host_idx] = pack_node(op, sa, sb);
        t.state[(size_t)dst_u * TOTAL_CELLS + host_idx] = 0ULL;
        t.node_count[host_idx] = dst_u + 1U;
    }

    uint32_t new_last = t.node_count[host_idx] - 1U;
    uint32_t ro_opt = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (ro_opt < 45U && t.node_count[host_idx] < MAX_K) {
        uint16_t src_old = (uint16_t)((t.out_s[host_idx] % start_kh) + 3U);
        uint16_t src_new = (uint16_t)(new_last + 3U);
        uint8_t comb_op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        uint32_t cu = t.node_count[host_idx];
        t.nodes[(size_t)cu * TOTAL_CELLS + host_idx] = pack_node(comb_op, src_old, src_new);
        t.state[(size_t)cu * TOTAL_CELLS + host_idx] = 0ULL;
        t.node_count[host_idx] = cu + 1U;
        t.out_s[host_idx] = cu;
    } else if (ro_opt < 75U) {
        t.out_s[host_idx] = new_last;
    }
    apoptosis_soa(t, host_idx);
}

__device__ static void mutate_soa(TissueSoA& t, uint32_t idx, uint64_t& rng) {
    uint32_t K = t.node_count[idx];
    if (K == 0U) return;
    uint32_t r = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (r < 30U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        uint8_t op; uint16_t sa, sb;
        unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
        op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
    } else if (r < 65U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        uint8_t op; uint16_t sa, sb;
        unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
        uint16_t ns = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
        if (xorshift64_dev(rng) & 1ULL) sa = ns; else sb = ns;
        t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
    } else if (r < 80U) {
        wire_split_mitosis_soa(t, idx, rng);
    } else if (r < 88U) {
        graft_subgraph_soa(t, idx, idx, rng);
    } else if (r < 97U) {
        prune_one_node_soa(t, idx, rng);
        if (r >= 93U) prune_one_node_soa(t, idx, rng);
    } else {
        t.out_s[idx] = pick_upper_biased_node(K, rng);
    }
    if (t.node_count[idx] > 16U) apoptosis_soa(t, idx);
}

// Kernel 1: 16,384 threads (1 per cell), 100% stride-1 coalesced SoA, nodes+vals cached in __shared__ SRAM!
__global__ void eval_all_cells_kernel(
    TissueSoA t,
    const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t, const uint64_t* targ_t,
    uint64_t* d_fallback_val, uint32_t* d_fallback_nodes
) {
    // 64 threads * 32 nodes * (8 + 4 bytes) = 24 KB + 2 KB stream = 26 KB __shared__ (2 blocks/SM on T4)
    __shared__ uint64_t sh_val[EVAL_BLOCK][SHARED_K];
    __shared__ uint32_t sh_nodes[EVAL_BLOCK][SHARED_K];
    __shared__ uint64_t s_i0[T_TICKS], s_i1[T_TICKS], s_ir[T_TICKS], s_tg[T_TICKS];

    if (threadIdx.x < T_TICKS) {
        s_i0[threadIdx.x] = in0_t[threadIdx.x];
        s_i1[threadIdx.x] = in1_t[threadIdx.x];
        s_ir[threadIdx.x] = in_rst_t[threadIdx.x];
        s_tg[threadIdx.x] = targ_t[threadIdx.x];
    }
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_CELLS) return;

    int32_t fuel = t.fuel[idx];
    if (fuel <= 0) return;

    const uint32_t K = t.node_count[idx];
    if (K == 0U) return;
    const uint32_t os = t.out_s[idx] % K;

    uint64_t* val_buf = (K <= SHARED_K) ? &sh_val[threadIdx.x][0] : (d_fallback_val + (size_t)idx * MAX_K);
    uint32_t* nod_buf = (K <= SHARED_K) ? &sh_nodes[threadIdx.x][0] : (d_fallback_nodes + (size_t)idx * MAX_K);

    // 100% stride-1 coalesced load ONCE per sweep + zero runtime UDIV in inner loop!
    for (uint32_t u = 0; u < K; ++u) {
        uint32_t p = t.nodes[(size_t)u * TOTAL_CELLS + idx];
        uint8_t op = (uint8_t)(p & 3U);
        uint16_t sa = (uint16_t)((p >> 2) & 0x7FFF);
        uint16_t sb = (uint16_t)((p >> 17) & 0x7FFF);
        if (sa >= 3U + K) sa = (uint16_t)((sa - 3U) % K + 3U);
        if (sb >= 3U + K) sb = (uint16_t)((sb - 3U) % K + 3U);
        nod_buf[u] = pack_node(op, sa, sb);
        val_buf[u] = t.state[(size_t)u * TOTAL_CELLS + idx];
    }

    uint32_t matches = 0U;
    for (int tick = 0; tick < T_TICKS; ++tick) {
        const uint64_t i0 = s_i0[tick], i1 = s_i1[tick], ir = s_ir[tick];
        for (uint32_t u = 0; u < K; ++u) {
            uint32_t p = nod_buf[u];
            uint8_t op = (uint8_t)(p & 3U);
            uint32_t sa = (p >> 2) & 0x7FFFU;
            uint32_t sb = (p >> 17) & 0x7FFFU;

            uint64_t a = (sa < 3U) ? ((sa == 0U) ? i0 : ((sa == 1U) ? i1 : ir)) : val_buf[sa - 3U];
            uint64_t b = (sb < 3U) ? ((sb == 0U) ? i0 : ((sb == 1U) ? i1 : ir)) : val_buf[sb - 3U];
            uint64_t r = 0ULL;
            if (op == 0U) r = ~(a & b);
            else if (op == 1U) r = ~(a | b);
            else if (op == 2U) r = a ^ b;
            else r = ~(a ^ b);
            val_buf[u] = r;
        }
        matches += (uint32_t)__popcll(~(val_buf[os] ^ s_tg[tick]));
    }

    // 100% stride-1 coalesced write back ONCE per sweep!
    for (uint32_t u = 0; u < K; ++u) {
        t.state[(size_t)u * TOTAL_CELLS + idx] = val_buf[u];
    }

    const uint32_t tot_bits = (uint32_t)T_TICKS * 64U;
    int32_t rew = (int32_t)(((2LL * (int64_t)matches - (int64_t)tot_bits) * 72LL) / (int64_t)tot_bits);
    int32_t ncost = (int32_t)((K + 7U) >> 3);
    int32_t fuel_next = fuel + rew - ncost - ((fuel >> 6) + 1);
    t.fuel[idx] = (fuel_next < 0) ? 0 : fuel_next;
}

// Kernel 2: 8,192 threads, checkerboard thermodynamic pairing & mitosis (no eval loop!)
__global__ void pair_interact_kernel(TissueSoA t, uint32_t n_pairs, uint32_t pass_mode) {
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

    uint64_t lr = t.rng[oi];
    int32_t fi = t.fuel[oi], fj = t.fuel[oj];
    if (fi <= 0 && fj <= 0) {
        mutate_soa(t, oi, lr); mutate_soa(t, oj, lr);
        t.fuel[oi] = 256;
    } else {
        uint32_t ai = (fi >= fj) ? oi : oj, bi = (fi >= fj) ? oj : oi;
        int32_t fa = t.fuel[ai], fb = t.fuel[bi];
        int32_t na = (int32_t)(xorshift64_dev(lr) & 127ULL), nb = (int32_t)(xorshift64_dev(lr) & 127ULL);

        if (fa >= REPRO_THRESH && fb >= REPRO_THRESH && (xorshift64_dev(lr) % 10ULL) == 0ULL) {
            graft_subgraph_soa(t, ai, bi, lr);
        } else if (fa >= REPRO_THRESH && (fa + na > fb + 256 + nb)) {
            uint32_t Ka = t.node_count[ai];
            t.node_count[bi] = Ka;
            t.out_s[bi] = t.out_s[ai];
            for (uint32_t u = 0; u < Ka; ++u) {
                t.nodes[(size_t)u * TOTAL_CELLS + bi] = t.nodes[(size_t)u * TOTAL_CELLS + ai];
                t.state[(size_t)u * TOTAL_CELLS + bi] = 0ULL;
            }
            if ((xorshift64_dev(lr) % 100ULL) < 50ULL) {
                mutate_soa(t, bi, lr);
            }
            int32_t share = fa >> 1;
            t.fuel[ai] = fa - share;
            t.fuel[bi] = share;
        } else if (fb <= 0) {
            mutate_soa(t, bi, lr);
        }
    }
    t.rng[oi] = lr;
}

static uint32_t eval_org_host(
    const std::vector<uint32_t>& h_nodes, uint32_t K, uint32_t out_s, uint32_t idx,
    const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t, const uint64_t* targ_t,
    int ticks, uint64_t* val_buf
) {
    if (K == 0U) return 0U;
    uint32_t matches = 0U, os = out_s % K;
    for (int tick = 0; tick < ticks; ++tick) {
        const uint64_t i0 = in0_t[tick], i1 = in1_t[tick], ir = in_rst_t[tick];
        for (uint32_t u = 0; u < K; ++u) {
            uint8_t op; uint16_t sa, sb;
            unpack_node(h_nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
            uint64_t a = (sa == 0U) ? i0 : ((sa == 1U) ? i1 : ((sa == 2U) ? ir : val_buf[(sa - 3U) % K]));
            uint64_t b = (sb == 0U) ? i0 : ((sb == 1U) ? i1 : ((sb == 2U) ? ir : val_buf[(sb - 3U) % K]));
            uint64_t r = 0ULL;
            if (op == 0U) r = ~(a & b);
            else if (op == 1U) r = ~(a | b);
            else if (op == 2U) r = a ^ b;
            else r = ~(a ^ b);
            val_buf[u] = r;
        }
        matches += (uint32_t)__builtin_popcountll(~(val_buf[os] ^ targ_t[tick]));
    }
    return matches;
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

static void print_active_circuit_host(const std::vector<uint32_t>& h_nodes, uint32_t K, uint32_t out_s, uint32_t idx) {
    if (K == 0U) return;
    uint32_t root = out_s % K;
    std::set<uint32_t> active_nodes;
    std::vector<uint32_t> stack;
    stack.push_back(root); active_nodes.insert(root);

    while (!stack.empty()) {
        uint32_t u = stack.back(); stack.pop_back();
        uint8_t op; uint16_t sa, sb;
        unpack_node(h_nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U) % K;
            if (active_nodes.find(j) == active_nodes.end()) { active_nodes.insert(j); stack.push_back(j); }
        }
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U) % K;
            if (active_nodes.find(j) == active_nodes.end()) { active_nodes.insert(j); stack.push_back(j); }
        }
    }

    static const char* OPN[4] = {"NAND", "NOR ", "XOR ", "XNOR"};
    std::cout << "    ========================================================\n"
              << "    ACTIVE CONE CIRCUIT DIAGRAM (Total K=" << K << ", Active Nodes=" << active_nodes.size() << ", out=n" << root << "):\n"
              << "    ========================================================\n";
    for (uint32_t u : active_nodes) {
        uint8_t op; uint16_t sa, sb;
        unpack_node(h_nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
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

    std::vector<uint32_t> h_nc(N), h_out(N), h_nodes((size_t)MAX_K * N, 0U);
    std::vector<int32_t> h_fuel(N);
    std::vector<uint64_t> h_rng(N), h_state((size_t)MAX_K * N, 0ULL);

    for (uint32_t i = 0; i < N; ++i) {
        uint64_t s = seed ^ ((uint64_t)(i + 1U) * 0x9E3779B97F4A7C15ULL);
        h_nc[i] = 4;
        for (int u = 0; u < 4; ++u) {
            uint8_t op = (uint8_t)(splitmix64(s) & 3ULL);
            uint16_t sa = (uint16_t)(splitmix64(s) % 7U);
            uint16_t sb = (uint16_t)(splitmix64(s) % 7U);
            h_nodes[(size_t)u * N + i] = pack_node(op, sa, sb);
        }
        h_out[i] = (uint32_t)(splitmix64(s) % 4U);
        h_fuel[i] = 2048;
        uint64_t r = splitmix64(s);
        h_rng[i] = r ? r : 0xCAFEBABE12345678ULL;
    }

    TissueSoA d_t;
    cudaMalloc(&d_t.node_count, N * sizeof(uint32_t));
    cudaMalloc(&d_t.out_s, N * sizeof(uint32_t));
    cudaMalloc(&d_t.fuel, N * sizeof(int32_t));
    cudaMalloc(&d_t.rng, N * sizeof(uint64_t));
    cudaMalloc(&d_t.nodes, (size_t)MAX_K * N * sizeof(uint32_t));
    cudaMalloc(&d_t.state, (size_t)MAX_K * N * sizeof(uint64_t));

    cudaMemcpy(d_t.node_count, h_nc.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.out_s, h_out.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.fuel, h_fuel.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.rng, h_rng.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.nodes, h_nodes.data(), (size_t)MAX_K * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.state, h_state.data(), (size_t)MAX_K * N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t *d_i0, *d_i1, *d_ir, *d_tg, *d_fb_val;
    uint32_t *d_fb_nodes;
    cudaMalloc(&d_fb_val, (size_t)N * MAX_K * sizeof(uint64_t));
    cudaMalloc(&d_fb_nodes, (size_t)N * MAX_K * sizeof(uint32_t));
    cudaMalloc(&d_i0, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_i1, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_ir, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_tg, (size_t)STREAM_BATCH_SWEEPS * T_TICKS * sizeof(uint64_t));

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
    std::cout << "--- GPU #" << gpu_id << " SEED " << seed_idx << " (100% Coalesced SoA 128x128, max_sw=" << max_sweeps << ") ---\n";

    int eval_blocks = (N + EVAL_BLOCK - 1) / EVAL_BLOCK;
    int pair_blocks = (n_pairs + PAIR_BLOCK - 1) / PAIR_BLOCK;
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
                eval_all_cells_kernel<<<eval_blocks, EVAL_BLOCK, 0, stream>>>(
                    d_t, d_i0 + off, d_i1 + off, d_ir + off, d_tg + off, d_fb_val, d_fb_nodes);
                pair_interact_kernel<<<pair_blocks, PAIR_BLOCK, 0, stream>>>(d_t, n_pairs, pm);
            }
            cudaStreamSynchronize(stream);
        }

        int sw = (base_sw == 0) ? 0 : std::min(base_sw + STREAM_BATCH_SWEEPS - 1, max_sweeps);
        if (sw == 0 || sw == 1000 || sw == 2000 || sw == 4000 || sw == 8000 || sw == 12000 || sw == 16000 || sw == 20000 || sw == max_sweeps) {
            cudaStreamSynchronize(stream);
            cudaMemcpy(h_nc.data(), d_t.node_count, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_out.data(), d_t.out_s, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fuel.data(), d_t.fuel, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_nodes.data(), d_t.nodes, (size_t)MAX_K * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            uint64_t erng = 0xDEADBEEFCAFE1234ULL ^ (uint64_t)sw;
            uint64_t ei0[T_TICKS], ei1[T_TICKS], eir[T_TICKS], etg[T_TICKS];
            gen_continuous_stream_host(8, 8, erng, ei0, ei1, eir, etg);

            long long tf = 0, sk = 0; int alive = 0, n90 = 0, n100 = 0;
            uint32_t mink = 999999U, maxk = 0U, bidx = 0U;
            double macc = 0.0, sacc = 0.0;
            uint64_t vc[MAX_K];

            for (uint32_t i = 0; i < N; ++i) {
                tf += h_fuel[i]; if (h_fuel[i] > 0) alive++;
                uint32_t K = h_nc[i];
                sk += K; mink = std::min(mink, K); maxk = std::max(maxk, K);
                for (uint32_t u = 0; u < K; ++u) vc[u] = 0ULL;
                uint32_t matches = eval_org_host(h_nodes, K, h_out[i], i, ei0, ei1, eir, etg, T_TICKS, vc);

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
                    uint32_t bK = h_nc[bidx], bout = h_out[bidx];
                    for (uint32_t u = 0; u < bK; ++u) vc[u] = 0ULL;

                    for (int b = 0; b < 16; ++b) {
                        gen_continuous_stream_host(2, 32, orng, ei0, ei1, eir, etg);
                        for (int tick = 0; tick < T_TICKS; ++tick) {
                            uint32_t m1 = eval_org_host(h_nodes, bK, bout, bidx, &ei0[tick], &ei1[tick], &eir[tick], &etg[tick], 1, vc);
                            om += m1;
                            if ((tick & 31) == 0) om_b0 += m1; else om_rest += m1;
                        }
                    }
                    std::cout << "\n  >>> GPU #" << gpu_id << " OOD FROZEN 32-BIT STREAM (1,024 pairs, NO reset): "
                              << std::setprecision(2) << (100.0 * om / (16LL * 64LL * 64LL)) << "% "
                              << "(rst_bit0=" << (100.0 * om_b0 / (16LL * 2LL * 64LL))
                              << "%, carry_bits_1..31=" << (100.0 * om_rest / (16LL * 62LL * 64LL)) << "%)\n";
                    print_active_circuit_host(h_nodes, bK, bout, bidx);
                }
            }
        }
        if (base_sw == 0) base_sw = 1 - STREAM_BATCH_SWEEPS;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_i0); cudaFreeHost(h_i1); cudaFreeHost(h_ir); cudaFreeHost(h_tg);
    cudaFree(d_t.node_count); cudaFree(d_t.out_s); cudaFree(d_t.fuel); cudaFree(d_t.rng);
    cudaFree(d_t.nodes); cudaFree(d_t.state);
    cudaFree(d_fb_val); cudaFree(d_fb_nodes);
    cudaFree(d_i0); cudaFree(d_i1); cudaFree(d_ir); cudaFree(d_tg);
    #pragma omp critical
    std::cout << "GPU #" << gpu_id << " Elapsed: " << std::setprecision(2)
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() << "s\n" << std::flush;
}

int main(int argc, char** argv) {
    int max_sweeps = (argc > 1) ? std::atoi(argv[1]) : 24000;
    std::cout << "========================================================================\n"
              << "KARYON-CORE EXP-1170h: PURE SoA COALESCED DUAL-GPU BITWISE TISSUE\n"
              << "Standard: KEP v51.0 | Stride-1 Coalesced SoA | Shared Node+Val | max_sw=" << max_sweeps << "\n"
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
