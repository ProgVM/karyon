#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <set>
#include <string>
#include <cuda_runtime.h>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1172: PURE UNBROKEN CONTINUOUS LIVING TISSUE (NO WRAPPERS)
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// Architecture & Scientific Protocol:
// 1. One Unbroken Physical Timeline (t = 0 .. 1,536,000 ticks, 12,000 sweeps):
//    - ZERO train/test split, ZERO freeze-and-extract N=1 wrappers.
//    - ZERO 120-step evaluation windows, ZERO M_BRANCHES pools, ZERO hardcoded thresholds.
//    - Continuous World Evolution on 128x128 spatial torus (16,384 living cells per GPU):
//        * Phase I   (t = 0 .. 512k):      Stationary Home World (Regime 0)
//        * Phase II  (t = 512k .. 1024k):  Live Non-Stationary Regime Shifts (0..3 every 2048 ticks)
//        * Phase III (t = 1024k .. 1536k): Live Full 64-bit Wire Permutation perm[64] (zero resets!)
// 2. Direct Side-by-Side Comparison (No Hardcoded Thresholds):
//    - Mode A (Pure Baseline): Uniform random sensory wire mutation across 0..63.
//    - Mode B (ContinuousWireField, F -> 0): Every cell updates continuous 1-step bit prediction
//      accuracy pred_acc[w] on every tick t (tau ~ 125). Mutations sample wires smoothly via
//      continuous Boltzmann excess predictability (max(0, pred_acc[w] - 0.52))^3 + eps.
//    - Direct thermodynamic fuel update (no per-cell r_bar baseline subtraction).
// 3. Dual-GPU Parallel Execution:
//    - GPU #0: Grid2D 8x8 (44 bits of cryptographic noise + 20 physical wires)
//    - GPU #1: ActiveWorld 1D 16-ring (50 bits of cryptographic noise + 14 physical wires)

#define MAX_K 64
#define EVAL_BLOCK 64
#define PAIR_BLOCK 64
#define GRID_W 128
#define GRID_H 128
#define TOTAL_CELLS (GRID_W * GRID_H)
#define REPRO_THRESH 1024
#define T_STEPS_PER_SWEEP 128
#define SHIFT_PERIOD 2048

struct WorldState {
    uint8_t  ax, ay;
    uint8_t  fx, fy;
    uint8_t  tx0, ty0;
    int8_t   last_dE_sign;
    uint8_t  prev_act;
    uint64_t prev_obs;
    uint64_t wrng;
    uint32_t food_hits;
    uint32_t toxin_hits;
    uint32_t deaths;
};

struct ContinuousTissueSoA {
    uint32_t*   node_count;
    uint32_t*   out0;
    uint32_t*   out1;
    int32_t*    fuel;
    uint64_t*   rng;
    uint32_t*   nodes;
    uint64_t*   state;
    WorldState* world;
    // ContinuousWireField SoA: 100% stride-1 coalesced across threads
    // pred_acc: [64 * TOTAL_CELLS]
    // pred_ema: [64 * 4 * 2 * TOTAL_CELLS] = [512 * TOTAL_CELLS]
    float*      pred_acc;
    float*      pred_ema;
    uint8_t*    d_perm64;
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

// Smooth Boltzmann wire sampling driven purely by excess predictability (pred_acc[w] - 0.52)+^3
// Zero discrete cutoffs or thresholds
__device__ static inline uint16_t sample_wire_continuous(
    const ContinuousTissueSoA& t, uint32_t idx, uint64_t& rng, bool use_wire_field
) {
    if (!use_wire_field) {
        return (uint16_t)(xorshift64_dev(rng) % 64ULL);
    }
    float tot = 0.0f;
    for (int w = 0; w < 64; ++w) {
        float ex = fmaxf(0.0f, t.pred_acc[(size_t)w * TOTAL_CELLS + idx] - 0.52f);
        tot += ex * ex * ex + 1e-5f;
    }
    float r = ((float)(xorshift64_dev(rng) & 0xFFFFFFULL) / 16777216.0f) * tot;
    float acc = 0.0f;
    for (int w = 0; w < 64; ++w) {
        float ex = fmaxf(0.0f, t.pred_acc[(size_t)w * TOTAL_CELLS + idx] - 0.52f);
        acc += ex * ex * ex + 1e-5f;
        if (acc >= r) return (uint16_t)w;
    }
    return 63U;
}

__device__ static void apoptosis_embodied(ContinuousTissueSoA& t, uint32_t idx) {
    uint32_t K = t.node_count[idx];
    if (K <= 8U) return;
    uint32_t r0 = t.out0[idx] % K;
    uint32_t r1 = t.out1[idx] % K;

    uint64_t active = (1ULL << r0) | (1ULL << r1);
    uint64_t stack = active;

    while (stack) {
        uint32_t u = 63U - (uint32_t)__clzll(stack);
        stack &= ~(1ULL << u);
        uint8_t op; uint16_t sa, sb;
        unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
        if (sa >= 64U) {
            uint32_t j = (uint32_t)(sa - 64U) % K;
            if (!((active >> j) & 1ULL)) { active |= (1ULL << j); stack |= (1ULL << j); }
        }
        if (sb >= 64U) {
            uint32_t j = (uint32_t)(sb - 64U) % K;
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
            if (sa >= 64U) sa = (uint16_t)(new_idx[(uint32_t)(sa - 64U) % K] + 64U);
            if (sb >= 64U) sb = (uint16_t)(new_idx[(uint32_t)(sb - 64U) % K] + 64U);
            t.nodes[(size_t)dst * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
            dst++;
        }
    }
    t.node_count[idx] = new_K;
    t.out0[idx] = (uint32_t)new_idx[r0];
    t.out1[idx] = (uint32_t)new_idx[r1];
}

__device__ static void wire_split_mitosis_embodied(
    ContinuousTissueSoA& t, uint32_t idx, uint64_t& rng, bool use_wire_field
) {
    uint32_t K = t.node_count[idx];
    if (K >= MAX_K - 2U || K == 0U) return;

    uint32_t rtype = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (rtype < 65U) {
        uint32_t v = (uint32_t)(xorshift64_dev(rng) % K);
        uint32_t pos = v;
        uint8_t op_v; uint16_t sa_v, sb_v;
        unpack_node(t.nodes[(size_t)v * TOTAL_CELLS + idx], op_v, sa_v, sb_v);

        bool use_a = (xorshift64_dev(rng) & 1ULL) != 0ULL;
        uint16_t orig_wire = use_a ? sa_v : sb_v;

        uint16_t second_wire = (xorshift64_dev(rng) & 1ULL)
            ? sample_wire_continuous(t, idx, rng, use_wire_field)
            : (uint16_t)((xorshift64_dev(rng) % K) + 64ULL);
        uint8_t new_op = (uint8_t)(xorshift64_dev(rng) & 3ULL);

        for (uint32_t i = K; i > pos; --i) {
            t.nodes[(size_t)i * TOTAL_CELLS + idx] = t.nodes[(size_t)(i - 1U) * TOTAL_CELLS + idx];
        }

        uint32_t nK = K + 1U;
        t.node_count[idx] = nK;

        auto shift_wire = [&](uint16_t w) -> uint16_t {
            if (w >= 64U) {
                uint32_t j = (uint32_t)(w - 64U);
                if (j >= pos) w = (uint16_t)(j + 1U + 64U);
            }
            if (w >= 64U + nK) w = (uint16_t)((w - 64U) % nK + 64U);
            return w;
        };

        for (uint32_t u = 0; u < nK; ++u) {
            if (u == pos) continue;
            uint8_t nop; uint16_t nsa, nsb;
            unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], nop, nsa, nsb);
            nsa = shift_wire(nsa); nsb = shift_wire(nsb);
            t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(nop, nsa, nsb);
        }
        if (t.out0[idx] >= pos) t.out0[idx]++;
        if (t.out0[idx] >= nK) t.out0[idx] %= nK;
        if (t.out1[idx] >= pos) t.out1[idx]++;
        if (t.out1[idx] >= nK) t.out1[idx] %= nK;

        uint16_t new_sa = shift_wire(orig_wire);
        uint16_t new_sb = shift_wire(second_wire);
        t.nodes[(size_t)pos * TOTAL_CELLS + idx] = pack_node(new_op, new_sa, new_sb);

        uint32_t new_v = pos + 1U;
        uint8_t vop; uint16_t vsa, vsb;
        unpack_node(t.nodes[(size_t)new_v * TOTAL_CELLS + idx], vop, vsa, vsb);
        if (use_a) vsa = (uint16_t)(pos + 64U);
        else vsb = (uint16_t)(pos + 64U);
        t.nodes[(size_t)new_v * TOTAL_CELLS + idx] = pack_node(vop, vsa, vsb);
    } else {
        uint8_t op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        uint16_t sa = (xorshift64_dev(rng) & 1ULL)
            ? sample_wire_continuous(t, idx, rng, use_wire_field)
            : (uint16_t)((xorshift64_dev(rng) % K) + 64ULL);
        uint16_t sb = (xorshift64_dev(rng) & 1ULL)
            ? sample_wire_continuous(t, idx, rng, use_wire_field)
            : (uint16_t)((xorshift64_dev(rng) % K) + 64ULL);
        t.nodes[(size_t)K * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
        t.node_count[idx] = K + 1U;
        uint32_t new_u = K;
        uint32_t ro = (uint32_t)(xorshift64_dev(rng) % 100ULL);
        if (ro < 50U && t.node_count[idx] < MAX_K) {
            bool target_out0 = (xorshift64_dev(rng) & 1ULL) != 0ULL;
            uint32_t old_root = target_out0 ? (t.out0[idx] % K) : (t.out1[idx] % K);
            uint16_t s_old = (uint16_t)(old_root + 64U);
            uint16_t s_new = (uint16_t)(new_u + 64U);
            uint8_t cop = (uint8_t)(xorshift64_dev(rng) & 3ULL);
            uint32_t comb_u = t.node_count[idx];
            t.nodes[(size_t)comb_u * TOTAL_CELLS + idx] = pack_node(cop, s_old, s_new);
            t.node_count[idx]++;
            if (target_out0) t.out0[idx] = comb_u; else t.out1[idx] = comb_u;
        } else if (ro < 80U) {
            if (xorshift64_dev(rng) & 1ULL) t.out0[idx] = new_u; else t.out1[idx] = new_u;
        }
    }
}

__device__ static void prune_one_node_embodied(ContinuousTissueSoA& t, uint32_t idx, uint64_t& rng) {
    uint32_t K = t.node_count[idx];
    if (K <= 4U) return;
    uint32_t du = (uint32_t)(xorshift64_dev(rng) % K);
    for (uint32_t i = du; i < K - 1U; ++i) {
        t.nodes[(size_t)i * TOTAL_CELLS + idx] = t.nodes[(size_t)(i + 1U) * TOTAL_CELLS + idx];
    }
    uint32_t nK = K - 1U;
    t.node_count[idx] = nK;
    for (uint32_t u = 0; u < nK; ++u) {
        uint8_t op; uint16_t sa, sb;
        unpack_node(t.nodes[(size_t)u * TOTAL_CELLS + idx], op, sa, sb);
        if (sa >= 64U) {
            uint32_t j = (uint32_t)(sa - 64U);
            if (j == du) sa = (uint16_t)((xorshift64_dev(rng) % nK) + 64U);
            else if (j > du) sa--;
            if (sa >= 64U + nK) sa = (uint16_t)((sa - 64U) % nK + 64U);
        }
        if (sb >= 64U) {
            uint32_t j = (uint32_t)(sb - 64U);
            if (j == du) sb = (uint16_t)((xorshift64_dev(rng) % nK) + 64U);
            else if (j > du) sb--;
            if (sb >= 64U + nK) sb = (uint16_t)((sb - 64U) % nK + 64U);
        }
        t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
    }
    if (t.out0[idx] == du) t.out0[idx] = (uint32_t)(xorshift64_dev(rng) % nK);
    else if (t.out0[idx] > du) t.out0[idx]--;
    if (t.out0[idx] >= nK) t.out0[idx] %= nK;

    if (t.out1[idx] == du) t.out1[idx] = (uint32_t)(xorshift64_dev(rng) % nK);
    else if (t.out1[idx] > du) t.out1[idx]--;
    if (t.out1[idx] >= nK) t.out1[idx] %= nK;
}

__device__ static void graft_subgraph_embodied(
    ContinuousTissueSoA& t, uint32_t donor_idx, uint32_t host_idx, uint64_t& rng
) {
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
        if (sa >= 64U) {
            uint32_t j = (uint32_t)(sa - 64U);
            if (j >= su && j < su + cnt) sa = (uint16_t)(start_kh + (j - su) + 64U);
            else sa = (uint16_t)((j % start_kh) + 64U);
            if (sa >= 64U + target_k) sa = (uint16_t)((sa - 64U) % target_k + 64U);
        }
        if (sb >= 64U) {
            uint32_t j = (uint32_t)(sb - 64U);
            if (j >= su && j < su + cnt) sb = (uint16_t)(start_kh + (j - su) + 64U);
            else sb = (uint16_t)((j % start_kh) + 64U);
            if (sb >= 64U + target_k) sb = (uint16_t)((sb - 64U) % target_k + 64U);
        }
        uint32_t dst_u = t.node_count[host_idx];
        t.nodes[(size_t)dst_u * TOTAL_CELLS + host_idx] = pack_node(op, sa, sb);
        t.node_count[host_idx] = dst_u + 1U;
    }

    uint32_t new_last = t.node_count[host_idx] - 1U;
    uint32_t ro_opt = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (ro_opt < 45U && t.node_count[host_idx] < MAX_K) {
        bool target_out0 = (xorshift64_dev(rng) & 1ULL) != 0ULL;
        uint32_t old_root = target_out0 ? (t.out0[host_idx] % start_kh) : (t.out1[host_idx] % start_kh);
        uint16_t src_old = (uint16_t)(old_root + 64U);
        uint16_t src_new = (uint16_t)(new_last + 64U);
        uint8_t comb_op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        uint32_t cu = t.node_count[host_idx];
        t.nodes[(size_t)cu * TOTAL_CELLS + host_idx] = pack_node(comb_op, src_old, src_new);
        t.node_count[host_idx] = cu + 1U;
        if (target_out0) t.out0[host_idx] = cu; else t.out1[host_idx] = cu;
    } else if (ro_opt < 75U) {
        if (xorshift64_dev(rng) & 1ULL) t.out0[host_idx] = new_last; else t.out1[host_idx] = new_last;
    }
    apoptosis_embodied(t, host_idx);
}

__device__ static void mutate_embodied(
    ContinuousTissueSoA& t, uint32_t idx, uint64_t& rng, bool use_wire_field
) {
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
        uint16_t ns = (xorshift64_dev(rng) & 1ULL)
            ? sample_wire_continuous(t, idx, rng, use_wire_field)
            : (uint16_t)((xorshift64_dev(rng) % K) + 64ULL);
        if (xorshift64_dev(rng) & 1ULL) sa = ns; else sb = ns;
        t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
    } else if (r < 80U) {
        wire_split_mitosis_embodied(t, idx, rng, use_wire_field);
    } else if (r < 88U) {
        graft_subgraph_embodied(t, idx, idx, rng);
    } else if (r < 97U) {
        prune_one_node_embodied(t, idx, rng);
        if (r >= 93U) prune_one_node_embodied(t, idx, rng);
    } else {
        if (xorshift64_dev(rng) & 1ULL) t.out0[idx] = pick_upper_biased_node(K, rng);
        else t.out1[idx] = pick_upper_biased_node(K, rng);
    }
    if (t.node_count[idx] > 16U) apoptosis_embodied(t, idx);
}

__global__ void continuous_tissue_tick_kernel(
    ContinuousTissueSoA t, uint32_t global_sweep_idx, uint32_t env_mode, bool use_wire_field
) {
    __shared__ uint32_t sh_nodes[EVAL_BLOCK][MAX_K];
    __shared__ uint8_t  sh_perm[64];

    if (threadIdx.x < 64) {
        sh_perm[threadIdx.x] = t.d_perm64[threadIdx.x];
    }
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_CELLS) return;

    int32_t fuel = t.fuel[idx];
    if (fuel <= 0) return;

    const uint32_t K = t.node_count[idx];
    if (K == 0U) return;

    const uint32_t o0 = t.out0[idx] % K;
    const uint32_t o1 = t.out1[idx] % K;

    for (uint32_t u = 0; u < K; ++u) {
        uint32_t p = t.nodes[(size_t)u * TOTAL_CELLS + idx];
        uint8_t op = (uint8_t)(p & 3U);
        uint16_t sa = (uint16_t)((p >> 2) & 0x7FFFU);
        uint16_t sb = (uint16_t)((p >> 17) & 0x7FFFU);
        if (sa >= 64U + K) sa = (uint16_t)((sa - 64U) % K + 64U);
        if (sb >= 64U + K) sb = (uint16_t)((sb - 64U) % K + 64U);
        sh_nodes[threadIdx.x][u] = pack_node(op, sa, sb);
    }

    uint64_t gs = t.state[(size_t)idx];
    WorldState w = t.world[idx];

    int32_t sweep_delta_fuel = 0;

    for (int step = 0; step < T_STEPS_PER_SWEEP; ++step) {
        uint64_t tick = (uint64_t)global_sweep_idx * T_STEPS_PER_SWEEP + (uint64_t)step;

        // Unbroken continuous timeline (t = 0 .. 1,536,000):
        //   0 .. 512k      : Stationary Home World (Regime 0)
        //   512k .. 1024k  : Live Non-Stationary Regime Shifts every 2048 ticks (Regime 0..3)
        //   1024k .. 1536k : Live Full 64-bit Wire Permutation perm64 (Regime 0, zero resets)
        uint32_t regime = (tick >= 512000ULL && tick < 1024000ULL)
            ? (uint32_t)(((tick - 512000ULL) / SHIFT_PERIOD) % 4ULL)
            : 0U;
        bool apply_perm = (tick >= 1024000ULL);
        bool a_is_food = (regime == 0U || regime == 2U);
        bool invert_motors = (regime == 2U || regime == 3U);

        uint64_t noise = xorshift64_dev(w.wrng);
        uint64_t raw_obs = 0ULL;

        if (env_mode == 0U) {
            uint64_t gain_b = (w.last_dE_sign > 0) ? 1ULL : 0ULL;
            uint64_t pain_b = (w.last_dE_sign < 0) ? 1ULL : 0ULL;
            raw_obs = (noise & ~0xFFFFFULL) | (gain_b << 18) | (pain_b << 19)
                | (uint64_t)(w.ax & 7U) | ((uint64_t)(w.ay & 7U) << 3)
                | ((uint64_t)(w.fx & 7U) << 6) | ((uint64_t)(w.fy & 7U) << 9)
                | ((uint64_t)(w.tx0 & 7U) << 12) | ((uint64_t)(w.ty0 & 7U) << 15);
        } else {
            uint64_t gain_b = (w.last_dE_sign > 0) ? 1ULL : 0ULL;
            uint64_t pain_b = (w.last_dE_sign < 0) ? 1ULL : 0ULL;
            raw_obs = (noise & ~0x3FFFULL) | (gain_b << 12) | (pain_b << 13)
                | (uint64_t)(w.ax & 15U) | ((uint64_t)(w.fx & 15U) << 4) | ((uint64_t)(w.tx0 & 15U) << 8);
        }

        uint64_t obs = raw_obs;
        if (apply_perm) {
            uint64_t p_obs = 0ULL;
            #pragma unroll
            for (int b = 0; b < 64; ++b) {
                if ((raw_obs >> b) & 1ULL) p_obs |= (1ULL << sh_perm[b]);
            }
            obs = p_obs;
        }

        // Continuous F -> 0 prediction dissipation on EVERY tick t
        // Exact causal alignment: (prev_obs, prev_act) -> obs
        if (use_wire_field && tick > 0ULL) {
            uint32_t pa = (uint32_t)(w.prev_act & 3U);
            for (int b = 0; b < 64; ++b) {
                uint32_t pb = (uint32_t)((w.prev_obs >> b) & 1ULL);
                uint32_t cur = (uint32_t)((obs >> b) & 1ULL);
                size_t ema_idx = (size_t)((b * 8U) + (pa * 2U) + pb) * TOTAL_CELLS + idx;
                size_t acc_idx = (size_t)b * TOTAL_CELLS + idx;

                float p_est = t.pred_ema[ema_idx];
                float hit = ((p_est >= 0.0f) == (cur == 1U)) ? 1.0f : 0.0f;
                t.pred_acc[acc_idx] = 0.992f * t.pred_acc[acc_idx] + 0.008f * hit;
                t.pred_ema[ema_idx] = 0.96f * p_est + 0.04f * (cur ? 1.0f : -1.0f);
            }
        }

        for (uint32_t u = 0; u < K; ++u) {
            uint32_t p = sh_nodes[threadIdx.x][u];
            uint8_t op = (uint8_t)(p & 3U);
            uint32_t sa = (p >> 2) & 0x7FFFU;
            uint32_t sb = (p >> 17) & 0x7FFFU;

            uint64_t a = (sa < 64U) ? ((obs >> sa) & 1ULL) : ((gs >> (sa - 64U)) & 1ULL);
            uint64_t b = (sb < 64U) ? ((obs >> sb) & 1ULL) : ((gs >> (sb - 64U)) & 1ULL);

            uint64_t r = 0ULL;
            if (op == 0U) r = ~(a & b) & 1ULL;
            else if (op == 1U) r = ~(a | b) & 1ULL;
            else if (op == 2U) r = (a ^ b) & 1ULL;
            else r = (~(a ^ b)) & 1ULL;

            gs = (gs & ~(1ULL << u)) | (r << u);
        }

        uint8_t m0 = (uint8_t)((gs >> o0) & 1ULL);
        uint8_t m1 = (uint8_t)((gs >> o1) & 1ULL);
        uint8_t act = (m1 << 1) | m0;

        w.prev_obs = obs;
        w.prev_act = act;

        int32_t dE = 0;
        if ((step & 3) == 0) dE -= 1;

        if (env_mode == 0U) {
            if (invert_motors) {
                if (act == 0) act = 1; else if (act == 1) act = 0;
                else if (act == 2) act = 3; else if (act == 3) act = 2;
            }
            if (act == 0) w.ay = (w.ay - 1U) & 7U;
            else if (act == 1) w.ay = (w.ay + 1U) & 7U;
            else if (act == 2) w.ax = (w.ax - 1U) & 7U;
            else if (act == 3) w.ax = (w.ax + 1U) & 7U;

            if ((step & 1) == 0) {
                uint32_t td = (uint32_t)(xorshift64_dev(w.wrng) & 3ULL);
                if (td == 0) w.ty0 = (w.ty0 - 1U) & 7U;
                else if (td == 1) w.ty0 = (w.ty0 + 1U) & 7U;
                else if (td == 2) w.tx0 = (w.tx0 - 1U) & 7U;
                else w.tx0 = (w.tx0 + 1U) & 7U;
            }

            if (w.ax == w.fx && w.ay == w.fy) {
                if (a_is_food) { dE += 48; w.food_hits++; }
                else { dE -= 64; w.toxin_hits++; }
                w.fx = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
                w.fy = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
            }
            if (w.ax == w.tx0 && w.ay == w.ty0) {
                if (a_is_food) { dE -= 64; w.toxin_hits++; }
                else { dE += 48; w.food_hits++; }
                w.tx0 = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
                w.ty0 = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
            }
        } else {
            int32_t dp = 0;
            if (act == 1) dp = invert_motors ? 1 : -1;
            else if (act == 2) dp = invert_motors ? -1 : 1;
            w.ax = (uint8_t)((int32_t)w.ax + dp + 16) & 15U;

            if (w.ax == w.fx) {
                if (a_is_food) { dE += 48; w.food_hits++; }
                else { dE -= 64; w.toxin_hits++; }
                w.fx = (uint8_t)(xorshift64_dev(w.wrng) & 15ULL);
            }
            if (w.ax == w.tx0) {
                if (a_is_food) { dE -= 64; w.toxin_hits++; }
                else { dE += 48; w.food_hits++; }
                w.tx0 = (uint8_t)(xorshift64_dev(w.wrng) & 15ULL);
            }
        }

        w.last_dE_sign = (dE > 0) ? 1 : ((dE < -1) ? -1 : 0);
        sweep_delta_fuel += dE;
    }

    int32_t ncost = (int32_t)((K + 7U) >> 3);
    int32_t fuel_next = fuel + sweep_delta_fuel - ncost - ((fuel >> 6) + 1);
    if (fuel_next <= 0) {
        fuel_next = 0;
        w.deaths++;
    }
    t.state[(size_t)idx] = gs;
    t.world[idx] = w;
    t.fuel[idx] = fuel_next;
}

__global__ void pair_interact_continuous_kernel(
    ContinuousTissueSoA t, uint32_t n_pairs, uint32_t pass_mode, uint32_t env_mode, bool use_wire_field
) {
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

    auto respawn_world_keep_sensors = [&](uint32_t target_idx) {
        WorldState& w = t.world[target_idx];
        if (env_mode == 0U) {
            w.ax = (uint8_t)(xorshift64_dev(lr) & 7ULL); w.ay = (uint8_t)(xorshift64_dev(lr) & 7ULL);
            w.fx = (uint8_t)(xorshift64_dev(lr) & 7ULL); w.fy = (uint8_t)(xorshift64_dev(lr) & 7ULL);
            w.tx0 = (uint8_t)(xorshift64_dev(lr) & 7ULL); w.ty0 = (uint8_t)(xorshift64_dev(lr) & 7ULL);
        } else {
            w.ax = (uint8_t)(xorshift64_dev(lr) & 15ULL);
            w.fx = (uint8_t)(xorshift64_dev(lr) & 15ULL);
            w.tx0 = (uint8_t)(xorshift64_dev(lr) & 15ULL);
        }
        w.last_dE_sign = 0;
        t.state[(size_t)target_idx] = 0ULL;
    };

    if (fi <= 0 && fj <= 0) {
        mutate_embodied(t, oi, lr, use_wire_field);
        mutate_embodied(t, oj, lr, use_wire_field);
        t.fuel[oi] = 512;
        respawn_world_keep_sensors(oi);
    } else {
        uint32_t ai = (fi >= fj) ? oi : oj, bi = (fi >= fj) ? oj : oi;
        int32_t fa = t.fuel[ai], fb = t.fuel[bi];
        int32_t na = (int32_t)(xorshift64_dev(lr) & 127ULL), nb = (int32_t)(xorshift64_dev(lr) & 127ULL);

        if (fa >= REPRO_THRESH && fb >= REPRO_THRESH && (xorshift64_dev(lr) % 10ULL) == 0ULL) {
            graft_subgraph_embodied(t, ai, bi, lr);
        } else if (fa >= REPRO_THRESH && (fa + na > fb + 256 + nb)) {
            uint32_t Ka = t.node_count[ai];
            t.node_count[bi] = Ka;
            t.out0[bi] = t.out0[ai];
            t.out1[bi] = t.out1[ai];
            for (uint32_t u = 0; u < Ka; ++u) {
                t.nodes[(size_t)u * TOTAL_CELLS + bi] = t.nodes[(size_t)u * TOTAL_CELLS + ai];
            }
            if ((xorshift64_dev(lr) % 100ULL) < 50ULL) {
                mutate_embodied(t, bi, lr, use_wire_field);
            }
            respawn_world_keep_sensors(bi);
            int32_t share = fa >> 1;
            t.fuel[ai] = fa - share;
            t.fuel[bi] = share;
        } else if (fb <= 0) {
            mutate_embodied(t, bi, lr, use_wire_field);
            t.fuel[bi] = 256;
            respawn_world_keep_sensors(bi);
        }
    }
    t.rng[oi] = lr;
}

static void run_continuous_gpu(
    int gpu_id, uint32_t env_mode, bool use_wire_field, int max_sweeps, uint64_t seed
) {
    cudaSetDevice(gpu_id);
    const uint32_t N = TOTAL_CELLS, n_pairs = N / 2;

    std::vector<uint8_t> h_perm64(64);
    for (int b = 0; b < 64; ++b) h_perm64[b] = (uint8_t)b;
    uint64_t prng = 0x123456789ABCDEF0ULL;
    for (int b = 63; b > 0; --b) {
        prng = prng * 6364136223846793005ULL + 1442695040888963407ULL;
        int j = (int)(prng % (uint64_t)(b + 1));
        uint8_t tmp = h_perm64[b]; h_perm64[b] = h_perm64[j]; h_perm64[j] = tmp;
    }

    std::vector<uint32_t> h_nc(N), h_out0(N), h_out1(N), h_nodes((size_t)MAX_K * N, 0U);
    std::vector<int32_t> h_fuel(N);
    std::vector<uint64_t> h_rng(N), h_state(N, 0ULL);
    std::vector<WorldState> h_world(N);
    std::vector<float> h_pred_acc((size_t)64 * N, 0.5f);
    std::vector<float> h_pred_ema;
    if (use_wire_field) h_pred_ema.assign((size_t)512 * N, 0.0f);

    for (uint32_t i = 0; i < N; ++i) {
        uint64_t s = seed ^ ((uint64_t)(i + 1U) * 0x9E3779B97F4A7C15ULL);
        h_nc[i] = 4;
        for (int u = 0; u < 4; ++u) {
            uint8_t op = (uint8_t)(splitmix64(s) & 3ULL);
            uint16_t sa = (uint16_t)(splitmix64(s) % 68U);
            uint16_t sb = (uint16_t)(splitmix64(s) % 68U);
            h_nodes[(size_t)u * N + i] = pack_node(op, sa, sb);
        }
        h_out0[i] = (uint32_t)(splitmix64(s) % 4U);
        h_out1[i] = (uint32_t)(splitmix64(s) % 4U);
        h_fuel[i] = 1024;
        uint64_t r = splitmix64(s);
        h_rng[i] = r ? r : 0xCAFEBABE12345678ULL;

        WorldState& w = h_world[i];
        w.wrng = splitmix64(s);
        if (env_mode == 0U) {
            w.ax = (uint8_t)(splitmix64(s) & 7ULL); w.ay = (uint8_t)(splitmix64(s) & 7ULL);
            w.fx = (uint8_t)(splitmix64(s) & 7ULL); w.fy = (uint8_t)(splitmix64(s) & 7ULL);
            w.tx0 = (uint8_t)(splitmix64(s) & 7ULL); w.ty0 = (uint8_t)(splitmix64(s) & 7ULL);
        } else {
            w.ax = (uint8_t)(splitmix64(s) & 15ULL);
            w.fx = (uint8_t)(splitmix64(s) & 15ULL);
            w.tx0 = (uint8_t)(splitmix64(s) & 15ULL);
        }
        w.last_dE_sign = 0;
        w.prev_act = 0;
        w.prev_obs = 0ULL;
        w.food_hits = 0U;
        w.toxin_hits = 0U;
        w.deaths = 0U;
    }

    ContinuousTissueSoA d_t;
    cudaMalloc(&d_t.node_count, N * sizeof(uint32_t));
    cudaMalloc(&d_t.out0, N * sizeof(uint32_t));
    cudaMalloc(&d_t.out1, N * sizeof(uint32_t));
    cudaMalloc(&d_t.fuel, N * sizeof(int32_t));
    cudaMalloc(&d_t.rng, N * sizeof(uint64_t));
    cudaMalloc(&d_t.nodes, (size_t)MAX_K * N * sizeof(uint32_t));
    cudaMalloc(&d_t.state, N * sizeof(uint64_t));
    cudaMalloc(&d_t.world, N * sizeof(WorldState));
    cudaMalloc(&d_t.d_perm64, 64 * sizeof(uint8_t));
    if (use_wire_field) {
        cudaMalloc(&d_t.pred_acc, (size_t)64 * N * sizeof(float));
        cudaMalloc(&d_t.pred_ema, (size_t)512 * N * sizeof(float));
    } else {
        d_t.pred_acc = nullptr;
        d_t.pred_ema = nullptr;
    }

    cudaMemcpy(d_t.node_count, h_nc.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.out0, h_out0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.out1, h_out1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.fuel, h_fuel.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.rng, h_rng.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.nodes, h_nodes.data(), (size_t)MAX_K * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.state, h_state.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.world, h_world.data(), N * sizeof(WorldState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.d_perm64, h_perm64.data(), 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (use_wire_field) {
        cudaMemcpy(d_t.pred_acc, h_pred_acc.data(), (size_t)64 * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t.pred_ema, h_pred_ema.data(), (size_t)512 * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp critical
    std::cout << "--- GPU #" << gpu_id << " (" << (env_mode == 0 ? "Grid2D 8x8" : "ActiveWorld 1D")
              << " | " << (use_wire_field ? "ContinuousWireField F->0" : "Pure Baseline")
              << " | Unbroken t=0.." << ((uint64_t)max_sweeps * T_STEPS_PER_SWEEP / 1000) << "k) ---\n" << std::flush;

    int eval_blocks = (N + EVAL_BLOCK - 1) / EVAL_BLOCK;
    int pair_blocks = (n_pairs + PAIR_BLOCK - 1) / PAIR_BLOCK;
    uint64_t hrng = seed ^ 0xFEEDFACE12345678ULL;

    uint64_t prev_food = 0ULL, prev_toxin = 0ULL, prev_deaths = 0ULL;

    for (int sw = 0; sw <= max_sweeps; ++sw) {
        uint32_t pm = (uint32_t)(xorshift64_dev(hrng) & 3ULL);
        continuous_tissue_tick_kernel<<<eval_blocks, EVAL_BLOCK, 0, stream>>>(d_t, sw, env_mode, use_wire_field);
        pair_interact_continuous_kernel<<<pair_blocks, PAIR_BLOCK, 0, stream>>>(d_t, n_pairs, pm, env_mode, use_wire_field);

        if (sw == 0 || sw == 500 || sw == 1000 || sw == 2000 || sw == 3999 ||
            sw == 4500 || sw == 6000 || sw == 7999 ||
            sw == 8250 || sw == 8500 || sw == 9000 || sw == 10000 || sw == max_sweeps) {
            cudaStreamSynchronize(stream);
            cudaMemcpy(h_nc.data(), d_t.node_count, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fuel.data(), d_t.fuel, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_world.data(), d_t.world, N * sizeof(WorldState), cudaMemcpyDeviceToHost);
            if (use_wire_field) {
                cudaMemcpy(h_pred_acc.data(), d_t.pred_acc, (size_t)64 * N * sizeof(float), cudaMemcpyDeviceToHost);
            }

            long long tf = 0, sk = 0;
            int alive = 0, bidx = 0;
            int32_t max_f = -999999;
            uint64_t tot_food = 0ULL, tot_toxin = 0ULL, tot_deaths = 0ULL;
            for (uint32_t i = 0; i < N; ++i) {
                tf += h_fuel[i];
                if (h_fuel[i] > 0) alive++;
                sk += h_nc[i];
                if (h_fuel[i] > max_f) { max_f = h_fuel[i]; bidx = i; }
                tot_food += h_world[i].food_hits;
                tot_toxin += h_world[i].toxin_hits;
                tot_deaths += h_world[i].deaths;
            }

            uint64_t df = tot_food - prev_food;
            uint64_t dt = tot_toxin - prev_toxin;
            uint64_t dd = tot_deaths - prev_deaths;
            prev_food = tot_food; prev_toxin = tot_toxin; prev_deaths = tot_deaths;

            double ratio = (dt > 0ULL) ? ((double)df / (double)dt) : (double)df;
            uint64_t tick = (uint64_t)(sw + 1) * T_STEPS_PER_SWEEP;
            const char* phase_tag = (tick <= 512000ULL) ? "HOME " : ((tick <= 1024000ULL) ? "SHIFT" : "PERM ");

            float min_sig = 1.0f, max_sig = 0.0f, max_noi = 0.0f;
            if (use_wire_field) {
                int n_sig = (env_mode == 0U) ? 20 : 14;
                bool is_perm = (tick > 1024000ULL);
                for (int b = 0; b < 64; ++b) {
                    int wire_id = is_perm ? h_perm64[b] : b;
                    float pa = h_pred_acc[(size_t)wire_id * N + bidx];
                    if (b < n_sig) {
                        if (pa < min_sig) min_sig = pa;
                        if (pa > max_sig) max_sig = pa;
                    } else {
                        if (pa > max_noi) max_noi = pa;
                    }
                }
            }

            #pragma omp critical
            {
                std::cout << "[GPU#" << gpu_id << " " << (env_mode == 0 ? "2D" : "1D")
                          << " " << (use_wire_field ? "CWF" : "BAS") << " " << phase_tag << "] "
                          << "t=" << std::setw(4) << (tick / 1000ULL) << "k"
                          << " | AvgF=" << std::setw(4) << (tf / N)
                          << " | Alive=" << std::setw(5) << alive
                          << " | AvgK=" << std::fixed << std::setprecision(1) << ((double)sk / N)
                          << " | dFood=" << std::setw(8) << df
                          << " | dTox=" << std::setw(7) << dt
                          << " | F/T=" << std::setprecision(2) << std::setw(6) << ratio
                          << " | dDead=" << std::setw(6) << dd;
                if (use_wire_field) {
                    std::cout << " | SigAcc=" << std::setprecision(2) << min_sig << ".." << max_sig
                              << " NoiMax=" << max_noi;
                }
                std::cout << "\n" << std::flush;
            }
        }
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_t.node_count); cudaFree(d_t.out0); cudaFree(d_t.out1); cudaFree(d_t.fuel);
    cudaFree(d_t.rng); cudaFree(d_t.nodes); cudaFree(d_t.state); cudaFree(d_t.world);
    cudaFree(d_t.d_perm64);
    if (use_wire_field) {
        cudaFree(d_t.pred_acc); cudaFree(d_t.pred_ema);
    }

    #pragma omp critical
    std::cout << "GPU #" << gpu_id << " (" << (use_wire_field ? "CWF" : "Baseline") << ") Elapsed: "
              << std::setprecision(2) << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count()
              << "s\n" << std::flush;
}

int main(int argc, char** argv) {
    int max_sweeps = (argc > 1) ? std::atoi(argv[1]) : 11999; // 12,000 sweeps = 1,536,000 unbroken ticks
    std::cout << "================================================================================\n"
              << "KARYON-CORE EXP-1172: PURE UNBROKEN CONTINUOUS TISSUE (NO WRAPPERS, NO WINDOWS)\n"
              << "Timeline: 0..512k Home | 512k..1024k Regime Shifts | 1024k..1536k Live Perm64\n"
              << "================================================================================\n\n" << std::flush;

    std::cout << ">>> RUN 1: PURE BASELINE (Uniform Wire Sampling, Zero F->0 Field) <<<\n" << std::flush;
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) run_continuous_gpu(0, 0U, false, max_sweeps, 0x9E3779B97F4A7C15ULL);
        else          run_continuous_gpu(1, 1U, false, max_sweeps, 0x123456789ABCDEF0ULL);
    }

    std::cout << "\n>>> RUN 2: CONTINUOUS WIRE FIELD (F -> 0 Per-Tick Bit Prediction, Zero Thresholds) <<<\n" << std::flush;
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) run_continuous_gpu(0, 0U, true, max_sweeps, 0x9E3779B97F4A7C15ULL);
        else          run_continuous_gpu(1, 1U, true, max_sweeps, 0x123456789ABCDEF0ULL);
    }

    return 0;
}
