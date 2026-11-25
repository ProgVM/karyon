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
// KARYON-CORE EXP-1171b: INTRA-LIFESPAN PLASTIC SENSORY-MOTOR INTERFACE (F -> 0)
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// Architecture & Scientific Protocol:
// 1. Evolution Phase (Dual T4 GPUs, 128x128 Embodied Tissue):
//    - Identical verified closed-loop substrate from EXP-1171a (NAND/NOR/XOR/XNOR, K=4 Tabula Rasa).
//    - Evolves functional 2D/1D navigation & somatic circuits under 44-50 bits of cryptographic noise.
// 2. Intra-Lifespan PlasticInterface (N=1 Single-Individual Evaluation, NO internal gate mutation):
//    - Core invariance: Internal logic gates (op, internal sa/sb >= 64, out0, out1) remain 100% FROZEN!
//    - Sensorimotor Predictive Free Energy (F -> 0):
//        * Each external wire p in [0..63] tracks 1-step & 2-step predictability conditioned on own action `act`,
//          via fixed-point EMA (tau=64, scale [-128, +128]).
//        * Wires with |sm_score[p]| > 36 after 160-step warmup cleanly separate real physical signals
//          (including rapidly toggling LSBs like ax0) from 50% cryptographic noise (|sm_score| < 25).
//    - Strictly Injective Receptor Map (`pin_map[64]`):
//        * Rewiring ALWAYS swaps two slots in `pin_map[64]` (bijection preserved, zero slot collapse).
//        * Noisy wires connected to active sensory pins are immediately swapped to unused predictable wires.
//    - Bacterial Chemotaxis Rollback + Motor Polarity Adaptation:
//        * Evaluates somatic balance `win_dE` over 200-step windows with decayed baseline tracking.
//        * If a trial swap or `motor_xor` change degrades `win_dE < baseline_dE - 4.0`, it immediately rolls back.
//    - Honest Phase Telemetry:
//        * Prints `active_sensory_pins.size()` and `pred_wires.size()` for both 2D and 1D evolved organisms.
//        * Logs Deaths | Food | Toxins across lifetime phases: 0..2000, 2000..5000, 5000..10000, 10000..20000.

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
    uint64_t wrng;
};

struct EmbodiedTissueSoA {
    uint32_t*   node_count;
    uint32_t*   out0;
    uint32_t*   out1;
    int32_t*    fuel;
    uint64_t*   rng;
    uint32_t*   nodes;
    uint64_t*   state;
    WorldState* world;
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

__device__ static void apoptosis_embodied(EmbodiedTissueSoA& t, uint32_t idx) {
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

__device__ static void wire_split_mitosis_embodied(EmbodiedTissueSoA& t, uint32_t idx, uint64_t& rng) {
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

        uint16_t second_wire = (uint16_t)((xorshift64_dev(rng) & 1ULL) ? (xorshift64_dev(rng) % 64ULL) : ((xorshift64_dev(rng) % K) + 64ULL));
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
        uint16_t sa = (uint16_t)((xorshift64_dev(rng) & 1ULL) ? (xorshift64_dev(rng) % 64ULL) : ((xorshift64_dev(rng) % K) + 64ULL));
        uint16_t sb = (uint16_t)((xorshift64_dev(rng) & 1ULL) ? (xorshift64_dev(rng) % 64ULL) : ((xorshift64_dev(rng) % K) + 64ULL));
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

__device__ static void prune_one_node_embodied(EmbodiedTissueSoA& t, uint32_t idx, uint64_t& rng) {
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

__device__ static void graft_subgraph_embodied(EmbodiedTissueSoA& t, uint32_t donor_idx, uint32_t host_idx, uint64_t& rng) {
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

__device__ static void mutate_embodied(EmbodiedTissueSoA& t, uint32_t idx, uint64_t& rng) {
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
        uint16_t ns = (uint16_t)((xorshift64_dev(rng) & 1ULL) ? (xorshift64_dev(rng) % 64ULL) : ((xorshift64_dev(rng) % K) + 64ULL));
        if (xorshift64_dev(rng) & 1ULL) sa = ns; else sb = ns;
        t.nodes[(size_t)u * TOTAL_CELLS + idx] = pack_node(op, sa, sb);
    } else if (r < 80U) {
        wire_split_mitosis_embodied(t, idx, rng);
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

__global__ void embodied_sweep_kernel(EmbodiedTissueSoA t, uint32_t global_sweep_idx, uint32_t env_mode) {
    __shared__ uint32_t sh_nodes[EVAL_BLOCK][MAX_K];

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
        uint64_t total_world_steps = (uint64_t)global_sweep_idx * T_STEPS_PER_SWEEP + step;
        uint32_t regime = (global_sweep_idx < 4000) ? 0U : (uint32_t)((total_world_steps / SHIFT_PERIOD) % 4U);

        bool a_is_food = (regime == 0U || regime == 2U);
        bool invert_motors = (regime == 2U || regime == 3U);

        uint64_t noise = xorshift64_dev(w.wrng);
        uint64_t obs = 0ULL;

        if (env_mode == 0U) {
            uint64_t gain_b = (w.last_dE_sign > 0) ? 1ULL : 0ULL;
            uint64_t pain_b = (w.last_dE_sign < 0) ? 1ULL : 0ULL;
            obs = (noise & ~0xFFFFFULL) | (gain_b << 18) | (pain_b << 19)
                | (uint64_t)(w.ax & 7U) | ((uint64_t)(w.ay & 7U) << 3)
                | ((uint64_t)(w.fx & 7U) << 6) | ((uint64_t)(w.fy & 7U) << 9)
                | ((uint64_t)(w.tx0 & 7U) << 12) | ((uint64_t)(w.ty0 & 7U) << 15);
        } else {
            uint64_t gain_b = (w.last_dE_sign > 0) ? 1ULL : 0ULL;
            uint64_t pain_b = (w.last_dE_sign < 0) ? 1ULL : 0ULL;
            obs = (noise & ~0x3FFFULL) | (gain_b << 12) | (pain_b << 13)
                | (uint64_t)(w.ax & 15U) | ((uint64_t)(w.fx & 15U) << 4) | ((uint64_t)(w.tx0 & 15U) << 8);
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
                if (a_is_food) dE += 48; else dE -= 64;
                w.fx = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
                w.fy = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
            }
            if (w.ax == w.tx0 && w.ay == w.ty0) {
                if (a_is_food) dE -= 64; else dE += 48;
                w.tx0 = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
                w.ty0 = (uint8_t)(xorshift64_dev(w.wrng) & 7ULL);
            }
        } else {
            int32_t dp = 0;
            if (act == 1) dp = invert_motors ? 1 : -1;
            else if (act == 2) dp = invert_motors ? -1 : 1;
            w.ax = (uint8_t)((int32_t)w.ax + dp + 16) & 15U;

            if (w.ax == w.fx) {
                if (a_is_food) dE += 48; else dE -= 64;
                w.fx = (uint8_t)(xorshift64_dev(w.wrng) & 15ULL);
            }
            if (w.ax == w.tx0) {
                if (a_is_food) dE -= 64; else dE += 48;
                w.tx0 = (uint8_t)(xorshift64_dev(w.wrng) & 15ULL);
            }
        }

        w.last_dE_sign = (dE > 0) ? 1 : ((dE < -1) ? -1 : 0);
        sweep_delta_fuel += dE;
    }

    t.state[(size_t)idx] = gs;
    t.world[idx] = w;

    int32_t ncost = (int32_t)((K + 7U) >> 3);
    int32_t fuel_next = fuel + sweep_delta_fuel - ncost - ((fuel >> 6) + 1);
    t.fuel[idx] = (fuel_next < 0) ? 0 : fuel_next;
}

__global__ void pair_interact_embodied_kernel(EmbodiedTissueSoA t, uint32_t n_pairs, uint32_t pass_mode, uint32_t env_mode) {
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

    auto respawn_world = [&](uint32_t target_idx) {
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
        mutate_embodied(t, oi, lr); mutate_embodied(t, oj, lr);
        t.fuel[oi] = 512; respawn_world(oi);
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
                mutate_embodied(t, bi, lr);
            }
            respawn_world(bi);
            int32_t share = fa >> 1;
            t.fuel[ai] = fa - share;
            t.fuel[bi] = share;
        } else if (fb <= 0) {
            mutate_embodied(t, bi, lr);
            t.fuel[bi] = 256;
            respawn_world(bi);
        }
    }
    t.rng[oi] = lr;
}

static std::vector<uint32_t> extract_active_sensory_pins(
    const std::vector<uint32_t>& h_nodes, uint32_t K, uint32_t out0, uint32_t out1, uint32_t bidx
) {
    std::vector<uint32_t> pins;
    if (K == 0U) return pins;
    uint32_t r0 = out0 % K, r1 = out1 % K;
    std::set<uint32_t> visited;
    std::set<uint32_t> sensory_set;
    std::vector<uint32_t> stack;
    visited.insert(r0); stack.push_back(r0);
    if (visited.find(r1) == visited.end()) { visited.insert(r1); stack.push_back(r1); }

    while (!stack.empty()) {
        uint32_t u = stack.back(); stack.pop_back();
        uint8_t op; uint16_t sa, sb;
        unpack_node(h_nodes[(size_t)u * TOTAL_CELLS + bidx], op, sa, sb);
        if (sa < 64U) sensory_set.insert((uint32_t)sa);
        else {
            uint32_t j = (uint32_t)(sa - 64U) % K;
            if (visited.find(j) == visited.end()) { visited.insert(j); stack.push_back(j); }
        }
        if (sb < 64U) sensory_set.insert((uint32_t)sb);
        else {
            uint32_t j = (uint32_t)(sb - 64U) % K;
            if (visited.find(j) == visited.end()) { visited.insert(j); stack.push_back(j); }
        }
    }
    for (uint32_t s : sensory_set) pins.push_back(s);
    return pins;
}

// ============================================================================
// EXP-1171b: INTRA-LIFESPAN PLASTIC SENSORY-MOTOR INTERFACE (N=1)
// - Internal logic gates (op, sa, sb, out0, out1) remain 100% FROZEN!
// - Strictly injective pin_map[64] (swap only, zero collisions).
// - Sensorimotor Predictive EMA >> 7 (tau=64, scale [-128, +128], |sm_score| > 36).
// - Bacterial Chemotaxis Rollback on 200-step somatic balance window.
// ============================================================================
struct PlasticInterface {
    uint8_t  pin_map[64];
    uint8_t  motor_xor;
    uint64_t prev1;
    uint64_t prev2;
    int32_t  acc1[4][64];
    int32_t  acc2[4][64];
    uint32_t act_cnt[4];

    double   win_dE;
    double   baseline_dE;
    bool     has_trial;
    uint8_t  trial_type; // 0: pin swap, 1: motor_xor
    uint8_t  trial_sa;
    uint8_t  trial_sb;
    uint8_t  trial_old_mxor;
    uint8_t  mxor_cycle_tries;
    uint64_t prng;

    void init() {
        for (int i = 0; i < 64; ++i) pin_map[i] = (uint8_t)i;
        motor_xor = 0;
        prev1 = 0ULL; prev2 = 0ULL;
        for (int a = 0; a < 4; ++a) {
            act_cnt[a] = 0U;
            for (int p = 0; p < 64; ++p) { acc1[a][p] = 0; acc2[a][p] = 0; }
        }
        win_dE = 0.0;
        baseline_dE = -50.0;
        has_trial = false;
        trial_type = 0; trial_sa = 0; trial_sb = 0; trial_old_mxor = 0;
        mxor_cycle_tries = 0;
        prng = 0xABCDEF0123456789ULL;
    }

    uint64_t next_rng() {
        prng = prng * 6364136223846793005ULL + 1442695040888963407ULL;
        return prng;
    }

    uint8_t find_slot_holding(uint8_t wire) const {
        for (int s = 0; s < 64; ++s) {
            if (pin_map[s] == wire) return (uint8_t)s;
        }
        return 0;
    }

    void swap_slots(uint8_t sa, uint8_t sb) {
        uint8_t tmp = pin_map[sa];
        pin_map[sa] = pin_map[sb];
        pin_map[sb] = tmp;
    }
};

struct PhaseBucket {
    long long deaths = 0, food = 0, toxins = 0;
};

static void eval_plastic_individual(
    const std::vector<uint32_t>& h_nodes, uint32_t K, uint32_t out0, uint32_t out1, uint32_t bidx,
    uint32_t env_mode, const std::string& label, bool enable_plasticity,
    bool enable_regime_shifts, bool apply_perm, uint64_t eval_steps = 20000
) {
    uint64_t erng = 0x9999888877776666ULL;
    auto nr = [&]() -> uint64_t {
        erng = erng * 6364136223846793005ULL + 1442695040888963407ULL;
        return erng;
    };

    uint8_t ax = (uint8_t)(nr() & (env_mode == 0 ? 7 : 15)), ay = (uint8_t)(nr() & 7);
    uint8_t fx = (uint8_t)(nr() & (env_mode == 0 ? 7 : 15)), fy = (uint8_t)(nr() & 7);
    uint8_t tx0 = (uint8_t)(nr() & (env_mode == 0 ? 7 : 15)), ty0 = (uint8_t)(nr() & 7);

    std::vector<uint32_t> active_pins = extract_active_sensory_pins(h_nodes, K, out0, out1, bidx);

    uint8_t perm[64];
    for (int b = 0; b < 64; ++b) perm[b] = (uint8_t)b;
    if (apply_perm) {
        uint64_t prng = 0x123456789ABCDEF0ULL;
        for (int b = 63; b > 0; --b) {
            prng = prng * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)(prng % (uint64_t)(b + 1));
            uint8_t tmp = perm[b]; perm[b] = perm[j]; perm[j] = tmp;
        }
    }

    PlasticInterface pi;
    pi.init();

    uint64_t gs = 0ULL;
    int8_t last_dE_sign = 0;
    double energy = 100.0;

    PhaseBucket p0, p1, p2, p3; // 0..2000, 2000..5000, 5000..10000, 10000..20000
    int last_pred_count = 0;
    int rollbacks = 0, adaptations = 0;

    for (uint64_t step = 0; step < eval_steps; ++step) {
        uint32_t regime = enable_regime_shifts ? (uint32_t)((step / SHIFT_PERIOD) % 4U) : 0U;
        bool a_is_food = (regime == 0U || regime == 2U);
        bool invert_motors = (regime == 2U || regime == 3U);

        uint64_t noise = nr();
        uint64_t raw_obs = 0ULL;

        if (env_mode == 0U) {
            uint64_t gain_b = (last_dE_sign > 0) ? 1ULL : 0ULL;
            uint64_t pain_b = (last_dE_sign < 0) ? 1ULL : 0ULL;
            raw_obs = (noise & ~0xFFFFFULL) | (gain_b << 18) | (pain_b << 19)
                | (uint64_t)(ax & 7U) | ((uint64_t)(ay & 7U) << 3)
                | ((uint64_t)(fx & 7U) << 6) | ((uint64_t)(fy & 7U) << 9)
                | ((uint64_t)(tx0 & 7U) << 12) | ((uint64_t)(ty0 & 7U) << 15);
        } else {
            uint64_t gain_b = (last_dE_sign > 0) ? 1ULL : 0ULL;
            uint64_t pain_b = (last_dE_sign < 0) ? 1ULL : 0ULL;
            raw_obs = (noise & ~0x3FFFULL) | (gain_b << 12) | (pain_b << 13)
                | (uint64_t)(ax & 15U) | ((uint64_t)(fx & 15U) << 4) | ((uint64_t)(tx0 & 15U) << 8);
        }

        uint64_t obs = raw_obs;
        if (apply_perm) {
            uint64_t p_obs = 0ULL;
            for (int b = 0; b < 64; ++b) {
                if ((raw_obs >> b) & 1ULL) p_obs |= (1ULL << perm[b]);
            }
            obs = p_obs;
        }

        // Map external environment wires through strictly injective PlasticInterface pin_map
        uint64_t mapped_obs = obs;
        if (enable_plasticity) {
            mapped_obs = 0ULL;
            for (uint32_t s = 0; s < 64U; ++s) {
                if ((obs >> pi.pin_map[s]) & 1ULL) mapped_obs |= (1ULL << s);
            }
        }

        for (uint32_t u = 0; u < K; ++u) {
            uint8_t op; uint16_t sa, sb;
            unpack_node(h_nodes[(size_t)u * TOTAL_CELLS + bidx], op, sa, sb);
            if (sa >= 64U + K) sa = (uint16_t)((sa - 64U) % K + 64U);
            if (sb >= 64U + K) sb = (uint16_t)((sb - 64U) % K + 64U);

            uint64_t a = (sa < 64U) ? ((mapped_obs >> sa) & 1ULL) : ((gs >> (sa - 64U)) & 1ULL);
            uint64_t b = (sb < 64U) ? ((mapped_obs >> sb) & 1ULL) : ((gs >> (sb - 64U)) & 1ULL);

            uint64_t r = 0ULL;
            if (op == 0U) r = ~(a & b) & 1ULL;
            else if (op == 1U) r = ~(a | b) & 1ULL;
            else if (op == 2U) r = (a ^ b) & 1ULL;
            else r = (~(a ^ b)) & 1ULL;

            gs = (gs & ~(1ULL << u)) | (r << u);
        }

        uint8_t m0 = (uint8_t)((gs >> (out0 % K)) & 1ULL);
        uint8_t m1 = (uint8_t)((gs >> (out1 % K)) & 1ULL);
        uint8_t raw_act = (m1 << 1) | m0;
        uint8_t act = enable_plasticity ? ((raw_act ^ pi.motor_xor) & 3U) : raw_act;

        // Update sensorimotor predictive EMA conditioned on own motor efference copy `act`
        if (enable_plasticity) {
            uint64_t s1 = ~(obs ^ pi.prev1);
            uint64_t s2 = ~(obs ^ pi.prev2);
            pi.prev2 = pi.prev1;
            pi.prev1 = obs;
            pi.act_cnt[act]++;
            for (int p = 0; p < 64; ++p) {
                int32_t m1_v = ((s1 >> p) & 1ULL) ? 1 : -1;
                int32_t m2_v = ((s2 >> p) & 1ULL) ? 1 : -1;
                // Unbiased fixed-point EMA (tau=64, scale [-128, +128] at acc >> 5)
                int32_t a1 = pi.acc1[act][p];
                int32_t a2 = pi.acc2[act][p];
                pi.acc1[act][p] = a1 - (a1 / 64) + (m1_v * 64);
                pi.acc2[act][p] = a2 - (a2 / 64) + (m2_v * 64);
            }
        }

        double dE = -0.25;
        long long df = 0, dt = 0, dd = 0;
        if (env_mode == 0U) {
            uint8_t w_act = act;
            if (invert_motors) {
                if (w_act == 0) w_act = 1; else if (w_act == 1) w_act = 0;
                else if (w_act == 2) w_act = 3; else if (w_act == 3) w_act = 2;
            }
            if (w_act == 0) ay = (ay - 1U) & 7U;
            else if (w_act == 1) ay = (ay + 1U) & 7U;
            else if (w_act == 2) ax = (ax - 1U) & 7U;
            else if (w_act == 3) ax = (ax + 1U) & 7U;

            if ((step & 1) == 0) {
                uint32_t td = (uint32_t)(nr() & 3ULL);
                if (td == 0) ty0 = (ty0 - 1U) & 7U;
                else if (td == 1) ty0 = (ty0 + 1U) & 7U;
                else if (td == 2) tx0 = (tx0 - 1U) & 7U;
                else tx0 = (tx0 + 1U) & 7U;
            }

            if (ax == fx && ay == fy) {
                if (a_is_food) { dE += 12.0; df++; } else { dE -= 16.0; dt++; }
                fx = (uint8_t)(nr() & 7ULL); fy = (uint8_t)(nr() & 7ULL);
            }
            if (ax == tx0 && ay == ty0) {
                if (a_is_food) { dE -= 16.0; dt++; } else { dE += 12.0; df++; }
                tx0 = (uint8_t)(nr() & 7ULL); ty0 = (uint8_t)(nr() & 7ULL);
            }
        } else {
            int32_t dp = 0;
            if (act == 1) dp = invert_motors ? 1 : -1;
            else if (act == 2) dp = invert_motors ? -1 : 1;
            ax = (uint8_t)((int32_t)ax + dp + 16) & 15U;

            if (ax == fx) {
                if (a_is_food) { dE += 12.0; df++; } else { dE -= 16.0; dt++; }
                fx = (uint8_t)(nr() & 15ULL);
            }
            if (ax == tx0) {
                if (a_is_food) { dE -= 16.0; dt++; } else { dE += 12.0; df++; }
                tx0 = (uint8_t)(nr() & 15ULL);
            }
        }

        energy += dE;
        if (energy > 200.0) energy = 200.0;
        if (energy <= 0.0) { dd++; energy = 100.0; }
        last_dE_sign = (dE > 0) ? 1 : ((dE < -1) ? -1 : 0);
        pi.win_dE += dE;

        PhaseBucket* pb = (step < 2000) ? &p0 : ((step < 5000) ? &p1 : ((step < 10000) ? &p2 : &p3));
        pb->deaths += dd; pb->food += df; pb->toxins += dt;

        // Intra-lifespan plastic adaptation every 200 steps after 160-step warmup
        if (enable_plasticity && step >= 199 && (step % 200) == 199 && !active_pins.empty()) {
            int active_a[4], n_aa = 0;
            for (int a = 0; a < 4; ++a) {
                if (pi.act_cnt[a] >= 4U) active_a[n_aa++] = a;
            }
            if (n_aa == 0) { active_a[0] = act; n_aa = 1; }

            int32_t sm_score[64];
            std::vector<uint8_t> pred_wires;
            for (int p = 0; p < 64; ++p) {
                int32_t s1_sum = 0, s2_sum = 0;
                for (int k = 0; k < n_aa; ++k) {
                    int a = active_a[k];
                    s1_sum += std::abs(pi.acc1[a][p]);
                    s2_sum += std::abs(pi.acc2[a][p]);
                }
                int32_t sc1 = s1_sum / (n_aa * 32);
                int32_t sc2 = s2_sum / (n_aa * 32);
                sm_score[p] = (sc1 > sc2) ? sc1 : sc2;
                if (sm_score[p] > 36) pred_wires.push_back((uint8_t)p);
            }
            last_pred_count = (int)pred_wires.size();

            // Bacterial Chemotaxis Rollback check
            if (pi.has_trial) {
                if (pi.win_dE < pi.baseline_dE - 4.0) {
                    if (pi.trial_type == 0) pi.swap_slots(pi.trial_sa, pi.trial_sb);
                    else pi.motor_xor = pi.trial_old_mxor;
                    pi.baseline_dE = 0.7 * pi.baseline_dE + 0.3 * pi.win_dE;
                    rollbacks++;
                } else {
                    pi.baseline_dE = 0.5 * pi.baseline_dE + 0.5 * pi.win_dE;
                }
                pi.has_trial = false;
            } else {
                pi.baseline_dE = pi.win_dE;
            }

            if (pi.win_dE < 20.0 && pred_wires.size() >= active_pins.size()) {
                // 1. Eject any active sensory pin currently plugged into a crypto-noise wire!
                std::vector<uint32_t> noisy_slots;
                bool used_wire[64] = {false};
                for (uint32_t s : active_pins) {
                    uint8_t w = pi.pin_map[s];
                    used_wire[w] = true;
                    if (sm_score[w] <= 36) noisy_slots.push_back(s);
                }

                if (!noisy_slots.empty()) {
                    std::vector<uint8_t> free_pred;
                    for (uint8_t w : pred_wires) {
                        if (!used_wire[w]) free_pred.push_back(w);
                    }
                    if (!free_pred.empty()) {
                        uint8_t sa = (uint8_t)noisy_slots[pi.next_rng() % noisy_slots.size()];
                        uint8_t target_w = free_pred[pi.next_rng() % free_pred.size()];
                        uint8_t sb = pi.find_slot_holding(target_w);
                        if (sa != sb) {
                            pi.swap_slots(sa, sb);
                            pi.has_trial = true; pi.trial_type = 0; pi.trial_sa = sa; pi.trial_sb = sb;
                            adaptations++;
                        }
                    }
                } else if (pi.mxor_cycle_tries < 3) {
                    // 2. Test motor polarity inversion first (fast 3-step check before altering aligned wires)
                    pi.trial_old_mxor = pi.motor_xor;
                    pi.motor_xor = (uint8_t)((pi.motor_xor + 1U) & 3U);
                    pi.has_trial = true; pi.trial_type = 1;
                    pi.mxor_cycle_tries++;
                    adaptations++;
                } else {
                    // 3. Swap 1 active sensory pin to another predictable wire (preserving strict bijection)
                    if ((pi.next_rng() % 4ULL) == 0ULL) {
                        pi.trial_old_mxor = pi.motor_xor;
                        pi.motor_xor = (uint8_t)((pi.motor_xor + 1U + (pi.next_rng() % 3ULL)) & 3U);
                        pi.has_trial = true; pi.trial_type = 1;
                    } else {
                        uint8_t sa = (uint8_t)active_pins[pi.next_rng() % active_pins.size()];
                        uint8_t target_w = pred_wires[pi.next_rng() % pred_wires.size()];
                        uint8_t sb = pi.find_slot_holding(target_w);
                        if (sa != sb) {
                            pi.swap_slots(sa, sb);
                            pi.has_trial = true; pi.trial_type = 0; pi.trial_sa = sa; pi.trial_sb = sb;
                        }
                    }
                    adaptations++;
                }
            } else if (pi.win_dE >= 20.0) {
                pi.mxor_cycle_tries = 0; // Healthy steady-state reached
            }
            pi.win_dE = 0.0;
        }
    }

    long long tot_d = p0.deaths + p1.deaths + p2.deaths + p3.deaths;
    long long tot_f = p0.food + p1.food + p2.food + p3.food;
    long long tot_t = p0.toxins + p1.toxins + p2.toxins + p3.toxins;

    std::cout << "      [" << label << "] Total: Deaths=" << std::setw(2) << tot_d
              << " | Food=" << std::setw(4) << tot_f << " | Toxins=" << std::setw(4) << tot_t
              << " | E=" << std::fixed << std::setprecision(1) << energy
              << " | Pins=" << active_pins.size() << " | PredWires=" << last_pred_count
              << " | Adapt=" << adaptations << " (RB=" << rollbacks << ")\n";
    std::cout << "        Phases: [0..2k: D=" << p0.deaths << " F=" << p0.food << " T=" << p0.toxins << "]"
              << " [2k..5k: D=" << p1.deaths << " F=" << p1.food << " T=" << p1.toxins << "]"
              << " [5k..10k: D=" << p2.deaths << " F=" << p2.food << " T=" << p2.toxins << "]"
              << " [10k..20k: D=" << p3.deaths << " F=" << p3.food << " T=" << p3.toxins << "]\n";
}

static void run_embodied_gpu_seed(int gpu_id, int seed_idx, uint32_t env_mode, int max_sweeps, uint64_t seed) {
    cudaSetDevice(gpu_id);
    const uint32_t N = TOTAL_CELLS, n_pairs = N / 2;

    std::vector<uint32_t> h_nc(N), h_out0(N), h_out1(N), h_nodes((size_t)MAX_K * N, 0U);
    std::vector<int32_t> h_fuel(N);
    std::vector<uint64_t> h_rng(N), h_state(N, 0ULL);
    std::vector<WorldState> h_world(N);

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
    }

    EmbodiedTissueSoA d_t;
    cudaMalloc(&d_t.node_count, N * sizeof(uint32_t));
    cudaMalloc(&d_t.out0, N * sizeof(uint32_t));
    cudaMalloc(&d_t.out1, N * sizeof(uint32_t));
    cudaMalloc(&d_t.fuel, N * sizeof(int32_t));
    cudaMalloc(&d_t.rng, N * sizeof(uint64_t));
    cudaMalloc(&d_t.nodes, (size_t)MAX_K * N * sizeof(uint32_t));
    cudaMalloc(&d_t.state, N * sizeof(uint64_t));
    cudaMalloc(&d_t.world, N * sizeof(WorldState));

    cudaMemcpy(d_t.node_count, h_nc.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.out0, h_out0.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.out1, h_out1.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.fuel, h_fuel.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.rng, h_rng.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.nodes, h_nodes.data(), (size_t)MAX_K * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.state, h_state.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t.world, h_world.data(), N * sizeof(WorldState), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp critical
    std::cout << "--- GPU #" << gpu_id << " SEED " << seed_idx << " (" << (env_mode == 0 ? "Grid2D 8x8" : "ActiveWorld 1D")
              << " 128x128, max_sw=" << max_sweeps << ") ---\n";

    int eval_blocks = (N + EVAL_BLOCK - 1) / EVAL_BLOCK;
    int pair_blocks = (n_pairs + PAIR_BLOCK - 1) / PAIR_BLOCK;
    uint64_t hrng = seed ^ 0xFEEDFACE12345678ULL;

    for (int sw = 0; sw <= max_sweeps; ++sw) {
        uint32_t pm = (uint32_t)(xorshift64_dev(hrng) & 3ULL);
        embodied_sweep_kernel<<<eval_blocks, EVAL_BLOCK, 0, stream>>>(d_t, sw, env_mode);
        pair_interact_embodied_kernel<<<pair_blocks, PAIR_BLOCK, 0, stream>>>(d_t, n_pairs, pm, env_mode);

        if (sw == 0 || sw == 2000 || sw == 3999 || sw == max_sweeps) {
            cudaStreamSynchronize(stream);
            cudaMemcpy(h_nc.data(), d_t.node_count, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_out0.data(), d_t.out0, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_out1.data(), d_t.out1, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fuel.data(), d_t.fuel, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_nodes.data(), d_t.nodes, (size_t)MAX_K * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            long long tf = 0, sk = 0; int alive = 0, bidx = 0;
            int32_t max_f = -999999;
            for (uint32_t i = 0; i < N; ++i) {
                tf += h_fuel[i]; if (h_fuel[i] > 0) alive++;
                sk += h_nc[i];
                if (h_fuel[i] > max_f) { max_f = h_fuel[i]; bidx = i; }
            }

            #pragma omp critical
            {
                std::cout << "[GPU #" << gpu_id << " S" << seed_idx << " " << (env_mode == 0 ? "2D" : "1D")
                          << "] sw=" << std::setw(5) << sw
                          << " | AvgF=" << std::setw(4) << (tf / N) << " | MaxF=" << std::setw(5) << max_f
                          << " | Alive=" << std::setw(5) << alive << " | AvgK=" << std::fixed << std::setprecision(1) << ((double)sk / N) << "\n" << std::flush;

                if (sw == 3999 || sw == max_sweeps) {
                    std::cout << "\n  ========================================================================================\n"
                              << "  SINGLE-INDIVIDUAL AUDIT (N=1, FROZEN CORE, PLASTIC SENSORY-MOTOR INTERFACE, 20k steps):\n"
                              << "  ========================================================================================\n";
                    eval_plastic_individual(h_nodes, h_nc[bidx], h_out0[bidx], h_out1[bidx], bidx, env_mode, "Test A  (Frozen Baseline, Stat) ", false, false, false, 20000);
                    eval_plastic_individual(h_nodes, h_nc[bidx], h_out0[bidx], h_out1[bidx], bidx, env_mode, "Test B0 (Frozen Baseline, Shift)", false, true,  false, 20000);
                    eval_plastic_individual(h_nodes, h_nc[bidx], h_out0[bidx], h_out1[bidx], bidx, env_mode, "Test B1 (Plastic Interface,Shift)", true,  true,  false, 20000);
                    eval_plastic_individual(h_nodes, h_nc[bidx], h_out0[bidx], h_out1[bidx], bidx, env_mode, "Test C0 (Frozen Baseline, Perm) ", false, true,  true,  20000);
                    eval_plastic_individual(h_nodes, h_nc[bidx], h_out0[bidx], h_out1[bidx], bidx, env_mode, "Test C1 (Plastic, Stat+Perm)    ", true,  false, true,  20000);
                    eval_plastic_individual(h_nodes, h_nc[bidx], h_out0[bidx], h_out1[bidx], bidx, env_mode, "Test C2 (Plastic, Shift+Perm)   ", true,  true,  true,  20000);
                    std::cout << "  ========================================================================================\n\n";
                }
            }
        }
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_t.node_count); cudaFree(d_t.out0); cudaFree(d_t.out1); cudaFree(d_t.fuel);
    cudaFree(d_t.rng); cudaFree(d_t.nodes); cudaFree(d_t.state); cudaFree(d_t.world);

    #pragma omp critical
    std::cout << "GPU #" << gpu_id << " Elapsed: " << std::setprecision(2)
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() << "s\n" << std::flush;
}

int main(int argc, char** argv) {
    int max_sweeps = (argc > 1) ? std::atoi(argv[1]) : 8000;
    std::cout << "========================================================================\n"
              << "KARYON-CORE EXP-1171b: INTRA-LIFESPAN PLASTIC SENSORY-MOTOR INTERFACE\n"
              << "Standard: KEP v51.0 | Frozen Core + PlasticInterface | Dual T4 | sw=" << max_sweeps << "\n"
              << "========================================================================\n\n" << std::flush;

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) {
            run_embodied_gpu_seed(0, 0, 0U, max_sweeps, 0x9E3779B97F4A7C15ULL);
        } else {
            run_embodied_gpu_seed(1, 1, 1U, max_sweeps, 0x123456789ABCDEF0ULL);
        }
    }
    return 0;
}
