#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <thread>
#include <sstream>

// ============================================================================
// KARYON-CORE EXP-1168: STAGE 3 - EMERGENT MEMORY IN CONTINUOUS TIME STREAM
// Lead Architect: Bazilevs (ProgVM) | Guardian: Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// Minimal physical delta over EXP-1167:
// 1. Persistent tape state across ticks & epochs:
//    `env_ticks_kernel` loads `tapes[idx]` into `sh_solo[tid]` ONCE before tick 0,
//    does NOT reset `sh_solo[tid]` between ticks, and writes `sh_solo[tid]` back
//    to `tapes[idx]` after the ticks.
// 2. Single-bit continuous sensory stream `X_t`:
//    At each tick t, ONLY ONE random sensory bit `x_curr` is injected at bit 254,
//    plus random actuator init `y_init` at bit 255. Bits 0..253 belong 100% to
//    the organism (zero memory register, zero `prev` opcode, zero reset).
// 3. Temporal tasks impossible without emergent internal state memory:
//    - Regime 0 (Delay1)   : Y_t = X_{t-1}
//    - Regime 1 (TempXOR)  : Y_t = X_t ^ X_{t-1}
//    - Regime 2 (Temp~XOR) : Y_t = ~(X_t ^ X_{t-1})
//    - Regime 3 (OOD:CurrX): Y_t = X_t (adaptation back to memoryless law)

struct alignas(32) Tape256 {
    uint64_t w[4];
};

__host__ __device__ __forceinline__ uint8_t get_bit_n(const uint64_t* w, uint32_t idx) {
    return (uint8_t)((w[idx >> 6] >> (idx & 63U)) & 1ULL);
}

__host__ __device__ __forceinline__ void set_bit_n(uint64_t* w, uint32_t idx, uint8_t val) {
    uint64_t mask = 1ULL << (idx & 63U);
    if (val) w[idx >> 6] |= mask;
    else     w[idx >> 6] &= ~mask;
}

__host__ __device__ __forceinline__ void xor_bit_n(uint64_t* w, uint32_t idx) {
    w[idx >> 6] ^= (1ULL << (idx & 63U));
}

__host__ __device__ __forceinline__ uint8_t get_nib_aligned(const uint64_t* w, uint32_t pc) {
    return (uint8_t)((w[pc >> 6] >> (pc & 60U)) & 0xFULL);
}

__host__ __device__ __forceinline__ int map_op4(uint8_t nib) {
    switch (nib) {
        case 3:  return 1;  // <  h0--
        case 12: return 2;  // >  h0++
        case 5:  return 3;  // {  h1--
        case 10: return 4;  // }  h1++
        case 6:  return 5;  // C  mem[h1] = mem[h0]
        case 9:  return 6;  // c  mem[h0] = mem[h1]
        case 7:  return 7;  // [  loop start
        case 8:  return 8;  // ]  loop end
        case 1:  return 9;  // !  mem[h0] ^= 1
        case 14: return 10; // ?  if (mem[h0]==0) skip next
        default: return 0;  // NOP
    }
}

__host__ __device__ __forceinline__ uint64_t xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

__host__ __device__ __forceinline__ uint64_t sparse_bitmask(uint64_t& rng, int log2_inv_p) {
    uint64_t mask = ~0ULL;
    for (int i = 0; i < log2_inv_p; ++i) mask &= xorshift64(rng);
    return mask;
}

__host__ __device__ __forceinline__ uint8_t target_temporal(uint32_t regime, uint8_t x_curr, uint8_t x_prev) {
    switch (regime) {
        case 0U: return x_prev;                  // Delay1: Y_t = X_{t-1}
        case 1U: return x_curr ^ x_prev;         // TempXOR: Y_t = X_t ^ X_{t-1}
        case 2U: return (x_curr ^ x_prev) ^ 1U;  // Temp~XOR: Y_t = ~(X_t ^ X_{t-1})
        case 3U: return x_curr;                  // OOD:CurrX: Y_t = X_t
        default: return x_prev;
    }
}

__host__ __device__ uint32_t eval_vm_packed(
    uint64_t* w, uint16_t* stack, uint32_t stack_cap,
    uint32_t len_mask, uint32_t init_h0, uint32_t init_h1, uint32_t max_steps
) {
    uint32_t len = len_mask + 1U;
    uint32_t pc = 0U, h0 = init_h0, h1 = init_h1, steps = 0U;
    int sp = 0;

    while (pc + 3U < len && steps < max_steps) {
        uint8_t nib = get_nib_aligned(w, pc);
        int op = map_op4(nib);
        switch (op) {
            case 0: break;
            case 1: h0 = (h0 - 1U) & len_mask; break;
            case 2: h0 = (h0 + 1U) & len_mask; break;
            case 3: h1 = (h1 - 1U) & len_mask; break;
            case 4: h1 = (h1 + 1U) & len_mask; break;
            case 5: set_bit_n(w, h1, get_bit_n(w, h0)); break;
            case 6: set_bit_n(w, h0, get_bit_n(w, h1)); break;
            case 7: if (sp < (int)stack_cap) stack[sp++] = (uint16_t)pc; break;
            case 8: if (sp > 0) { pc = (uint32_t)stack[sp - 1] + 4U; steps++; continue; } break;
            case 9: xor_bit_n(w, h0); break;
            case 10: {
                if (get_bit_n(w, h0) == 0U) {
                    pc += 4U;
                    steps++;
                    if (pc + 3U < len && steps < max_steps) {
                        int nop = map_op4(get_nib_aligned(w, pc));
                        if (nop == 8) {
                            if (sp > 0) sp--;
                            pc += 4U;
                            steps++;
                        } else if (nop == 7) {
                            int depth = 1;
                            pc += 4U;
                            steps++;
                            while (depth > 0 && pc + 3U < len && steps < max_steps) {
                                int lop = map_op4(get_nib_aligned(w, pc));
                                if (lop == 7) depth++;
                                else if (lop == 8) depth--;
                                pc += 4U;
                                steps++;
                            }
                            if (depth > 0) return steps;
                        } else {
                            pc += 4U;
                            steps++;
                        }
                    }
                    continue;
                }
                break;
            }
        }
        pc += 4U;
        steps++;
    }
    return steps;
}

__global__ void init_population_kernel(
    Tape256* tapes, int32_t* fuel, uint64_t* rngs, uint8_t* prev_x_arr,
    uint32_t n_tapes, uint64_t base_seed, int32_t init_fuel, bool invert
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tapes) return;

    uint64_t s = base_seed ^ ((uint64_t)(idx + 1U) * 0x9E3779B97F4A7C15ULL);
    for (int k = 0; k < 4; ++k) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        tapes[idx].w[k] = invert ? (~z) : z;
    }
    fuel[idx] = init_fuel;
    prev_x_arr[idx] = (uint8_t)(s & 1ULL);
    rngs[idx] = s ? s : 0xCAFEBABE12345678ULL;
}

__global__ void env_ticks_kernel(
    Tape256* tapes, int32_t* fuel, uint64_t* rngs, uint8_t* prev_x_arr,
    uint32_t n_tapes, uint32_t regime, int ticks_per_epoch, int32_t* corr_out
) {
    __shared__ uint64_t sh_solo[256][4];
    __shared__ uint16_t sh_stack[256][16];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tapes) return;

    uint32_t tid = threadIdx.x;
    uint64_t local_rng = rngs[idx];
    uint8_t prev_x = prev_x_arr[idx];
    int32_t f = fuel[idx];
    int32_t local_corr = 0;

    for (int k = 0; k < 4; ++k) sh_solo[tid][k] = tapes[idx].w[k];

    for (int t = 0; t < ticks_per_epoch; ++t) {
        uint8_t x_curr = (uint8_t)(xorshift64(local_rng) >> 63);
        uint8_t y_init = (uint8_t)(xorshift64(local_rng) >> 63);
        uint8_t y_star = target_temporal(regime, x_curr, prev_x);

        set_bit_n(sh_solo[tid], 254U, x_curr);
        set_bit_n(sh_solo[tid], 255U, y_init);

        eval_vm_packed(sh_solo[tid], sh_stack[tid], 16U, 255U, 0U, 0U, 256U);

        if (get_bit_n(sh_solo[tid], 255U) == y_star) {
            f += 32;
            if (f > 4096) f = 4096;
            local_corr++;
        } else {
            f -= 32;
            if (f < 0) f = 0;
        }
        f -= ((f >> 7) + 1);
        if (f < 0) f = 0;

        prev_x = x_curr;
    }

    for (int k = 0; k < 4; ++k) tapes[idx].w[k] = sh_solo[tid][k];
    prev_x_arr[idx] = prev_x;
    rngs[idx] = local_rng;
    fuel[idx] = f;
    corr_out[idx] = local_corr;
}

__global__ void checkerboard_pair_kernel(
    Tape256* tapes, int32_t* fuel, uint64_t* rngs,
    uint32_t W, uint32_t H, uint32_t pass_mode, uint32_t n_pairs, int32_t* repro_out
) {
    __shared__ uint64_t sh_buf[256][8];
    __shared__ uint16_t sh_stack[256][24];

    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;

    uint32_t tid = threadIdx.x;
    uint32_t orig_i, orig_j;
    if (pass_mode < 2U) {
        uint32_t pairs_per_row = W >> 1;
        uint32_t y = p / pairs_per_row;
        uint32_t x = ((p % pairs_per_row) << 1) + (pass_mode & 1U);
        uint32_t x2 = (x + 1U) & (W - 1U);
        orig_i = y * W + x;
        orig_j = y * W + x2;
    } else {
        uint32_t x = p % W;
        uint32_t y = ((p / W) << 1) + (pass_mode & 1U);
        uint32_t y2 = (y + 1U) & (H - 1U);
        orig_i = y * W + x;
        orig_j = y2 * W + x;
    }

    uint64_t local_rng = rngs[orig_i];
    uint32_t a_idx = orig_i, b_idx = orig_j;
    int32_t fa = fuel[a_idx];
    int32_t fb = fuel[b_idx];

    int32_t na = (int32_t)(xorshift64(local_rng) & 255ULL);
    int32_t nb = (int32_t)(xorshift64(local_rng) & 255ULL);
    if (fb + nb > fa + na) {
        a_idx = orig_j;
        b_idx = orig_i;
        int32_t tf = fa; fa = fb; fb = tf;
    }

    if (fa <= 0) {
        if (repro_out) repro_out[p] = 0;
        rngs[orig_i] = local_rng;
        return;
    }

    for (int k = 0; k < 4; ++k) {
        sh_buf[tid][k]     = tapes[a_idx].w[k];
        sh_buf[tid][k + 4] = tapes[b_idx].w[k];
    }
    uint64_t old_b[4] = { sh_buf[tid][4], sh_buf[tid][5], sh_buf[tid][6], sh_buf[tid][7] };

    uint32_t step_budget = (uint32_t)min(4096, fa);
    eval_vm_packed(sh_buf[tid], sh_stack[tid], 24U, 511U, 0U, 256U, step_budget);

    for (int w = 0; w < 8; ++w) sh_buf[tid][w] ^= sparse_bitmask(local_rng, 13);

    int diff_bits = 0;
    for (int k = 0; k < 4; ++k) diff_bits += __popcll(old_b[k] ^ sh_buf[tid][k + 4]);

    for (int k = 0; k < 4; ++k) {
        tapes[a_idx].w[k] = sh_buf[tid][k];
        tapes[b_idx].w[k] = sh_buf[tid][k + 4];
    }

    if (diff_bits > 32 && fa > fb) {
        int32_t give = fa >> 2;
        fuel[a_idx] = fa - give;
        fuel[b_idx] = min(4096, fb + give);
        if (repro_out) repro_out[p] = 1;
    } else {
        if (repro_out) repro_out[p] = 0;
    }

    rngs[orig_i] = local_rng;
}

__global__ void thermal_melt_kernel(Tape256* tapes, const int32_t* fuel, uint64_t* rngs, uint32_t n_tapes) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tapes || fuel[idx] > 0) return;

    uint64_t local_rng = rngs[idx];
    for (int w = 0; w < 4; ++w) tapes[idx].w[w] ^= sparse_bitmask(local_rng, 7);
    rngs[idx] = local_rng;
}

__global__ void eval_temporal_table_kernel(
    const Tape256* tapes, uint32_t n_tapes, uint32_t regime, uint8_t* score_out
) {
    __shared__ uint64_t sh_s[256][4];
    __shared__ uint16_t sh_stack[256][16];
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tapes) return;

    uint32_t tid = threadIdx.x;
    uint8_t ok = 0;
    for (int xp = 0; xp <= 1; ++xp) {
        for (int xc = 0; xc <= 1; ++xc) {
            for (int y1 = 0; y1 <= 1; ++y1) {
                for (int y2 = 0; y2 <= 1; ++y2) {
                    for (int k = 0; k < 4; ++k) sh_s[tid][k] = tapes[idx].w[k];
                    set_bit_n(sh_s[tid], 254U, (uint8_t)xp);
                    set_bit_n(sh_s[tid], 255U, (uint8_t)y1);
                    eval_vm_packed(sh_s[tid], sh_stack[tid], 16U, 255U, 0U, 0U, 256U);

                    set_bit_n(sh_s[tid], 254U, (uint8_t)xc);
                    set_bit_n(sh_s[tid], 255U, (uint8_t)y2);
                    eval_vm_packed(sh_s[tid], sh_stack[tid], 16U, 255U, 0U, 0U, 256U);

                    if (get_bit_n(sh_s[tid], 255U) == target_temporal(regime, (uint8_t)xc, (uint8_t)xp)) ok++;
                }
            }
        }
    }
    score_out[idx] = ok;
}

static std::string disasm_packed(const Tape256& t, size_t& active_ops_out) {
    const char* names[] = {".", "<", ">", "{", "}", "C", "c", "[", "]", "!", "?"};
    std::string s;
    active_ops_out = 0;
    for (uint32_t pc = 0; pc < 256; pc += 4) {
        int op = map_op4(get_nib_aligned(t.w, pc));
        s += names[op];
        if (op > 0) active_ops_out++;
    }
    return s;
}

static std::string probe_emergent_memory_bits(const Tape256& t, uint32_t regime) {
    uint64_t s0[4], s1[4];
    uint16_t st0[16], st1[16];
    for (int k = 0; k < 4; ++k) { s0[k] = t.w[k]; s1[k] = t.w[k]; }
    set_bit_n(s0, 254U, 0); set_bit_n(s0, 255U, 0);
    set_bit_n(s1, 254U, 1); set_bit_n(s1, 255U, 0);
    eval_vm_packed(s0, st0, 16U, 255U, 0U, 0U, 256U);
    eval_vm_packed(s1, st1, 16U, 255U, 0U, 0U, 256U);

    std::string bits;
    for (uint32_t b = 0; b < 254U; ++b) {
        if (get_bit_n(s0, b) != get_bit_n(s1, b)) {
            if (!bits.empty()) bits += ",";
            bits += std::to_string(b);
        }
    }
    if (bits.empty()) bits = "none";

    int ok = 0;
    for (int xp = 0; xp <= 1; ++xp) {
        for (int xc = 0; xc <= 1; ++xc) {
            for (int y1 = 0; y1 <= 1; ++y1) {
                for (int y2 = 0; y2 <= 1; ++y2) {
                    uint64_t s[4] = {t.w[0], t.w[1], t.w[2], t.w[3]};
                    set_bit_n(s, 254U, (uint8_t)xp); set_bit_n(s, 255U, (uint8_t)y1);
                    eval_vm_packed(s, st0, 16U, 255U, 0U, 0U, 256U);
                    set_bit_n(s, 254U, (uint8_t)xc); set_bit_n(s, 255U, (uint8_t)y2);
                    eval_vm_packed(s, st0, 16U, 255U, 0U, 0U, 256U);
                    if (get_bit_n(s, 255U) == target_temporal(regime, (uint8_t)xc, (uint8_t)xp)) ok++;
                }
            }
        }
    }
    return "tt=" + std::to_string(ok) + "/16, mem_bits=[" + bits + "]";
}

struct TapeHash {
    size_t operator()(const Tape256& t) const {
        uint64_t h = t.w[0] ^ (t.w[1] * 0x9E3779B97F4A7C15ULL) ^ (t.w[2] * 0xBF58476D1CE4E5B9ULL) ^ (t.w[3] * 0x94D049BB133111EBULL);
        return (size_t)(h ^ (h >> 32));
    }
};

struct TapeEq {
    bool operator()(const Tape256& a, const Tape256& b) const {
        return a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.w[2] == b.w[2] && a.w[3] == b.w[3];
    }
};

struct RegimeBlock {
    uint32_t end_ep;
    uint32_t regime;
    const char* name;
};

struct SeedResult {
    std::string report;
};

SeedResult run_gpu_temporal(
    int gpu_id, uint64_t seed, uint32_t W, uint32_t H, bool invert,
    const std::vector<RegimeBlock>& schedule, const char* label
) {
    cudaSetDevice(gpu_id);
    const uint32_t N = W * H;
    Tape256 *d_tapes = nullptr;
    int32_t *d_fuel = nullptr, *d_corr = nullptr, *d_repro = nullptr;
    uint64_t *d_rngs = nullptr;
    uint8_t *d_prev_x = nullptr, *d_tt = nullptr;

    cudaMalloc(&d_tapes, N * sizeof(Tape256));
    cudaMalloc(&d_fuel, N * sizeof(int32_t));
    cudaMalloc(&d_corr, N * sizeof(int32_t));
    cudaMalloc(&d_repro, (N / 2) * sizeof(int32_t));
    cudaMalloc(&d_rngs, N * sizeof(uint64_t));
    cudaMalloc(&d_prev_x, N * sizeof(uint8_t));
    cudaMalloc(&d_tt, N * sizeof(uint8_t));

    init_population_kernel<<<(N + 255) / 256, 256>>>(d_tapes, d_fuel, d_rngs, d_prev_x, N, seed, 4096, invert);
    cudaDeviceSynchronize();

    std::vector<Tape256> h_tapes(N);
    std::vector<int32_t> h_fuel(N), h_corr(N), h_repro(N / 2);
    std::vector<uint8_t> h_tt(N);
    uint64_t host_rng = seed ^ 0xA5A5A5A5A5A5A5A5ULL;

    std::ostringstream oss;
    oss << "========================================================================\n";
    oss << "KARYON-CORE EXP-1168 (GPU #" << gpu_id << " SM_75, " << W << "x" << H << "): " << label << "\n";
    oss << "Standard: KEP v51.0 | Continuous Persistent Stream + Single Bit X_t (N=" << N << ")\n";
    oss << "========================================================================\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t total_ep = schedule.back().end_ep;

    for (uint32_t ep = 0; ep <= total_ep; ++ep) {
        uint32_t regime = schedule.back().regime;
        const char* reg_str = schedule.back().name;
        bool is_boundary = (ep == 0);
        for (const auto& rb : schedule) {
            if (ep <= rb.end_ep) { regime = rb.regime; reg_str = rb.name; break; }
        }
        for (const auto& rb : schedule) if (ep == rb.end_ep) is_boundary = true;

        env_ticks_kernel<<<(N + 255) / 256, 256>>>(d_tapes, d_fuel, d_rngs, d_prev_x, N, regime, 8, d_corr);

        uint32_t h_mode = (uint32_t)(xorshift64(host_rng) & 1ULL);
        uint32_t v_mode = 2U | (uint32_t)(xorshift64(host_rng) & 1ULL);
        checkerboard_pair_kernel<<<((N / 2) + 255) / 256, 256>>>(d_tapes, d_fuel, d_rngs, W, H, h_mode, N / 2, nullptr);
        checkerboard_pair_kernel<<<((N / 2) + 255) / 256, 256>>>(d_tapes, d_fuel, d_rngs, W, H, v_mode, N / 2, d_repro);
        thermal_melt_kernel<<<(N + 255) / 256, 256>>>(d_tapes, d_fuel, d_rngs, N);

        if (is_boundary || ep % 400 == 0) {
            eval_temporal_table_kernel<<<(N + 255) / 256, 256>>>(d_tapes, N, regime, d_tt);
            cudaDeviceSynchronize();
            cudaMemcpy(h_tapes.data(), d_tapes, N * sizeof(Tape256), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fuel.data(), d_fuel, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_corr.data(), d_corr, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_repro.data(), d_repro, (N / 2) * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tt.data(), d_tt, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

            long long total_corr = 0, total_fuel = 0;
            int32_t max_fuel = 0;
            int repro_sum = 0, n100 = 0, n75 = 0;
            size_t zero_bits = 0;
            std::unordered_map<Tape256, int, TapeHash, TapeEq> top;
            top.reserve(N);

            for (uint32_t i = 0; i < N; ++i) {
                total_corr += h_corr[i];
                total_fuel += h_fuel[i];
                if (h_fuel[i] > max_fuel) max_fuel = h_fuel[i];
                if (h_tt[i] == 16) n100++;
                else if (h_tt[i] >= 12) n75++;
                top[h_tapes[i]]++;
                for (int k = 0; k < 4; ++k) zero_bits += (64 - __builtin_popcountll(h_tapes[i].w[k]));
            }
            for (uint32_t i = 0; i < N / 2; ++i) repro_sum += h_repro[i];

            std::vector<std::pair<int, Tape256>> sorted;
            sorted.reserve(top.size());
            for (const auto& kv : top) sorted.push_back({kv.second, kv.first});
            std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

            double acc = 100.0 * total_corr / (N * 8.0);
            double zero_pct = 100.0 * zero_bits / (N * 256.0);
            oss << "  ep=" << std::setw(4) << ep << " [" << std::setw(9) << reg_str << "]"
                << " | Acc=" << std::fixed << std::setprecision(2) << std::setw(6) << acc << "%"
                << " | AvgF=" << std::setw(4) << (total_fuel / (long long)N)
                << " | MaxF=" << std::setw(4) << max_fuel
                << " | Repro=" << std::setw(5) << repro_sum
                << " | 100%=" << std::setw(5) << n100 << " 75%=" << std::setw(5) << n75
                << " | zero=" << std::setprecision(1) << zero_pct << "%"
                << " | u=" << sorted.size() << "/" << N << "\n";

            if (is_boundary && ep > 0) {
                for (int r = 0; r < std::min<int>(2, sorted.size()); ++r) {
                    size_t ao = 0;
                    std::string ds = disasm_packed(sorted[r].second, ao);
                    std::string probe = probe_emergent_memory_bits(sorted[r].second, regime);
                    oss << "   #" << r << " (" << sorted[r].first << "/" << N << ", act="
                        << std::setprecision(1) << (100.0 * ao / ds.size()) << "%, " << probe << "): " << ds << "\n";
                }
            }
        }
    }

    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    oss << "  Elapsed GPU #" << gpu_id << " Time: " << std::fixed << std::setprecision(3) << sec << "s\n\n";

    cudaFree(d_tapes);
    cudaFree(d_fuel);
    cudaFree(d_corr);
    cudaFree(d_repro);
    cudaFree(d_rngs);
    cudaFree(d_prev_x);
    cudaFree(d_tt);
    return {oss.str()};
}

int main() {
    std::cout << std::unitbuf;
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    int g0 = 0, g1 = (gpu_count > 1) ? 1 : 0;

    std::vector<RegimeBlock> temp_sched = {
        {800U, 0U, "Delay1"},
        {2200U, 1U, "TempXOR"},
        {3200U, 2U, "Temp~XOR"}
    };
    std::vector<RegimeBlock> temp_ood_sched = {
        {800U, 0U, "Delay1"},
        {2200U, 1U, "TempXOR"},
        {3200U, 2U, "Temp~XOR"},
        {3800U, 3U, "OOD:CurrX"}
    };

    auto t_all = std::chrono::high_resolution_clock::now();

    SeedResult r0, r1;
    std::thread th0([&]() { r0 = run_gpu_temporal(g0, 0x9E3779B97F4A7C15ULL, 128, 128, false, temp_sched, "Seed 0 (128x128)"); });
    std::thread th1([&]() { r1 = run_gpu_temporal(g1, 0x123456789ABCDEF0ULL, 128, 128, false, temp_sched, "Seed 1 (128x128)"); });
    th0.join(); th1.join();
    std::cout << r0.report << r1.report;

    SeedResult r_inv, r_256;
    std::thread th2([&]() { r_inv = run_gpu_temporal(g0, 0x9E3779B97F4A7C15ULL, 128, 128, true, temp_sched, "Seed 0 Inverted 0<->1 (128x128)"); });
    std::thread th3([&]() { r_256 = run_gpu_temporal(g1, 0x5555555512345678ULL, 256, 256, false, temp_ood_sched, "Scale 256x256 + OOD:CurrX (N=65536)"); });
    th2.join(); th3.join();
    std::cout << r_inv.report << r_256.report;

    double tot_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_all).count();
    std::cout << "Total Wall-Clock Time (EXP-1168 Stage 3 Emergent Memory): "
              << std::fixed << std::setprecision(2) << tot_sec << "s\n";
    return 0;
}
