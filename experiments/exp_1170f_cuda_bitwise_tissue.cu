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
// KARYON-CORE EXP-1170f: DUAL-GPU CUDA BITWISE TISSUE (APOPTOSIS + COPY FIDELITY)
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | Unbounded Graph | Dual Tesla T4 CUDA Substrate
// ============================================================================
// Clean physical deltas from EXP-1170e:
// 1. Dual Tesla T4 GPU Acceleration (sm_75, nvcc -O3):
//    - Concurrent GPU #0 (Seed 0) & GPU #1 (Seed 1) evaluation on 128x128 grid (16,384 cells).
//    - Shared-memory broadcast of continuous input/target streams per block.
// 2. Active Cone Apoptosis (On-GPU Hanging Node Cleansing):
//    - Traces backward from out_s to remove non-functional introns directly on GPU every 250 sweeps,
//      preserving exact causal topology order (j < u invariant) and keeping K compact.
// 3. Copy Fidelity (50% Exact Clones / 50% Mutated Clones):
//    - Prevents error catastrophe, allowing high-performing (92%-99%) lineages to form stable colonies.
// 4. Smart Readout Combiner & Algebraic Dependency Analysis:
//    - Traces active dependency cone at end of run and prints exact Boolean gates.

#define MAX_K 256
#define GRID_W 128
#define GRID_H 128
#define TOTAL_CELLS (GRID_W * GRID_H)
#define REPRO_THRESH 1024
#define MAX_FUEL 4096
#define T_TICKS 64

struct Node {
    uint8_t op;
    uint16_t src_a, src_b;
};

struct alignas(16) OrganismGPU {
    uint32_t node_count;
    uint32_t out_s;
    int32_t fuel;
    uint64_t rng;
    Node nodes[MAX_K];
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

__host__ __device__ static inline uint64_t read_wire_dev(
    uint16_t s, uint32_t u, uint32_t K,
    uint64_t i0, uint64_t i1, uint64_t ir,
    const uint64_t* vc, const uint64_t* state
) {
    if (s == 0U) return i0;
    if (s == 1U) return i1;
    if (s == 2U) return ir;
    uint32_t j = (uint32_t)(s - 3U);
    if (j >= K) j %= K;
    return (j < u) ? vc[j] : state[j];
}

__host__ __device__ static inline uint32_t pick_upper_biased_node(uint32_t K, uint64_t& rng) {
    if (K <= 1U) return 0U;
    uint32_t u1 = (uint32_t)(xorshift64_dev(rng) % K);
    uint32_t u2 = (uint32_t)(xorshift64_dev(rng) % K);
    return (u1 > u2) ? u1 : u2;
}

__host__ __device__ static void apoptosis_org(OrganismGPU& org) {
    uint32_t K = org.node_count;
    if (K <= 4U) return;
    uint32_t root = org.out_s % K;

    bool active[MAX_K];
    uint32_t stack[MAX_K];
    for (uint32_t u = 0; u < K; ++u) active[u] = false;

    uint32_t sp = 0U;
    stack[sp++] = root;
    active[root] = true;

    while (sp > 0U) {
        uint32_t u = stack[--sp];
        uint16_t sa = org.nodes[u].src_a;
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U) % K;
            if (!active[j]) { active[j] = true; stack[sp++] = j; }
        }
        uint16_t sb = org.nodes[u].src_b;
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U) % K;
            if (!active[j]) { active[j] = true; stack[sp++] = j; }
        }
    }

    uint32_t new_idx[MAX_K];
    uint32_t new_K = 0U;
    for (uint32_t u = 0; u < K; ++u) {
        new_idx[u] = active[u] ? new_K++ : 0U;
    }

    if (new_K == K || new_K == 0U) return;

    Node new_nodes[MAX_K];
    uint64_t new_state[MAX_K];
    for (uint32_t u = 0; u < K; ++u) {
        if (active[u]) {
            uint32_t nu = new_idx[u];
            Node nd = org.nodes[u];
            if (nd.src_a >= 3U) {
                uint32_t j = (uint32_t)(nd.src_a - 3U) % K;
                nd.src_a = (uint16_t)(new_idx[j] + 3U);
            }
            if (nd.src_b >= 3U) {
                uint32_t j = (uint32_t)(nd.src_b - 3U) % K;
                nd.src_b = (uint16_t)(new_idx[j] + 3U);
            }
            new_nodes[nu] = nd;
            new_state[nu] = org.state[u];
        }
    }

    org.node_count = new_K;
    org.out_s = new_idx[root];
    for (uint32_t u = 0; u < new_K; ++u) {
        org.nodes[u] = new_nodes[u];
        org.state[u] = new_state[u];
    }
}

__host__ __device__ static void prune_one_node(OrganismGPU& org, uint64_t& rng) {
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
        uint16_t sa = org.nodes[u].src_a;
        if (sa >= 3U) {
            uint32_t j = (uint32_t)(sa - 3U);
            if (j == du) org.nodes[u].src_a = (uint16_t)(xorshift64_dev(rng) % (nK + 3U));
            else if (j > du) org.nodes[u].src_a = (uint16_t)(sa - 1U);
        }
        uint16_t sb = org.nodes[u].src_b;
        if (sb >= 3U) {
            uint32_t j = (uint32_t)(sb - 3U);
            if (j == du) org.nodes[u].src_b = (uint16_t)(xorshift64_dev(rng) % (nK + 3U));
            else if (j > du) org.nodes[u].src_b = (uint16_t)(sb - 1U);
        }
    }
    if (org.out_s == du) org.out_s = (uint32_t)(xorshift64_dev(rng) % nK);
    else if (org.out_s > du) org.out_s--;
}

__host__ __device__ static void graft_subgraph(const OrganismGPU& donor, OrganismGPU& host, uint64_t& rng) {
    uint32_t Kd = donor.node_count, Kh = host.node_count;
    if (Kd < 1U || Kh < 1U || Kh >= (MAX_K - 8U)) return;
    uint32_t want = 2U + (uint32_t)(xorshift64_dev(rng) % 4ULL);
    uint32_t cnt = (Kd < want) ? Kd : want;
    uint32_t su = (uint32_t)(xorshift64_dev(rng) % (Kd - cnt + 1U));
    uint32_t start_kh = Kh;

    for (uint32_t c = 0; c < cnt; ++c) {
        Node nd = donor.nodes[su + c];
        if (nd.src_a >= 3U) {
            uint32_t j = (uint32_t)(nd.src_a - 3U);
            if (j >= su && j < su + cnt) nd.src_a = (uint16_t)(start_kh + (j - su) + 3U);
            else nd.src_a = (uint16_t)((j % start_kh) + 3U);
        }
        if (nd.src_b >= 3U) {
            uint32_t j = (uint32_t)(nd.src_b - 3U);
            if (j >= su && j < su + cnt) nd.src_b = (uint16_t)(start_kh + (j - su) + 3U);
            else nd.src_b = (uint16_t)((j % start_kh) + 3U);
        }
        host.nodes[host.node_count] = nd;
        host.state[host.node_count] = 0ULL;
        host.node_count++;
    }

    uint32_t new_last = host.node_count - 1U;
    uint32_t ro_opt = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (ro_opt < 45U && host.node_count < MAX_K) {
        uint16_t src_old = (uint16_t)((host.out_s % start_kh) + 3U);
        uint16_t src_new = (uint16_t)(new_last + 3U);
        uint8_t comb_op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
        host.nodes[host.node_count] = {comb_op, src_old, src_new};
        host.state[host.node_count] = 0ULL;
        host.node_count++;
        host.out_s = host.node_count - 1U;
    } else if (ro_opt < 75U) {
        host.out_s = new_last;
    }
}

__host__ __device__ static void mutate_organism(OrganismGPU& org, uint64_t& rng) {
    uint32_t K = org.node_count;
    if (K == 0U) return;
    uint32_t r = (uint32_t)(xorshift64_dev(rng) % 100ULL);
    if (r < 35U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        org.nodes[u].op = (uint8_t)(xorshift64_dev(rng) & 3ULL);
    } else if (r < 70U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        uint16_t ns = (uint16_t)(xorshift64_dev(rng) % (K + 3U));
        if (xorshift64_dev(rng) & 1ULL) org.nodes[u].src_a = ns; else org.nodes[u].src_b = ns;
    } else if (r < 80U) {
        if (K < (MAX_K - 2U)) {
            org.nodes[K] = {(uint8_t)(xorshift64_dev(rng) & 3ULL),
                            (uint16_t)(xorshift64_dev(rng) % (K + 3U)),
                            (uint16_t)(xorshift64_dev(rng) % (K + 3U))};
            org.state[K] = 0ULL;
            org.node_count++;
            uint32_t new_u = org.node_count - 1U;
            uint32_t ro_opt = (uint32_t)(xorshift64_dev(rng) % 100ULL);
            if (ro_opt < 40U && org.node_count < MAX_K) {
                uint16_t src_old = (uint16_t)((org.out_s % K) + 3U);
                uint16_t src_new = (uint16_t)(new_u + 3U);
                org.nodes[org.node_count] = {(uint8_t)(xorshift64_dev(rng) & 3ULL), src_old, src_new};
                org.state[org.node_count] = 0ULL;
                org.node_count++;
                org.out_s = org.node_count - 1U;
            } else if (ro_opt < 70U) {
                org.out_s = new_u;
            }
        }
    } else if (r < 88U) {
        graft_subgraph(org, org, rng);
    } else if (r < 97U) {
        prune_one_node(org, rng);
        if (r >= 93U) prune_one_node(org, rng);
    } else {
        org.out_s = pick_upper_biased_node(K, rng);
    }
}

__host__ __device__ static uint32_t eval_org(
    OrganismGPU& org, const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t,
    const uint64_t* targ_t, int ticks, uint64_t* vc
) {
    const uint32_t K = org.node_count;
    if (K == 0U) return 0U;
    uint32_t matches = 0U, os = org.out_s % K;
    for (int t = 0; t < ticks; ++t) {
        const uint64_t i0 = in0_t[t], i1 = in1_t[t], ir = in_rst_t[t];
        for (uint32_t u = 0; u < K; ++u) {
            const Node nd = org.nodes[u];
            uint64_t a = read_wire_dev(nd.src_a, u, K, i0, i1, ir, vc, org.state);
            uint64_t b = read_wire_dev(nd.src_b, u, K, i0, i1, ir, vc, org.state);
            uint64_t r = 0ULL;
            switch (nd.op & 3U) {
                case 0: r = ~(a & b); break;
                case 1: r = ~(a | b); break;
                case 2: r = a ^ b;    break;
                case 3: r = ~(a ^ b); break;
            }
            vc[u] = r;
        }
#ifdef __CUDA_ARCH__
        matches += (uint32_t)__popcll(~(vc[os] ^ targ_t[t]));
#else
        matches += (uint32_t)__builtin_popcountll(~(vc[os] ^ targ_t[t]));
#endif
        for (uint32_t u = 0; u < K; ++u) org.state[u] = vc[u];
    }
    return matches;
}

__device__ static void eval_and_update_fuel(
    OrganismGPU& org, const uint64_t* s_i0, const uint64_t* s_i1,
    const uint64_t* s_ir, const uint64_t* s_tg, uint64_t* vc_mem
) {
    if (org.fuel <= 0) return;
    const uint32_t tot_bits = (uint32_t)T_TICKS * 64U;
    uint32_t m = eval_org(org, s_i0, s_i1, s_ir, s_tg, T_TICKS, vc_mem);
    int32_t rew = (int32_t)(((2LL * (int64_t)m - (int64_t)tot_bits) * 72LL) / (int64_t)tot_bits);
    int32_t ncost = (int32_t)((org.node_count + 7U) >> 3);
    int32_t fuel_next = org.fuel + rew - ncost - ((org.fuel >> 6) + 1);
    if (fuel_next < 0) fuel_next = 0;
    if (fuel_next > MAX_FUEL) fuel_next = MAX_FUEL;
    org.fuel = fuel_next;
}

__global__ void tissue_sweep_kernel(
    OrganismGPU* orgs, uint32_t n_pairs, uint32_t pass_mode,
    const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t, const uint64_t* targ_t
) {
    __shared__ uint64_t s_i0[T_TICKS];
    __shared__ uint64_t s_i1[T_TICKS];
    __shared__ uint64_t s_ir[T_TICKS];
    __shared__ uint64_t s_tg[T_TICKS];

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

    uint64_t vc_mem[MAX_K];
    uint64_t lr = orgs[oi].rng;

    eval_and_update_fuel(orgs[oi], s_i0, s_i1, s_ir, s_tg, vc_mem);
    eval_and_update_fuel(orgs[oj], s_i0, s_i1, s_ir, s_tg, vc_mem);

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
            // Copy Fidelity: 50% exact clone, 50% mutated clone
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

__global__ void apoptosis_kernel(OrganismGPU* orgs, uint32_t n_cells) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    if (orgs[i].node_count > 12U) {
        apoptosis_org(orgs[i]);
    }
}

static void gen_continuous_stream(int n_words, int bits_per_word, uint64_t& rng, uint64_t* i0, uint64_t* i1, uint64_t* ir, uint64_t* tg) {
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

static void print_active_circuit(const OrganismGPU& org) {
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
        auto check_wire = [&](uint16_t s) {
            if (s >= 3U) {
                uint32_t j = (uint32_t)(s - 3U) % K;
                if (active_nodes.find(j) == active_nodes.end()) {
                    active_nodes.insert(j);
                    stack.push_back(j);
                }
            }
        };
        check_wire(org.nodes[u].src_a);
        check_wire(org.nodes[u].src_b);
    }

    static const char* OPN[4] = {"NAND", "NOR ", "XOR ", "XNOR"};
    std::cout << "    ========================================================\n"
              << "    ACTIVE CONE CIRCUIT DIAGRAM (Total K=" << K << ", Active Nodes=" << active_nodes.size() << ", out=n" << root << "):\n"
              << "    ========================================================\n";
    for (uint32_t u : active_nodes) {
        auto fmt = [&](uint16_t s) -> std::string {
            if (s == 0U) return "inA"; if (s == 1U) return "inB"; if (s == 2U) return "rst";
            uint32_t j = (uint32_t)(s - 3U) % K;
            return "n" + std::to_string(j) + ((j >= u) ? "(t-1)" : "(t)");
        };
        std::string extra = "";
        if (org.nodes[u].src_a == org.nodes[u].src_b) {
            uint8_t op = org.nodes[u].op & 3U;
            if (op == 0U || op == 1U) extra = "  [NOT " + fmt(org.nodes[u].src_a) + "]";
            else if (op == 2U) extra = "  [CONST 0]";
            else extra = "  [CONST 1]";
        }
        std::cout << "      n" << u << (u == root ? "*" : " ") << " = "
                  << OPN[org.nodes[u].op & 3U] << "(" << fmt(org.nodes[u].src_a) << ", " << fmt(org.nodes[u].src_b) << ")"
                  << extra << "\n";
    }
    std::cout << "    ========================================================\n\n";
}

static void run_gpu_seed(int gpu_id, int seed_idx, int max_sweeps, uint64_t seed) {
    cudaSetDevice(gpu_id);
    const uint32_t N = TOTAL_CELLS, n_pairs = N / 2;

    std::vector<OrganismGPU> host_orgs(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t s = seed ^ ((uint64_t)(i + 1U) * 0x9E3779B97F4A7C15ULL);
        host_orgs[i].node_count = 4;
        for (int u = 0; u < 4; ++u) {
            host_orgs[i].nodes[u] = {(uint8_t)(splitmix64(s) & 3ULL),
                                     (uint16_t)(splitmix64(s) % 7U),
                                     (uint16_t)(splitmix64(s) % 7U)};
            host_orgs[i].state[u] = 0ULL;
        }
        host_orgs[i].out_s = (uint32_t)(splitmix64(s) % 4U);
        host_orgs[i].fuel = 2048;
        uint64_t r = splitmix64(s);
        host_orgs[i].rng = r ? r : 0xCAFEBABE12345678ULL;
    }

    OrganismGPU* d_orgs = nullptr;
    cudaMalloc(&d_orgs, N * sizeof(OrganismGPU));
    cudaMemcpy(d_orgs, host_orgs.data(), N * sizeof(OrganismGPU), cudaMemcpyHostToDevice);

    uint64_t *d_i0, *d_i1, *d_ir, *d_tg;
    cudaMalloc(&d_i0, T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_i1, T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_ir, T_TICKS * sizeof(uint64_t));
    cudaMalloc(&d_tg, T_TICKS * sizeof(uint64_t));

    uint64_t hrng = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
    uint64_t i0[T_TICKS], i1[T_TICKS], ir[T_TICKS], tg[T_TICKS];

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp critical
    std::cout << "--- GPU #" << gpu_id << " SEED " << seed_idx << " (Grid 128x128, 16,384 Tissue Cells, max_sw=" << max_sweeps << ") ---\n";

    int threads = 256;
    int blocks_pairs = (n_pairs + threads - 1) / threads;
    int blocks_cells = (N + threads - 1) / threads;
    const uint32_t tot_bits = T_TICKS * 64U;

    for (int sw = 0; sw <= max_sweeps; ++sw) {
        gen_continuous_stream(8, 8, hrng, i0, i1, ir, tg);
        cudaMemcpy(d_i0, i0, T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_i1, i1, T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ir, ir, T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tg, tg, T_TICKS * sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint32_t pm = (uint32_t)(xorshift64_dev(hrng) & 3ULL);
        tissue_sweep_kernel<<<blocks_pairs, threads>>>(d_orgs, n_pairs, pm, d_i0, d_i1, d_ir, d_tg);

        if (sw > 0 && sw % 250 == 0) {
            apoptosis_kernel<<<blocks_cells, threads>>>(d_orgs, N);
        }

        if (sw == 0 || sw == 2000 || sw == 4000 || sw == 8000 || sw == 12000 || sw == 16000 || sw == 20000 || sw == max_sweeps) {
            cudaDeviceSynchronize();
            cudaMemcpy(host_orgs.data(), d_orgs, N * sizeof(OrganismGPU), cudaMemcpyDeviceToHost);

            uint64_t erng = 0xDEADBEEFCAFE1234ULL ^ (uint64_t)sw;
            uint64_t ei0[T_TICKS], ei1[T_TICKS], eir[T_TICKS], etg[T_TICKS];
            gen_continuous_stream(8, 8, erng, ei0, ei1, eir, etg);

            long long tf = 0, sk = 0; int alive = 0, n90 = 0, n100 = 0;
            uint32_t mink = 999999U, maxk = 0U, bidx = 0U;
            double macc = 0.0, sacc = 0.0;
            uint64_t vc[MAX_K];

            for (uint32_t i = 0; i < N; ++i) {
                tf += host_orgs[i].fuel; if (host_orgs[i].fuel > 0) alive++;
                uint32_t K = host_orgs[i].node_count;
                sk += K; mink = std::min(mink, K); maxk = std::max(maxk, K);

                OrganismGPU tmp = host_orgs[i];
                for (uint32_t u = 0; u < K; ++u) tmp.state[u] = 0ULL;
                uint32_t matches = eval_org(tmp, ei0, ei1, eir, etg, T_TICKS, vc);

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
                    OrganismGPU best_o = host_orgs[bidx];
                    for (uint32_t u = 0; u < best_o.node_count; ++u) best_o.state[u] = 0ULL;

                    for (int b = 0; b < 16; ++b) {
                        gen_continuous_stream(2, 32, orng, ei0, ei1, eir, etg);
                        for (int t = 0; t < T_TICKS; ++t) {
                            uint32_t m1 = eval_org(best_o, &ei0[t], &ei1[t], &eir[t], &etg[t], 1, vc);
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
    }

    cudaDeviceSynchronize();
    cudaFree(d_orgs);
    cudaFree(d_i0); cudaFree(d_i1); cudaFree(d_ir); cudaFree(d_tg);
    #pragma omp critical
    std::cout << "GPU #" << gpu_id << " Elapsed: " << std::setprecision(2)
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() << "s\n" << std::flush;
}

int main(int argc, char** argv) {
    int max_sweeps = (argc > 1) ? std::atoi(argv[1]) : 24000;
    std::cout << "========================================================================\n"
              << "KARYON-CORE EXP-1170f: DUAL-GPU CUDA BITWISE TISSUE (APOPTOSIS + COPY FIDELITY)\n"
              << "Standard: KEP v51.0 | Dual Tesla T4 | Grid 128x128 (16,384) | max_sw=" << max_sweeps << "\n"
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
