#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1170e: CONTINUOUS BITWISE TISSUE (100% SOLVER PURSUIT & GRAPH TRACER)
// Lead Architect: Bazilevs (ProgVM) | Lead AI Cyberneticist
// Standard: KEP v51.0 | Unbounded Graph | Zero Fixed Ceiling | Zero Curriculum
// ============================================================================
// Clean physical deltas from EXP-1170d:
// 1. Extended Evolutionary Horizon (18,000 - 20,000 sweeps):
//    - Gives Seed 0 (98.38% carry cascade pioneer) and Seed 1 (100% rst_bit0 pioneer)
//      sufficient sweeps for symbiogenesis to attach rst_bit0 gating atop the 8/8 carry core.
// 2. Active Cone Dependency Analysis:
//    - Traces backward from out_s to extract only the functional nodes in the active graph cone,
//      printing clean algebraic Boolean equations for Bazilevs' direct inspection.

struct Node { uint8_t op; uint16_t src_a, src_b; };
struct alignas(64) Organism {
    std::vector<Node> nodes;
    std::vector<uint64_t> state;
    uint32_t out_s;
    int32_t fuel;
    uint64_t rng;
};

static inline uint64_t xorshift64(uint64_t& s) {
    uint64_t x = s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline uint64_t splitmix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL; uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint32_t eval_org_continuous(
    Organism& org, const uint64_t* in0_t, const uint64_t* in1_t, const uint64_t* in_rst_t,
    const uint64_t* targ_t, int T_ticks, std::vector<uint64_t>& vc
) {
    const uint32_t K = (uint32_t)org.nodes.size();
    if (K == 0U) return 0U;
    if (org.state.size() != K) org.state.resize(K, 0ULL);
    if (vc.size() < K) vc.resize(K);
    uint32_t matches = 0U, os = org.out_s % K;
    for (int t = 0; t < T_ticks; ++t) {
        const uint64_t i0 = in0_t[t], i1 = in1_t[t], ir = in_rst_t[t];
        for (uint32_t u = 0; u < K; ++u) {
            const Node& nd = org.nodes[u];
            auto rw = [&](uint16_t s) -> uint64_t {
                if (s == 0U) return i0;
                if (s == 1U) return i1;
                if (s == 2U) return ir;
                uint32_t j = (uint32_t)(s - 3U);
                if (j >= K) j %= K;
                return (j < u) ? vc[j] : org.state[j];
            };
            uint64_t a = rw(nd.src_a), b = rw(nd.src_b), r = 0ULL;
            switch (nd.op & 3U) {
                case 0: r = ~(a & b); break;
                case 1: r = ~(a | b); break;
                case 2: r = a ^ b;    break;
                case 3: r = ~(a ^ b); break;
            }
            vc[u] = r;
        }
        matches += (uint32_t)__builtin_popcountll(~(vc[os] ^ targ_t[t]));
        for (uint32_t u = 0; u < K; ++u) org.state[u] = vc[u];
    }
    return matches;
}

static inline uint32_t pick_upper_biased_node(uint32_t K, uint64_t& rng) {
    uint32_t u1 = (uint32_t)(xorshift64(rng) % K);
    uint32_t u2 = (uint32_t)(xorshift64(rng) % K);
    return (u1 > u2) ? u1 : u2;
}

static void prune_one_node(Organism& org, uint64_t& rng) {
    uint32_t K = (uint32_t)org.nodes.size();
    if (K <= 4U) return;
    uint32_t du = (uint32_t)(xorshift64(rng) % K);
    org.nodes.erase(org.nodes.begin() + du);
    org.state.erase(org.state.begin() + du);
    uint32_t nK = (uint32_t)org.nodes.size();
    auto rm = [&](uint16_t& s) {
        if (s >= 3U) {
            uint32_t j = (uint32_t)(s - 3U);
            if (j == du) s = (uint16_t)(xorshift64(rng) % (nK + 3U));
            else if (j > du) s--;
        }
    };
    for (auto& nd : org.nodes) { rm(nd.src_a); rm(nd.src_b); }
    if (org.out_s == du) org.out_s = (uint32_t)(xorshift64(rng) % nK);
    else if (org.out_s > du) org.out_s--;
}

static void graft_subgraph(const Organism& donor, Organism& host, uint64_t& rng) {
    uint32_t Kd = (uint32_t)donor.nodes.size(), Kh = (uint32_t)host.nodes.size();
    if (Kd < 2U || Kh >= 256U) return;
    uint32_t cnt = std::min<uint32_t>(Kd, 2U + (uint32_t)(xorshift64(rng) % 4ULL));
    uint32_t su = (uint32_t)(xorshift64(rng) % (Kd - cnt + 1U));
    uint32_t start_kh = Kh;
    for (uint32_t c = 0; c < cnt; ++c) {
        Node nd = donor.nodes[su + c];
        auto map_w = [&](uint16_t& s) {
            if (s >= 3U) {
                uint32_t j = (uint32_t)(s - 3U);
                if (j >= su && j < su + cnt) s = (uint16_t)(start_kh + (j - su) + 3U);
                else s = (uint16_t)((j % start_kh) + 3U);
            }
        };
        map_w(nd.src_a); map_w(nd.src_b);
        host.nodes.push_back(nd);
        host.state.push_back(0ULL);
    }

    uint32_t new_last = (uint32_t)host.nodes.size() - 1U;
    uint32_t ro_opt = (uint32_t)(xorshift64(rng) % 100ULL);
    if (ro_opt < 45U && host.nodes.size() < 256U) {
        uint16_t src_old = (uint16_t)(host.out_s + 3U);
        uint16_t src_new = (uint16_t)(new_last + 3U);
        uint8_t comb_op = (uint8_t)(xorshift64(rng) & 3ULL);
        host.nodes.push_back({comb_op, src_old, src_new});
        host.state.push_back(0ULL);
        host.out_s = (uint32_t)host.nodes.size() - 1U;
    } else if (ro_opt < 75U) {
        host.out_s = new_last;
    }
}

static void mutate_organism(Organism& org, uint64_t& rng) {
    uint32_t K = (uint32_t)org.nodes.size();
    uint32_t r = (uint32_t)(xorshift64(rng) % 100ULL);
    if (r < 35U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        org.nodes[u].op = (uint8_t)(xorshift64(rng) & 3ULL);
    } else if (r < 70U) {
        uint32_t u = pick_upper_biased_node(K, rng);
        uint16_t ns = (uint16_t)(xorshift64(rng) % (K + 3U));
        if (xorshift64(rng) & 1ULL) org.nodes[u].src_a = ns; else org.nodes[u].src_b = ns;
    } else if (r < 80U) {
        if (K < 256U) {
            org.nodes.push_back({(uint8_t)(xorshift64(rng) & 3ULL),
                                 (uint16_t)(xorshift64(rng) % (K + 3U)),
                                 (uint16_t)(xorshift64(rng) % (K + 3U))});
            org.state.push_back(0ULL);
            uint32_t new_u = (uint32_t)org.nodes.size() - 1U;
            uint32_t ro_opt = (uint32_t)(xorshift64(rng) % 100ULL);
            if (ro_opt < 40U && org.nodes.size() < 256U) {
                uint16_t src_old = (uint16_t)(org.out_s + 3U), src_new = (uint16_t)(new_u + 3U);
                org.nodes.push_back({(uint8_t)(xorshift64(rng) & 3ULL), src_old, src_new});
                org.state.push_back(0ULL);
                org.out_s = (uint32_t)org.nodes.size() - 1U;
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

static void gen_continuous_stream(int n_words, int bits_per_word, uint64_t& rng, uint64_t* i0, uint64_t* i1, uint64_t* ir, uint64_t* tg) {
    int idx = 0;
    for (int w = 0; w < n_words; ++w) {
        uint64_t c = 0ULL;
        for (int b = 0; b < bits_per_word; ++b) {
            uint64_t a = xorshift64(rng), bv = xorshift64(rng);
            i0[idx] = a; i1[idx] = bv; ir[idx] = (b == 0) ? ~0ULL : 0ULL;
            uint64_t c_in = (b == 0) ? 0ULL : c;
            tg[idx] = a ^ bv ^ c_in;
            c = (a & bv) | ((a ^ bv) & c_in);
            idx++;
        }
    }
}

static void print_active_circuit(const Organism& org) {
    uint32_t K = (uint32_t)org.nodes.size();
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
        std::cout << "      n" << u << (u == root ? "*" : " ") << " = "
                  << OPN[org.nodes[u].op & 3U] << "(" << fmt(org.nodes[u].src_a) << ", " << fmt(org.nodes[u].src_b) << ")\n";
    }
    std::cout << "    ========================================================\n\n";
}

static void run_seed(int seed_idx, int max_sweeps, uint64_t seed) {
    const uint32_t W = 64, H = 64, N = W * H, n_pairs = N / 2;
    const int32_t MAX_FUEL = 4096, REPRO_THRESH = 1024;
    std::vector<Organism> orgs(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t s = seed ^ ((uint64_t)(i + 1U) * 0x9E3779B97F4A7C15ULL);
        orgs[i].nodes.resize(4);
        orgs[i].state.assign(4, 0ULL);
        for (int u = 0; u < 4; ++u) {
            orgs[i].nodes[u] = {(uint8_t)(splitmix64(s) & 3ULL),
                                (uint16_t)(splitmix64(s) % 7U),
                                (uint16_t)(splitmix64(s) % 7U)};
        }
        orgs[i].out_s = (uint32_t)(splitmix64(s) % 4U);
        orgs[i].fuel = 2048;
        uint64_t r = splitmix64(s);
        orgs[i].rng = r ? r : 0xCAFEBABE12345678ULL;
    }

    uint64_t hrng = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "--- SEED " << seed_idx << " (Grid 64x64, Unbounded Tissue, max_sw=" << max_sweeps << ") ---\n";
    uint64_t i0[64], i1[64], ir[64], tg[64];
    long long wr = 0, wsym = 0;
    const int T_ticks = 64; // 8 x 8-bit words = 4096 bits/sweep
    const uint32_t tot_bits = T_ticks * 64U;

    for (int sw = 0; sw <= max_sweeps; ++sw) {
        gen_continuous_stream(8, 8, hrng, i0, i1, ir, tg);
        uint32_t pm = (uint32_t)(xorshift64(hrng) & 3ULL);
        long long sr = 0, ssym = 0;

        #pragma omp parallel reduction(+:sr,ssym)
        {
            std::vector<uint64_t> vc(256);
            #pragma omp for schedule(static)
            for (uint32_t p = 0; p < n_pairs; ++p) {
                uint32_t oi, oj;
                if (pm < 2U) { uint32_t y = p / (W >> 1), x = ((p % (W >> 1)) << 1) + (pm & 1U); oi = y * W + x; oj = y * W + ((x + 1U) & (W - 1U)); }
                else { uint32_t x = p % W, y = ((p / W) << 1) + (pm & 1U); oi = y * W + x; oj = ((y + 1U) & (H - 1U)) * W + x; }

                uint64_t lr = orgs[oi].rng;
                auto eval_one = [&](uint32_t idx) {
                    if (orgs[idx].fuel <= 0) return;
                    uint32_t m = eval_org_continuous(orgs[idx], i0, i1, ir, tg, T_ticks, vc);
                    int32_t rew = (int32_t)(((2LL * m - (int64_t)tot_bits) * 72LL) / (int64_t)tot_bits);
                    int32_t ncost = (int32_t)((orgs[idx].nodes.size() + 7U) >> 3);
                    orgs[idx].fuel = std::clamp(orgs[idx].fuel + rew - ncost - ((orgs[idx].fuel >> 6) + 1), 0, MAX_FUEL);
                };
                eval_one(oi); eval_one(oj);

                int32_t fi = orgs[oi].fuel, fj = orgs[oj].fuel;
                if (fi <= 0 && fj <= 0) {
                    mutate_organism(orgs[oi], lr); mutate_organism(orgs[oj], lr);
                    orgs[oi].fuel = 256;
                } else {
                    uint32_t ai = (fi >= fj) ? oi : oj, bi = (fi >= fj) ? oj : oi;
                    int32_t fa = orgs[ai].fuel, fb = orgs[bi].fuel;
                    int32_t na = (int32_t)(xorshift64(lr) & 127ULL), nb = (int32_t)(xorshift64(lr) & 127ULL);
                    if (fa >= REPRO_THRESH && fb >= REPRO_THRESH && (xorshift64(lr) % 10ULL) == 0ULL) {
                        graft_subgraph(orgs[ai], orgs[bi], lr);
                        ssym++;
                    } else if (fa >= REPRO_THRESH && (fa + na > fb + 256 + nb)) {
                        orgs[bi].nodes = orgs[ai].nodes;
                        orgs[bi].out_s = orgs[ai].out_s;
                        orgs[bi].state.assign(orgs[ai].nodes.size(), 0ULL);
                        mutate_organism(orgs[bi], lr);
                        int32_t share = fa >> 1;
                        orgs[ai].fuel = fa - share;
                        orgs[bi].fuel = share;
                        sr++;
                    } else if (fb <= 0) {
                        mutate_organism(orgs[bi], lr);
                    }
                }
                orgs[oi].rng = lr;
            }
        }
        wr += sr; wsym += ssym;

        if (sw == 0 || sw == 2000 || sw == 4000 || sw == 6000 || sw == 8000 || sw == 12000 || sw == 16000 || sw == max_sweeps) {
            uint64_t erng = 0xDEADBEEFCAFE1234ULL ^ (uint64_t)sw;
            uint64_t ei0[64], ei1[64], eir[64], etg[64];
            gen_continuous_stream(8, 8, erng, ei0, ei1, eir, etg);
            long long tf = 0, sk = 0; int alive = 0, n90 = 0, n100 = 0;
            uint32_t mink = 999999U, maxk = 0U, bidx = 0U;
            double macc = 0.0, sacc = 0.0;
            std::vector<uint64_t> vc(256);
            for (uint32_t i = 0; i < N; ++i) {
                tf += orgs[i].fuel; if (orgs[i].fuel > 0) alive++;
                uint32_t K = (uint32_t)orgs[i].nodes.size();
                sk += K; mink = std::min(mink, K); maxk = std::max(maxk, K);
                Organism tmp = orgs[i]; tmp.state.assign(K, 0ULL);
                uint32_t m = eval_org_continuous(tmp, ei0, ei1, eir, etg, T_ticks, vc);
                double acc = 100.0 * m / tot_bits; sacc += acc;
                if (acc > macc) { macc = acc; bidx = i; }
                if (acc >= 90.0) n90++;
                if (m == tot_bits) n100++;
            }
            std::cout << "  sw=" << std::setw(5) << sw << " | AvgAcc=" << std::fixed << std::setprecision(2) << (sacc / N)
                      << "% | MaxAcc=" << macc << "% | >=90%=" << std::setw(4) << n90 << " | 100%=" << std::setw(4) << n100
                      << " | AvgF=" << std::setw(4) << (tf / N) << " | Alive=" << std::setw(4) << alive
                      << " | AvgK=" << std::setprecision(1) << ((double)sk / N) << " [" << mink << ".." << maxk << "]"
                      << " | Repro=" << wr << " Sym=" << wsym << "\n";
            wr = 0; wsym = 0;

            if (sw == max_sweeps) {
                uint64_t orng = 0x7777777712345678ULL;
                long long om = 0, om_b0 = 0, om_rest = 0;
                Organism best_o = orgs[bidx]; best_o.state.assign(best_o.nodes.size(), 0ULL);
                for (int b = 0; b < 16; ++b) {
                    gen_continuous_stream(2, 32, orng, ei0, ei1, eir, etg);
                    for (int t = 0; t < 64; ++t) {
                        uint32_t m1 = eval_org_continuous(best_o, &ei0[t], &ei1[t], &eir[t], &etg[t], 1, vc);
                        om += m1;
                        if ((t & 31) == 0) om_b0 += m1; else om_rest += m1;
                    }
                }
                std::cout << "  >>> OOD FROZEN 32-BIT STREAM (1,024 pairs, NO reset): "
                          << std::setprecision(2) << (100.0 * om / (16LL * 64LL * 64LL)) << "% "
                          << "(rst_bit0=" << (100.0 * om_b0 / (16LL * 2LL * 64LL))
                          << "%, carry_bits_1..31=" << (100.0 * om_rest / (16LL * 62LL * 64LL)) << "%)\n";
                print_active_circuit(best_o);
            }
        }
    }
    std::cout << "Elapsed: " << std::setprecision(2) << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() << "s\n\n";
}

int main(int argc, char** argv) {
    int max_sweeps = (argc > 1) ? std::atoi(argv[1]) : 18000;
    std::cout << "========================================================================\n"
              << "KARYON-CORE EXP-1170e: CONTINUOUS BITWISE TISSUE (100% SOLVER PURSUIT & GRAPH TRACER)\n"
              << "Standard: KEP v51.0 | T_ticks=64 | rew_scale=72 | Active Cone Tracer | max_sw=" << max_sweeps << "\n"
              << "========================================================================\n\n";
    run_seed(0, max_sweeps, 0x9E3779B97F4A7C15ULL);
    run_seed(1, max_sweeps, 0x123456789ABCDEF0ULL);
    return 0;
}
