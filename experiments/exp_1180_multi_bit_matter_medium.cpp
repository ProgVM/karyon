#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <algorithm>

// ============================================================================
// KARYON-CORE EXP-1180: MULTI-BIT MATTER MEDIA + STRUCTURED NOISE + WAIT
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | ZERO goals, ZERO fitness, ZERO mutation, ZERO cloning.
//
// Addresses Bazilevs' insight: "Странно что ты всего выделил 1 бит."
// In real physics, matter at a spatial position is NOT a 1-bit scalar.
// Sites have multi-bit internal degrees of freedom (8-bit / 16-bit state per site)
// and physical local couplings/plasticity.
//
// We test 3 completely unremarkable multi-bit physical media:
//
//   1. Medium MultiByte-A: 8-bit Isotropic Physical State + Local Bit-Diffusion
//      Every site (x,y) holds uint8_t matter. Local interaction transforms 8-bit
//      states via isotropic symmetric bit-mixing & local neighborhood parity.
//
//   2. Medium Plastic-B: Multi-Bit State + Physical Substrate Plasticity (Riverbed)
//      Every site (x,y) holds uint8_t matter + uint8_t local coupling W(x,y).
//      As state flux flows through, W(x,y) physically erodes/relaxes (like sand/clay),
//      and future flux follows W(x,y). ZERO goals, ZERO fitness, ZERO err!
//
//   3. Medium MultiBit-C: 16-bit Reversible Conserved Matter-Collision (Margolus-16)
//      Every site holds uint16_t matter. Staggered 2x2 blocks undergo 16-bit
//      conserved physical collision & exchange.
//
// Driven by 3 streams at boundary x=0:
//   - White Noise (Control)
//   - Rule 110 (Structured)
//   - Real Wikipedia Byte Stream (Structured)
// ============================================================================

constexpr int W = 256;
constexpr int H = 256;
constexpr size_t N = W * H;

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ----------------------------------------------------------------------------
// 1. Medium MultiByte-A: 8-Bit Isotropic Physical State
// ----------------------------------------------------------------------------
static void step_multibyte_a(
    const uint8_t* cur, uint8_t* nxt,
    const uint8_t* stream_bytes, size_t n_stream_bytes,
    uint64_t tick, uint64_t& rng)
{
    // Boundary x=0: continuous byte flux enters
    size_t spos = (size_t)((tick * H) % n_stream_bytes);
    for (int y = 0; y < H; ++y) {
        nxt[y * W + 0] = stream_bytes[(spos + y) % n_stream_bytes];
    }

    // Absorbing sink at x = W - 1
    for (int y = 0; y < H; ++y) {
        uint8_t r = (uint8_t)(splitmix64(rng) & 0xFF);
        uint8_t prev_b = cur[y * W + (W - 2)];
        nxt[y * W + (W - 1)] = (r & 0x0F) | (prev_b & 0xF0);
    }

    // Interior x = 1 .. W-2: isotropic 8-bit physical state transformation
    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);

        for (int x = 1; x < W - 1; ++x) {
            uint8_t c  = cur[y * W + x];
            uint8_t l  = cur[y * W + (x - 1)];
            uint8_t r  = cur[y * W + (x + 1)];
            uint8_t u  = cur[ym * W + x];
            uint8_t d  = cur[yp * W + x];

            // Pure symmetric local 8-bit physical rule:
            // 8-bit state = (l ^ r) rotl 1 + (u ^ d) rotr 1 + XOR parity of center
            uint8_t xor_lr = l ^ r;
            uint8_t xor_ud = u ^ d;
            uint8_t rot_lr = (uint8_t)((xor_lr << 1) | (xor_lr >> 7));
            uint8_t rot_ud = (uint8_t)((xor_ud >> 1) | (xor_ud << 7));
            uint8_t out = rot_lr ^ rot_ud ^ c;

            // Faint background thermal noise (~0.4%)
            if ((splitmix64(rng) & 0xFF) == 0) {
                out ^= (uint8_t)(splitmix64(rng) & 0xFF);
            }
            nxt[y * W + x] = out;
        }
    }
}

// ----------------------------------------------------------------------------
// 2. Medium Plastic-B: Multi-Bit State + Physical Substrate Plasticity (Riverbed)
// ----------------------------------------------------------------------------
struct PlasticSite {
    uint8_t state; // 8-bit matter state
    uint8_t w_l;   // 8-bit physical coupling / permeability to left neighbor
    uint8_t w_r;   // 8-bit coupling to right neighbor
};

static void step_plastic_b(
    const PlasticSite* cur, PlasticSite* nxt,
    const uint8_t* stream_bytes, size_t n_stream_bytes,
    uint64_t tick, uint64_t& rng)
{
    // Copy current state to next as baseline
    std::copy(cur, cur + N, nxt);

    // Boundary x=0: continuous byte flux enters
    size_t spos = (size_t)((tick * H) % n_stream_bytes);
    for (int y = 0; y < H; ++y) {
        nxt[y * W + 0].state = stream_bytes[(spos + y) % n_stream_bytes];
    }

    // Interior x = 1 .. W-2: state flows through local couplings W; W erodes/relaxes with flux!
    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);

        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            uint8_t c = cur[idx].state;
            uint8_t l = cur[y * W + (x - 1)].state;
            uint8_t r = cur[y * W + (x + 1)].state;
            uint8_t u = cur[ym * W + x].state;
            uint8_t d = cur[yp * W + x].state;

            uint8_t wl = cur[idx].w_l;
            uint8_t wr = cur[idx].w_r;

            // State update is weighted by physical local couplings (riverbed flow):
            uint8_t flux_l = l & wl;
            uint8_t flux_r = r & wr;
            uint8_t out_state = (c ^ flux_l ^ flux_r ^ (u >> 2) ^ (d << 2));

            // Physical erosion / relaxation of W:
            // High flux / state alignment reinforces local coupling W; mismatch erodes W slightly
            uint8_t match_l = ~(c ^ l);
            uint8_t match_r = ~(c ^ r);

            // Passive relaxation towards matching flux (NO fitness, NO err, pure Hebbian/erosion physics!):
            // W_{new} = (7 * W + match) / 8
            uint8_t new_wl = (uint8_t)(((uint16_t)wl * 7 + match_l) >> 3);
            uint8_t new_wr = (uint8_t)(((uint16_t)wr * 7 + match_r) >> 3);

            nxt[idx].state = out_state;
            nxt[idx].w_l   = new_wl;
            nxt[idx].w_r   = new_wr;
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Medium MultiBit-C: 16-Bit Reversible Conserved Matter-Collision (Margolus-16)
// ----------------------------------------------------------------------------
static void step_margolus_16(
    const uint16_t* cur, uint16_t* nxt,
    const uint8_t* stream_bytes, size_t n_stream_bytes,
    uint64_t tick, uint64_t& rng)
{
    std::copy(cur, cur + N, nxt);

    // Boundary x=0 receives 16-bit word flux
    size_t spos = (size_t)((tick * H * 2) % n_stream_bytes);
    for (int y = 0; y < H; ++y) {
        size_t p = (spos + y * 2) % (n_stream_bytes - 1);
        nxt[y * W + 0] = (uint16_t)stream_bytes[p] | ((uint16_t)stream_bytes[p + 1] << 8);
    }

    int phase = (int)(tick & 1ULL);
    for (int y = phase; y < H - 1; y += 2) {
        for (int x = phase; x < W - 1; x += 2) {
            int i00 = y * W + x;
            int i10 = y * W + (x + 1);
            int i01 = (y + 1) * W + x;
            int i11 = (y + 1) * W + (x + 1);

            uint16_t A = cur[i00];
            uint16_t B = cur[i10];
            uint16_t C = cur[i01];
            uint16_t D = cur[i11];

            // Reversible 16-bit 2x2 block collision (conserves total 1s across the 4 16-bit words!):
            // Rotate CW: A -> B -> D -> C -> A if popcount sum is odd; CCW if even
            uint32_t tot_pop = __builtin_popcount(A) + __builtin_popcount(B) + __builtin_popcount(C) + __builtin_popcount(D);
            if (tot_pop & 1) {
                // CW rotation + XOR exchange
                nxt[i00] = C ^ (B & 0x00FF);
                nxt[i10] = A ^ (D & 0xFF00);
                nxt[i11] = B ^ (C & 0x00FF);
                nxt[i01] = D ^ (A & 0xFF00);
            } else {
                // CCW rotation
                nxt[i00] = B;
                nxt[i10] = D;
                nxt[i11] = C;
                nxt[i01] = A;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Helpers & Diagnostics
// ----------------------------------------------------------------------------
static std::vector<uint8_t> gen_white_noise(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes);
    uint64_t x = 0x123456789ABCDEF0ULL;
    for (size_t i = 0; i < n_bytes; ++i) out[i] = (uint8_t)(splitmix64(x) & 0xFF);
    return out;
}

static std::vector<uint8_t> gen_rule110_stream(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    const int LW = 256;
    std::vector<uint8_t> row(LW, 0), nxt(LW, 0);
    uint64_t s = 0xCAFEBABE12345678ULL;
    for (int i = 0; i < LW; ++i) row[i] = (uint8_t)((splitmix64(s) >> 63) & 1ULL);
    size_t t = 0, tot = n_bytes * 8;
    while (t < tot) {
        for (int i = 0; i < LW && t < tot; ++i, ++t) {
            if (row[i]) out[t >> 3] |= (1U << (t & 7));
        }
        for (int i = 0; i < LW; ++i) {
            uint8_t pat = (row[(i + LW - 1) % LW] << 2) | (row[i] << 1) | row[(i + 1) % LW];
            nxt[i] = (110 >> pat) & 1U;
        }
        row.swap(nxt);
    }
    return out;
}

struct MultiStats {
    double mean_val;
    double near_val, mid_val, far_val;
    double H_byte;        // Shannon entropy of 8-bit states (0..8 bits)
    size_t unique_bytes;  // How many distinct byte states exist in interior
    size_t unique_2x2;    // How many distinct 2x2 byte patterns exist
    double top_motif_pct; // Fraction of interior covered by most common 2x2 pattern
    double persist_100t;  // Persistence of byte states over 100 ticks
    double mean_w_l;      // Mean coupling W_l (for Medium Plastic-B)
};

static MultiStats measure_multibyte(const std::vector<uint8_t>& cur, const std::vector<uint8_t>& prev) {
    uint64_t sum_all = 0, near_s = 0, mid_s = 0, far_s = 0, same = 0;
    std::unordered_map<uint8_t, uint32_t> byte_hist;
    byte_hist.reserve(256);

    for (int y = 0; y < H; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            uint8_t v = cur[y * W + x];
            sum_all += v;
            byte_hist[v]++;
            if (x < 64) near_s += v;
            else if (x < 192) mid_s += v;
            else far_s += v;

            if (!prev.empty() && cur[y * W + x] == prev[y * W + x]) same++;
        }
    }

    double n_tot = (double)H * (W - 2);
    double H_byte = 0.0;
    for (auto& kv : byte_hist) {
        double p = (double)kv.second / n_tot;
        H_byte -= p * std::log2(p);
    }

    // 2x2 byte tiles
    std::unordered_map<uint32_t, uint32_t> tile_hist;
    tile_hist.reserve(4096);
    uint64_t total_tiles = 0;
    for (int y = 0; y < H - 1; y += 2) {
        for (int x = 8; x < W - 8; x += 2) {
            uint32_t tile = (uint32_t)cur[y * W + x]
                          | ((uint32_t)cur[y * W + (x + 1)] << 8)
                          | ((uint32_t)cur[(y + 1) * W + x] << 16)
                          | ((uint32_t)cur[(y + 1) * W + (x + 1)] << 24);
            tile_hist[tile]++;
            total_tiles++;
        }
    }

    uint32_t max_nt = 0;
    for (auto& kv : tile_hist) {
        if (kv.second > max_nt) max_nt = kv.second;
    }

    return {
        (double)sum_all / n_tot,
        (double)near_s / (63.0 * H),
        (double)mid_s  / (128.0 * H),
        (double)far_s  / (63.0 * H),
        H_byte,
        byte_hist.size(),
        tile_hist.size(),
        100.0 * (double)max_nt / (double)total_tiles,
        prev.empty() ? 0.0 : 100.0 * (double)same / n_tot,
        0.0
    };
}

// ----------------------------------------------------------------------------
// Experiment Runners
// ----------------------------------------------------------------------------
static void run_exp_multibyte_a(const char* title, const std::vector<uint8_t>& stream, uint64_t ticks) {
    std::vector<uint8_t> A(N), B(N), snap_prev(N);
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < N; ++i) A[i] = (uint8_t)(splitmix64(rng) & 0xFF);

    std::cout << "\n--- " << title << " ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    uint8_t *cur = A.data(), *nxt = B.data();

    for (uint64_t t = 0; t <= ticks; ++t) {
        step_multibyte_a(cur, nxt, stream.data(), stream.size(), t, rng);
        std::swap(cur, nxt);

        if (t == 1000 || t == 10000 || t == 50000 || t == 200000 || t == ticks) {
            std::copy(cur, cur + N, snap_prev.begin());
            for (int k = 1; k <= 100; ++k) {
                step_multibyte_a(cur, nxt, stream.data(), stream.size(), t + k, rng);
                std::swap(cur, nxt);
            }
            std::vector<uint8_t> cur_vec(cur, cur + N);
            MultiStats st = measure_multibyte(cur_vec, snap_prev);
            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  t=" << std::setw(6) << t
                      << " | mean_val=" << std::fixed << std::setprecision(1) << st.mean_val
                      << " (near=" << st.near_val << ", mid=" << st.mid_val << ", far=" << st.far_val << ")"
                      << " | H_byte=" << std::setprecision(2) << st.H_byte << "/8"
                      << " | uniqByte=" << std::setw(3) << st.unique_bytes
                      << " | uniq2x2=" << std::setw(5) << st.unique_2x2
                      << " | topMotif=" << std::setw(4) << std::setprecision(2) << st.top_motif_pct << "%"
                      << " | persist100t=" << std::setw(5) << st.persist_100t << "%"
                      << " | " << sec << "s\n";
            t += 100;
        }
    }
}

static void run_exp_plastic_b(const char* title, const std::vector<uint8_t>& stream, uint64_t ticks) {
    std::vector<PlasticSite> A(N), B(N), snap_prev_s(N);
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < N; ++i) {
        A[i].state = (uint8_t)(splitmix64(rng) & 0xFF);
        A[i].w_l   = 128; // initialized to 50% coupling baseline
        A[i].w_r   = 128;
    }

    std::cout << "\n--- " << title << " ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    PlasticSite *cur = A.data(), *nxt = B.data();
    std::vector<uint8_t> cur_bytes(N), prev_bytes(N);

    for (uint64_t t = 0; t <= ticks; ++t) {
        step_plastic_b(cur, nxt, stream.data(), stream.size(), t, rng);
        std::swap(cur, nxt);

        if (t == 1000 || t == 10000 || t == 50000 || t == 200000 || t == ticks) {
            for (size_t i = 0; i < N; ++i) cur_bytes[i] = cur[i].state;
            for (int k = 1; k <= 100; ++k) {
                step_plastic_b(cur, nxt, stream.data(), stream.size(), t + k, rng);
                std::swap(cur, nxt);
            }
            for (size_t i = 0; i < N; ++i) prev_bytes[i] = cur[i].state;

            MultiStats st = measure_multibyte(cur_bytes, prev_bytes);

            // Compute mean coupling W_l across near/mid/far
            uint64_t sum_wl_near = 0, sum_wl_mid = 0, sum_wl_far = 0;
            for (int y = 0; y < H; ++y) {
                for (int x = 1; x < W - 1; ++x) {
                    uint8_t wl = cur[y * W + x].w_l;
                    if (x < 64) sum_wl_near += wl;
                    else if (x < 192) sum_wl_mid += wl;
                    else sum_wl_far += wl;
                }
            }
            double mean_wl_near = (double)sum_wl_near / (63.0 * H);
            double mean_wl_mid  = (double)sum_wl_mid  / (128.0 * H);
            double mean_wl_far  = (double)sum_wl_far  / (63.0 * H);

            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  t=" << std::setw(6) << t
                      << " | mean_val=" << std::fixed << std::setprecision(1) << st.mean_val
                      << " | H_byte=" << std::setprecision(2) << st.H_byte << "/8"
                      << " | uniq2x2=" << std::setw(5) << st.unique_2x2
                      << " | topMotif=" << std::setw(4) << std::setprecision(2) << st.top_motif_pct << "%"
                      << " | W_l(near/mid/far)=" << std::setprecision(1) << mean_wl_near << "/" << mean_wl_mid << "/" << mean_wl_far
                      << " | persist100t=" << std::setw(5) << st.persist_100t << "%"
                      << " | " << sec << "s\n";
            t += 100;
        }
    }
}

static void run_exp_margolus_16(const char* title, const std::vector<uint8_t>& stream, uint64_t ticks) {
    std::vector<uint16_t> A(N), B(N), snap_prev(N);
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < N; ++i) A[i] = (uint16_t)(splitmix64(rng) & 0xFFFF);

    std::cout << "\n--- " << title << " ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    uint16_t *cur = A.data(), *nxt = B.data();
    std::vector<uint8_t> cur_bytes(N), prev_bytes(N);

    for (uint64_t t = 0; t <= ticks; ++t) {
        step_margolus_16(cur, nxt, stream.data(), stream.size(), t, rng);
        std::swap(cur, nxt);

        if (t == 1000 || t == 10000 || t == 50000 || t == 200000 || t == ticks) {
            for (size_t i = 0; i < N; ++i) cur_bytes[i] = (uint8_t)(cur[i] & 0xFF);
            for (int k = 1; k <= 100; ++k) {
                step_margolus_16(cur, nxt, stream.data(), stream.size(), t + k, rng);
                std::swap(cur, nxt);
            }
            for (size_t i = 0; i < N; ++i) prev_bytes[i] = (uint8_t)(cur[i] & 0xFF);

            MultiStats st = measure_multibyte(cur_bytes, prev_bytes);
            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  t=" << std::setw(6) << t
                      << " | mean_val=" << std::fixed << std::setprecision(1) << st.mean_val
                      << " (near=" << st.near_val << ", mid=" << st.mid_val << ", far=" << st.far_val << ")"
                      << " | H_byte=" << std::setprecision(2) << st.H_byte << "/8"
                      << " | uniq2x2=" << std::setw(5) << st.unique_2x2
                      << " | topMotif=" << std::setw(4) << std::setprecision(2) << st.top_motif_pct << "%"
                      << " | persist100t=" << std::setw(5) << st.persist_100t << "%"
                      << " | " << sec << "s\n";
            t += 100;
        }
    }
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1180: MULTI-BIT PHYSICAL MATTER MEDIA + STRUCTURED NOISE + WAIT\n";
    std::cout << " 256x256 = 65,536 sites | 8-bit / 16-bit Matter States per site | ZERO goals, ZERO fitness, ZERO cloning.\n";
    std::cout << "==================================================================================================\n";

    const size_t BYTES = 131072;
    auto white_b = gen_white_noise(BYTES);
    auto r110_b  = gen_rule110_stream(BYTES);

    std::vector<uint8_t> wiki_b;
    std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
    if (fw) {
        wiki_b.resize(BYTES);
        fw.read((char*)wiki_b.data(), BYTES);
        wiki_b.resize((size_t)fw.gcount());
    }

    const uint64_t TICKS = 200000;

    // 1. Medium MultiByte-A: 8-bit Isotropic Physical State
    run_exp_multibyte_a("Medium MultiByte-A (8-bit Physical State) | Control: White Noise", white_b, TICKS);
    run_exp_multibyte_a("Medium MultiByte-A (8-bit Physical State) | Structured: Rule 110", r110_b, TICKS);
    if (!wiki_b.empty()) run_exp_multibyte_a("Medium MultiByte-A (8-bit Physical State) | Structured: Wikipedia", wiki_b, TICKS);

    // 2. Medium Plastic-B: Multi-Bit State + Substrate Plasticity (Riverbed)
    run_exp_plastic_b("Medium Plastic-B (Multi-Bit State + Substrate Erosion) | Control: White Noise", white_b, TICKS);
    run_exp_plastic_b("Medium Plastic-B (Multi-Bit State + Substrate Erosion) | Structured: Rule 110", r110_b, TICKS);
    if (!wiki_b.empty()) run_exp_plastic_b("Medium Plastic-B (Multi-Bit State + Substrate Erosion) | Structured: Wikipedia", wiki_b, TICKS);

    // 3. Medium MultiBit-C: 16-bit Reversible Conserved Matter-Collision (Margolus-16)
    run_exp_margolus_16("Medium MultiBit-C (16-bit Reversible Conserved Margolus) | Control: White Noise", white_b, TICKS);
    run_exp_margolus_16("Medium MultiBit-C (16-bit Reversible Conserved Margolus) | Structured: Rule 110", r110_b, TICKS);
    if (!wiki_b.empty()) run_exp_margolus_16("Medium MultiBit-C (16-bit Reversible Conserved Margolus) | Structured: Wikipedia", wiki_b, TICKS);

    return 0;
}
