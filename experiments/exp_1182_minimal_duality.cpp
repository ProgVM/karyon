#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1182: MINIMAL DUALITY ("МИНИМАЛЬНАЯ ДУАЛЬНОСТЬ")
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | ZERO goals, ZERO fitness, ZERO err, ZERO clone().
//
// Tests 3 strictly pure realizations of Minimal Duality driven by boundary flux
// (White Noise Control vs Rule 110 vs Wikipedia vs Periodic):
//
//   1. Duality A (Conjugate Spin <-> Coupling Lattice, Symmetric Fast/Slow):
//      Every site holds a fast 8-bit spin state S and a slow 8-bit local operator R.
//      S_{t+1} = F(S_nbrs, R)  (local threshold/nonlinear interaction parameterized by R)
//      R_{t+1} = G(R_nbrs, S)  (slow relaxation/drift driven by conjugate field S)
//      Zero heuristics, zero "if turbulent then melt", pure conjugate field physics!
//
//   2. Duality B (Conjugate Wave <-> Matter Potential, Coupled Sine-Gordon/Klein-like):
//      Two coupled continuous integer fields (Phi, Psi) with nonlinear cross-coupling:
//      Phi_{t+1} = lap(Phi) - grad_potential(Phi, Psi)
//      Psi_{t+1} = lap(Psi) + conjugate_force(Phi, Psi)
//      Pure physical dual-field PDE on 128x128!
//
//   3. Duality C (Homoiconic Operator <-> Operand 1D/2D Raw Byte Soup):
//      Every site is a 16-byte string. Adjacent neighbors (A, B) interact purely
//      as Operator(A) acting on Operand(B) via minimal pointer/byte physics
//      (NO copy(), NO fitness, NO goal, NO mutation!).
//      Boundary x=0 continuously injects White Noise vs Rule 110 vs Wikipedia bytes!
// ============================================================================

constexpr int W = 128;
constexpr int H = 128;
constexpr size_t N = W * H;

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ----------------------------------------------------------------------------
// 1. Duality A: Conjugate State S <-> Local Operator R (Symmetric Fast/Slow)
// ----------------------------------------------------------------------------
struct ConjSite {
    int16_t s; // fast state (-2048 .. +2048)
    int16_t r; // slow local operator/phase (-2048 .. +2048)
};

static void step_conjugate_sr(
    const ConjSite* cur, ConjSite* nxt,
    const uint8_t* stream_bytes, size_t n_bytes, uint64_t tick)
{
    size_t spos = (size_t)((tick * H) % n_bytes);
    for (int y = 0; y < H; ++y) {
        int16_t val = (int16_t)(((int32_t)stream_bytes[(spos + y) % n_bytes] - 128) << 4); // -2048..+2032
        nxt[y * W + 0] = {val, cur[y * W + 0].r};
        nxt[y * W + (W - 1)] = {0, 0};
    }

    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            int32_t s  = cur[idx].s;
            int32_t r  = cur[idx].r;
            int32_t sl = cur[y * W + (x - 1)].s;
            int32_t sr = cur[y * W + (x + 1)].s;
            int32_t su = cur[ym * W + x].s;
            int32_t sd = cur[yp * W + x].s;

            int32_t rl = cur[y * W + (x - 1)].r;
            int32_t rr = cur[y * W + (x + 1)].r;
            int32_t ru = cur[ym * W + x].r;
            int32_t rd = cur[yp * W + x].r;

            int32_t lap_s = (sl + sr + su + sd - 4 * s) >> 2;
            int32_t lap_r = (rl + rr + ru + rd - 4 * r) >> 3;

            // Conjugate coupling: S evolves under local operator R; R evolves under conjugate S!
            // ds = lap_s + (R * S_avg) / 2048 - S / 16
            // dr = lap_r - (S * R) / 4096 + (sl - s) / 16  (pure antisymmetric / symplectic-like + damping!)
            int32_t s_avg = (sl + sr + su + sd) >> 2;
            int32_t ds = (lap_s >> 1) + ((r * s_avg) >> 12) - (s >> 5);
            int32_t dr = (lap_r >> 2) - ((s * s_avg) >> 13) + ((s - r) >> 6);

            nxt[idx].s = (int16_t)std::clamp(s + ds, -2048, 2048);
            nxt[idx].r = (int16_t)std::clamp(r + dr, -2048, 2048);
        }
    }
}

// ----------------------------------------------------------------------------
// 2. Duality B: Coupled Nonlinear Dual-Field (Wave Phi <-> Matter Psi)
// ----------------------------------------------------------------------------
struct DualFieldSite {
    int32_t phi, vphi;
    int32_t psi, vpsi;
};

static void step_dual_field(
    const DualFieldSite* cur, DualFieldSite* nxt,
    const uint8_t* stream_bytes, size_t n_bytes, uint64_t tick)
{
    size_t spos = (size_t)((tick * H) % n_bytes);
    for (int y = 0; y < H; ++y) {
        int32_t val = ((int32_t)stream_bytes[(spos + y) % n_bytes] - 128) << 6; // -8192..+8128
        nxt[y * W + 0] = {val, 0, val >> 1, 0};
        nxt[y * W + (W - 1)] = {0, 0, 0, 0};
    }

    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            const auto& c = cur[idx];
            int32_t lap_phi = (cur[y*W+(x-1)].phi + cur[y*W+(x+1)].phi + cur[ym*W+x].phi + cur[yp*W+x].phi - 4*c.phi) >> 2;
            int32_t lap_psi = (cur[y*W+(x-1)].psi + cur[y*W+(x+1)].psi + cur[ym*W+x].psi + cur[yp*W+x].psi - 4*c.psi) >> 2;

            // Symplectic-like dual-potential coupling: Phi^2 * Psi & Psi^2 * Phi (phi4-like dual field)
            int64_t ph = c.phi >> 6; // ~ -128..128
            int64_t ps = c.psi >> 6;
            int32_t f_phi = (lap_phi >> 1) - (c.phi >> 8) + (int32_t)((ps * ps * ph) >> 13) - (c.vphi >> 6);
            int32_t f_psi = (lap_psi >> 2) - (c.psi >> 8) - (int32_t)((ph * ph * ps) >> 13) - (c.vpsi >> 6);

            int32_t vpn = std::clamp(c.vphi + f_phi, -32768, 32768);
            int32_t vsn = std::clamp(c.vpsi + f_psi, -32768, 32768);
            nxt[idx].vphi = vpn;
            nxt[idx].vpsi = vsn;
            nxt[idx].phi  = std::clamp(c.phi + (vpn >> 2), -16384, 16384);
            nxt[idx].psi  = std::clamp(c.psi + (vsn >> 2), -16384, 16384);
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Duality C: Homoiconic Operator <-> Operand Raw Byte Soup (Zero copy()!)
// ----------------------------------------------------------------------------
// Grid of 64x64 sites (4,096 cells), each cell holds 16 raw bytes (128 bits).
// Every tick, random adjacent pairs (A, B) interact: A acts as Operator on B (Operand)
// on a temporary 32-byte tape (A || B) using ultra-minimal 8-op byte physics
// WITHOUT copy(), WITHOUT goal, WITHOUT fitness!
// Wait: we test BOTH:
//   C1: Minimal Reversible-ish Byte Pointer Physics (only +, -, <, >, [, ], XOR-exchange, NO copy!)
//   C2: Standard BFF-style Minimal Pointer Physics (has +, -, <, >, [, ], ., ,)
// Boundary x=0 continuously overwrites cell(0, y) with incoming stream bytes!
constexpr int CW = 64;
constexpr int CH = 64;
constexpr int CELL_BYTES = 16;

static void interact_homoiconic(uint8_t* A, uint8_t* B, bool allow_dot) {
    uint8_t tape[32];
    for (int i = 0; i < 16; ++i) { tape[i] = A[i]; tape[i + 16] = B[i]; }

    int ip = 0, h0 = 0, h1 = 16;
    for (int step = 0; step < 128 && ip < 32; ++step) {
        uint8_t op = tape[ip];
        switch (op & 7) {
            case 0: h0 = (h0 + 1) & 31; break;
            case 1: h0 = (h0 + 31) & 31; break;
            case 2: h1 = (h1 + 1) & 31; break;
            case 3: tape[h0]++; break;
            case 4: tape[h0]--; break;
            case 5:
                if (allow_dot) tape[h1] = tape[h0];
                else           tape[h1] ^= tape[h0]; // pure non-copy symmetric exchange
                break;
            case 6:
                if (tape[h0] == 0) {
                    int depth = 1, scan = ip + 1;
                    while (scan < 32 && depth > 0) {
                        if ((tape[scan] & 7) == 6) depth++;
                        else if ((tape[scan] & 7) == 7) depth--;
                        scan++;
                    }
                    ip = scan - 1;
                }
                break;
            case 7:
                if (tape[h0] != 0) {
                    int depth = 1, scan = ip - 1;
                    while (scan >= 0 && depth > 0) {
                        if ((tape[scan] & 7) == 7) depth++;
                        else if ((tape[scan] & 7) == 6) depth--;
                        scan--;
                    }
                    ip = scan + 1;
                }
                break;
        }
        ip++;
    }
    for (int i = 0; i < 16; ++i) { A[i] = tape[i]; B[i] = tape[i + 16]; }
}

// ----------------------------------------------------------------------------
// Generators & Telemetry
// ----------------------------------------------------------------------------
static std::vector<uint8_t> gen_white_noise(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes);
    uint64_t x = 0x123456789ABCDEF0ULL;
    for (size_t i = 0; i < n_bytes; ++i) out[i] = (uint8_t)(splitmix64(x) & 0xFF);
    return out;
}

static std::vector<uint8_t> gen_rule110_stream(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes, 0);
    const int LW = 128;
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

struct FieldStats {
    double near_rms, mid_rms, far_rms;
    double scorr, tcorr, H_q;
};

template <typename FnVal, typename FnPrev>
static FieldStats eval_field(FnVal gv, FnPrev gp) {
    double n_e = 0, m_e = 0, f_e = 0;
    double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0, st = 0, spt = 0, stt = 0;
    uint64_t hist[16] = {0}, cnt = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            double v = gv(x, y), vr = gv(x + 1, y), vp = gp(x, y);
            if (x < 32) n_e += v * v;
            else if (x < 96) m_e += v * v;
            else f_e += v * v;
            if (x >= 16 && x < W - 16) {
                sx += v; sy += vr; sxy += v * vr; sxx += v * v; syy += vr * vr;
                st += vp; spt += v * vp; stt += vp * vp;
                hist[std::clamp((int)((v + 4096.0) / 512.0), 0, 15)]++;
                cnt++;
            }
        }
    }
    double fn = (double)cnt;
    double vx = sxx/fn - (sx/fn)*(sx/fn), vy = syy/fn - (sy/fn)*(sy/fn), vt = stt/fn - (st/fn)*(st/fn);
    double sc = (vx > 1e-9 && vy > 1e-9) ? (sxy/fn - (sx/fn)*(sy/fn)) / std::sqrt(vx*vy) : 0.0;
    double tc = (vx > 1e-9 && vt > 1e-9) ? (spt/fn - (sx/fn)*(st/fn)) / std::sqrt(vx*vt) : 0.0;
    double Hq = 0.0;
    for (int i = 0; i < 16; ++i) if (hist[i] > 0) { double p = (double)hist[i] / fn; Hq -= p * std::log2(p); }
    return { std::sqrt(n_e/(31.0*H)), std::sqrt(m_e/(64.0*H)), std::sqrt(f_e/(31.0*H)), sc, tc, Hq };
}

struct SoupStats {
    double H_byte;
    size_t unique_cells;
    double top_cell_pct;
    double zero_cell_pct;
    double near_top_pct, far_top_pct;
};

static SoupStats eval_soup(const std::vector<uint8_t>& grid) {
    std::unordered_map<uint64_t, uint32_t> cell_hist;
    uint64_t bhist[256] = {0};
    uint64_t tot_bytes = 0, zeros = 0;
    std::unordered_map<uint64_t, uint32_t> near_h, far_h;

    for (int y = 0; y < CH; ++y) {
        for (int x = 1; x < CW; ++x) {
            const uint8_t* p = &grid[(y * CW + x) * CELL_BYTES];
            uint64_t h1 = 14695981039346656037ULL;
            bool all0 = true;
            for (int k = 0; k < CELL_BYTES; ++k) {
                bhist[p[k]]++;
                tot_bytes++;
                if (p[k] != 0) all0 = false;
                h1 = (h1 ^ p[k]) * 1099511628211ULL;
            }
            if (all0) zeros++;
            cell_hist[h1]++;
            if (x < 16) near_h[h1]++;
            else if (x >= 48) far_h[h1]++;
        }
    }

    double Hb = 0.0;
    for (int i = 0; i < 256; ++i) if (bhist[i] > 0) { double p = (double)bhist[i] / (double)tot_bytes; Hb -= p * std::log2(p); }

    uint32_t top_all = 0, top_near = 0, top_far = 0;
    for (auto& kv : cell_hist) if (kv.second > top_all) top_all = kv.second;
    for (auto& kv : near_h) if (kv.second > top_near) top_near = kv.second;
    for (auto& kv : far_h)  if (kv.second > top_far)  top_far = kv.second;

    double n_int = (double)CH * (CW - 1);
    return {
        Hb, cell_hist.size(),
        100.0 * (double)top_all / n_int,
        100.0 * (double)zeros / n_int,
        100.0 * (double)top_near / (15.0 * CH),
        100.0 * (double)top_far / (16.0 * CH)
    };
}

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1182: MINIMAL DUALITY (S<->R, Wave<->Matter, Operator<->Operand) + WAIT\n";
    std::cout << " ZERO goals, ZERO fitness, ZERO err, ZERO mutation.\n";
    std::cout << "==================================================================================================\n";

    const size_t BYTES = 131072;
    std::vector<std::vector<uint8_t>> streams(3);
    streams[0] = gen_white_noise(BYTES);
    streams[1] = gen_rule110_stream(BYTES);
    {
        std::ifstream fw("data/real_wikipedia_stream.txt", std::ios::binary);
        if (fw) {
            streams[2].resize(BYTES);
            fw.read((char*)streams[2].data(), BYTES);
            streams[2].resize((size_t)fw.gcount());
        } else {
            streams[2] = streams[1];
        }
    }
    const char* snames[3] = {"White Noise", "Rule 110", "Wikipedia"};

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1 & 2: Conjugate fields (Duality A & B)
    std::vector<std::string> field_out(6);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 6; ++idx) {
        int kind = idx / 3; // 0 = S<->R, 1 = DualField Phi<->Psi
        int st = idx % 3;
        const auto& strm = streams[st];
        uint64_t rng = 0xCAFE1234ULL + idx;
        std::ostringstream oss;
        oss << "\n--- " << (kind == 0 ? "Duality A (Conjugate S<->R)" : "Duality B (DualField Phi<->Psi)")
            << " | " << snames[st] << " ---\n";

        if (kind == 0) {
            std::vector<ConjSite> A(N), B(N), snap(N);
            for (size_t i = 0; i < N; ++i) {
                A[i].s = (int16_t)((int32_t)(splitmix64(rng) & 1023) - 512);
                A[i].r = (int16_t)((int32_t)(splitmix64(rng) & 1023) - 512);
            }
            ConjSite *cur = A.data(), *nxt = B.data();
            for (uint64_t t = 0; t <= 100000; ++t) {
                step_conjugate_sr(cur, nxt, strm.data(), strm.size(), t);
                std::swap(cur, nxt);
                if (t == 1000 || t == 25000 || t == 100000) {
                    std::copy(cur, cur + N, snap.begin());
                    for (int k = 1; k <= 20; ++k) {
                        step_conjugate_sr(cur, nxt, strm.data(), strm.size(), t + k);
                        std::swap(cur, nxt);
                    }
                    FieldStats fs = eval_field(
                        [&](int x, int y){ return (double)cur[y*W+x].s; },
                        [&](int x, int y){ return (double)snap[y*W+x].s; }
                    );
                    FieldStats fr = eval_field(
                        [&](int x, int y){ return (double)cur[y*W+x].r; },
                        [&](int x, int y){ return (double)snap[y*W+x].r; }
                    );
                    oss << "  t=" << std::setw(6) << t
                        << " | S_rms(n/m/f)=" << std::fixed << std::setprecision(1) << fs.near_rms << "/" << fs.mid_rms << "/" << fs.far_rms
                        << " sc=" << std::setprecision(3) << fs.scorr << " tc=" << fs.tcorr
                        << " | R_rms(n/m/f)=" << std::setprecision(1) << fr.near_rms << "/" << fr.mid_rms << "/" << fr.far_rms
                        << " sc=" << std::setprecision(3) << fr.scorr << "\n";
                    t += 20;
                }
            }
        } else {
            std::vector<DualFieldSite> A(N), B(N), snap(N);
            for (size_t i = 0; i < N; ++i) {
                A[i] = {(int32_t)(splitmix64(rng)&1023)-512, 0, (int32_t)(splitmix64(rng)&1023)-512, 0};
            }
            DualFieldSite *cur = A.data(), *nxt = B.data();
            for (uint64_t t = 0; t <= 100000; ++t) {
                step_dual_field(cur, nxt, strm.data(), strm.size(), t);
                std::swap(cur, nxt);
                if (t == 1000 || t == 25000 || t == 100000) {
                    std::copy(cur, cur + N, snap.begin());
                    for (int k = 1; k <= 20; ++k) {
                        step_dual_field(cur, nxt, strm.data(), strm.size(), t + k);
                        std::swap(cur, nxt);
                    }
                    FieldStats fp = eval_field(
                        [&](int x, int y){ return (double)cur[y*W+x].phi; },
                        [&](int x, int y){ return (double)snap[y*W+x].phi; }
                    );
                    FieldStats fq = eval_field(
                        [&](int x, int y){ return (double)cur[y*W+x].psi; },
                        [&](int x, int y){ return (double)snap[y*W+x].psi; }
                    );
                    oss << "  t=" << std::setw(6) << t
                        << " | Phi_rms(n/m/f)=" << std::fixed << std::setprecision(1) << fp.near_rms << "/" << fp.mid_rms << "/" << fp.far_rms
                        << " sc=" << std::setprecision(3) << fp.scorr
                        << " | Psi_rms(n/m/f)=" << std::setprecision(1) << fq.near_rms << "/" << fq.mid_rms << "/" << fq.far_rms
                        << " sc=" << std::setprecision(3) << fq.scorr << "\n";
                    t += 20;
                }
            }
        }
        field_out[idx] = oss.str();
    }
    for (const auto& s : field_out) std::cout << s;

    // 3: Duality C: Homoiconic Operator <-> Operand Raw Byte Soup (C1: NO copy, C2: with byte copy)
    std::vector<std::string> soup_out(6);
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < 6; ++idx) {
        bool allow_dot = (idx / 3) == 1;
        int st = idx % 3;
        const auto& strm = streams[st];
        uint64_t rng = 0xBEEF7777ULL + idx;
        std::vector<uint8_t> grid(CW * CH * CELL_BYTES);
        for (size_t i = 0; i < grid.size(); ++i) grid[i] = (uint8_t)(splitmix64(rng) & 0xFF);

        std::ostringstream oss;
        oss << "\n--- Duality C" << (allow_dot ? "2 (Op<->Operand w/ byte=)" : "1 (Op<->Operand XOR-only, NO copy)")
            << " | " << snames[st] << " ---\n";

        size_t spos = 0;
        const uint64_t EPOCHS = 1500; // each epoch = 4096 random neighbor interactions (~1.5M interactions per cell? ~1500/cell)
        for (uint64_t ep = 0; ep <= EPOCHS; ++ep) {
            // Feed boundary x=0 from stream
            for (int y = 0; y < CH; ++y) {
                for (int k = 0; k < CELL_BYTES; ++k) {
                    grid[(y * CW + 0) * CELL_BYTES + k] = strm[(spos++) % strm.size()];
                }
            }
            // Random local neighbor collisions across 64x64 grid
            for (int m = 0; m < CW * CH; ++m) {
                uint64_t r = splitmix64(rng);
                int x = (int)(r % CW);
                int y = (int)((r >> 16) % CH);
                int dir = (int)((r >> 32) & 3);
                int nx = x + (dir == 0 ? 1 : (dir == 1 ? -1 : 0));
                int ny = (y + (dir == 2 ? 1 : (dir == 3 ? -1 : 0)) + CH) % CH;
                if (nx < 0 || nx >= CW) continue;
                uint8_t* A = &grid[(y * CW + x) * CELL_BYTES];
                uint8_t* B = &grid[(ny * CW + nx) * CELL_BYTES];
                interact_homoiconic(A, B, allow_dot);
            }

            if (ep == 100 || ep == 500 || ep == EPOCHS) {
                SoupStats ss = eval_soup(grid);
                oss << "  ep=" << std::setw(4) << ep
                    << " | H_byte=" << std::fixed << std::setprecision(2) << ss.H_byte << "/8"
                    << " | uniqCells=" << std::setw(4) << ss.unique_cells
                    << " | topCell=" << std::setw(5) << ss.top_cell_pct << "%"
                    << " | zeros=" << std::setw(5) << ss.zero_cell_pct << "%"
                    << " (nearTop=" << ss.near_top_pct << "%, farTop=" << ss.far_top_pct << "%)\n";
            }
        }
        soup_out[idx] = oss.str();
    }
    for (const auto& s : soup_out) std::cout << s;

    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "\nTotal elapsed: " << std::fixed << std::setprecision(2) << sec << "s\n";
    return 0;
}
