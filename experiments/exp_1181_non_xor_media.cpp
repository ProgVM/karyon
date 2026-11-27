#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <omp.h>

// ============================================================================
// KARYON-CORE EXP-1181: NON-XOR DISSIPATIVE / ERODING MULTI-BIT MEDIA + WAIT
// Lead Architect: Bazilevs (ProgVM) | Autonomous Lead AI Cyberneticist
// Standard: KEP v51.0 | ZERO goals, ZERO fitness, ZERO mutation, ZERO cloning.
//
// Tests whether removing XOR cryptographic mixing in favor of genuinely
// physical non-XOR multi-bit dynamics causes Structured Noise (Rule 110 /
// Wikipedia) to carve dramatically different structures vs White Noise:
//
//   1. Medium ViscoWave-A: Damped Nonlinear Elastic Wave Field (int32 fixed-point)
//   2. Medium Riverbed-B: Diffusive Potential + Passive Erosion (water & sand)
//   3. Medium Excitable-C: Integer Excitable Reaction-Diffusion (u, v)
//
// Strict anti-falsification test (Bazilevs):
//   "If it doesn't differ at all or differs only slightly - the guess is wrong."
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
// 1. Medium ViscoWave-A: Nonlinear Damped Elastic Wave Sheet (Q8 fixed-point)
// ----------------------------------------------------------------------------
struct WaveSite {
    int32_t u; // displacement
    int32_t v; // velocity
};

static void step_viscowave(
    const WaveSite* cur, WaveSite* nxt,
    const uint8_t* stream_bytes, size_t n_bytes, uint64_t tick)
{
    // Drive boundary x=0 continuously with centered stream signal (-128..+127 in Q8)
    size_t spos = (size_t)((tick * H) % n_bytes);
    for (int y = 0; y < H; ++y) {
        int32_t val = ((int32_t)stream_bytes[(spos + y) % n_bytes] - 128) << 8;
        nxt[y * W + 0] = {val, 0};
        nxt[y * W + (W - 1)] = {0, 0}; // absorbing edge
    }

    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            int32_t u  = cur[idx].u;
            int32_t v  = cur[idx].v;
            int32_t ul = cur[y * W + (x - 1)].u;
            int32_t ur = cur[y * W + (x + 1)].u;
            int32_t uu = cur[ym * W + x].u;
            int32_t ud = cur[yp * W + x].u;

            // Laplacian coupling + weak nonlinear restoring potential - viscous damping
            int32_t lap = (ul + ur + uu + ud - 4 * u) >> 2;
            // Nonlinear cubic-like soft saturation force: -u/64 - (u^3) scaled safely
            int64_t u_norm = u >> 8;
            int32_t nl = (int32_t)((u_norm * u_norm * u_norm) >> 14);
            int32_t force = (lap >> 1) - (u >> 7) - nl - (v >> 5);
            int32_t vn = std::clamp(v + force, -65536, 65536);
            int32_t un = std::clamp(u + (vn >> 2), -65536, 65536);
            nxt[idx] = {un, vn};
        }
    }
}

// ----------------------------------------------------------------------------
// 2. Medium Riverbed-B: Diffusive Current + Passive Substrate Erosion (Sand/Clay)
// ----------------------------------------------------------------------------
struct RiverSite {
    int32_t u;     // fluid/pressure potential
    uint16_t k_r;  // horizontal conductance (rightward)
    uint16_t k_d;  // vertical conductance (downward)
};

static void step_riverbed(
    const RiverSite* cur, RiverSite* nxt,
    const uint8_t* stream_bytes, size_t n_bytes, uint64_t tick)
{
    size_t spos = (size_t)((tick * H) % n_bytes);
    for (int y = 0; y < H; ++y) {
        // Source pressure at x=0, zero sink at x=W-1
        int32_t src = ((int32_t)stream_bytes[(spos + y) % n_bytes]) << 6; // 0..16320
        nxt[y * W + 0] = {src, 4096, 4096};
        nxt[y * W + (W - 1)] = {0, 4096, 4096};
    }

    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            int32_t u  = cur[idx].u;
            int32_t ul = cur[y * W + (x - 1)].u;
            int32_t ur = cur[y * W + (x + 1)].u;
            int32_t uu = cur[ym * W + x].u;
            int32_t ud = cur[yp * W + x].u;

            uint16_t kr_l = cur[y * W + (x - 1)].k_r;
            uint16_t kr_r = cur[idx].k_r;
            uint16_t kd_u = cur[ym * W + x].k_d;
            uint16_t kd_d = cur[idx].k_d;

            // Physical current proportional to conductance * gradient (Q14 fixed-point)
            int64_t f_l = ((int64_t)kr_l * (ul - u)) >> 14;
            int64_t f_r = ((int64_t)kr_r * (ur - u)) >> 14;
            int64_t f_u = ((int64_t)kd_u * (uu - u)) >> 14;
            int64_t f_d = ((int64_t)kd_d * (ud - u)) >> 14;

            int32_t un = (int32_t)std::clamp<int64_t>(u + ((f_l + f_r + f_u + f_d) >> 2), -65536, 65536);

            // Passive erosion: strong local flow carves higher conductance; quiet heals back toward 1024
            int64_t abs_fr = std::abs(f_r);
            int64_t abs_fd = std::abs(f_d);
            int32_t new_kr = (int32_t)kr_r + (int32_t)((abs_fr >> 3) - (((int32_t)kr_r - 1024) >> 7));
            int32_t new_kd = (int32_t)kd_d + (int32_t)((abs_fd >> 3) - (((int32_t)kd_d - 1024) >> 7));

            nxt[idx].u   = un;
            nxt[idx].k_r = (uint16_t)std::clamp(new_kr, 64, 16384);
            nxt[idx].k_d = (uint16_t)std::clamp(new_kd, 64, 16384);
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Medium Excitable-C: Nonlinear Reaction-Diffusion (u: activator, v: inhibitor)
// ----------------------------------------------------------------------------
struct ExciteSite {
    int32_t u; // 0 .. 4096
    int32_t v; // 0 .. 4096
};

static void step_excitable(
    const ExciteSite* cur, ExciteSite* nxt,
    const uint8_t* stream_bytes, size_t n_bytes, uint64_t tick)
{
    size_t spos = (size_t)((tick * H) % n_bytes);
    for (int y = 0; y < H; ++y) {
        int32_t inj = ((int32_t)stream_bytes[(spos + y) % n_bytes]) << 4; // 0..4080
        nxt[y * W + 0] = {inj, 0};
        nxt[y * W + (W - 1)] = {0, 0};
    }

    for (int y = 0; y < H; ++y) {
        int ym = (y == 0) ? (H - 1) : (y - 1);
        int yp = (y == H - 1) ? 0 : (y + 1);
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            int32_t u = cur[idx].u;
            int32_t v = cur[idx].v;
            int32_t lap_u = (cur[y * W + (x - 1)].u + cur[y * W + (x + 1)].u +
                             cur[ym * W + x].u + cur[yp * W + x].u - 4 * u) >> 2;
            int32_t lap_v = (cur[y * W + (x - 1)].v + cur[y * W + (x + 1)].v +
                             cur[ym * W + x].v + cur[yp * W + x].v - 4 * v) >> 3;

            // FitzHugh-Nagumo-like integer dynamics: self-amplifying threshold + slow recovery
            int64_t unorm = u; // 0..4096
            int64_t react = (unorm * (4096 - unorm) * (unorm - 1024)) >> 23;
            int32_t du = (lap_u >> 1) + (int32_t)react - (v >> 3);
            int32_t dv = lap_v + ((u - 2 * v) >> 6);

            nxt[idx].u = std::clamp(u + du, 0, 4096);
            nxt[idx].v = std::clamp(v + dv, 0, 4096);
        }
    }
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

// Periodic wave control to test resonance as well
static std::vector<uint8_t> gen_periodic_stream(size_t n_bytes) {
    std::vector<uint8_t> out(n_bytes);
    for (size_t i = 0; i < n_bytes; ++i) {
        out[i] = (uint8_t)(128 + 120.0 * std::sin(2.0 * 3.14159265 * (double)(i % 32) / 32.0));
    }
    return out;
}

struct PhysStats {
    double near_rms, mid_rms, far_rms; // Energy/amplitude penetration across space
    double spat_corr_x;                // Adjacent spatial correlation (structure vs noise)
    double temporal_corr;              // Autocorrelation over 20 ticks
    double aux_stat;                   // Extra state (e.g. Riverbed conductance variance / std)
    double H_quant;                    // Spatial quantized state entropy (4-bit bins, max 4.0)
};

template <typename FnGetVal, typename FnGetPrev, typename FnGetAux>
static PhysStats eval_field(
    FnGetVal get_val, FnGetPrev get_prev, FnGetAux get_aux)
{
    double n_e = 0, m_e = 0, f_e = 0;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0, sum_yy = 0;
    double sum_t = 0, sum_pt = 0, sum_tt = 0;
    double aux_sum = 0, aux_sq = 0;
    uint64_t hist[16] = {0};
    uint64_t cnt = 0;

    for (int y = 0; y < H; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            double v  = get_val(x, y);
            double vr = get_val(x + 1, y);
            double vp = get_prev(x, y);
            double a  = get_aux(x, y);

            if (x < 32) n_e += v * v;
            else if (x < 96) m_e += v * v;
            else f_e += v * v;

            if (x >= 16 && x < W - 16) {
                sum_x += v; sum_y += vr;
                sum_xy += v * vr; sum_xx += v * v; sum_yy += vr * vr;

                sum_t += vp; sum_pt += v * vp; sum_tt += vp * vp;
                aux_sum += a; aux_sq += a * a;

                int bin = std::clamp((int)((v + 4096.0) / 512.0), 0, 15);
                hist[bin]++;
                cnt++;
            }
        }
    }

    double fn = (double)cnt;
    double cov_xy = sum_xy / fn - (sum_x / fn) * (sum_y / fn);
    double var_x  = sum_xx / fn - (sum_x / fn) * (sum_x / fn);
    double var_y  = sum_yy / fn - (sum_y / fn) * (sum_y / fn);
    double scorr  = (var_x > 1e-9 && var_y > 1e-9) ? cov_xy / std::sqrt(var_x * var_y) : 0.0;

    double cov_t = sum_pt / fn - (sum_x / fn) * (sum_t / fn);
    double var_t = sum_tt / fn - (sum_t / fn) * (sum_t / fn);
    double tcorr = (var_x > 1e-9 && var_t > 1e-9) ? cov_t / std::sqrt(var_x * var_t) : 0.0;

    double aux_var = std::max(0.0, aux_sq / fn - (aux_sum / fn) * (aux_sum / fn));

    double Hq = 0.0;
    for (int i = 0; i < 16; ++i) {
        if (hist[i] > 0) {
            double p = (double)hist[i] / fn;
            Hq -= p * std::log2(p);
        }
    }

    return {
        std::sqrt(n_e / (31.0 * H)),
        std::sqrt(m_e / (64.0 * H)),
        std::sqrt(f_e / (31.0 * H)),
        scorr, tcorr, std::sqrt(aux_var), Hq
    };
}

struct TaskDef {
    int medium_type; // 0=Wave, 1=River, 2=Excitable
    int stream_type; // 0=White, 1=Rule110, 2=Wiki, 3=Periodic
    std::string label;
};

int main() {
    std::cout << "==================================================================================================\n";
    std::cout << " KARYON-CORE EXP-1181: NON-XOR DISSIPATIVE / ERODING MULTI-BIT MEDIA + WAIT (128x128, 100k TICKS)\n";
    std::cout << " Strict anti-falsification test: ZERO goals, ZERO fitness, ZERO mutation, ZERO cloning.\n";
    std::cout << "==================================================================================================\n";

    const size_t BYTES = 131072;
    std::vector<std::vector<uint8_t>> streams(4);
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
    streams[3] = gen_periodic_stream(BYTES);

    std::vector<TaskDef> tasks = {
        {0, 0, "ViscoWave-A | Control: White Noise"},
        {0, 1, "ViscoWave-A | Structured: Rule 110"},
        {0, 2, "ViscoWave-A | Structured: Wikipedia"},
        {0, 3, "ViscoWave-A | Structured: Sine Wave"},
        {1, 0, "Riverbed-B  | Control: White Noise"},
        {1, 1, "Riverbed-B  | Structured: Rule 110"},
        {1, 2, "Riverbed-B  | Structured: Wikipedia"},
        {1, 3, "Riverbed-B  | Structured: Sine Wave"},
        {2, 0, "Excitable-C | Control: White Noise"},
        {2, 1, "Excitable-C | Structured: Rule 110"},
        {2, 2, "Excitable-C | Structured: Wikipedia"},
        {2, 3, "Excitable-C | Structured: Sine Wave"}
    };

    std::vector<std::string> outputs(tasks.size());
    const uint64_t TICKS = 100000;
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int ti = 0; ti < (int)tasks.size(); ++ti) {
        const auto& tk = tasks[ti];
        const auto& strm = streams[tk.stream_type];
        uint64_t rng = 0x13572468ACE01234ULL + ti;

        std::ostringstream oss;
        oss << "\n--- " << tk.label << " ---\n";

        if (tk.medium_type == 0) {
            std::vector<WaveSite> A(N), B(N), snap(N);
            for (size_t i = 0; i < N; ++i) A[i] = {(int32_t)(splitmix64(rng) & 511) - 256, 0};
            WaveSite *cur = A.data(), *nxt = B.data();
            for (uint64_t t = 0; t <= TICKS; ++t) {
                step_viscowave(cur, nxt, strm.data(), strm.size(), t);
                std::swap(cur, nxt);
                if (t == 1000 || t == 25000 || t == TICKS) {
                    std::copy(cur, cur + N, snap.begin());
                    for (int k = 1; k <= 20; ++k) {
                        step_viscowave(cur, nxt, strm.data(), strm.size(), t + k);
                        std::swap(cur, nxt);
                    }
                    PhysStats ps = eval_field(
                        [&](int x, int y) { return (double)cur[y * W + x].u; },
                        [&](int x, int y) { return (double)snap[y * W + x].u; },
                        [&](int x, int y) { return (double)cur[y * W + x].v; }
                    );
                    oss << "  t=" << std::setw(6) << t
                        << " | rms(near/mid/far)=" << std::fixed << std::setprecision(1)
                        << ps.near_rms << "/" << ps.mid_rms << "/" << ps.far_rms
                        << " | scorr=" << std::setprecision(3) << ps.spat_corr_x
                        << " | tcorr=" << ps.temporal_corr
                        << " | std_v=" << std::setprecision(1) << ps.aux_stat
                        << " | H_q=" << std::setprecision(2) << ps.H_quant << "\n";
                    t += 20;
                }
            }
        } else if (tk.medium_type == 1) {
            std::vector<RiverSite> A(N), B(N), snap(N);
            for (size_t i = 0; i < N; ++i) A[i] = {(int32_t)(splitmix64(rng) & 1023), 1024, 1024};
            RiverSite *cur = A.data(), *nxt = B.data();
            for (uint64_t t = 0; t <= TICKS; ++t) {
                step_riverbed(cur, nxt, strm.data(), strm.size(), t);
                std::swap(cur, nxt);
                if (t == 1000 || t == 25000 || t == TICKS) {
                    std::copy(cur, cur + N, snap.begin());
                    for (int k = 1; k <= 20; ++k) {
                        step_riverbed(cur, nxt, strm.data(), strm.size(), t + k);
                        std::swap(cur, nxt);
                    }
                    PhysStats ps = eval_field(
                        [&](int x, int y) { return (double)cur[y * W + x].u; },
                        [&](int x, int y) { return (double)snap[y * W + x].u; },
                        [&](int x, int y) { return (double)cur[y * W + x].k_r; }
                    );
                    oss << "  t=" << std::setw(6) << t
                        << " | rms(near/mid/far)=" << std::fixed << std::setprecision(1)
                        << ps.near_rms << "/" << ps.mid_rms << "/" << ps.far_rms
                        << " | scorr=" << std::setprecision(3) << ps.spat_corr_x
                        << " | tcorr=" << ps.temporal_corr
                        << " | std_kr=" << std::setprecision(1) << ps.aux_stat
                        << " | H_q=" << std::setprecision(2) << ps.H_quant << "\n";
                    t += 20;
                }
            }
        } else {
            std::vector<ExciteSite> A(N), B(N), snap(N);
            for (size_t i = 0; i < N; ++i) A[i] = {(int32_t)(splitmix64(rng) & 2047), 0};
            ExciteSite *cur = A.data(), *nxt = B.data();
            for (uint64_t t = 0; t <= TICKS; ++t) {
                step_excitable(cur, nxt, strm.data(), strm.size(), t);
                std::swap(cur, nxt);
                if (t == 1000 || t == 25000 || t == TICKS) {
                    std::copy(cur, cur + N, snap.begin());
                    for (int k = 1; k <= 20; ++k) {
                        step_excitable(cur, nxt, strm.data(), strm.size(), t + k);
                        std::swap(cur, nxt);
                    }
                    PhysStats ps = eval_field(
                        [&](int x, int y) { return (double)cur[y * W + x].u; },
                        [&](int x, int y) { return (double)snap[y * W + x].u; },
                        [&](int x, int y) { return (double)cur[y * W + x].v; }
                    );
                    oss << "  t=" << std::setw(6) << t
                        << " | rms(near/mid/far)=" << std::fixed << std::setprecision(1)
                        << ps.near_rms << "/" << ps.mid_rms << "/" << ps.far_rms
                        << " | scorr=" << std::setprecision(3) << ps.spat_corr_x
                        << " | tcorr=" << ps.temporal_corr
                        << " | std_v=" << std::setprecision(1) << ps.aux_stat
                        << " | H_q=" << std::setprecision(2) << ps.H_quant << "\n";
                    t += 20;
                }
            }
        }
        outputs[ti] = oss.str();
    }

    for (const auto& s : outputs) std::cout << s;
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "\nTotal elapsed: " << std::fixed << std::setprecision(2) << sec << "s\n";
    return 0;
}
