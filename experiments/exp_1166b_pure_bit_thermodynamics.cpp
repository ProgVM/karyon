#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <set>
#include <map>
#include <string>

// ============================================================================
// KARYON-CORE EXP-1166b: PURE BIT THERMODYNAMICS V=2 (STAGE 2 - AUDIT FIXES)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// Pure 1-bit code=data thermodynamics substrate under open environmental coupling.
// 10-operator bitwise-complement symmetric basis (6 NOPs = 37.5% neutral space).
// Audit fixes:
// 1. Fixed 4-bit opcode step (pc += 4 for NOPs as well) so '?' never breaks phase.
// 2. Unmatched ']' (sp == 0) acts as NOP (break) instead of aborting execution.
// 3. Zero-drift baseline membrane coupling (+40 / -40, 0 base tax).
// 4. Honest energy conservation on macro-replication (>32 bits changed):
//    share = fuel[best_i] / 2 with no MIN_FUEL floor; skip collision if fuel < 512.

struct PureBitThermodynamics {
    static constexpr size_t NUM_TAPES = 2048;
    static constexpr size_t TAPE_BITS = 256;
    static constexpr int MIN_FUEL = 0;
    static constexpr int MAX_FUEL = 4096;
    static constexpr int REPRO_MIN_FUEL = 512;

    std::vector<uint8_t> bits;
    std::vector<int> fuel;
    uint64_t rng = 0x9E3779B97F4A7C15ULL;

    inline uint64_t next_rand() {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return rng ^ (rng >> 31);
    }

    void init(uint64_t seed) {
        rng = seed;
        bits.resize(NUM_TAPES * TAPE_BITS);
        fuel.assign(NUM_TAPES, 2000);
        for (size_t i = 0; i < bits.size(); ++i) bits[i] = (next_rand() >> 63) & 1;
    }

    // Exact bitwise-complement symmetric 4-bit opcode mapping:
    // 0011:< <-> 1100:> | 0101:{ <-> 1010:} | 0110:C <-> 1001:c
    // 0111:[ <-> 1000:] | 0001:! <-> 1110:? | remaining 6 nibbles = NOP
    static inline int map_op4(uint8_t nib) {
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

    static size_t eval_vm(uint8_t* mem, size_t len, size_t init_h0, size_t init_h1, size_t max_steps) {
        size_t pc = 0, h0 = init_h0, h1 = init_h1, steps = 0;
        size_t stack[64]; int sp = 0;
        while (pc + 3 < len && steps < max_steps) {
            uint8_t nib = (mem[pc] << 3) | (mem[pc+1] << 2) | (mem[pc+2] << 1) | mem[pc+3];
            int op = map_op4(nib);
            switch (op) {
                case 0: break; // NOP: fixed 4-bit advance below
                case 1: h0 = (h0 == 0) ? (len - 1) : (h0 - 1); break;
                case 2: h0 = (h0 + 1) % len; break;
                case 3: h1 = (h1 == 0) ? (len - 1) : (h1 - 1); break;
                case 4: h1 = (h1 + 1) % len; break;
                case 5: mem[h1] = mem[h0]; break;
                case 6: mem[h0] = mem[h1]; break;
                case 7: if (sp < 64) stack[sp++] = pc; break;
                case 8:
                    if (sp > 0) { pc = stack[sp - 1] + 4; steps++; continue; }
                    break; // Unmatched ']' acts as NOP
                case 9: mem[h0] ^= 1; break;
                case 10: {
                    if (mem[h0] == 0) {
                        pc += 4;
                        if (pc + 3 < len) {
                            uint8_t nn = (mem[pc] << 3) | (mem[pc+1] << 2) | (mem[pc+2] << 1) | mem[pc+3];
                            int nop = map_op4(nn);
                            if (nop == 8) { if (sp > 0) sp--; pc += 4; }
                            else if (nop == 7) {
                                int depth = 1; pc += 4;
                                while (depth > 0 && pc + 3 < len) {
                                    uint8_t ln = (mem[pc] << 3) | (mem[pc+1] << 2) | (mem[pc+2] << 1) | mem[pc+3];
                                    int lop = map_op4(ln);
                                    if (lop == 7) depth++; else if (lop == 8) depth--;
                                    pc += 4;
                                }
                                if (depth > 0) return steps;
                            } else pc += 4;
                        }
                        steps++; continue;
                    }
                    break;
                }
            }
            pc += 4; steps++;
        }
        return steps;
    }

    static std::string disasm(const uint8_t* mem, size_t len, size_t& active_ops_out) {
        const char* names[] = {".", "<", ">", "{", "}", "C", "c", "[", "]", "!", "?"};
        std::string s; size_t pc = 0; active_ops_out = 0;
        while (pc + 3 < len) {
            uint8_t nib = (mem[pc] << 3) | (mem[pc+1] << 2) | (mem[pc+2] << 1) | mem[pc+3];
            int op = map_op4(nib);
            s += names[op];
            if (op > 0) active_ops_out++;
            pc += 4;
        }
        return s;
    }

    double entropy_16() const {
        std::map<uint16_t, uint32_t> freq;
        size_t nw = bits.size() / 16;
        for (size_t i = 0; i < nw; ++i) {
            uint16_t w = 0;
            for (int k = 0; k < 16; ++k) w = (w << 1) | bits[i * 16 + k];
            freq[w]++;
        }
        double H = 0.0;
        for (auto& kv : freq) { double p = kv.second / (double)nw; H -= p * std::log2(p); }
        return H;
    }
};

void run_experiment_stage2(uint64_t seed, const char* label) {
    PureBitThermodynamics sim;
    sim.init(seed);
    uint8_t buf[512], solo_buf[256];
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "========================================================================\n";
    std::cout << "KARYON-CORE EXP-1166b: " << label << "\n";
    std::cout << "Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15\n";
    std::cout << "========================================================================\n\n";

    // 3 Regimes: 2a (NOT, ep 0..300), 2b (XOR, ep 301..2000), 2c (~XOR, ep 2001..2300)
    for (size_t ep = 0; ep <= 2300; ++ep) {
        int corr = 0, tot = 0;

        // 1. Environmental interaction ticks (8 ticks per epoch, symmetric +40 / -40)
        for (int tick = 0; tick < 8; ++tick) {
            uint8_t x0 = (sim.next_rand() >> 60) & 1;
            uint8_t x1 = (sim.next_rand() >> 59) & 1;
            uint8_t y_star = (ep <= 300) ? (x0 ^ 1) : ((ep <= 2000) ? (x0 ^ x1) : ((x0 ^ x1) ^ 1));

            for (size_t i = 0; i < PureBitThermodynamics::NUM_TAPES; ++i) {
                uint8_t* tape = &sim.bits[i * 256];
                std::memcpy(solo_buf, tape, 256);
                solo_buf[253] = x1;
                solo_buf[254] = x0;
                solo_buf[255] = (sim.next_rand() >> 62) & 1;

                PureBitThermodynamics::eval_vm(solo_buf, 256, 0, 0, 256);

                if (solo_buf[255] == y_star) {
                    sim.fuel[i] = std::min(PureBitThermodynamics::MAX_FUEL, sim.fuel[i] + 40);
                    corr++;
                } else {
                    sim.fuel[i] = std::max(PureBitThermodynamics::MIN_FUEL, sim.fuel[i] - 40);
                }
                tot++;
            }
        }

        // 2. Collisions & replication: dynamic max_steps = fuel[best_i], skip if fuel < 512
        int repro_events = 0;
        for (size_t c = 0; c < PureBitThermodynamics::NUM_TAPES / 2; ++c) {
            size_t best_i = sim.next_rand() % PureBitThermodynamics::NUM_TAPES;
            for (int k = 0; k < 3; ++k) {
                size_t cand = sim.next_rand() % PureBitThermodynamics::NUM_TAPES;
                if (sim.fuel[cand] > sim.fuel[best_i]) best_i = cand;
            }

            if (sim.fuel[best_i] < PureBitThermodynamics::REPRO_MIN_FUEL) continue;

            size_t j = sim.next_rand() % PureBitThermodynamics::NUM_TAPES;
            if (best_i == j) j = (best_i + 1) % PureBitThermodynamics::NUM_TAPES;

            std::memcpy(buf, &sim.bits[best_i * 256], 256);
            std::memcpy(buf + 256, &sim.bits[j * 256], 256);

            PureBitThermodynamics::eval_vm(buf, 512, 0, 256, (size_t)sim.fuel[best_i]);

            for (int k = 0; k < 512; ++k) {
                if ((sim.next_rand() & 16383ULL) == 0ULL) buf[k] ^= 1;
            }

            int diff_bits = 0;
            for (int k = 0; k < 256; ++k) {
                if (sim.bits[j * 256 + k] != buf[256 + k]) diff_bits++;
            }

            std::memcpy(&sim.bits[best_i * 256], buf, 256);
            std::memcpy(&sim.bits[j * 256], buf + 256, 256);

            // Honest fuel division (no MIN_FUEL floor) only if >32 bits modified in target
            if (diff_bits > 32) {
                repro_events++;
                int share = sim.fuel[best_i] / 2;
                sim.fuel[best_i] = share;
                sim.fuel[j] = share;
            }
        }

        // Gentle background thermal noise on starved tapes (< 512 fuel)
        for (size_t i = 0; i < PureBitThermodynamics::NUM_TAPES; ++i) {
            if (sim.fuel[i] < PureBitThermodynamics::REPRO_MIN_FUEL) {
                for (int k = 0; k < 256; ++k) {
                    if ((sim.next_rand() & 4095ULL) == 0ULL) sim.bits[i * 256 + k] ^= 1;
                }
            }
        }

        if (ep % 100 == 0 || ep == 300 || ep == 301 || ep == 2000 || ep == 2001 || ep == 2300) {
            double acc = (tot > 0) ? (100.0 * corr / tot) : 0.0;
            double avg_f = 0; int hi = 0;
            std::map<std::vector<uint8_t>, int> top;
            for (size_t k = 0; k < PureBitThermodynamics::NUM_TAPES; ++k) {
                avg_f += sim.fuel[k];
                if (sim.fuel[k] > 2000) hi++;
                top[std::vector<uint8_t>(sim.bits.begin() + k*256, sim.bits.begin() + (k+1)*256)]++;
            }
            avg_f /= PureBitThermodynamics::NUM_TAPES;
            double H16 = sim.entropy_16();

            std::vector<std::pair<int, std::vector<uint8_t>>> sorted;
            for (auto& kv : top) sorted.push_back({kv.second, kv.first});
            std::sort(sorted.rbegin(), sorted.rend());

            const char* reg = (ep <= 300) ? "2a:NOT" : ((ep <= 2000) ? "2b:XOR" : "2c:~XOR");
            std::cout << "  ep=" << std::setw(4) << ep << " [" << reg << "]"
                      << " | Acc=" << std::fixed << std::setprecision(2) << std::setw(6) << acc << "%"
                      << " | AvgFuel=" << std::setw(5) << std::setprecision(0) << avg_f
                      << " | HighFuel(>2k)=" << std::setw(4) << hi << "/2048"
                      << " | Repro=" << std::setw(3) << repro_events
                      << " | H16=" << std::setprecision(4) << H16 << " b/16b"
                      << " | u_tapes=" << sorted.size() << "/2048\n";

            if (ep == 300 || ep == 2000 || ep == 2300) {
                std::cout << "\n  --- Top Dominant Quasispecies at ep=" << ep << " [" << reg << "] ---\n";
                for (int r = 0; r < std::min<int>(2, sorted.size()); ++r) {
                    size_t ao = 0;
                    std::string ds = PureBitThermodynamics::disasm(sorted[r].second.data(), 256, ao);
                    std::cout << "   #" << r << " (" << sorted[r].first << "/2048, active_ops="
                              << std::setprecision(1) << (100.0*ao/ds.size()) << "%): " << ds << "\n";
                }
                std::cout << "\n";
            }
        }
    }
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  Elapsed Time: " << std::fixed << std::setprecision(2) << sec << "s\n\n";
}

int main() {
    std::cout << std::unitbuf;
    run_experiment_stage2(0x9E3779B97F4A7C15ULL, "Run #0 (Seed 0)");
    run_experiment_stage2(0x123456789ABCDEF0ULL, "Run #1 (Seed 1)");
    return 0;
}
