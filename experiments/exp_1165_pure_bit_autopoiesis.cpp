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
// KARYON-CORE EXP-1165: PURE BIT AUTOPOIETIC SUBSTRATE V=2 (STAGE 1)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Lead AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// Pure 1-bit Code=Data substrate (zero byte boxes, zero task/oracle bias).
// 10-operator bitwise-complement symmetric basis (6 NOPs = 37.5% neutral space).
// Initial head geometry per collision A||B (512 bits): h0 = 0 (A), h1 = 256 (B).

struct PureBitAutopoiesis {
    static constexpr size_t NUM_TAPES = 2048;
    static constexpr size_t TAPE_BITS = 256;
    static constexpr size_t MAX_STEPS = 4096;
    std::vector<uint8_t> bits;
    uint64_t rng = 0;

    inline uint64_t next_rand() {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return rng ^ (rng >> 31);
    }

    void init_noise(uint64_t seed, bool invert = false) {
        rng = seed;
        bits.resize(NUM_TAPES * TAPE_BITS);
        for (size_t i = 0; i < bits.size(); ++i)
            bits[i] = ((next_rand() >> 63) & 1) ^ (invert ? 1 : 0);
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
            case 14: return 10; // ?  if (mem[h0]==0) skip next instruction
            default: return 0;  // NOP
        }
    }

    static size_t eval_pair(uint8_t* mem, size_t len, size_t init_h0, size_t init_h1) {
        size_t pc = 0, h0 = init_h0, h1 = init_h1, steps = 0;
        size_t stack[64];
        int sp = 0;
        while (pc + 3 < len && steps < MAX_STEPS) {
            uint8_t nib = (mem[pc] << 3) | (mem[pc+1] << 2) | (mem[pc+2] << 1) | mem[pc+3];
            int op = map_op4(nib);
            if (op == 0) { pc += 1; steps++; continue; }
            switch (op) {
                case 1: h0 = (h0 == 0) ? (len - 1) : (h0 - 1); break;
                case 2: h0 = (h0 + 1) % len; break;
                case 3: h1 = (h1 == 0) ? (len - 1) : (h1 - 1); break;
                case 4: h1 = (h1 + 1) % len; break;
                case 5: mem[h1] = mem[h0]; break;
                case 6: mem[h0] = mem[h1]; break;
                case 7: if (sp < 64) stack[sp++] = pc; break;
                case 8: {
                    if (sp > 0) { pc = stack[sp - 1] + 4; steps++; continue; }
                    return steps;
                }
                case 9: mem[h0] ^= 1; break;
                case 10: { // ? : skip next instruction cleanly without stack leak
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
                                    pc += (lop == 0 ? 1 : 4);
                                }
                                if (depth > 0) return steps;
                            } else pc += (nop == 0 ? 1 : 4);
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
            pc += (op == 0 ? 1 : 4);
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

void run_stage1(uint64_t seed, bool invert, const char* label) {
    PureBitAutopoiesis sim;
    sim.init_noise(seed, invert);
    uint8_t buf[512];
    double mean_steps = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "=== " << label << " ===\n";
    for (size_t ep = 0; ep <= 500; ++ep) {
        if (ep == 0 || ep == 50 || ep == 100 || ep == 250 || ep == 500) {
            std::set<uint64_t> u64;
            std::map<std::vector<uint8_t>, int> top;
            for (size_t i = 0; i < sim.bits.size() / 64; ++i) {
                uint64_t w = 0; for (int k = 0; k < 64; ++k) w = (w << 1) | sim.bits[i * 64 + k];
                u64.insert(w);
            }
            for (size_t i = 0; i < PureBitAutopoiesis::NUM_TAPES; ++i)
                top[std::vector<uint8_t>(sim.bits.begin() + i*256, sim.bits.begin() + (i+1)*256)]++;
            double H16 = sim.entropy_16();
            size_t z = 0; for (uint8_t b : sim.bits) if (b == 0) z++;
            std::vector<std::pair<int, std::vector<uint8_t>>> sorted;
            for (auto& kv : top) sorted.push_back({kv.second, kv.first});
            std::sort(sorted.rbegin(), sorted.rend());
            size_t act = 0;
            std::string dis = PureBitAutopoiesis::disasm(sorted[0].second.data(), 256, act);
            std::cout << "  ep=" << std::setw(3) << ep
                      << " | H16=" << std::fixed << std::setprecision(4) << std::setw(7) << H16 << " b/16b"
                      << " | u64=" << std::setw(4) << u64.size() << "/8192"
                      << " | tapes=" << std::setw(4) << top.size() << "/2048"
                      << " | top=" << std::setw(3) << sorted[0].first
                      << " | zero=" << std::setprecision(1) << (100.0*z/sim.bits.size()) << "%"
                      << " | active_ops=" << std::setprecision(1) << (100.0*act/dis.size()) << "%"
                      << " | steps=" << std::setprecision(0) << mean_steps << "\n";
            if (ep == 500) {
                std::cout << "  Top-5 Dominant Tapes (ep=500):\n";
                for (int r = 0; r < std::min<int>(5, sorted.size()); ++r) {
                    size_t ao = 0;
                    std::string ds = PureBitAutopoiesis::disasm(sorted[r].second.data(), 256, ao);
                    std::cout << "   #" << r << " (" << sorted[r].first << "/2048, act="
                              << std::setprecision(1) << (100.0*ao/ds.size()) << "%): " << ds << "\n";
                }
            }
        }
        if (ep == 500) break;
        size_t tot_s = 0;
        for (size_t c = 0; c < PureBitAutopoiesis::NUM_TAPES / 2; ++c) {
            size_t i = sim.next_rand() % PureBitAutopoiesis::NUM_TAPES;
            size_t j = sim.next_rand() % PureBitAutopoiesis::NUM_TAPES;
            if (i == j) j = (i + 1) % PureBitAutopoiesis::NUM_TAPES;
            std::memcpy(buf, &sim.bits[i * 256], 256);
            std::memcpy(buf + 256, &sim.bits[j * 256], 256);
            tot_s += PureBitAutopoiesis::eval_pair(buf, 512, 0, 256);
            for (int k = 0; k < 512; ++k) if ((sim.next_rand() & 16383ULL) == 0ULL) buf[k] ^= 1;
            std::memcpy(&sim.bits[i * 256], buf, 256);
            std::memcpy(&sim.bits[j * 256], buf + 256, 256);
        }
        mean_steps = tot_s / (double)(PureBitAutopoiesis::NUM_TAPES / 2);
    }
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  elapsed=" << std::fixed << std::setprecision(2) << sec << "s\n\n";
}

int main() {
    std::cout << std::unitbuf;
    run_stage1(0x9E3779B97F4A7C15ULL, false, "Seed 0 (Normal)");
    run_stage1(0x9E3779B97F4A7C15ULL, true,  "Seed 0 (Bitwise Inverted 0<->1)");
    run_stage1(0x123456789ABCDEF0ULL, false, "Seed 1 (Normal)");
    run_stage1(0xCAFEBABEDEADBEEFULL, false, "Seed 2 (Normal)");
    return 0;
}
