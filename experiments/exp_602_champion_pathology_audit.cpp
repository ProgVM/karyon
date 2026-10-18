// ======================================================================================
// EXP-602: Endoscopic Causal Pathology Diagnostic of the EXP-601 Champion
// --------------------------------------------------------------------------------------
// Objective:
// Step-by-step probe into the internal registers, operand routing, bit-shifts,
// and saturation bounds of the 9-node cell on tasks 6 (Language), 7 (Phonotactics),
// 8 (Bracket Stack), and 9 (Periodic Sine).
// Uncover EXACTLY why it failed or plateaued on these specific reality challenges.
// ======================================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

// Emulate the exact 9-node Champion Cell from EXP-601
struct ChampionCellAuditor {
    int16_t registers[16] = {0};

    inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
        return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
    }

    void reset() {
        for (int r = 0; r < 16; ++r) registers[r] = 0;
    }

    int16_t step(int16_t in0, int16_t in1, int16_t in2, int16_t in3, int16_t in4, int16_t in5, bool verbose = false) {
        int16_t in[6] = {in0, in1, in2, in3, in4, in5};

        // Node 0: ADD(in[2], in[5]) * 9 / 16
        int16_t n0 = dev_clamp16(((in[2] + in[5]) * 9) >> 4, -4096, 4095);

        // Node 1: NOT(n0) * 14 / 16 (in[4] ignored by NOT)
        int16_t n1 = dev_clamp16(((~n0) * 14) >> 4, -4096, 4095);

        // Node 2: ACCUMULATE: registers[in[3] % 16] += in[1]; res = registers[in[3] % 16] * 16 / 16
        int reg2_idx = static_cast<uint8_t>(in[3]) % 16;
        registers[reg2_idx] = dev_clamp16(registers[reg2_idx] + in[1], -4096, 4095);
        int16_t n2 = dev_clamp16(registers[reg2_idx] * 1, -4096, 4095);

        // Node 3: SUB(n2, in[1]) * 9 / 16
        int16_t n3 = dev_clamp16(((n2 - in[1]) * 9) >> 4, -4096, 4095);

        // Node 4: ADD(in[1], n3) * -7 / 16
        int16_t n4 = dev_clamp16(((in[1] + n3) * -7) >> 4, -4096, 4095);

        // Node 5: SIGN(in[0]) * -16 / 16
        int16_t s5 = (in[0] > 0) ? 1 : ((in[0] < 0) ? -1 : 0);
        int16_t n5 = dev_clamp16((s5 * -16) >> 4, -4096, 4095);

        // Node 6: STORE_REG: registers[n4 % 16] = n3; res = n3 * -6 / 16
        int reg6_idx = static_cast<uint8_t>(n4) % 16;
        registers[reg6_idx] = n3;
        int16_t n6 = dev_clamp16((n3 * -6) >> 4, -4096, 4095);

        // Node 7: MAX(in[0], n0) * -10 / 16
        int16_t max_val = (in[0] > n0) ? in[0] : n0;
        int16_t n7 = dev_clamp16((max_val * -10) >> 4, -4096, 4095);

        // Node 8: SUB(n2, n5) * 16 / 16
        int16_t n8 = dev_clamp16(((n2 - n5) * 16) >> 4, -4096, 4095);

        if (verbose) {
            std::cout << "    [In: in0=" << in0 << ", in1=" << in1 << ", in2=" << in2 << ", in3=" << in3 << "] "
                      << "n2=" << n2 << ", n5=" << n5 << ", n8(out)=" << n8 
                      << " (out&0xFF=" << (n8 & 0xFF) << " / char:'" << (char)(n8 & 0xFF) << "')\n";
        }

        return n8;
    }
};

int main() {
    ChampionCellAuditor cell;

    std::cout << "========================================================================\n";
    std::cout << "=== 🔬 ENDOSCOPIC AUTOPSY: ROOT-CAUSE FAILURE ANALYSIS OF EXP-601    ===\n";
    std::cout << "========================================================================\n\n";

    // ----------------------------------------------------------------------------------
    // AUDIT 1: TASK 6 (Natural Language Alpaca Stream)
    // Input format in EXP-601: in[0] = cur_byte, in[1] = t, in[2..5] = 0
    // Target: next_byte
    // ----------------------------------------------------------------------------------
    std::cout << ">>> AUDITING TASK 6: Natural Language Stream ('instruction_response_karyon_core')\n";
    cell.reset();
    const char phrase[33] = "instruction_response_karyon_core";
    for (int t = 0; t < 8; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        std::cout << "Step " << t << " | In Byte: '" << (char)cur << "' (" << (int)cur << "), Tgt: '" << (char)nxt << "' (" << (int)nxt << ")\n";
        int16_t out = cell.step(cur, t, 0, 0, 0, 0, true);
    }

    // ----------------------------------------------------------------------------------
    // AUDIT 2: TASK 8 (Bracket Balancing Stack)
    // Input: in[0] = b ('(' or ')'), in[1] = t
    // Target: stack depth
    // ----------------------------------------------------------------------------------
    std::cout << "\n>>> AUDITING TASK 8: Context-Free Bracket Balancing Stack\n";
    cell.reset();
    const char brackets[17] = "(())((()))()(())";
    int depth = 0;
    for (int t = 0; t < 8; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        std::cout << "Step " << t << " | In: '" << b << "' (" << (int)b << "), Tgt Depth: " << depth << "\n";
        int16_t out = cell.step(b, t, 0, 0, 0, 0, true);
    }

    // ----------------------------------------------------------------------------------
    // AUDIT 3: MATHEMATICAL DISSECTION OF THE OUTPUT FORMULA (node_8)
    // ----------------------------------------------------------------------------------
    std::cout << "\n>>> MATHEMATICAL DISSECTION OF THE OUTPUT NODE:\n";
    std::cout << "  node_8 = SUB(node_2, node_5) * 1\n";
    std::cout << "  node_5 = SIGN(in[0]) * -1\n";
    std::cout << "  node_2 = registers[in[3] % 16] += in[1]\n";
    std::cout << "  Therefore: out = (cumulative sum of in[1]) - (-1) = sum(in[1]) + 1\n";
    std::cout << "  LOOK AT WHAT in[1] WAS IN TASK 6 AND TASK 8:\n";
    std::cout << "  in[1] was passed as 't' (the time step: 0, 1, 2, 3...)!\n";
    std::cout << "  So out(t) = t*(t+1)/2 + 1 ! It is purely a quadratic ramp function of time!\n";
    std::cout << "========================================================================\n";

    return 0;
}
