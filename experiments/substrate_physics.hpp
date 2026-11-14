#pragma once
// ============================================================================
// KARYON-CORE MINIMAL SUBSTRATE PHYSICS (EXP-1163c)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// STRICT PHYSICAL LAWS (No matrices, No backprop, No lookup tables):
//   1. Sparse 1-Bit Receptors: Cells spawn with single-bit input masks (1/64),
//      naturally isolating signal from noise without pre-engineered filters.
//   2. Whole-Word Decisions & Parity Voting: Cell fires (res != 0) over full 64-bit
//      ALU word. Inputs include current raw_obs, ~raw_obs, prev_obs, ~prev_obs.
//   3. Decaying Multi-Step Eligibility Trace: 8-step decaying trace (0.85^t)
//      propagates global energy rewards/penalties back across multi-step sequences.
//   4. Proportional Chemotactic Plasticity: Constant low exploration (0.5% noise)
//      prevents metabolic coma under hunger; heavy pain (toxins) triggers
//      proportional mutations only on cells that voted FOR erroneous actions.
// ============================================================================

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <bit>

enum CellALUOp : uint8_t {
    ALU_XOR    = 0, // (va & mask_a) ^ (vb & mask_b)
    ALU_AND    = 1, // (va & mask_a) & (vb & mask_b)
    ALU_ROT    = 2, // std::rotl(va & mask_a, shift)
    ALU_POPCNT = 3  // __builtin_popcountll((va & mask_a) ^ (vb & mask_b)) & 1 ? ~0ULL : 0ULL
};

struct SubstrateCell {
    uint8_t   src_a       = 0;     // Index into register bank
    uint8_t   src_b       = 0;
    uint64_t  mask_a      = 1ULL;  // Sparse single-bit receptor mask
    uint64_t  mask_b      = 1ULL;
    uint8_t   op          = ALU_XOR;
    double    energy      = 10.0;  // Metabolism
    double    trace0      = 0.0;   // Decaying eligibility trace for motor bit 0
    double    trace1      = 0.0;   // Decaying eligibility trace for motor bit 1
    uint64_t  output_word = 0ULL;
};

class SubstratePhysicsEngine {
public:
    static constexpr size_t   MAX_CELLS     = 64;
    static constexpr size_t   NUM_SENSORS   = 16;
    static constexpr double   CELL_INIT_E   = 10.0;
    static constexpr double   CELL_MAX_E    = 50.0;
    static constexpr double   CELL_STEP_COST= 0.02; // Sustainable metabolism
    static constexpr double   GAIN_MULT     = 1.5;
    static constexpr double   PENALTY_MULT  = 2.0;
    static constexpr double   TRACE_DECAY   = 0.85;

    std::vector<SubstrateCell> cells;
    uint64_t registers[NUM_SENSORS + MAX_CELLS];
    uint64_t prev_obs = 0ULL;
    uint8_t  last_chosen_action = 0;
    uint64_t rng = 0xCAFEEBABEDEADBEEULL;

    SubstratePhysicsEngine() { reset(); }

    inline uint64_t next_rand() {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return rng;
    }

    void reset() {
        cells.clear();
        cells.resize(MAX_CELLS);
        for (size_t i = 0; i < MAX_CELLS; ++i) respawn_cell(i);
        std::fill(std::begin(registers), std::end(registers), 0ULL);
        prev_obs = 0ULL;
        last_chosen_action = 0;
    }

    void respawn_cell(size_t i) {
        SubstrateCell& c = cells[i];
        // Allow connection to sensors AND outputs of existing lower-index cells
        c.src_a = static_cast<uint8_t>(next_rand() % (NUM_SENSORS + i));
        c.src_b = static_cast<uint8_t>(next_rand() % (NUM_SENSORS + i));
        // Sparse 1-bit initial mask (isolates single bits from noise)
        c.mask_a = 1ULL << (next_rand() & 63U);
        c.mask_b = 1ULL << (next_rand() & 63U);
        c.op    = static_cast<uint8_t>((next_rand() >> 8) & 3U);
        c.energy = CELL_INIT_E;
        c.trace0 = 0.0;
        c.trace1 = 0.0;
        c.output_word = 0ULL;
    }

    // Step 1: Forward Execution (Sensory Registers -> Cell ALU Network -> Majority Vote Motors)
    uint8_t forward(uint64_t raw_obs) {
        // Distribute sensory signals: current raw_obs, ~raw_obs, prev_obs, ~prev_obs
        for (size_t i = 0; i < 4; ++i) {
            registers[i]     = std::rotl(raw_obs, static_cast<int>(i * 16));
            registers[i + 4] = ~registers[i];
            registers[i + 8] = std::rotl(prev_obs, static_cast<int>(i * 16));
            registers[i + 12]= ~registers[i + 8];
        }
        prev_obs = raw_obs;

        uint32_t votes0_1 = 0, votes0_0 = 0;
        uint32_t votes1_1 = 0, votes1_0 = 0;

        for (size_t i = 0; i < MAX_CELLS; ++i) {
            SubstrateCell& c = cells[i];
            if (c.energy <= 0.0) respawn_cell(i);

            uint64_t va = registers[c.src_a] & c.mask_a;
            uint64_t vb = registers[c.src_b] & c.mask_b;
            uint64_t res = 0ULL;

            if (c.op == ALU_XOR) res = va ^ vb;
            else if (c.op == ALU_AND) res = va & vb;
            else if (c.op == ALU_ROT) res = std::rotl(va, static_cast<int>(vb & 63ULL));
            else res = (__builtin_popcountll(va ^ vb) & 1U) ? ~0ULL : 0ULL;

            c.output_word = res;
            registers[NUM_SENSORS + i] = res;
            c.energy -= CELL_STEP_COST;

            // Whole-word decision & parity voting (no single-bit bottlenecks)
            uint8_t vote0 = (res != 0ULL) ? 1U : 0U;
            uint8_t vote1 = ((__builtin_popcountll(res) & 1U)) ? 1U : 0U;

            // Decaying multi-step eligibility trace update
            c.trace0 = c.trace0 * TRACE_DECAY + (vote0 ? 1.0 : -1.0);
            c.trace1 = c.trace1 * TRACE_DECAY + (vote1 ? 1.0 : -1.0);

            if (i < MAX_CELLS / 2) {
                if (vote0) votes0_1++; else votes0_0++;
            } else {
                if (vote1) votes1_1++; else votes1_0++;
            }
        }

        // Majority vote efference
        uint8_t a0 = (votes0_1 >= votes0_0) ? 1U : 0U;
        uint8_t a1 = (votes1_1 >= votes1_0) ? 1U : 0U;
        last_chosen_action = (a1 << 1) | a0;
        return last_chosen_action;
    }

    // Step 2: Proportional Multi-Step Chemotactic Adaptation (3-Factor Plasticity)
    void adapt(double delta_E_global) {
        uint8_t act0 = last_chosen_action & 1U;
        uint8_t act1 = (last_chosen_action >> 1) & 1U;

        for (size_t i = 0; i < MAX_CELLS; ++i) {
            SubstrateCell& c = cells[i];
            if (c.energy <= 0.0) continue;

            double tr = (i < MAX_CELLS / 2) ? c.trace0 : c.trace1;
            double chosen_sign = ((i < MAX_CELLS / 2) ? act0 : act1) ? 1.0 : -1.0;
            double alignment = tr * chosen_sign; // Positive if cell trace supported chosen action

            if (delta_E_global > 0.0) {
                // Success (food eaten): reward cells whose multi-step trace supported the action
                if (alignment > 0.0) {
                    c.energy += delta_E_global * GAIN_MULT * (alignment / 2.0);
                    if (c.energy > CELL_MAX_E) c.energy = CELL_MAX_E;
                }
            } else {
                // Hunger / Toxin damage
                double pain = std::abs(delta_E_global);
                if (alignment > 0.0) {
                    c.energy -= pain * PENALTY_MULT * (alignment / 2.0);
                    
                    // Chemotactic mutation rate proportional to pain & alignment
                    double mut_prob = std::clamp(0.005 + 0.05 * pain, 0.005, 0.50);
                    uint64_t threshold = static_cast<uint64_t>(mut_prob * 18446744073709551615.0);

                    if (next_rand() < threshold) {
                        c.op = static_cast<uint8_t>((next_rand() >> 4) & 3U);
                    }
                    if (next_rand() < (threshold / 2)) {
                        c.src_a = static_cast<uint8_t>(next_rand() % (NUM_SENSORS + i));
                    }
                    if (next_rand() < (threshold / 2)) {
                        c.src_b = static_cast<uint8_t>(next_rand() % (NUM_SENSORS + i));
                    }
                    if (next_rand() < threshold) {
                        uint64_t r = next_rand();
                        // Mutate sparse receptor masks while maintaining low popcount (<= 4)
                        if (__builtin_popcountll(c.mask_a) > 4) c.mask_a = 1ULL << (r & 63U);
                        else c.mask_a ^= (1ULL << (r & 63U));

                        if (__builtin_popcountll(c.mask_b) > 4) c.mask_b = 1ULL << ((r >> 6) & 63U);
                        else c.mask_b ^= (1ULL << ((r >> 6) & 63U));
                    }
                }
            }
        }
    }
};
