#pragma once
// ============================================================================
// KARYON-CORE MINIMAL SUBSTRATE PHYSICS (EXP-1163d)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 1, 3, 4 (V=2), 6, 8, 9, 11, 13, 14, 15
// ============================================================================
// STRICT PHYSICAL LAWS (No matrices, No backprop, No lookup tables):
//   1. Boolean-Level Receptor Filtering & Logic ALU:
//      After sparse 1-bit masking, inputs collapse to full 64-bit boolean
//      levels (~0ULL or 0ULL). ALU ops: XOR, AND, OR, AND_NOT (ba & ~bb,
//      elementary 1-bit comparator a > b). Cell-to-cell compositionality works
//      at 100% reliability without bit-position misses.
//   2. Active Energy-Weighted Motor Efference:
//      Only ACTIVE cells (res != 0) cast energy-weighted votes for motor bits.
//      Silent cells do not vote, allowing active detectors to drive action.
//   3. Homeostatic Survival & Crystallization Protection:
//      Normal steps (-0.5) incur CELL_STEP_COST (0.02) + 0.5% exploratory noise.
//      Heavy penalty & mutation trigger ONLY on toxin damage (delta_E < -1.0),
//      damped inversely by accumulated cell energy (crystallization).
// ============================================================================

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <bit>

enum CellALUOp : uint8_t {
    ALU_XOR     = 0, // ba ^ bb
    ALU_AND     = 1, // ba & bb
    ALU_OR      = 2, // ba | bb
    ALU_AND_NOT = 3  // ba & ~bb (elementary 1-bit comparator: a > b)
};

struct SubstrateCell {
    uint16_t  src_a       = 0;     // Index into register bank (0..271)
    uint16_t  src_b       = 0;
    uint64_t  mask_a      = 1ULL;  // Sparse receptor mask
    uint64_t  mask_b      = 1ULL;
    uint8_t   op          = ALU_XOR;
    double    energy      = 10.0;  // Metabolism
    double    trace0      = 0.0;   // Decaying eligibility trace for motor bit 0
    double    trace1      = 0.0;   // Decaying eligibility trace for motor bit 1
    uint64_t  output_word = 0ULL;
};

class SubstratePhysicsEngine {
public:
    static constexpr size_t   MAX_CELLS     = 256;
    static constexpr size_t   NUM_SENSORS   = 16;
    static constexpr double   CELL_INIT_E   = 10.0;
    static constexpr double   CELL_MAX_E    = 100.0;
    static constexpr double   CELL_STEP_COST= 0.02;
    static constexpr double   GAIN_MULT     = 1.5;
    static constexpr double   PENALTY_MULT  = 1.0;
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
        // 50% chance to connect directly to sensors, 50% to lower cells
        if ((next_rand() & 1ULL) == 0ULL || i == 0) {
            c.src_a = static_cast<uint16_t>(next_rand() % NUM_SENSORS);
            c.src_b = static_cast<uint16_t>(next_rand() % NUM_SENSORS);
        } else {
            c.src_a = static_cast<uint16_t>(next_rand() % (NUM_SENSORS + i));
            c.src_b = static_cast<uint16_t>(next_rand() % (NUM_SENSORS + i));
        }
        c.mask_a = 1ULL << (next_rand() & 63U);
        c.mask_b = 1ULL << (next_rand() & 63U);
        c.op    = static_cast<uint8_t>((next_rand() >> 8) & 3U);
        c.energy = CELL_INIT_E;
        c.trace0 = 0.0;
        c.trace1 = 0.0;
        c.output_word = 0ULL;
    }

    // Step 1: Forward Execution (Sensory Registers -> Boolean Cell ALU -> Active Energy Voting)
    uint8_t forward(uint64_t raw_obs) {
        for (size_t i = 0; i < 4; ++i) {
            registers[i]     = std::rotl(raw_obs, static_cast<int>(i * 16));
            registers[i + 4] = ~registers[i];
            registers[i + 8] = std::rotl(prev_obs, static_cast<int>(i * 16));
            registers[i + 12]= ~registers[i + 8];
        }
        prev_obs = raw_obs;

        double active_votes0 = 0.0;
        double active_votes1 = 0.0;
        double total_energy0 = 0.0;
        double total_energy1 = 0.0;

        for (size_t i = 0; i < MAX_CELLS; ++i) {
            SubstrateCell& c = cells[i];
            if (c.energy <= 0.0) respawn_cell(i);

            uint64_t ba = ((registers[c.src_a] & c.mask_a) != 0ULL) ? ~0ULL : 0ULL;
            uint64_t bb = ((registers[c.src_b] & c.mask_b) != 0ULL) ? ~0ULL : 0ULL;
            uint64_t res = 0ULL;

            if (c.op == ALU_XOR)          res = ba ^ bb;
            else if (c.op == ALU_AND)     res = ba & bb;
            else if (c.op == ALU_OR)      res = ba | bb;
            else                          res = ba & ~bb; // ALU_AND_NOT

            c.output_word = res;
            registers[NUM_SENSORS + i] = res;
            c.energy -= CELL_STEP_COST;

            bool fired = (res != 0ULL);

            if (i < MAX_CELLS / 2) {
                total_energy0 += c.energy;
                c.trace0 = c.trace0 * TRACE_DECAY + (fired ? 1.0 : -1.0);
                if (fired) active_votes0 += c.energy;
            } else {
                total_energy1 += c.energy;
                c.trace1 = c.trace1 * TRACE_DECAY + (fired ? 1.0 : -1.0);
                if (fired) active_votes1 += c.energy;
            }
        }

        // Active fraction voting: motor bit is 1 if >35% of energy pool in that hemisphere fires
        uint8_t a0 = (active_votes0 > (total_energy0 * 0.35)) ? 1U : 0U;
        uint8_t a1 = (active_votes1 > (total_energy1 * 0.35)) ? 1U : 0U;
        last_chosen_action = (a1 << 1) | a0;
        return last_chosen_action;
    }

    // Step 2: Homeostatic Chemotactic Adaptation with Crystallization Protection
    void adapt(double delta_E_global) {
        uint8_t act0 = last_chosen_action & 1U;
        uint8_t act1 = (last_chosen_action >> 1) & 1U;

        for (size_t i = 0; i < MAX_CELLS; ++i) {
            SubstrateCell& c = cells[i];
            if (c.energy <= 0.0) continue;

            double tr = (i < MAX_CELLS / 2) ? c.trace0 : c.trace1;
            double chosen_sign = ((i < MAX_CELLS / 2) ? act0 : act1) ? 1.0 : -1.0;
            double alignment = tr * chosen_sign;

            // Crystallization protection: higher energy = lower mutation probability
            double crystal_damp = (c.energy > CELL_INIT_E) ? (CELL_INIT_E / c.energy) : 1.0;

            if (delta_E_global > 0.0) {
                // Success (food eaten): reward cells whose trace supported the action
                if (alignment > 0.0) {
                    c.energy += delta_E_global * GAIN_MULT * (alignment / 2.0);
                    if (c.energy > CELL_MAX_E) c.energy = CELL_MAX_E;
                }
            } else if (delta_E_global < -1.0) {
                // Real damage (toxin): penalize & mutate cells that drove the faulty action
                double pain = std::abs(delta_E_global);
                if (alignment > 0.0) {
                    c.energy -= pain * PENALTY_MULT * (alignment / 2.0);

                    double mut_prob = std::clamp(0.02 * pain * crystal_damp, 0.01, 0.50);
                    uint64_t threshold = static_cast<uint64_t>(mut_prob * 18446744073709551615.0);

                    if (next_rand() < threshold) {
                        c.op = static_cast<uint8_t>((next_rand() >> 4) & 3U);
                    }
                    if (next_rand() < (threshold / 2)) {
                        c.src_a = static_cast<uint16_t>(next_rand() % (NUM_SENSORS + i));
                    }
                    if (next_rand() < (threshold / 2)) {
                        c.src_b = static_cast<uint16_t>(next_rand() % (NUM_SENSORS + i));
                    }
                    if (next_rand() < threshold) {
                        uint64_t r = next_rand();
                        if (__builtin_popcountll(c.mask_a) > 4) c.mask_a = 1ULL << (r & 63U);
                        else c.mask_a ^= (1ULL << (r & 63U));

                        if (__builtin_popcountll(c.mask_b) > 4) c.mask_b = 1ULL << ((r >> 6) & 63U);
                        else c.mask_b ^= (1ULL << ((r >> 6) & 63U));
                    }
                }
            } else {
                // Normal step (-0.5): no pain energy drain, only gentle 0.5% exploratory noise
                double explore_prob = 0.005 * crystal_damp;
                uint64_t threshold = static_cast<uint64_t>(explore_prob * 18446744073709551615.0);
                if (next_rand() < threshold) {
                    uint64_t r = next_rand();
                    if ((r & 3ULL) == 0ULL) c.op = static_cast<uint8_t>((r >> 4) & 3U);
                    else if ((r & 3ULL) == 1ULL) c.src_a = static_cast<uint16_t>((r >> 8) % (NUM_SENSORS + i));
                    else if ((r & 3ULL) == 2ULL) c.src_b = static_cast<uint16_t>((r >> 16) % (NUM_SENSORS + i));
                    else {
                        if (__builtin_popcountll(c.mask_a) > 4) c.mask_a = 1ULL << (r & 63U);
                        else c.mask_a ^= (1ULL << (r & 63U));
                    }
                }
            }
        }
    }
};
