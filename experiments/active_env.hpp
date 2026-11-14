#pragma once
// ============================================================================
// KARYON-CORE FROZEN ACTIVE ENVIRONMENT (EXP-1163a)
// Minimal Causal Sensorimotor World with Noise & Non-Stationary Regime Shifts
// ============================================================================
#include <cstdint>
#include <algorithm>

struct ActiveWorld {
    static constexpr uint32_t RING_SIZE = 16;
    static constexpr double   E_INIT    = 100.0;
    static constexpr double   E_MAX     = 200.0;
    static constexpr double   STEP_COST = 0.5;
    static constexpr double   FOOD_GAIN = 15.0;
    static constexpr double   TOXIN_PEN = 20.0;
    static constexpr uint64_t SHIFT_PERIOD = 2000;

    uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
    uint64_t tick      = 0;
    uint32_t pos       = 0;
    uint32_t posA      = 4;
    uint32_t posB      = 12;
    double   energy    = E_INIT;
    double   prev_energy = E_INIT;
    uint32_t regime    = 0;
    uint64_t food_eaten = 0;
    uint64_t toxins_hit = 0;

    inline uint64_t next_rand() {
        uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    void reset(uint64_t seed = 0x123456789ABCDEF0ULL) {
        rng_state = seed;
        tick = 0;
        pos = static_cast<uint32_t>(next_rand() % RING_SIZE);
        respawn(posA);
        respawn(posB);
        energy = E_INIT;
        prev_energy = E_INIT;
        regime = 0;
        food_eaten = 0;
        toxins_hit = 0;
    }

    void respawn(uint32_t& target) {
        do {
            target = static_cast<uint32_t>(next_rand() % RING_SIZE);
        } while (target == pos);
    }

    // Returns 64-bit observation:
    //   bits  0..3 : agent position (4 bits)
    //   bits  4..7 : object A position (4 bits)
    //   bits  8..11: object B position (4 bits)
    //   bits 12..63: 52 bits of pure random environmental noise
    inline uint64_t observe() {
        uint64_t noise = next_rand() & ~0xFFFULL;
        uint64_t signal = static_cast<uint64_t>(pos & 0xF)
                        | (static_cast<uint64_t>(posA & 0xF) << 4)
                        | (static_cast<uint64_t>(posB & 0xF) << 8);
        return noise | signal;
    }

    // Executes 2-bit motor action (0=stay, 1=left, 2=right, 3=stay)
    // Returns delta_E = energy(t+1) - energy(t). Alive iff energy > 0.
    double step(uint8_t action2b) {
        prev_energy = energy;
        tick++;
        regime = static_cast<uint32_t>((tick / SHIFT_PERIOD) % 4);

        bool invert_motors = (regime == 2 || regime == 3);
        bool a_is_food     = (regime == 0 || regime == 2);

        uint8_t act = action2b & 3U;
        int32_t delta_pos = 0;
        if (act == 1) delta_pos = invert_motors ? 1 : -1;
        else if (act == 2) delta_pos = invert_motors ? -1 : 1;

        pos = static_cast<uint32_t>((static_cast<int32_t>(pos) + delta_pos + RING_SIZE) % RING_SIZE);
        energy -= STEP_COST;

        if (pos == posA) {
            if (a_is_food) { energy += FOOD_GAIN; food_eaten++; }
            else           { energy -= TOXIN_PEN; toxins_hit++; }
            respawn(posA);
        }
        if (pos == posB) {
            if (!a_is_food) { energy += FOOD_GAIN; food_eaten++; }
            else            { energy -= TOXIN_PEN; toxins_hit++; }
            respawn(posB);
        }

        if (energy > E_MAX) energy = E_MAX;
        if (energy < 0.0)   energy = 0.0;
        return energy - prev_energy;
    }

    inline bool alive() const { return energy > 0.0; }
};
