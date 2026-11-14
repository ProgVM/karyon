// ============================================================================
// KARYON-CORE EXP-1163a: HONEST ACTIVE ENVIRONMENT WITH ORACLE BASELINE
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v51.0 | Principles 0, 3, 4, 8, 9, 11, 13, 14, 15
// ============================================================================
// Evaluates THREE baseline agents in the frozen ActiveWorld (active_env.hpp):
//   1. Random Agent: selects 2-bit motor actions purely uniformly.
//   2. Context Hash Q-Agent: 12-bit clean signal lookup table (fails on regime shift).
//   3. Oracle Agent: PERFECT environment knowledge (knows active regime laws).
// Benchmarks performance across 100,000 steps with periodic Regime Shifts
// (Regime 0..3: Object Swap & Motor Inversion every 2,000 steps).
// ============================================================================

#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "active_env.hpp"

struct RandomAgent {
    uint64_t rng = 0x9876543210FEDCBAULL;
    inline uint8_t act(uint64_t obs) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint8_t>((rng >> 32) & 3U);
    }
    inline void learn(uint64_t obs, uint8_t act, double delta_E, uint64_t next_obs) {}
};

struct ContextHashQAgent {
    std::vector<double> q_table; // 4096 states x 4 actions
    double lr = 0.1;
    double gamma = 0.9;
    double epsilon = 0.1;
    uint64_t rng = 0x13579BDF2468ACE0ULL;

    ContextHashQAgent() : q_table(4096 * 4, 0.0) {}

    inline uint16_t state_key(uint64_t obs) const {
        return static_cast<uint16_t>(obs & 0xFFFULL);
    }

    inline uint8_t act(uint64_t obs) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        if (((rng >> 16) & 0xFFFF) < static_cast<uint32_t>(epsilon * 65535.0)) {
            return static_cast<uint8_t>((rng >> 32) & 3U);
        }
        uint16_t s = state_key(obs);
        uint8_t best_a = 0;
        double max_q = q_table[s * 4];
        for (uint8_t a = 1; a < 4; ++a) {
            double q = q_table[s * 4 + a];
            if (q > max_q) { max_q = q; best_a = a; }
        }
        return best_a;
    }

    inline void learn(uint64_t obs, uint8_t a, double delta_E, uint64_t next_obs) {
        uint16_t s = state_key(obs);
        uint16_t s_next = state_key(next_obs);
        double max_next_q = q_table[s_next * 4];
        for (uint8_t na = 1; na < 4; ++na) {
            if (q_table[s_next * 4 + na] > max_next_q) max_next_q = q_table[s_next * 4 + na];
        }
        double target = delta_E + gamma * max_next_q;
        q_table[s * 4 + a] += lr * (target - q_table[s * 4 + a]);
    }
};

// Oracle Agent: inspects the environment's internal regime state to move directly toward food
// while avoiding toxins on the 16-element ring!
struct OracleAgent {
    inline uint8_t act_oracle(const ActiveWorld& env) {
        uint32_t pos = env.pos;
        bool a_is_food = (env.regime == 0 || env.regime == 2);
        uint32_t target_food  = a_is_food ? env.posA : env.posB;
        uint32_t target_toxin = a_is_food ? env.posB : env.posA;
        bool invert = (env.regime == 2 || env.regime == 3);

        if (pos == target_food) return 0; // Stay

        uint32_t cw_dist_food  = (target_food  - pos + 16) % 16;
        uint32_t cw_dist_toxin = (target_toxin - pos + 16) % 16;
        uint32_t ccw_dist_food  = 16 - cw_dist_food;
        uint32_t ccw_dist_toxin = (16 - cw_dist_toxin) % 16;

        bool cw_blocked  = (cw_dist_toxin > 0 && cw_dist_toxin < cw_dist_food);
        bool ccw_blocked = (ccw_dist_toxin > 0 && ccw_dist_toxin < ccw_dist_food);

        bool want_right = (cw_dist_food <= ccw_dist_food);
        if (want_right && cw_blocked && !ccw_blocked) want_right = false;
        else if (!want_right && ccw_blocked && !cw_blocked) want_right = true;

        if (!invert) {
            return want_right ? 2 : 1;
        } else {
            return want_right ? 1 : 2;
        }
    }
};

template <typename Agent>
void run_active_benchmark(const std::string& name, Agent& agent, uint64_t total_steps = 100000) {
    ActiveWorld env;
    env.reset(0x9999888877776666ULL);

    uint64_t deaths = 0;
    std::vector<double> regime_rewards(4, 0.0);
    std::vector<uint64_t> regime_steps(4, 0);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t step = 0; step < total_steps; ++step) {
        uint64_t obs = env.observe();
        uint8_t a = 0;

        if constexpr (std::is_same_v<Agent, OracleAgent>) {
            a = agent.act_oracle(env);
        } else {
            a = agent.act(obs);
        }

        double dE = env.step(a);
        uint64_t next_obs = env.observe();

        if constexpr (!std::is_same_v<Agent, OracleAgent>) {
            agent.learn(obs, a, dE, next_obs);
        }

        uint32_t r = env.regime;
        regime_rewards[r] += dE;
        regime_steps[r]++;

        if (!env.alive()) {
            deaths++;
            env.energy = ActiveWorld::E_INIT;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n======================================================================" << std::endl;
    std::cout << " BASELINE BENCHMARK: " << name << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << " Total Steps     : " << total_steps << " | Total Deaths: " << deaths << std::endl;
    std::cout << " Food Eaten      : " << env.food_eaten << " | Toxins Hit: " << env.toxins_hit << std::endl;
    std::cout << " Final Energy    : " << std::fixed << std::setprecision(1) << env.energy << " / " << ActiveWorld::E_MAX << std::endl;
    std::cout << " Execution Speed : " << std::setprecision(2) << ms << " ms (" << (total_steps / (ms / 1000.0)) << " steps/sec)" << std::endl;
    std::cout << " Regime Delta E Breakdown:" << std::endl;
    std::cout << "   Regime 0 (Base Law)       : dE = " << std::setprecision(1) << regime_rewards[0]
              << " (" << regime_steps[0] << " steps)" << std::endl;
    std::cout << "   Regime 1 (Toxin Swap)     : dE = " << regime_rewards[1]
              << " (" << regime_steps[1] << " steps)" << std::endl;
    std::cout << "   Regime 2 (Motor Inversion) : dE = " << regime_rewards[2]
              << " (" << regime_steps[2] << " steps)" << std::endl;
    std::cout << "   Regime 3 (Dual Swap + Inv) : dE = " << regime_rewards[3]
              << " (" << regime_steps[3] << " steps)" << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << " KARYON-CORE EXP-1163a: FROZEN ACTIVE ENVIRONMENT BASELINES" << std::endl;
    std::cout << " Standard: KEP v51.0 | Principles 0, 3, 4, 8, 9, 11, 13, 14, 15" << std::endl;
    std::cout << "======================================================================" << std::endl;

    RandomAgent rand_agent;
    run_active_benchmark("1. RANDOM AGENT (Uniform 2-bit Motors)", rand_agent, 100000);

    ContextHashQAgent q_agent;
    run_active_benchmark("2. CONTEXT HASH Q-AGENT (Clean 12-bit Signal Lookup)", q_agent, 100000);

    OracleAgent oracle_agent;
    run_active_benchmark("3. ORACLE AGENT (Perfect Regime Knowledge)", oracle_agent, 100000);

    return 0;
}
