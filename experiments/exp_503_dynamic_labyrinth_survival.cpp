// experiments/exp_503_dynamic_labyrinth_survival.cpp
// EXP-503: Non-Stationary Dynamic Labyrinth with Moving Hazard Orbs
// Evaluates Karyon Native Core v7.7 on spatial navigation, dead-end avoidance, and dynamic obstacle evasion.
// Standard: KEP v20.0 Sovereign Master (Strict Principles 2 to 28 Compliant)

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

struct HazardOrb {
    float x, y;
    float vx, vy;
    float radius;
};

class DynamicLabyrinth {
public:
    static constexpr float ARENA_SIZE = 12.0f;
    static constexpr size_t NUM_HAZARDS = 4;

    // Agent state
    float agent_x = 6.0f;
    float agent_y = 6.0f;

    // Goal state
    float goal_x = 2.0f;
    float goal_y = 10.0f;

    std::vector<HazardOrb> hazards;
    std::mt19937 rng{1337};

    DynamicLabyrinth() {
        reset_hazards();
    }

    void reset_hazards() {
        hazards.clear();
        std::uniform_real_distribution<float> dist_pos(2.5f, 9.5f);
        std::uniform_real_distribution<float> dist_vel(-0.04f, 0.04f);

        for (size_t i = 0; i < NUM_HAZARDS; ++i) {
            HazardOrb orb;
            orb.x = dist_pos(rng);
            orb.y = dist_pos(rng);
            orb.vx = dist_vel(rng);
            orb.vy = dist_vel(rng);
            orb.radius = 1.2f;
            hazards.push_back(orb);
        }
    }

    void respawn_goal() {
        std::uniform_real_distribution<float> dist(2.0f, 10.0f);
        goal_x = dist(rng);
        goal_y = dist(rng);
    }

    struct StepResult {
        bool wall_collision;
        bool hazard_collision;
        bool goal_catch;
        float valence;
    };

    StepResult step(size_t action) {
        StepResult res{false, false, false, 0.0f};

        float step_dist = 0.22f;
        float nx = agent_x;
        float ny = agent_y;

        // Actions: 0=Up, 1=Down, 2=Left, 3=Right
        if (action == 0) ny += step_dist;
        else if (action == 1) ny -= step_dist;
        else if (action == 2) nx -= step_dist;
        else if (action == 3) nx += step_dist;

        // Wall collisions
        if (nx <= 1.0f || nx >= ARENA_SIZE - 1.0f || ny <= 1.0f || ny >= ARENA_SIZE - 1.0f) {
            res.wall_collision = true;
            res.valence -= 1.0f;
        } else {
            agent_x = nx;
            agent_y = ny;
        }

        // Step dynamic hazards
        for (auto& orb : hazards) {
            orb.x += orb.vx;
            orb.y += orb.vy;

            if (orb.x <= 1.8f || orb.x >= ARENA_SIZE - 1.8f) orb.vx = -orb.vx;
            if (orb.y <= 1.8f || orb.y >= ARENA_SIZE - 1.8f) orb.vy = -orb.vy;

            float dx = agent_x - orb.x;
            float dy = agent_y - orb.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < orb.radius) {
                res.hazard_collision = true;
                res.valence -= 1.2f;
            }
        }

        // Goal check
        float gdx = agent_x - goal_x;
        float gdy = agent_y - goal_y;
        float gdist = std::sqrt(gdx * gdx + gdy * gdy);
        if (gdist < 0.65f) {
            res.goal_catch = true;
            res.valence += 1.0f;
            respawn_goal();
        } else if (!res.wall_collision && !res.hazard_collision) {
            res.valence += 0.02f; // Mild living incentive
        }

        return res;
    }

    // Sensory: 16 channels
    // 0..3: Goal relative (dx, dy, dist, angle)
    // 4..7: Wall boundaries proximity (North, South, West, East)
    // 8..15: 8-sector dynamic hazard proximity
    std::vector<float> get_sensory() const {
        std::vector<float> s(16, 0.0f);
        float dx = goal_x - agent_x;
        float dy = goal_y - agent_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        s[0] = dx / ARENA_SIZE;
        s[1] = dy / ARENA_SIZE;
        s[2] = std::clamp(dist / ARENA_SIZE, 0.0f, 1.0f);
        s[3] = std::atan2(dy, dx) / 3.14159f;

        // Proximity to walls
        s[4] = (ARENA_SIZE - 1.0f - agent_y < 1.5f) ? 1.0f : 0.0f; // North
        s[5] = (agent_y - 1.0f < 1.5f) ? 1.0f : 0.0f;              // South
        s[6] = (agent_x - 1.0f < 1.5f) ? 1.0f : 0.0f;              // West
        s[7] = (ARENA_SIZE - 1.0f - agent_x < 1.5f) ? 1.0f : 0.0f; // East

        // Hazards 8-sector radar
        for (size_t sec = 0; sec < 8; ++sec) {
            float sec_angle = (static_cast<float>(sec) / 8.0f) * 2.0f * 3.14159f - 3.14159f;
            float min_hdist = 100.0f;
            for (const auto& orb : hazards) {
                float ox = orb.x - agent_x;
                float oy = orb.y - agent_y;
                float odist = std::sqrt(ox * ox + oy * oy);
                float oangle = std::atan2(oy, ox);
                float adiff = std::abs(oangle - sec_angle);
                if (adiff > 3.14159f) adiff = 2.0f * 3.14159f - adiff;
                if (adiff < (3.14159f / 4.0f) && odist < min_hdist) {
                    min_hdist = odist;
                }
            }
            s[8 + sec] = std::clamp(1.0f - (min_hdist / 3.5f), 0.0f, 1.0f);
        }
        return s;
    }

    std::vector<float> get_goal_state() const {
        std::vector<float> g(16, 0.0f);
        // All zeros: at target, zero wall peril, zero hazard proximity
        return g;
    }
};

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-503: NON-STATIONARY DYNAMIC LABYRINTH & MOVING HAZARDS   ===\n";
    std::cout << "=== Evaluating Karyon's Spatial Navigation & Threat Avoidance    ===\n";
    std::cout << "====================================================================\n\n";

    constexpr size_t TOTAL_STEPS = 25000;

    // Condition A: Baseline (No Predictive Threat Horizon, Blind FE)
    std::cout << "Running Condition A (Reactive Baseline: No Predictive Threat Horizon)...\n";
    karyon::CoreConfig cfg_a;
    cfg_a.sensory_dim = 16;
    cfg_a.latent_dim = 24;
    cfg_a.action_dim = 4;
    cfg_a.min_horizon_k = 1;
    cfg_a.max_horizon_k = 1;
    cfg_a.aversive_weight = 0.0f;
    cfg_a.plasticity_rate = 0.03f;

    karyon::KaryonSovereignCore core_a(cfg_a);
    DynamicLabyrinth arena_a;
    size_t catches_a = 0, wall_hits_a = 0, hazard_hits_a = 0;
    float fe_sum_a = 0.0f;

    auto t0_a = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        auto s_curr = arena_a.get_sensory();
        auto s_tgt = arena_a.get_goal_state();

        size_t a = core_a.decide_action(s_curr, s_tgt);
        auto outcome = arena_a.step(a);

        if (outcome.goal_catch) catches_a++;
        if (outcome.wall_collision) wall_hits_a++;
        if (outcome.hazard_collision) hazard_hits_a++;

        auto s_nxt = arena_a.get_sensory();
        auto telem = core_a.step_environment(s_curr, s_tgt, s_nxt, a, 0.0f); // Blind
        fe_sum_a += telem.free_energy;
    }
    auto t1_a = std::chrono::high_resolution_clock::now();
    double time_a = std::chrono::duration<double>(t1_a - t0_a).count();

    // Condition B: Native Core v7.7 with Predictive Threat Horizon
    std::cout << "Running Condition B (Karyon Native Core v7.7 with Predictive Peril Barrier)...\n";
    karyon::CoreConfig cfg_b;
    cfg_b.sensory_dim = 16;
    cfg_b.latent_dim = 24;
    cfg_b.action_dim = 4;
    cfg_b.min_horizon_k = 1;
    cfg_b.max_horizon_k = 5;
    cfg_b.aversive_weight = 6.0f;
    cfg_b.peril_threshold = 0.50f;
    cfg_b.plasticity_rate = 0.04f;

    karyon::KaryonSovereignCore core_b(cfg_b);
    DynamicLabyrinth arena_b;
    size_t catches_b = 0, wall_hits_b = 0, hazard_hits_b = 0;
    float fe_sum_b = 0.0f;

    auto t0_b = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        auto s_curr = arena_b.get_sensory();
        auto s_tgt = arena_b.get_goal_state();

        size_t a = core_b.decide_action(s_curr, s_tgt);
        auto outcome = arena_b.step(a);

        if (outcome.goal_catch) catches_b++;
        if (outcome.wall_collision) wall_hits_b++;
        if (outcome.hazard_collision) hazard_hits_b++;

        auto s_nxt = arena_b.get_sensory();
        auto telem = core_b.step_environment(s_curr, s_tgt, s_nxt, a, outcome.valence);
        fe_sum_b += telem.free_energy;
    }
    auto t1_b = std::chrono::high_resolution_clock::now();
    double time_b = std::chrono::duration<double>(t1_b - t0_b).count();

    std::cout << "\n====================================================================\n";
    std::cout << "=== EMPIRICAL COMPARISON & STATISTICAL AUDIT (25,000 TICKS)      ===\n";
    std::cout << "====================================================================\n";
    std::cout << std::left << std::setw(28) << "Metric"
              << std::setw(24) << "Condition A (Reactive)"
              << std::setw(24) << "Condition B (v7.7 Peril)"
              << "Delta (Advantage)\n";
    std::cout << std::string(88, '-') << "\n";

    std::cout << std::left << std::setw(28) << "Target Goal Catches:"
              << std::setw(24) << catches_a
              << std::setw(24) << catches_b
              << "+" << (catches_b >= catches_a ? catches_b - catches_a : 0) << "\n";

    std::cout << std::left << std::setw(28) << "Wall Collisions:"
              << std::setw(24) << wall_hits_a
              << std::setw(24) << wall_hits_b
              << "-" << (wall_hits_a >= wall_hits_b ? wall_hits_a - wall_hits_b : 0)
              << " (" << (1.0f - static_cast<float>(wall_hits_b) / std::max(wall_hits_a, 1UL)) * 100.0f << "%)\n";

    std::cout << std::left << std::setw(28) << "Moving Hazard Hits:"
              << std::setw(24) << hazard_hits_a
              << std::setw(24) << hazard_hits_b
              << "-" << (hazard_hits_a >= hazard_hits_b ? hazard_hits_a - hazard_hits_b : 0)
              << " (" << (1.0f - static_cast<float>(hazard_hits_b) / std::max(hazard_hits_a, 1UL)) * 100.0f << "%)\n";

    float pain_a = static_cast<float>(wall_hits_a + hazard_hits_a);
    float pain_b = static_cast<float>(wall_hits_b + hazard_hits_b);
    float ratio_a = static_cast<float>(catches_a) / std::max(pain_a, 1.0f);
    float ratio_b = static_cast<float>(catches_b) / std::max(pain_b, 1.0f);
    std::cout << std::left << std::setw(28) << "Mastery Ratio (Goal/Pain):"
              << std::setw(24) << ratio_a
              << std::setw(24) << ratio_b
              << (ratio_b / std::max(ratio_a, 1e-4f)) << "x Mastery\n";

    std::cout << std::left << std::setw(28) << "Average Free Energy:"
              << std::setw(24) << (fe_sum_a / TOTAL_STEPS)
              << std::setw(24) << (fe_sum_b / TOTAL_STEPS)
              << "\n";

    std::cout << std::left << std::setw(28) << "Throughput (Ticks/sec):"
              << std::setw(24) << (TOTAL_STEPS / time_a)
              << std::setw(24) << (TOTAL_STEPS / time_b)
              << "\n";
    std::cout << "====================================================================\n";

    return 0;
}
