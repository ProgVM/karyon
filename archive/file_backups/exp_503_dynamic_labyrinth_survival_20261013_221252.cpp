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
    static constexpr float ARENA_SIZE = 20.0f;
    static constexpr size_t NUM_HAZARDS = 6;

    // Agent state
    float agent_x = 10.0f;
    float agent_y = 10.0f;
    float agent_vx = 0.0f;
    float agent_vy = 0.0f;

    // Goal state
    float goal_x = 3.0f;
    float goal_y = 17.0f;

    std::vector<HazardOrb> hazards;
    std::mt19937 rng{1337};

    DynamicLabyrinth() {
        reset_hazards();
    }

    void reset_hazards() {
        hazards.clear();
        std::uniform_real_distribution<float> dist_pos(3.0f, 17.0f);
        std::uniform_real_distribution<float> dist_vel(-0.08f, 0.08f);

        for (size_t i = 0; i < NUM_HAZARDS; ++i) {
            HazardOrb orb;
            orb.x = dist_pos(rng);
            orb.y = dist_pos(rng);
            orb.vx = dist_vel(rng);
            orb.vy = dist_vel(rng);
            orb.radius = 1.6f;
            hazards.push_back(orb);
        }
    }

    void respawn_goal() {
        std::uniform_real_distribution<float> dist(2.5f, 17.5f);
        goal_x = dist(rng);
        goal_y = dist(rng);
    }

    // Step physics, returns {pain_event, catch_event, external_valence}
    struct StepResult {
        bool wall_collision;
        bool hazard_collision;
        bool goal_catch;
        float valence;
    };

    StepResult step(float ax, float ay) {
        StepResult res{false, false, false, 0.0f};

        // Inertial dynamics with damping
        agent_vx = agent_vx * 0.85f + ax * 0.18f;
        agent_vy = agent_vy * 0.85f + ay * 0.18f;

        float next_x = agent_x + agent_vx;
        float next_y = agent_y + agent_vy;

        // Perimeter Wall Boundaries
        if (next_x <= 1.0f) {
            next_x = 1.0f;
            agent_vx = -agent_vx * 0.5f;
            res.wall_collision = true;
        } else if (next_x >= ARENA_SIZE - 1.0f) {
            next_x = ARENA_SIZE - 1.0f;
            agent_vx = -agent_vx * 0.5f;
            res.wall_collision = true;
        }

        if (next_y <= 1.0f) {
            next_y = 1.0f;
            agent_vy = -agent_vy * 0.5f;
            res.wall_collision = true;
        } else if (next_y >= ARENA_SIZE - 1.0f) {
            next_y = ARENA_SIZE - 1.0f;
            agent_vy = -agent_vy * 0.5f;
            res.wall_collision = true;
        }

        agent_x = next_x;
        agent_y = next_y;

        // Step dynamic hazards
        for (auto& orb : hazards) {
            orb.x += orb.vx;
            orb.y += orb.vy;

            if (orb.x <= 2.0f || orb.x >= ARENA_SIZE - 2.0f) orb.vx = -orb.vx;
            if (orb.y <= 2.0f || orb.y >= ARENA_SIZE - 2.0f) orb.vy = -orb.vy;

            // Check collision with agent
            float dx = agent_x - orb.x;
            float dy = agent_y - orb.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < orb.radius) {
                res.hazard_collision = true;
            }
        }

        // Check goal interception
        float gdx = agent_x - goal_x;
        float gdy = agent_y - goal_y;
        float gdist = std::sqrt(gdx * gdx + gdy * gdy);
        if (gdist < 1.4f) {
            res.goal_catch = true;
            respawn_goal();
        }

        // Compute valence feedback
        if (res.hazard_collision) {
            res.valence -= 1.0f;
        }
        if (res.wall_collision) {
            res.valence -= 0.8f;
        }
        if (res.goal_catch) {
            res.valence += 1.0f;
        }

        return res;
    }

    // Sensory encoding: 16-channel spatial radar
    // [0..3]: relative goal (dx, dy, dist, angle)
    // [4..7]: agent velocity and wall proximity
    // [8..15]: 8-sector hazard proximity distance radar
    void get_sensory_vector(std::vector<float>& sensory) const {
        sensory.assign(16, 0.0f);

        // Relative goal
        float dx = goal_x - agent_x;
        float dy = goal_y - agent_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        sensory[0] = dx / ARENA_SIZE;
        sensory[1] = dy / ARENA_SIZE;
        sensory[2] = std::clamp(dist / ARENA_SIZE, 0.0f, 1.0f);
        sensory[3] = std::atan2(dy, dx) / 3.14159f;

        // Velocity & Wall proximity
        sensory[4] = agent_vx;
        sensory[5] = agent_vy;
        sensory[6] = (agent_x - 1.0f) / ARENA_SIZE;
        sensory[7] = (agent_y - 1.0f) / ARENA_SIZE;

        // 8-Sector Hazard Radar (1.0 = touch, 0.0 = far)
        for (size_t s = 0; s < 8; ++s) {
            float sector_angle = (static_cast<float>(s) / 8.0f) * 2.0f * 3.14159f - 3.14159f;
            float min_hazard_dist = 100.0f;

            for (const auto& orb : hazards) {
                float ox = orb.x - agent_x;
                float oy = orb.y - agent_y;
                float odist = std::sqrt(ox * ox + oy * oy);
                float oangle = std::atan2(oy, ox);
                float angle_diff = std::abs(oangle - sector_angle);
                if (angle_diff > 3.14159f) angle_diff = 2.0f * 3.14159f - angle_diff;

                if (angle_diff < (3.14159f / 4.0f) && odist < min_hazard_dist) {
                    min_hazard_dist = odist;
                }
            }
            float proximity = std::clamp(1.0f - (min_hazard_dist / 6.0f), 0.0f, 1.0f);
            sensory[8 + s] = proximity;
        }
    }

    void get_ideal_goal_sensory(std::vector<float>& goal_sensory) const {
        goal_sensory.assign(16, 0.0f);
        goal_sensory[0] = 0.0f;
        goal_sensory[1] = 0.0f;
        goal_sensory[2] = 0.0f; // Zero distance to goal
        goal_sensory[3] = 0.0f;
        goal_sensory[4] = 0.0f; // Rest velocity
        goal_sensory[5] = 0.0f;
        goal_sensory[6] = 0.5f; // Safe distance from walls
        goal_sensory[7] = 0.5f;
        // Zero hazard proximity across all 8 sectors
        for (size_t s = 0; s < 8; ++s) {
            goal_sensory[8 + s] = 0.0f;
        }
    }
};

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-503: NON-STATIONARY DYNAMIC LABYRINTH & MOVING HAZARDS   ===\n";
    std::cout << "=== Autonomous Obstacle Avoidance & Goal Interception Evaluation ===\n";
    std::cout << "====================================================================\n\n";

    constexpr size_t TOTAL_STEPS = 30000;

    // Evaluate Condition A: Blind Open-Loop Reactive Baseline (No Peril Barrier)
    std::cout << "Executing Condition A (Reactive Baseline: No Predictive Threat Horizon)...\n";
    karyon::CoreConfig cfg_a;
    cfg_a.sensory_dim = 16;
    cfg_a.latent_dim = 24;
    cfg_a.action_dim = 4;
    cfg_a.min_horizon_k = 1;
    cfg_a.max_horizon_k = 1; // Pure reactive horizon
    cfg_a.aversive_weight = 0.0f; // No threat avoidance
    cfg_a.plasticity_rate = 0.03f;

    karyon::KaryonSovereignCore core_a(cfg_a);
    DynamicLabyrinth arena_a;
    std::vector<float> s_curr(16), s_tgt(16), s_nxt(16), act(4, 0.0f);
    arena_a.get_ideal_goal_sensory(s_tgt);

    size_t catches_a = 0, wall_hits_a = 0, hazard_hits_a = 0;
    double fe_accum_a = 0.0;

    auto t0_a = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        arena_a.get_sensory_vector(s_curr);
        core_a.decide_continuous_action(s_curr, s_tgt, act);

        float ax = act[0] - act[1];
        float ay = act[2] - act[3];

        auto step_res = arena_a.step(ax, ay);
        if (step_res.goal_catch) catches_a++;
        if (step_res.wall_collision) wall_hits_a++;
        if (step_res.hazard_collision) hazard_hits_a++;

        arena_a.get_sensory_vector(s_nxt);
        auto telem = core_a.step_environment(s_curr, s_tgt, s_nxt, act, step_res.valence);
        fe_accum_a += telem.free_energy;
    }
    auto t1_a = std::chrono::high_resolution_clock::now();
    double time_a = std::chrono::duration<double>(t1_a - t0_a).count();

    // Evaluate Condition B: Karyon Native Core v7.7 with Predictive Threat Horizon
    std::cout << "Executing Condition B (Karyon Native Core v7.7 with Predictive Peril Barrier)...\n";
    karyon::CoreConfig cfg_b;
    cfg_b.sensory_dim = 16;
    cfg_b.latent_dim = 24;
    cfg_b.action_dim = 4;
    cfg_b.min_horizon_k = 1;
    cfg_b.max_horizon_k = 5; // Dynamic multi-branch mental rollout
    cfg_b.aversive_weight = 6.0f;
    cfg_b.peril_threshold = 0.50f;
    cfg_b.plasticity_rate = 0.04f;

    karyon::KaryonSovereignCore core_b(cfg_b);
    DynamicLabyrinth arena_b;
    arena_b.get_ideal_goal_sensory(s_tgt);

    size_t catches_b = 0, wall_hits_b = 0, hazard_hits_b = 0;
    double fe_accum_b = 0.0;

    auto t0_b = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        arena_b.get_sensory_vector(s_curr);
        core_b.decide_continuous_action(s_curr, s_tgt, act);

        float ax = act[0] - act[1];
        float ay = act[2] - act[3];

        auto step_res = arena_b.step(ax, ay);
        if (step_res.goal_catch) catches_b++;
        if (step_res.wall_collision) wall_hits_b++;
        if (step_res.hazard_collision) hazard_hits_b++;

        arena_b.get_sensory_vector(s_nxt);
        auto telem = core_b.step_environment(s_curr, s_tgt, s_nxt, act, step_res.valence);
        fe_accum_b += telem.free_energy;
    }
    auto t1_b = std::chrono::high_resolution_clock::now();
    double time_b = std::chrono::duration<double>(t1_b - t0_b).count();

    std::cout << "\n====================================================================\n";
    std::cout << "=== EMPIRICAL COMPARISON & STATISTICAL AUDIT (30,000 TICKS)      ===\n";
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

    float ratio_a = static_cast<float>(catches_a) / std::max(wall_hits_a + hazard_hits_a, 1UL);
    float ratio_b = static_cast<float>(catches_b) / std::max(wall_hits_b + hazard_hits_b, 1UL);
    std::cout << std::left << std::setw(28) << "Mastery Ratio (Goal/Pain):"
              << std::setw(24) << ratio_a
              << std::setw(24) << ratio_b
              << (ratio_b / std::max(ratio_a, 1e-4f)) << "x Mastery\n";

    std::cout << std::left << std::setw(28) << "Average Free Energy:"
              << std::setw(24) << (fe_accum_a / TOTAL_STEPS)
              << std::setw(24) << (fe_accum_b / TOTAL_STEPS)
              << "\n";

    std::cout << std::left << std::setw(28) << "Throughput (Ticks/sec):"
              << std::setw(24) << (TOTAL_STEPS / time_a)
              << std::setw(24) << (TOTAL_STEPS / time_b)
              << "\n";
    std::cout << "====================================================================\n";

    return 0;
}
