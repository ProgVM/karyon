// experiments/exp_504_sovereign_hormones_autopoiesis.cpp
// EXP-504: Sovereign Endogenous Hormones vs Static Anthropomorphic Baseline
// Standard: KEP v20.0 Sovereign Master (Principles 2, 6, 27, 28)
//
// Evaluates the autonomous genesis and dynamic modulatory regulation of sovereign hormones:
// Condition A: Fixed Static Baseline (No Endocrine Autopoiesis, Static Fixed Modulators)
// Condition B: Sovereign Endogenous Hormones v8.0 with Dynamic Sprouting & Effector Projections
// Environment: Multi-Hazard Evasive Spatial Pursuit Arena (25,000 steps)

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

class SovereignArena {
public:
    static constexpr float ARENA_SIZE = 12.0f;
    static constexpr size_t NUM_HAZARDS = 4;

    float agent_x = 6.0f;
    float agent_y = 6.0f;

    float goal_x = 3.0f;
    float goal_y = 9.0f;

    std::vector<HazardOrb> hazards;
    std::mt19937 rng{4242};

    SovereignArena() {
        reset_hazards();
    }

    void reset_hazards() {
        hazards.clear();
        std::uniform_real_distribution<float> dist_pos(2.5f, 9.5f);
        std::uniform_real_distribution<float> dist_vel(-0.045f, 0.045f);

        for (size_t i = 0; i < NUM_HAZARDS; ++i) {
            HazardOrb orb;
            orb.x = dist_pos(rng);
            orb.y = dist_pos(rng);
            orb.vx = dist_vel(rng);
            orb.vy = dist_vel(rng);
            orb.radius = 1.15f;
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

        // Hazards
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
            res.valence += 0.02f;
        }

        return res;
    }

    std::vector<float> get_sensory() const {
        std::vector<float> s(16, 0.0f);
        float dx = goal_x - agent_x;
        float dy = goal_y - agent_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        s[0] = dx / ARENA_SIZE;
        s[1] = dy / ARENA_SIZE;
        s[2] = std::clamp(dist / ARENA_SIZE, 0.0f, 1.0f);
        s[3] = std::atan2(dy, dx) / 3.14159f;

        // Wall proximity
        s[4] = (ARENA_SIZE - 1.0f - agent_y < 1.5f) ? 1.0f : 0.0f;
        s[5] = (agent_y - 1.0f < 1.5f) ? 1.0f : 0.0f;
        s[6] = (agent_x - 1.0f < 1.5f) ? 1.0f : 0.0f;
        s[7] = (ARENA_SIZE - 1.0f - agent_x < 1.5f) ? 1.0f : 0.0f;

        // Hazard radar
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
        return std::vector<float>(16, 0.0f);
    }
};

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-504: SOVEREIGN ENDOGENOUS HORMONES & AUTOPOIESIS (v8.0)  ===\n";
    std::cout << "=== Auditing Autonomous Dynamic Modulator Genesis in Karyon Core ===\n";
    std::cout << "====================================================================\n\n";

    constexpr size_t TOTAL_STEPS = 25000;

    // Condition A: Static Modulators (Fixed 2 hormones, zero sprouting)
    std::cout << "Running Condition A (Static Modulators: Fixed 2 Hormones, Max=2)...\n";
    karyon::CoreConfig cfg_a;
    cfg_a.sensory_dim = 16;
    cfg_a.latent_dim = 24;
    cfg_a.action_dim = 4;
    cfg_a.initial_hormones = 2;
    cfg_a.max_hormones = 2; // Locked
    cfg_a.min_horizon_k = 1;
    cfg_a.max_horizon_k = 5;
    cfg_a.aversive_weight = 6.0f;
    cfg_a.peril_threshold = 0.50f;
    cfg_a.plasticity_rate = 0.04f;

    karyon::KaryonSovereignCore core_a(cfg_a);
    SovereignArena arena_a;
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
        auto telem = core_a.step_environment(s_curr, s_tgt, s_nxt, a, outcome.valence);
        fe_sum_a += telem.free_energy;
    }
    auto t1_a = std::chrono::high_resolution_clock::now();
    double time_a = std::chrono::duration<double>(t1_a - t0_a).count();

    // Condition B: Sovereign Endogenous Hormones v8.0 with Dynamic Genesis
    std::cout << "Running Condition B (Sovereign Endogenous Hormones v8.0 with Autopoiesis)...\n";
    karyon::CoreConfig cfg_b;
    cfg_b.sensory_dim = 16;
    cfg_b.latent_dim = 24;
    cfg_b.action_dim = 4;
    cfg_b.initial_hormones = 3;
    cfg_b.max_hormones = 6; // Dynamic morphogenetic ceiling
    cfg_b.min_horizon_k = 1;
    cfg_b.max_horizon_k = 5;
    cfg_b.aversive_weight = 6.0f;
    cfg_b.peril_threshold = 0.50f;
    cfg_b.plasticity_rate = 0.04f;
    cfg_b.sprouting_fe_threshold = 0.08f;
    cfg_b.sprouting_cooldown_ticks = 800;

    karyon::KaryonSovereignCore core_b(cfg_b);
    SovereignArena arena_b;
    size_t catches_b = 0, wall_hits_b = 0, hazard_hits_b = 0;
    float fe_sum_b = 0.0f;
    size_t hormone_sprouts_b = 0;

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
        if (telem.event_hormone_sprouted) hormone_sprouts_b++;
    }
    auto t1_b = std::chrono::high_resolution_clock::now();
    double time_b = std::chrono::duration<double>(t1_b - t0_b).count();

    std::cout << "\n====================================================================\n";
    std::cout << "=== EMPIRICAL COMPARISON & STATISTICAL AUDIT (25,000 TICKS)      ===\n";
    std::cout << "====================================================================\n";
    std::cout << std::left << std::setw(28) << "Metric"
              << std::setw(24) << "Condition A (Fixed)"
              << std::setw(24) << "Condition B (Sovereign v8.0)"
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

    std::cout << std::left << std::setw(28) << "Final Active Hormones:"
              << std::setw(24) << core_a.get_hormones_count()
              << std::setw(24) << core_b.get_hormones_count()
              << "+" << (core_b.get_hormones_count() - core_a.get_hormones_count()) << " Axes\n";

    std::cout << std::left << std::setw(28) << "Hormone Sprout Events:"
              << std::setw(24) << 0
              << std::setw(24) << hormone_sprouts_b
              << "\n";

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
