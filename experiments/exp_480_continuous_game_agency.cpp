// experiments/exp_480_continuous_game_agency.cpp
// EXP-480: Closed-Loop Active Inference & Real-Time Continuous Game Agency
// Standard: KEP v20.0 Sovereign Master (Principle 7: Action in Continuous Time & Real-World Agency)

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

class ContinuousKineticArena {
public:
    static constexpr size_t state_dim = 8;
    static constexpr size_t num_actions = 4;

    float px = 0.0f, py = 0.0f;
    float vx = 0.0f, vy = 0.0f;

    float tx = 0.5f, ty = 0.5f;
    float hx = 0.0f, hy = 0.0f; // Hazard in center

    float orbit_angle = 0.0f;
    size_t score_catches = 0;
    size_t hazard_collisions = 0;
    size_t ticks_elapsed = 0;

    std::mt19937 rng{2026};

    ContinuousKineticArena() {
        reset_arena();
    }

    void reset_arena() {
        px = -0.7f; py = -0.7f;
        vx = 0.0f;  vy = 0.0f;
        hx = 0.0f;  hy = 0.0f;
        orbit_angle = 0.0f;
        update_target_orbit();
    }

    void update_target_orbit() {
        orbit_angle += 0.02f; // Continuous orbit
        tx = 0.65f * std::cos(orbit_angle);
        ty = 0.65f * std::sin(orbit_angle);
    }

    std::vector<float> get_observation() const {
        return {px, py, vx, vy, tx, ty, hx, hy};
    }

    std::vector<float> get_goal_attractor() const {
        return {tx, ty, 0.0f, 0.0f, tx, ty, hx, hy};
    }

    bool step(size_t action) {
        ticks_elapsed++;
        update_target_orbit();

        float force = 0.08f;
        if (action == 0) vx += force;      // +X
        else if (action == 1) vx -= force; // -X
        else if (action == 2) vy += force; // +Y
        else if (action == 3) vy -= force; // -Y

        // Friction / Inertia
        vx *= 0.92f;
        vy *= 0.92f;

        px += vx;
        py += vy;

        // Boundaries bounce
        if (px < -1.0f) { px = -1.0f; vx = -vx * 0.5f; }
        if (px > 1.0f)  { px = 1.0f;  vx = -vx * 0.5f; }
        if (py < -1.0f) { py = -1.0f; vy = -vy * 0.5f; }
        if (py > 1.0f)  { py = 1.0f;  vy = -vy * 0.5f; }

        // Hazard distance
        float dh_x = px - hx;
        float dh_y = py - hy;
        float dist_hazard = std::sqrt(dh_x * dh_x + dh_y * dh_y);
        if (dist_hazard < 0.20f) {
            hazard_collisions++;
            // Repulsive shock
            vx += (dh_x / (dist_hazard + 1e-4f)) * 0.2f;
            vy += (dh_y / (dist_hazard + 1e-4f)) * 0.2f;
        }

        // Target capture check
        float dt_x = px - tx;
        float dt_y = py - ty;
        float dist_target = std::sqrt(dt_x * dt_x + dt_y * dt_y);
        if (dist_target < 0.22f) {
            score_catches++;
            // Advance orbit jump
            orbit_angle += 1.57f;
            update_target_orbit();
            return true;
        }
        return false;
    }
};

int main() {
    std::cout << "=== EXP-480: Closed-Loop Active Inference & Real-Time Game Agency ===" << std::endl;
    std::cout << "Substrate: Karyon Native Core v7.3 Sovereign\n";
    std::cout << "Game Arena: Continuous 2D Physics (Inertia, Moving Orbital Target, Repulsive Hazard)\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 8;
    cfg.latent_dim = 8;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 4;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 4;
    cfg.plasticity_rate = 0.03f;

    karyon::KaryonSovereignCore core(cfg);
    ContinuousKineticArena arena;

    size_t total_game_ticks = 20000;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t period_catches = 0;
    size_t period_hazards = 0;
    float period_fe = 0.0f;

    std::cout << "Launching Game Session (" << total_game_ticks << " real-time physics ticks)...\n\n";

    for (size_t t = 1; t <= total_game_ticks; ++t) {
        auto obs_t = arena.get_observation();
        auto goal_t = arena.get_goal_attractor();

        // 1. Karyon analyzes sensory state and projects future action trajectory
        size_t action = core.decide_action(obs_t, goal_t);

        // 2. Physics environment step
        bool caught = arena.step(action);
        if (caught) period_catches++;

        auto obs_tp1 = arena.get_observation();

        // 3. Somatic active inference update
        auto telem = core.step_environment(obs_t, goal_t, obs_tp1, action);
        period_fe += telem.free_energy;

        // Periodic telemetry reporting
        if (t % 4000 == 0 || t == total_game_ticks) {
            float avg_fe = period_fe / 4000.0f;
            std::cout << "Tick " << std::setw(5) << t << "/" << total_game_ticks
                      << " | Target Catches: " << std::setw(4) << arena.score_catches
                      << " (+" << std::setw(3) << period_catches << " in window)"
                      << " | Hazard Hits: " << std::setw(3) << arena.hazard_collisions
                      << " | Avg FE: " << std::fixed << std::setprecision(4) << avg_fe
                      << " | Horizon K: " << core.get_current_horizon()
                      << " | Noradrenaline: " << std::fixed << std::setprecision(3) << core.get_noradrenaline()
                      << "\n";

            period_catches = 0;
            period_hazards = 0;
            period_fe = 0.0f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double ticks_per_sec = total_game_ticks / (elapsed_sec + 1e-6);

    std::cout << "\n=== FINAL TELEMETRY (EXP-480 GAME AGENCY) ===\n";
    std::cout << "Execution Speed: " << std::fixed << std::setprecision(1) << ticks_per_sec << " physics ticks/sec\n";
    std::cout << "Total Targets Intercepted: " << arena.score_catches << "\n";
    std::cout << "Total Hazard Collisions: " << arena.hazard_collisions << "\n";
    std::cout << "Catch-to-Hazard Mastery Ratio: " << std::fixed << std::setprecision(2)
              << (float)arena.score_catches / std::max((size_t)1, arena.hazard_collisions) << "x\n";
    std::cout << "Final Interoceptive Noradrenaline: " << std::fixed << std::setprecision(4) << core.get_noradrenaline() << "\n";

    return 0;
}
