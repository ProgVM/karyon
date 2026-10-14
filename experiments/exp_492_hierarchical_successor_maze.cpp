// experiments/exp_492_hierarchical_successor_maze.cpp
// EXP-492: Dual-Level Active Inference & Continuous Successor Representation in Non-Convex Maze
// Auditing true spatial pathfinding mastery and barrier avoidance under KEP Principle 28

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include "karyon_native_core.h"

struct Wall {
    float x1, y1, x2, y2;
};

struct DynamicThreat {
    float x, y;
    float vx, vy;
    float radius;
};

class ComplexNonlinearMaze {
public:
    static constexpr float ARENA_W = 30.0f;
    static constexpr float ARENA_H = 30.0f;

    float agent_x = 2.0f;
    float agent_y = 2.0f;
    float agent_vx = 0.0f;
    float agent_vy = 0.0f;

    float goal_x = 28.0f;
    float goal_y = 28.0f;

    std::vector<Wall> walls;
    std::vector<DynamicThreat> threats;
    std::mt19937 rng{42};

    ComplexNonlinearMaze() {
        init_maze();
    }

    void init_maze() {
        // Multi-chamber non-convex maze with dead ends and narrow corridors
        // Boundary walls
        walls.push_back({0.0f, 0.0f, ARENA_W, 0.0f});
        walls.push_back({0.0f, ARENA_H, ARENA_W, ARENA_H});
        walls.push_back({0.0f, 0.0f, 0.0f, ARENA_H});
        walls.push_back({ARENA_W, 0.0f, ARENA_W, ARENA_H});

        // Chamber 1 dividers with dead-end pocket
        walls.push_back({10.0f, 0.0f, 10.0f, 20.0f}); // Vertical barrier 1
        walls.push_back({5.0f, 15.0f, 10.0f, 15.0f});  // Dead end pocket wall
        walls.push_back({10.0f, 25.0f, 10.0f, 30.0f}); // Upper passage at [20, 25]

        // Chamber 2 dividers with central obstacle
        walls.push_back({20.0f, 10.0f, 20.0f, 30.0f}); // Vertical barrier 2
        walls.push_back({10.0f, 10.0f, 16.0f, 10.0f}); // Dead end corridor
        walls.push_back({14.0f, 18.0f, 20.0f, 18.0f}); // Barrier pocket

        // Dynamic wandering threats
        threats.clear();
        std::uniform_real_distribution<float> p_dist(5.0f, 25.0f);
        std::uniform_real_distribution<float> v_dist(-0.10f, 0.10f);
        for (size_t i = 0; i < 4; ++i) {
            threats.push_back({p_dist(rng), p_dist(rng), v_dist(rng), v_dist(rng), 1.8f});
        }
    }

    void respawn_goal() {
        std::vector<std::pair<float, float>> goal_spots = {
            {28.0f, 28.0f}, {2.0f, 28.0f}, {28.0f, 2.0f}, {15.0f, 15.0f}, {25.0f, 12.0f}
        };
        std::uniform_int_distribution<size_t> d(0, goal_spots.size() - 1);
        auto spot = goal_spots[d(rng)];
        goal_x = spot.first;
        goal_y = spot.second;
    }

    struct StepOutcome {
        bool wall_hit;
        bool threat_hit;
        bool goal_reached;
        float valence;
    };

    float dist_to_segment(float px, float py, float x1, float y1, float x2, float y2) const {
        float l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        if (l2 == 0.0f) return std::sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1));
        float t = std::max(0.0f, std::min(1.0f, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2));
        float proj_x = x1 + t * (x2 - x1);
        float proj_y = y1 + t * (y2 - y1);
        return std::sqrt((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y));
    }

    StepOutcome step(size_t discrete_action) {
        StepOutcome outcome{false, false, false, 0.0f};

        // 4 discrete actions: 0: +X, 1: -X, 2: +Y, 3: -Y
        float ax = 0.0f, ay = 0.0f;
        float force = 0.45f;
        if (discrete_action == 0) ax = force;
        else if (discrete_action == 1) ax = -force;
        else if (discrete_action == 2) ay = force;
        else if (discrete_action == 3) ay = -force;

        agent_vx = agent_vx * 0.82f + ax * 0.20f;
        agent_vy = agent_vy * 0.82f + ay * 0.20f;

        float nx = agent_x + agent_vx;
        float ny = agent_y + agent_vy;

        for (const auto& w : walls) {
            if (dist_to_segment(nx, ny, w.x1, w.y1, w.x2, w.y2) < 0.8f) {
                outcome.wall_hit = true;
                agent_vx = -agent_vx * 0.4f;
                agent_vy = -agent_vy * 0.4f;
                nx = agent_x;
                ny = agent_y;
                break;
            }
        }

        agent_x = std::clamp(nx, 0.5f, ARENA_W - 0.5f);
        agent_y = std::clamp(ny, 0.5f, ARENA_H - 0.5f);

        for (auto& th : threats) {
            th.x += th.vx;
            th.y += th.vy;
            if (th.x <= 2.0f || th.x >= ARENA_W - 2.0f) th.vx = -th.vx;
            if (th.y <= 2.0f || th.y >= ARENA_H - 2.0f) th.vy = -th.vy;

            float dtx = agent_x - th.x;
            float dty = agent_y - th.y;
            if (std::sqrt(dtx * dtx + dty * dty) < th.radius) {
                outcome.threat_hit = true;
            }
        }

        float gdx = agent_x - goal_x;
        float gdy = agent_y - goal_y;
        if (std::sqrt(gdx * gdx + gdy * gdy) < 1.8f) {
            outcome.goal_reached = true;
            respawn_goal();
        }

        if (outcome.goal_reached) outcome.valence += 1.0f;
        if (outcome.wall_hit) outcome.valence -= 0.6f;
        if (outcome.threat_hit) outcome.valence -= 1.0f;

        return outcome;
    }

    // Sensory encoding:
    // [0..3]: relative displacement (dx/W, dy/H, dist/W, angle)
    // [4..7]: velocities and nearest wall proximity
    // [8..15]: 8-sector distance radar
    void get_sensory(std::vector<float>& sensory) const {
        sensory.assign(16, 0.0f);

        float dx = goal_x - agent_x;
        float dy = goal_y - agent_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        sensory[0] = dx / ARENA_W;
        sensory[1] = dy / ARENA_H;
        sensory[2] = std::clamp(dist / ARENA_W, 0.0f, 1.0f);
        sensory[3] = std::atan2(dy, dx) / 3.14159f;

        sensory[4] = agent_vx;
        sensory[5] = agent_vy;

        // 8-Ray LiDAR
        for (size_t s = 0; s < 8; ++s) {
            float angle = (static_cast<float>(s) / 8.0f) * 2.0f * 3.14159f - 3.14159f;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float min_d = 8.0f;

            for (float r = 0.5f; r <= 8.0f; r += 0.5f) {
                float rx = agent_x + cos_a * r;
                float ry = agent_y + sin_a * r;

                bool hit = false;
                for (const auto& w : walls) {
                    if (dist_to_segment(rx, ry, w.x1, w.y1, w.x2, w.y2) < 0.6f) {
                        min_d = r;
                        hit = true;
                        break;
                    }
                }
                if (hit) break;

                for (const auto& th : threats) {
                    float tdx = rx - th.x;
                    float tdy = ry - th.y;
                    if (std::sqrt(tdx * tdx + tdy * tdy) < th.radius) {
                        min_d = r;
                        hit = true;
                        break;
                    }
                }
                if (hit) break;
            }
            sensory[8 + s] = 1.0f - (min_d / 8.0f);
        }
    }

    void get_ideal_goal(std::vector<float>& goal_s) const {
        goal_s.assign(16, 0.0f);
        // Desired relative state: dx=0, dy=0, dist=0, vx=0, vy=0, clear radar
    }
};

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-492: HIERARCHICAL ACTIVE INFERENCE & SUCCESSOR DIFFUSION ===\n";
    std::cout << "=== Validating Non-Convex Spatial Pathfinding in Multi-Chamber   ===\n";
    std::cout << "====================================================================\n\n";

    constexpr size_t TOTAL_STEPS = 50000;

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 16;
    cfg.initial_latent_dim = 16;
    cfg.max_latent_dim = 32;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 6;
    cfg.plasticity_rate = 0.03f;
    cfg.metric_curvature_scale = 10.0f;

    karyon::KaryonSovereignCore core(cfg);
    ComplexNonlinearMaze maze;

    std::vector<float> s_curr(16), s_tgt(16), s_nxt(16);
    maze.get_ideal_goal(s_tgt);

    size_t goals_achieved = 0;
    size_t wall_hits = 0;
    size_t threat_hits = 0;
    double cumulative_fe = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        maze.get_sensory(s_curr);
        size_t action = core.decide_action(s_curr, s_tgt);

        auto outcome = maze.step(action);
        if (outcome.goal_reached) goals_achieved++;
        if (outcome.wall_hit) wall_hits++;
        if (outcome.threat_hit) threat_hits++;

        maze.get_sensory(s_nxt);
        auto telem = core.step_environment(s_curr, s_tgt, s_nxt, action, outcome.valence);
        cumulative_fe += telem.free_energy;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(TOTAL_STEPS) / elapsed;

    std::cout << "=== BENCHMARK TELEMETRY RESULTS ===\n";
    std::cout << "Total Steps Executed:       " << TOTAL_STEPS << "\n";
    std::cout << "Elapsed Time:               " << elapsed << " s\n";
    std::cout << "Throughput:                 " << std::fixed << std::setprecision(1) << throughput << " steps/s\n";
    std::cout << "Goals Solved:               " << goals_achieved << " (vs EXP-491 baseline: 2 goals)\n";
    std::cout << "Wall Collisions:            " << wall_hits << " (" << std::fixed << std::setprecision(1) << (static_cast<double>(wall_hits) * 100.0 / TOTAL_STEPS) << "% of steps)\n";
    std::cout << "Dynamic Threat Hits:        " << threat_hits << "\n";
    std::cout << "Average Free Energy:        " << std::fixed << std::setprecision(5) << (cumulative_fe / TOTAL_STEPS) << " nats\n";
    std::cout << "Final Latent Dimension:     " << core.get_current_latent_dim() << "\n";
    std::cout << "Active Super-Primitives:    " << core.get_primitives_count() << "\n";
    std::cout << "Active Sovereign Hormones:  " << core.get_hormones_count() << "\n";
    std::cout << "Average Epigenetic Lock:    " << std::fixed << std::setprecision(3) << core.get_average_epigenetic_lock() << "\n\n";

    double goal_rate = (static_cast<double>(goals_achieved) * 1000.0) / TOTAL_STEPS;
    std::cout << "Goal Rate:                  " << std::fixed << std::setprecision(2) << goal_rate << " goals / 1000 steps\n";

    if (goals_achieved >= 15 && throughput > 10000.0) {
        std::cout << "VERDICT: POSITIVE (AUTHENTIC NON-CONVEX MAZE PROBLEM SOLVING)\n";
    } else {
        std::cout << "VERDICT: REJECTED (INSUFFICIENT AUTONOMOUS NAVIGATION)\n";
    }
    std::cout << "====================================================================\n";

    return 0;
}
