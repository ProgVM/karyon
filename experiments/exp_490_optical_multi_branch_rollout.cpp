// experiments/exp_490_optical_multi_branch_rollout.cpp
// EXP-490: Optical 256D Multi-Branch Latent Mental Rollout
// Standard: KEP v20.0 Sovereign Master (Principles 2, 5, 7, 27, 28)
//
// Hypothesis:
// Integrating multi-branch latent mental tree rollout directly into the high-dimensional
// continuous optical pixel vision arena (256D retina) will enable Karyon to actively predict
// evasive target orbital paths rather than purely reacting to instantaneous gradient errors,
// increasing visual target interceptions by >= 50% compared to baseline while maintaining
// super-realtime visual processing (> 10,000 FPS).

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <span>

class ContinuousPixelArena {
public:
    static constexpr size_t grid_w = 16;
    static constexpr size_t grid_h = 16;
    static constexpr size_t total_pixels = grid_w * grid_h;

    float px = 0.0f, py = 0.0f;
    float vx = 0.0f, vy = 0.0f;
    float tx = 0.5f, ty = 0.5f;
    float orbit_angle = 0.0f;
    size_t catches = 0;

    std::vector<float> fb_current;
    std::vector<float> fb_goal;
    std::vector<float> fb_next;

    ContinuousPixelArena() {
        fb_current.resize(total_pixels, 0.0f);
        fb_goal.resize(total_pixels, 0.0f);
        fb_next.resize(total_pixels, 0.0f);
        update_target();
    }

    void update_target() {
        orbit_angle += 0.030f; // Evasive target orbit
        tx = 0.65f * std::cos(orbit_angle);
        ty = 0.65f * std::sin(orbit_angle);
    }

    void render_into(std::span<float> fb, float self_x, float self_y, float target_x, float target_y) const {
        const float sigma_sq = 0.04f;
        for (size_t y = 0; y < grid_h; ++y) {
            float world_y = -1.0f + 2.0f * (static_cast<float>(y) / static_cast<float>(grid_h - 1));
            for (size_t x = 0; x < grid_w; ++x) {
                float world_x = -1.0f + 2.0f * (static_cast<float>(x) / static_cast<float>(grid_w - 1));

                float dtx = world_x - target_x;
                float dty = world_y - target_y;
                float target_intensity = std::exp(-(dtx * dtx + dty * dty) / (2.0f * sigma_sq));

                float dpx = world_x - self_x;
                float dpy = world_y - self_y;
                float self_intensity = 0.5f * std::exp(-(dpx * dpx + dpy * dpy) / (2.0f * sigma_sq));

                fb[y * grid_w + x] = std::clamp(target_intensity + self_intensity, 0.0f, 1.0f);
            }
        }
    }

    void render_current() {
        render_into(fb_current, px, py, tx, ty);
    }

    void render_next() {
        render_into(fb_next, px, py, tx, ty);
    }

    void render_goal() {
        render_into(fb_goal, tx, ty, tx, ty);
    }

    bool step_continuous(std::span<const float> u) {
        update_target();
        float force_scale = 0.09f;
        float ax = force_scale * (u[0] - u[1]);
        float ay = force_scale * (u[2] - u[3]);

        vx += ax;
        vy += ay;
        vx *= 0.90f; // Velocity damping
        vy *= 0.90f;

        px += vx;
        py += vy;

        if (px < -1.0f) { px = -1.0f; vx = -vx * 0.5f; }
        if (px > 1.0f)  { px = 1.0f;  vx = -vx * 0.5f; }
        if (py < -1.0f) { py = -1.0f; vy = -vy * 0.5f; }
        if (py > 1.0f)  { py = 1.0f;  vy = -vy * 0.5f; }

        float dx = px - tx;
        float dy = py - ty;
        if (std::sqrt(dx * dx + dy * dy) < 0.25f) {
            catches++;
            orbit_angle += 1.57f; // Phase jump on catch
            update_target();
            return true;
        }
        return false;
    }
};

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-490: OPTICAL 256D MULTI-BRANCH LATENT MENTAL ROLLOUT     ===\n";
    std::cout << "=== Anticipatory Active Inference on High-Dimensional Video      ===\n";
    std::cout << "====================================================================\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 256;      // Full 16x16 pixel retina
    cfg.latent_dim = 32;       // Compressed latent manifold
    cfg.action_dim = 4;        // 4D analog continuous actuation
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 4;      // Deep dynamic mental rollout
    cfg.plasticity_rate = 0.02f;

    karyon::KaryonSovereignCore core(cfg);
    ContinuousPixelArena arena;

    std::vector<float> continuous_action(4, 0.0f);

    size_t total_frames = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t window_catches = 0;
    float window_fe = 0.0f;

    std::cout << "Streaming " << total_frames << " frames through Optical Mental Rollout Pipeline...\n\n";

    for (size_t f = 1; f <= total_frames; ++f) {
        arena.render_current();
        arena.render_goal();

        // 1. Evaluate discrete mental rollout across candidate motor directions to bias continuous steering
        size_t discrete_plan = core.decide_action(arena.fb_current, arena.fb_goal);

        // Map discrete plan into continuous action initialization
        std::fill(continuous_action.begin(), continuous_action.end(), 0.0f);
        if (discrete_plan < 4) {
            continuous_action[discrete_plan] = 0.50f;
        }

        // 2. Refine via analytical continuous adjoint projection
        core.decide_continuous_action(arena.fb_current, arena.fb_goal, continuous_action);

        // 3. Step physical arena
        bool caught = arena.step_continuous(continuous_action);
        if (caught) window_catches++;

        arena.render_next();

        // 4. Update core with observed sensory transition
        auto telem = core.step_environment(arena.fb_current, arena.fb_goal, arena.fb_next, continuous_action);
        window_fe += telem.free_energy;

        if (f % 2500 == 0 || f == total_frames) {
            float avg_fe = window_fe / 2500.0f;
            std::cout << "Frame " << std::setw(5) << f << "/" << total_frames
                      << " | Optical Catches: " << std::setw(4) << arena.catches
                      << " (+" << std::setw(3) << window_catches << " in window)"
                      << " | Optical FE: " << std::fixed << std::setprecision(4) << avg_fe
                      << " | Mental Horizon K: " << core.get_current_horizon()
                      << " | Noradrenaline: " << std::fixed << std::setprecision(3) << core.get_noradrenaline()
                      << " | Discrete Plan: A" << discrete_plan
                      << "\n";
            window_catches = 0;
            window_fe = 0.0f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double fps = static_cast<double>(total_frames) / (elapsed_sec + 1e-6);

    std::cout << "\n=== FINAL TELEMETRY (EXP-490 OPTICAL LATENT ROLLOUT) ===\n";
    std::cout << "Optical Throughput: " << std::fixed << std::setprecision(1) << fps << " Frames/Sec (FPS)\n";
    std::cout << "Target 60 FPS Standard: " << (fps >= 60.0 ? "PASSED (SUPER-REALTIME)" : "FAILED") << "\n";
    std::cout << "Realtime Speedup Factor: " << std::fixed << std::setprecision(1) << (fps / 60.0) << "x\n";
    std::cout << "Total Visual Targets Intercepted: " << arena.catches << " catches\n";
    std::cout << "Elapsed Time: " << std::fixed << std::setprecision(3) << elapsed_sec << " s\n";
    std::cout << "Final Noradrenaline: " << std::fixed << std::setprecision(4) << core.get_noradrenaline() << "\n";

    return 0;
}
