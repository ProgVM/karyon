// experiments/exp_485_v7_supersonic_pixel_vision.cpp
// EXP-485: Karyon Native Core v7.0 Continuous Pixel Vision & Supersonic FPS Verification
// Standard: KEP v20.0 Sovereign Master (Principles 2, 5, 7, 27)

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
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
        orbit_angle += 0.035f;
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
        vx *= 0.90f; // Damping
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
            orbit_angle += 1.57f;
            update_target();
            return true;
        }
        return false;
    }
};

int main() {
    std::cout << "=== EXP-485: Karyon Native Core v7.3 Continuous Pixel Vision Benchmark ===\n";
    std::cout << "Sensory: 256D Optical Pixel Retina (16x16 Raw Framebuffer)\n";
    std::cout << "Motor: 4D Continuous Analog Thrusters via Hamiltonian Least-Action\n";
    std::cout << "Memory Policy: 100% Zero-Allocation Scratchpad\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 256;      // Full 16x16 pixel retina
    cfg.initial_latent_dim = 32;
    cfg.max_latent_dim = 64;       // Compressed latent manifold
    cfg.action_dim = 4;        // 4D analog continuous actuation
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 3;
    cfg.plasticity_rate = 0.02f;
    // cfg.hamiltonian_steps = 2; // Fast 2-step gradient steering in mental sandbox

    karyon::KaryonSovereignCore core(cfg);
    ContinuousPixelArena arena;

    std::vector<float> continuous_action(4, 0.0f);

    size_t total_frames = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t window_catches = 0;
    float window_fe = 0.0f;

    std::cout << "Streaming " << total_frames << " frames through Core v7.3 zero-allocation pipeline...\n\n";

    for (size_t f = 1; f <= total_frames; ++f) {
        arena.render_current();
        arena.render_goal();

        core.decide_continuous_action(arena.fb_current, arena.fb_goal, continuous_action);

        bool caught = arena.step_continuous(continuous_action);
        if (caught) window_catches++;

        arena.render_next();

        float valence = caught ? 1.0f : 0.0f;
        auto telem = core.step_environment(arena.fb_current, arena.fb_goal, arena.fb_next, continuous_action, valence);
        window_fe += telem.free_energy;

        if (f % 2500 == 0 || f == total_frames) {
            float avg_fe = window_fe / 2500.0f;
            std::cout << "Frame " << std::setw(5) << f << "/" << total_frames
                      << " | Visual Catches: " << std::setw(4) << arena.catches
                      << " (+" << std::setw(3) << window_catches << " in window)"
                      << " | Optical FE: " << std::fixed << std::setprecision(4) << avg_fe
                      << " | Horizon K: " << core.get_current_horizon()
                      << " | Plasticity: " << std::fixed << std::setprecision(3) << core.get_effective_plasticity()
                      << " | Thruster u0: " << std::fixed << std::setprecision(2) << continuous_action[0]
                      << "\n";
            window_catches = 0;
            window_fe = 0.0f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double fps = static_cast<double>(total_frames) / (elapsed_sec + 1e-6);

    std::cout << "\n=== FINAL TELEMETRY (EXP-485 SUPERSONIC VISION v7.3) ===\n";
    std::cout << "Visual Throughput: " << std::fixed << std::setprecision(1) << fps << " Frames/Sec (FPS)\n";
    std::cout << "Target 60 FPS Standard: " << (fps >= 60.0 ? "PASSED (SUPER-REALTIME)" : "FAILED") << "\n";
    std::cout << "Realtime Speedup Factor: " << std::fixed << std::setprecision(1) << (fps / 60.0) << "x\n";
    std::cout << "Total Optically Guided Catches: " << arena.catches << "\n";
    std::cout << "Elapsed Time: " << std::fixed << std::setprecision(3) << elapsed_sec << " s\n";

    return 0;
}
