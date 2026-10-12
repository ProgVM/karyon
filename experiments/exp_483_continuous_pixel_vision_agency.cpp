// experiments/exp_483_continuous_pixel_vision_agency.cpp
// EXP-483: Continuous Pixel Vision Field & Real-Time Sensorimotor Agency (60+ FPS)
// Standard: KEP v20.0 Sovereign Master (Principle 2, 3, 7: Universal Non-Biological Substrate & Real-World Agency)
//
// Sensory Modality:
// - Raw 2D Continuous Pixel Framebuffer: 16x16 monochromatic optical retinal field = 256 continuous floats.
// - Rendering: Dynamic moving target rendered as a Gaussian intensity blob in pixel space.
// - Zero Convolutional Crutches: Karyon super-primitives ingest raw 256D continuous pixel field directly,
//   compressing visual topology into latent phase space and actuating physical thrusters (4 directions).
// - Closed-loop frame rate benchmark targeting > 60 FPS (real-time video agency).

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

class PixelRetinaArena {
public:
    static constexpr size_t grid_w = 16;
    static constexpr size_t grid_h = 16;
    static constexpr size_t pixel_dim = grid_w * grid_h; // 256D continuous pixel vector
    static constexpr size_t num_actions = 4;

    float px = 0.0f, py = 0.0f;
    float vx = 0.0f, vy = 0.0f;
    float tx = 0.5f, ty = 0.5f;
    float orbit_angle = 0.0f;
    size_t catches = 0;

    PixelRetinaArena() {
        update_target();
    }

    void update_target() {
        orbit_angle += 0.035f;
        tx = 0.65f * std::cos(orbit_angle);
        ty = 0.65f * std::sin(orbit_angle);
    }

    // Renders continuous physical scene into 16x16 raw pixel framebuffer
    // Gaussian intensity: I(x, y) = exp(-((x - obj_x)^2 + (y - obj_y)^2) / (2 * sigma^2))
    std::vector<float> render_pixel_framebuffer() const {
        std::vector<float> fb(pixel_dim, 0.0f);
        float sigma_sq = 0.04f; // Optical Gaussian spread

        for (size_t y = 0; y < grid_h; ++y) {
            float world_y = -1.0f + 2.0f * ((float)y / (float)(grid_h - 1));
            for (size_t x = 0; x < grid_w; ++x) {
                float world_x = -1.0f + 2.0f * ((float)x / (float)(grid_w - 1));

                // Target photon emission (bright blob ~ 1.0)
                float dtx = world_x - tx;
                float dty = world_y - ty;
                float target_intensity = std::exp(-(dtx * dtx + dty * dty) / (2.0f * sigma_sq));

                // Self-embodiment photon marker (fainter blob ~ 0.5)
                float dpx = world_x - px;
                float dpy = world_y - py;
                float self_intensity = 0.5f * std::exp(-(dpx * dpx + dpy * dpy) / (2.0f * sigma_sq));

                fb[y * grid_w + x] = std::clamp(target_intensity + self_intensity, 0.0f, 1.0f);
            }
        }
        return fb;
    }

    // Goal visual state: self blob aligned with target blob (center of retina)
    std::vector<float> get_goal_framebuffer() const {
        std::vector<float> fb(pixel_dim, 0.0f);
        float sigma_sq = 0.04f;
        for (size_t y = 0; y < grid_h; ++y) {
            float world_y = -1.0f + 2.0f * ((float)y / (float)(grid_h - 1));
            for (size_t x = 0; x < grid_w; ++x) {
                float world_x = -1.0f + 2.0f * ((float)x / (float)(grid_w - 1));
                float dtx = world_x - tx;
                float dty = world_y - ty;
                fb[y * grid_w + x] = 1.0f * std::exp(-(dtx * dtx + dty * dty) / (2.0f * sigma_sq));
            }
        }
        return fb;
    }

    bool step(size_t action) {
        update_target();
        float force = 0.11f;
        if (action == 0) vx += force;      // +X
        else if (action == 1) vx -= force; // -X
        else if (action == 2) vy += force; // +Y
        else if (action == 3) vy -= force; // -Y

        vx *= 0.90f;
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
    std::cout << "=== EXP-483: Continuous Pixel Vision Field & Real-Time Sensorimotor Agency ===" << std::endl;
    std::cout << "Substrate: Karyon Native Core v6.0 Sovereign\n";
    std::cout << "Retinal Input: 16x16 Raw Monochromatic Framebuffer (256 continuous floats)\n";
    std::cout << "Target: Orbital Gaussian optical photon emitter in continuous 2D space\n\n";

    karyon::CoreConfig cfg;
    cfg.state_dim = 256;      // Full 16x16 pixel retina ingested directly!
    cfg.num_primitives = 2;
    cfg.num_actions = 4;
    cfg.min_horizon_k = 1;
    cfg.max_horizon_k = 4;
    cfg.plasticity_rate = 0.02f;

    karyon::KaryonSovereignCore core(cfg);
    PixelRetinaArena arena;

    size_t total_frames = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t window_catches = 0;
    float window_fe = 0.0f;

    std::cout << "Streaming " << total_frames << " continuous pixel frames to Karyon retina...\n\n";

    for (size_t f = 1; f <= total_frames; ++f) {
        // 1. Sensory transduction: Render raw 256D pixel array
        auto visual_frame = arena.render_pixel_framebuffer();
        auto visual_goal = arena.get_goal_framebuffer();

        // 2. Active Inference: Karyon perceives raw pixel field and decides motor actuation
        size_t action = core.decide_action(visual_frame, visual_goal);

        // 3. Physical engine step
        bool caught = arena.step(action);
        if (caught) window_catches++;

        auto next_visual_frame = arena.render_pixel_framebuffer();

        // 4. Endoscopic predictive coding across 256D visual field & causal plasticity update
        auto telem = core.step_environment(visual_frame, visual_goal, next_visual_frame, action);
        window_fe += telem.free_energy;

        if (f % 2500 == 0 || f == total_frames) {
            float avg_fe = window_fe / 2500.0f;
            std::cout << "Frame " << std::setw(5) << f << "/" << total_frames
                      << " | Visual Catches: " << std::setw(4) << arena.catches
                      << " (+" << std::setw(3) << window_catches << " in window)"
                      << " | Optical FE: " << std::fixed << std::setprecision(4) << avg_fe
                      << " | Horizon K: " << core.get_current_horizon()
                      << " | Noradrenaline: " << std::fixed << std::setprecision(3) << core.get_noradrenaline()
                      << "\n";
            window_catches = 0;
            window_fe = 0.0f;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double fps = total_frames / (elapsed_sec + 1e-6);

    std::cout << "\n=== FINAL TELEMETRY (EXP-483 PIXEL VISION AGENCY) ===\n";
    std::cout << "Visual Throughput: " << std::fixed << std::setprecision(1) << fps << " Frames/Sec (FPS)\n";
    std::cout << "Target 60 FPS Real-Time Standard: " << (fps >= 60.0 ? "PASSED (SUPER-REALTIME)" : "FAILED") << "\n";
    std::cout << "Speedup relative to 60 FPS: " << std::fixed << std::setprecision(1) << (fps / 60.0f) << "x realtime!\n";
    std::cout << "Total Optically Guided Catches: " << arena.catches << "\n";
    std::cout << "Final Noradrenaline: " << std::fixed << std::setprecision(4) << core.get_noradrenaline() << "\n";

    return 0;
}
