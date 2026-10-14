// experiments/exp_488_rigorous_behavioral_validation.cpp
// EXP-488: Rigorous Behavioral & Quality Validation of Karyon Native Core v7.4
// Standard: KEP v20.0 Sovereign Master (Principles 2, 5, 7, 27)

#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>

struct VisualArenaQuality {
    size_t total_frames = 10000;
    size_t catches = 0;
    double avg_distance_to_target = 0.0;
    double cumulative_fe = 0.0;
    double fps = 0.0;
};

VisualArenaQuality test_visual_behavioral_quality(karyon::KaryonSovereignCore& core) {
    VisualArenaQuality res{};

    float px = 0.0f, py = 0.0f;
    float tx = 0.5f, ty = 0.5f;
    float vx = 0.0f, vy = 0.0f;
    float orbit_angle = 0.0f;

    std::vector<float> fb_curr(256, 0.0f);
    std::vector<float> fb_goal(256, 0.0f);
    std::vector<float> fb_next(256, 0.0f);
    std::vector<float> action(4, 0.0f);

    const size_t grid_h = 16;
    const size_t grid_w = 16;
    const float sigma_sq = 0.04f;

    auto render_into = [&](std::vector<float>& fb, float self_x, float self_y, float target_x, float target_y) {
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
    };

    double total_dist = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t f = 1; f <= res.total_frames; ++f) {
        // Target dynamic orbit
        orbit_angle += 0.02f;
        tx = 0.65f * std::cos(orbit_angle);
        ty = 0.65f * std::sin(orbit_angle);

        render_into(fb_curr, px, py, tx, ty);
        render_into(fb_goal, tx, ty, tx, ty);

        // Core analog motor steering
        core.decide_continuous_action(fb_curr, fb_goal, action);

        // Physics step: thrusters control acceleration
        float force_scale = 0.09f;
        float ax = force_scale * (action[0] - action[1]);
        float ay = force_scale * (action[2] - action[3]);
        vx += ax;
        vy += ay;
        vx *= 0.90f;
        vy *= 0.90f;
        px += vx;
        py += vy;

        if (px < -1.0f) { px = -1.0f; vx = -vx * 0.5f; }
        if (px > 1.0f)  { px = 1.0f;  vx = -vx * 0.5f; }
        if (py < -1.0f) { py = -1.0f; vy = -vy * 0.5f; }
        if (py > 1.0f)  { py = 1.0f;  vy = -vy * 0.5f; }

        render_into(fb_next, px, py, tx, ty);

        // Measure true behavioral Euclidean distance
        float dx = px - tx;
        float dy = py - ty;
        float dist = std::sqrt(dx * dx + dy * dy);
        total_dist += dist;

        if (dist < 0.25f) { // Interception radius in normalized space [-1, 1]
            res.catches++;
            orbit_angle += 1.57f;
        }

        float valence = (dist < 0.25f) ? 1.0f : 0.0f;
        auto telem = core.step_environment(fb_curr, fb_goal, fb_next, action, valence);
        res.cumulative_fe += telem.free_energy;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    res.fps = static_cast<double>(res.total_frames) / (elapsed + 1e-6);
    res.avg_distance_to_target = total_dist / static_cast<double>(res.total_frames);

    return res;
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-488: RIGOROUS QUALITY AUDIT OF KARYON NATIVE CORE v7.4   ===\n";
    std::cout << "=== Ground-Truth Continuous Pursuit Quality (Real Metric Space)  ===\n";
    std::cout << "====================================================================\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 256;
    cfg.initial_latent_dim = 32;
    cfg.max_latent_dim = 64;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.plasticity_rate = 0.02f;

    karyon::KaryonSovereignCore core(cfg);

    std::cout << "[1] Auditing Behavioral Quality in Closed-Loop Pixel Pursuit Arena...\n";
    auto quality = test_visual_behavioral_quality(core);

    std::cout << "  - Frames Evaluated:         " << quality.total_frames << "\n";
    std::cout << "  - Processing Speed:         " << std::fixed << std::setprecision(1) << quality.fps << " FPS\n";
    std::cout << "  - Mean Normalized Distance: " << std::fixed << std::setprecision(3) << quality.avg_distance_to_target << " (Space: [-1, 1] x [-1, 1])\n";
    std::cout << "  - Target Interceptions:     " << quality.catches << " CATCHES\n";
    std::cout << "  - Interception Frequency:   " << std::fixed << std::setprecision(2) << (quality.catches * 1000.0 / quality.total_frames) << " catches / 1000 frames\n";
    std::cout << "  - Average Free Energy:      " << std::fixed << std::setprecision(5) << (quality.cumulative_fe / quality.total_frames) << " nats\n";
    std::cout << "  - Active Primitives:        " << core.get_primitives_count() << "\n";
    std::cout << "  - Epigenetic Invariant Lock:" << std::fixed << std::setprecision(3) << core.get_average_epigenetic_lock() << "\n\n";

    // Rigorous KEP Rule #2 Verdict Criteria
    bool behavioral_success = (quality.catches >= 60) && (quality.avg_distance_to_target < 0.80);
    bool speed_success = (quality.fps >= 5000.0);

    std::cout << "=== QUALITY AUDIT EVALUATION ===\n";
    std::cout << "Target Pursuit Competence: " << (behavioral_success ? "PASSED (AUTHENTIC PURSUIT ATTRACTION)" : "FAILED (BLIND/DEGENERATE)") << "\n";
    std::cout << "Throughput Standard:       " << (speed_success ? "PASSED (SUPER-REALTIME)" : "FAILED (BOTTLENECK)") << "\n";

    return (behavioral_success && speed_success) ? 0 : 1;
}
