#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-506: SOVEREIGN CONTINUOUS VARIATIONAL ROLLOUT & L-SPOUT  ===\n";
    std::cout << "=== Auditing Latent Morphogenesis & Continuous Trajectory Flow   ===\n";
    std::cout << "====================================================================\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 64;
    cfg.initial_latent_dim = 12;
    cfg.max_latent_dim = 32;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 6;
    cfg.sprouting_cooldown_ticks = 50;
    cfg.sprouting_fe_threshold = 0.03f;

    karyon::KaryonSovereignCore core(cfg);

    std::vector<float> curr_state(cfg.sensory_dim, 0.0f);
    std::vector<float> target_goal(cfg.sensory_dim, 0.5f);
    std::vector<float> next_state(cfg.sensory_dim, 0.0f);
    std::vector<float> action(cfg.action_dim, 0.0f);

    size_t total_ticks = 10000;
    size_t latent_sprouts = 0;
    size_t init_dim = core.get_current_latent_dim();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < total_ticks; ++t) {
        // Continuous action decision via variational adjoint rollout
        core.decide_continuous_action(curr_state, target_goal, action);

        // Environmental response (non-linear manifold)
        for (size_t i = 0; i < cfg.sensory_dim; ++i) {
            float motor_effect = 0.05f * (action[i % cfg.action_dim]);
            next_state[i] = std::sin(curr_state[i] * 1.5f + motor_effect);
        }

        // Compute step telemetry
        auto telem = core.step_environment(curr_state, target_goal, next_state, action, 0.1f);

        if (telem.current_latent_dim > init_dim) {
            latent_sprouts += (telem.current_latent_dim - init_dim);
            init_dim = telem.current_latent_dim;
        }

        curr_state = next_state;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double rate = total_ticks / elapsed;

    std::cout << "Total Ticks Executed:       " << total_ticks << "\n";
    std::cout << "Initial Latent Dimension:   12\n";
    std::cout << "Final Latent Dimension:     " << core.get_current_latent_dim() << "\n";
    std::cout << "Latent Axis Sprouts:        " << (core.get_current_latent_dim() - 12) << "\n";
    std::cout << "Active Super-Primitives:    " << core.get_primitives_count() << "\n";
    std::cout << "Active Sovereign Hormones:  " << core.get_hormones_count() << "\n";
    std::cout << "Elapsed Time:               " << elapsed << " s\n";
    std::cout << "Throughput (Ticks/sec):     " << rate << " ticks/s\n";
    std::cout << "====================================================================\n";

    return 0;
}