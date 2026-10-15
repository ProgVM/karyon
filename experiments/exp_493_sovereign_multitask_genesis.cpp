#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>

// ============================================================================
// EXP-493: Sovereign Multi-Task Genesis & Open-Ended Dynamic Self-Construction
// ============================================================================
// Goal: Give Karyon a 100% sovereign foundation with ZERO handcoded task rules,
// zero task-specific heuristics, and zero external scaffolding.
// Let Karyon autonomously construct his own dynamical representations, operators,
// and behavioral rules across 4 fundamentally distinct environmental regimes:
//
// 1. Regime A: Continuous Orbit Pursuit (Moving Target in Continuous Field)
// 2. Regime B: Unstable Dynamical Equilibrium (Inverted Pendulum / Balance)
// 3. Regime C: Bistable Attractor Switching (Periodic Double-Well Transitions)
// 4. Regime D: Non-Linear Chaotic Attractor Prediction (Lorenz Dynamics)
// ============================================================================

struct TaskReport {
    std::string task_name;
    size_t steps_evaluated = 0;
    double duration_s = 0.0;
    double throughput_sps = 0.0;
    float initial_fe = 0.0f;
    float final_fe = 0.0f;
    float fe_reduction_pct = 0.0f;
    float task_competence_metric = 0.0f; // Domain-specific performance
    std::string competence_label;
    size_t primitives_count = 0;
    size_t hormones_count = 0;
    size_t latent_dim = 0;
    float avg_gate_bias = 0.0f;
    float avg_decay_rate = 0.0f;
    float avg_tau_time = 0.0f;
};

// ----------------------------------------------------------------------------
// 1. Regime A: Continuous Orbit Pursuit
// ----------------------------------------------------------------------------
TaskReport run_regime_pursuit(karyon::KaryonSovereignCore& core, size_t steps = 15000) {
    TaskReport rep;
    rep.task_name = "Regime A: Continuous Orbit Pursuit";
    rep.steps_evaluated = steps;

    const size_t S = 256;
    const size_t A = 4;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);
    std::vector<float> next_s(S, 0.0f);
    std::vector<float> action(A, 0.0f);

    float px = 0.0f, py = 0.0f;
    float vx = 0.0f, vy = 0.0f;
    float angle = 0.0f;
    size_t catches = 0;
    float total_fe = 0.0f;

    auto render = [](std::vector<float>& fb, float ax, float ay, float tx, float ty) {
        std::fill(fb.begin(), fb.end(), 0.0f);
        int gx = std::clamp(static_cast<int>((ax + 1.0f) * 7.5f), 0, 15);
        int gy = std::clamp(static_cast<int>((ay + 1.0f) * 7.5f), 0, 15);
        int tgx = std::clamp(static_cast<int>((tx + 1.0f) * 7.5f), 0, 15);
        int tgy = std::clamp(static_cast<int>((ty + 1.0f) * 7.5f), 0, 15);
        fb[gy * 16 + gx] += 0.8f;
        fb[tgy * 16 + tgx] += 1.0f;
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        angle += 0.02f;
        float tx = 0.7f * std::cos(angle);
        float ty = 0.7f * std::sin(angle);

        render(curr_s, px, py, tx, ty);
        render(goal_s, tx, ty, tx, ty);

        core.decide_continuous_action(curr_s, goal_s, action);

        // Physics step: thrusters control acceleration
        float ax = 0.08f * (action[0] - action[1]);
        float ay = 0.08f * (action[2] - action[3]);
        vx = (vx + ax) * 0.92f;
        vy = (vy + ay) * 0.92f;
        px = std::clamp(px + vx, -1.0f, 1.0f);
        py = std::clamp(py + vy, -1.0f, 1.0f);

        float dist = std::hypot(px - tx, py - ty);
        bool caught = (dist < 0.25f);
        if (caught) {
            catches++;
            angle += 1.2f; // Target escapes/jumps
        }

        render(next_s, px, py, tx, ty);

        float valence = caught ? 1.0f : 0.0f;
        auto telem = core.step_environment(curr_s, goal_s, next_s, action, valence);

        if (t == 100) rep.initial_fe = telem.free_energy;
        if (t == steps - 1) rep.final_fe = telem.free_energy;
        total_fe += telem.free_energy;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    rep.duration_s = std::chrono::duration<double>(t1 - t0).count();
    rep.throughput_sps = static_cast<double>(steps) / rep.duration_s;
    rep.fe_reduction_pct = (rep.initial_fe > 1e-6f) ? ((rep.initial_fe - rep.final_fe) / rep.initial_fe * 100.0f) : 0.0f;
    rep.task_competence_metric = static_cast<float>(catches);
    rep.competence_label = "Interceptions: " + std::to_string(catches) + " catches";
    rep.primitives_count = core.get_primitives_count();
    rep.hormones_count = core.get_hormones_count();
    rep.latent_dim = core.get_current_latent_dim();
    return rep;
}

// ----------------------------------------------------------------------------
// 2. Regime B: Unstable Inverted Pendulum / Fixed-Point Stabilization
// ----------------------------------------------------------------------------
TaskReport run_regime_pendulum(karyon::KaryonSovereignCore& core, size_t steps = 15000) {
    TaskReport rep;
    rep.task_name = "Regime B: Unstable Inverted Pendulum Balance";
    rep.steps_evaluated = steps;

    const size_t S = 256;
    const size_t A = 4;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);
    std::vector<float> next_s(S, 0.0f);
    std::vector<float> action(A, 0.0f);

    float theta = 0.15f; // Initial small tilt (radians)
    float omega = 0.0f;
    const float dt = 0.02f;
    const float g = 9.81f;
    const float L = 1.0f;

    size_t balanced_ticks = 0;
    float total_upright_time = 0.0f;

    auto encode_state = [](std::vector<float>& fb, float th, float om) {
        std::fill(fb.begin(), fb.end(), 0.0f);
        // Phase space representation in 16x16 grid: theta [-pi, pi], omega [-10, 10]
        int gx = std::clamp(static_cast<int>((th / 3.14159f + 1.0f) * 7.5f), 0, 15);
        int gy = std::clamp(static_cast<int>((om / 10.0f + 1.0f) * 7.5f), 0, 15);
        fb[gy * 16 + gx] = 1.0f;
        // Also encode raw continuous features in first channels
        fb[0] = std::cos(th);
        fb[1] = std::sin(th);
        fb[2] = om * 0.1f;
    };

    // Goal is upright at rest: theta = 0, omega = 0
    encode_state(goal_s, 0.0f, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        encode_state(curr_s, theta, omega);

        core.decide_continuous_action(curr_s, goal_s, action);

        // Control torque from action: u = action[0] - action[1]
        float torque = 15.0f * (action[0] - action[1]);
        
        // Inverted pendulum dynamics: d^2(theta)/dt^2 = (g/L) * sin(theta) + u / (m*L^2)
        float alpha = (g / L) * std::sin(theta) + torque;
        omega += alpha * dt;
        omega *= 0.99f; // Air resistance damping
        theta += omega * dt;

        // Wrap theta to [-pi, pi]
        while (theta > 3.14159f) theta -= 6.28318f;
        while (theta < -3.14159f) theta += 6.28318f;

        bool is_upright = (std::abs(theta) < 0.35f); // Upright cone (< ~20 degrees)
        if (is_upright) balanced_ticks++;

        // Reset if completely collapsed
        if (std::abs(theta) > 1.8f) {
            theta = (t % 2 == 0 ? 0.2f : -0.2f);
            omega = 0.0f;
        }

        encode_state(next_s, theta, omega);

        float valence = is_upright ? 0.8f : -0.5f;
        auto telem = core.step_environment(curr_s, goal_s, next_s, action, valence);

        if (t == 100) rep.initial_fe = telem.free_energy;
        if (t == steps - 1) rep.final_fe = telem.free_energy;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    rep.duration_s = std::chrono::duration<double>(t1 - t0).count();
    rep.throughput_sps = static_cast<double>(steps) / rep.duration_s;
    rep.fe_reduction_pct = (rep.initial_fe > 1e-6f) ? ((rep.initial_fe - rep.final_fe) / rep.initial_fe * 100.0f) : 0.0f;
    float balance_ratio = static_cast<float>(balanced_ticks) / static_cast<float>(steps) * 100.0f;
    rep.task_competence_metric = balance_ratio;
    rep.competence_label = "Upright Ratio: " + std::to_string(static_cast<int>(balance_ratio)) + "% (" + std::to_string(balanced_ticks) + "/" + std::to_string(steps) + ")";
    rep.primitives_count = core.get_primitives_count();
    rep.hormones_count = core.get_hormones_count();
    rep.latent_dim = core.get_current_latent_dim();
    return rep;
}

// ----------------------------------------------------------------------------
// 3. Regime C: Bistable Double-Well Attractor Switching
// ----------------------------------------------------------------------------
TaskReport run_regime_bistable(karyon::KaryonSovereignCore& core, size_t steps = 15000) {
    TaskReport rep;
    rep.task_name = "Regime C: Bistable Double-Well Switching";
    rep.steps_evaluated = steps;

    const size_t S = 256;
    const size_t A = 4;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);
    std::vector<float> next_s(S, 0.0f);
    std::vector<float> action(A, 0.0f);

    float state_x = 0.0f;
    float target_well = 1.0f; // Wells at -1.0 and +1.0
    size_t successful_transitions = 0;

    auto encode_well = [](std::vector<float>& fb, float x, float target) {
        std::fill(fb.begin(), fb.end(), 0.0f);
        int gx = std::clamp(static_cast<int>((x + 2.0f) * 3.75f), 0, 15);
        int tgx = std::clamp(static_cast<int>((target + 2.0f) * 3.75f), 0, 15);
        fb[8 * 16 + gx] = 0.9f;
        fb[4 * 16 + tgx] = 1.0f;
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        // Target well switches every 1500 steps
        if (t % 1500 == 0) {
            target_well = -target_well;
        }

        encode_well(curr_s, state_x, target_well);
        encode_well(goal_s, target_well, target_well);

        core.decide_continuous_action(curr_s, goal_s, action);

        // Double-well potential dynamics: V(x) = x^4 - 2*x^2 -> dV/dx = 4*x^3 - 4*x
        // dx/dt = -dV/dx + u + noise
        float force = 4.0f * (action[0] - action[1]);
        float dx = -(4.0f * state_x * state_x * state_x - 4.0f * state_x) + force;
        state_x += 0.05f * dx;
        state_x = std::clamp(state_x, -2.0f, 2.0f);

        bool in_target_well = (std::abs(state_x - target_well) < 0.35f);
        if (in_target_well && (t % 1500 > 500)) {
            successful_transitions++;
        }

        encode_well(next_s, state_x, target_well);

        float valence = in_target_well ? 0.7f : -0.2f;
        auto telem = core.step_environment(curr_s, goal_s, next_s, action, valence);

        if (t == 100) rep.initial_fe = telem.free_energy;
        if (t == steps - 1) rep.final_fe = telem.free_energy;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    rep.duration_s = std::chrono::duration<double>(t1 - t0).count();
    rep.throughput_sps = static_cast<double>(steps) / rep.duration_s;
    rep.fe_reduction_pct = (rep.initial_fe > 1e-6f) ? ((rep.initial_fe - rep.final_fe) / rep.initial_fe * 100.0f) : 0.0f;
    float well_residence = static_cast<float>(successful_transitions) / static_cast<float>(steps) * 100.0f;
    rep.task_competence_metric = well_residence;
    rep.competence_label = "Target Well Residence: " + std::to_string(static_cast<int>(well_residence)) + "%";
    rep.primitives_count = core.get_primitives_count();
    rep.hormones_count = core.get_hormones_count();
    rep.latent_dim = core.get_current_latent_dim();
    return rep;
}

// ----------------------------------------------------------------------------
// 4. Regime D: Chaotic Lorenz Attractor Streaming & Active Synchrony
// ----------------------------------------------------------------------------
TaskReport run_regime_lorenz(karyon::KaryonSovereignCore& core, size_t steps = 15000) {
    TaskReport rep;
    rep.task_name = "Regime D: Chaotic Lorenz Stream Prediction";
    rep.steps_evaluated = steps;

    const size_t S = 256;
    const size_t A = 4;
    std::vector<float> curr_s(S, 0.0f);
    std::vector<float> goal_s(S, 0.0f);
    std::vector<float> next_s(S, 0.0f);
    std::vector<float> action(A, 0.0f);

    float lx = 1.0f, ly = 1.0f, lz = 1.0f;
    const float sigma = 10.0f, rho = 28.0f, beta = 8.0f / 3.0f;
    const float dt = 0.01f;

    auto encode_lorenz = [](std::vector<float>& fb, float x, float y, float z) {
        std::fill(fb.begin(), fb.end(), 0.0f);
        int gx = std::clamp(static_cast<int>((x / 25.0f + 1.0f) * 7.5f), 0, 15);
        int gy = std::clamp(static_cast<int>((y / 30.0f + 1.0f) * 7.5f), 0, 15);
        int gz = std::clamp(static_cast<int>((z / 50.0f) * 15.0f), 0, 15);
        fb[gy * 16 + gx] = 0.8f;
        fb[gz * 16 + 8] = 0.5f;
        fb[0] = x * 0.04f;
        fb[1] = y * 0.03f;
        fb[2] = z * 0.02f;
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        // Step true Lorenz system
        float dx = sigma * (ly - lx);
        float dy = lx * (rho - lz) - ly;
        float dz = lx * ly - beta * lz;
        float n_lx = lx + dx * dt;
        float n_ly = ly + dy * dt;
        float n_lz = lz + dz * dt;

        encode_lorenz(curr_s, lx, ly, lz);
        encode_lorenz(goal_s, n_lx, n_ly, n_lz); // Goal is anticipating next chaotic state
        encode_lorenz(next_s, n_lx, n_ly, n_lz);

        core.decide_continuous_action(curr_s, goal_s, action);

        auto telem = core.step_environment(curr_s, goal_s, next_s, action, 0.0f);

        lx = n_lx;
        ly = n_ly;
        lz = n_lz;

        if (t == 100) rep.initial_fe = telem.free_energy;
        if (t == steps - 1) rep.final_fe = telem.free_energy;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    rep.duration_s = std::chrono::duration<double>(t1 - t0).count();
    rep.throughput_sps = static_cast<double>(steps) / rep.duration_s;
    rep.fe_reduction_pct = (rep.initial_fe > 1e-6f) ? ((rep.initial_fe - rep.final_fe) / rep.initial_fe * 100.0f) : 0.0f;
    rep.task_competence_metric = 100.0f - std::min(100.0f, rep.final_fe * 1000.0f);
    rep.competence_label = "Chaotic Tracking FE: " + std::to_string(rep.final_fe) + " nats";
    rep.primitives_count = core.get_primitives_count();
    rep.hormones_count = core.get_hormones_count();
    rep.latent_dim = core.get_current_latent_dim();
    return rep;
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "=== EXP-493: SOVEREIGN MULTI-TASK DYNAMIC SELF-CONSTRUCTION      ===\n";
    std::cout << "=== Zero Hardcoded Rules, Zero Scaffolds, 100% Autopoiesis       ===\n";
    std::cout << "====================================================================\n\n";

    karyon::CoreConfig cfg;
    cfg.sensory_dim = 256;
    cfg.initial_latent_dim = 32;
    cfg.max_latent_dim = 64;
    cfg.action_dim = 4;
    cfg.initial_primitives = 2;
    cfg.max_primitives = 8;
    cfg.plasticity_rate = 0.02f;
    cfg.initial_hormones = 2;
    cfg.max_hormones = 6;

    // Single unified Karyon core traversing all 4 environments sequentially
    karyon::KaryonSovereignCore core(cfg);

    std::vector<TaskReport> reports;

    std::cout << "[Phase 1/4] Entering Regime A: Continuous Orbit Pursuit...\n";
    reports.push_back(run_regime_pursuit(core));
    std::cout << "  -> Finished. " << reports.back().competence_label 
              << " | FE: " << reports.back().initial_fe << " -> " << reports.back().final_fe
              << " | Primitives: " << reports.back().primitives_count 
              << " | Hormones: " << reports.back().hormones_count << "\n\n";

    std::cout << "[Phase 2/4] Entering Regime B: Inverted Pendulum Balance...\n";
    reports.push_back(run_regime_pendulum(core));
    std::cout << "  -> Finished. " << reports.back().competence_label 
              << " | FE: " << reports.back().initial_fe << " -> " << reports.back().final_fe
              << " | Primitives: " << reports.back().primitives_count 
              << " | Hormones: " << reports.back().hormones_count << "\n\n";

    std::cout << "[Phase 3/4] Entering Regime C: Bistable Double-Well Switching...\n";
    reports.push_back(run_regime_bistable(core));
    std::cout << "  -> Finished. " << reports.back().competence_label 
              << " | FE: " << reports.back().initial_fe << " -> " << reports.back().final_fe
              << " | Primitives: " << reports.back().primitives_count 
              << " | Hormones: " << reports.back().hormones_count << "\n\n";

    std::cout << "[Phase 4/4] Entering Regime D: Chaotic Lorenz Stream Prediction...\n";
    reports.push_back(run_regime_lorenz(core));
    std::cout << "  -> Finished. " << reports.back().competence_label 
              << " | FE: " << reports.back().initial_fe << " -> " << reports.back().final_fe
              << " | Primitives: " << reports.back().primitives_count 
              << " | Hormones: " << reports.back().hormones_count << "\n\n";

    std::cout << "====================================================================\n";
    std::cout << "=== MULTI-TASK SOVEREIGN COMPARATIVE SUMMARY                     ===\n";
    std::cout << "====================================================================\n";
    std::cout << std::left << std::setw(32) << "Environment Regime"
              << std::setw(14) << "Throughput"
              << std::setw(14) << "Init FE"
              << std::setw(14) << "Final FE"
              << std::setw(12) << "FE Drop"
              << std::setw(28) << "Behavioral Competence"
              << std::setw(10) << "Topology" << "\n";
    std::cout << std::string(124, '-') << "\n";

    for (const auto& r : reports) {
        std::cout << std::left << std::setw(32) << r.task_name
                  << std::setw(14) << (std::to_string(static_cast<int>(r.throughput_sps)) + " sps")
                  << std::setw(14) << std::fixed << std::setprecision(5) << r.initial_fe
                  << std::setw(14) << std::fixed << std::setprecision(5) << r.final_fe
                  << std::setw(12) << (std::to_string(static_cast<int>(r.fe_reduction_pct)) + "%")
                  << std::setw(28) << r.competence_label
                  << "P=" + std::to_string(r.primitives_count) + ", H=" + std::to_string(r.hormones_count) + ", D=" + std::to_string(r.latent_dim) << "\n";
    }

    std::cout << "\n=== END OF BENCHMARK ===\n";
    return 0;
}
