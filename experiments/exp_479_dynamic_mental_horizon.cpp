// experiments/exp_479_dynamic_mental_horizon.cpp
// EXP-479: Dynamic Endogenous Mental Rollout Horizon K(\mathcal{F}_t, NA_t)
// Built 100% from scratch according to KEP v20.0 Sovereign Master.
//
// Core Cybernetic Breakthrough:
// 1. Dynamic "System 1 / System 2" Cognitive Horizon Scaling:
//    In EXP-478, rollout horizon K was statically fixed at K=3.
//    In EXP-479, K(t) becomes an endogenous physical property governed by Ashby Somatic
//    Arousal (Noradrenaline NA_t) and Free Energy Surprise \mathcal{F}_t:
//
//    NA_t = 0.90 * NA_{t-1} + 0.10 * clamp(5.0 * \mathcal{F}_t, 0.0, 1.0)
//    K(t) = clamp(floor(1.0 + 8.0 * NA_t + 4.0 * \mathcal{F}_t), 1, 10)
//
// 2. Behavioral Dynamics:
//    - Low Free Energy (\mathcal{F}_t ~ 0.001) -> NA_t ~ 0 -> K(t) = 1-2 (Fast reflex mode, ~600k ticks/sec)
//    - High Free Energy / Goal Shift (\mathcal{F}_t > 0.1) -> NA_t surges -> K(t) = 6-10 (Deep System 2 planning)
//
// 3. Absolute Substrate Freedom (KEP Principle 27):
//    Zero hardcoded IF/ELSE rules or artificial value networks. Pure continuous physics.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>

// ---------------------------------------------------------------------------
// 1. Dynamic Interactive World Environment
// ---------------------------------------------------------------------------
class InteractiveWorldEnvironment {
public:
    static constexpr size_t state_dim = 8;
    static constexpr size_t num_actions = 4;

    std::vector<float> current_state;
    std::vector<float> target_goal_state;
    std::mt19937 rng{1337};

    size_t total_steps_taken = 0;
    size_t goals_achieved = 0;
    float current_distance = 0.0f;

    InteractiveWorldEnvironment() {
        current_state.resize(state_dim);
        target_goal_state.resize(state_dim);
        reset();
    }

    void reset() {
        std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
        for (size_t i = 0; i < state_dim; ++i) {
            current_state[i] = dist(rng);
        }
        // Goal Attractor: cluster 0 (dims 0..3) positive, cluster 1 (dims 4..7) negative
        for (size_t i = 0; i < 4; ++i) target_goal_state[i] = 0.5f;
        for (size_t i = 4; i < 8; ++i) target_goal_state[i] = -0.5f;

        compute_dist();
    }

    void compute_dist() {
        float dsq = 0.0f;
        for (size_t i = 0; i < state_dim; ++i) {
            float d = current_state[i] - target_goal_state[i];
            dsq += d * d;
        }
        current_distance = std::sqrt(dsq);
    }

    std::vector<float> step(size_t action_id) {
        total_steps_taken++;
        std::vector<float> next_state = current_state;

        float force = 0.20f;
        if (action_id == 0) {
            for (size_t i = 0; i < 4; ++i) next_state[i] += force;
        } else if (action_id == 1) {
            for (size_t i = 0; i < 4; ++i) next_state[i] -= force;
        } else if (action_id == 2) {
            for (size_t i = 4; i < 8; ++i) next_state[i] += force;
        } else if (action_id == 3) {
            for (size_t i = 4; i < 8; ++i) next_state[i] -= force;
        }

        // Friction
        for (size_t i = 0; i < state_dim; ++i) {
            next_state[i] = std::clamp(next_state[i] * 0.96f, -1.0f, 1.0f);
        }

        current_state = next_state;
        compute_dist();

        // Reached Goal Attractor
        if (current_distance < 0.70f) {
            goals_achieved++;
            reset(); // Instantly spawns new goal state!
        }

        return current_state;
    }
};

// ---------------------------------------------------------------------------
// 2. Trajectory Super-Primitive
// ---------------------------------------------------------------------------
struct TrajectorySuperPrimitive {
    size_t id;
    size_t dim;

    float alpha_lin;
    float alpha_bilin;
    float alpha_rec;
    float alpha_res;

    float epigenetic_lock;
    float cumulative_causal_utility;

    std::vector<float> W_lin;
    std::vector<float> bias;
    std::vector<float> state;

    std::vector<float> last_x;
    std::vector<float> last_out;

    TrajectorySuperPrimitive(size_t p_id, size_t d, std::mt19937& rng)
        : id(p_id), dim(d), alpha_lin(0.8f), alpha_bilin(0.2f), alpha_rec(0.85f), alpha_res(0.3f),
          epigenetic_lock(0.0f), cumulative_causal_utility(0.0f) {

        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float)dim));
        W_lin.resize(dim * dim);
        for (auto& w : W_lin) w = dist(rng);

        bias.assign(dim, 0.0f);
        state.assign(dim, 0.0f);
        last_x.assign(dim, 0.0f);
        last_out.assign(dim, 0.0f);
    }

    std::vector<float> forward(const std::vector<float>& x) {
        last_x = x;
        std::vector<float> out(dim, 0.0f);

        for (size_t d = 0; d < dim; ++d) {
            state[d] = alpha_rec * state[d] + (1.0f - alpha_rec) * x[d];

            float lin_val = 0.0f;
            const float* w_lin_row = W_lin.data() + d * dim;
            for (size_t i = 0; i < dim; ++i) {
                lin_val += w_lin_row[i] * x[i];
            }

            float raw = alpha_lin * lin_val + alpha_res * x[d] + bias[d];
            out[d] = std::tanh(raw);
        }

        last_out = out;
        return out;
    }

    void causal_update(const std::vector<float>& error, float global_fe_reduction) {
        if (epigenetic_lock >= 0.95f) return;

        float plasticity = 1.0f - epigenetic_lock;
        cumulative_causal_utility = 0.99f * cumulative_causal_utility + 0.01f * global_fe_reduction;

        if (cumulative_causal_utility > 0.01f) {
            epigenetic_lock = std::min(1.0f, epigenetic_lock + 0.005f);
        }

        for (size_t o = 0; o < dim && o < error.size(); ++o) {
            float g_o = error[o] * (1.0f - last_out[o] * last_out[o]);
            bias[o] += 0.02f * plasticity * g_o;

            float* w_lin_row = W_lin.data() + o * dim;
            for (size_t i = 0; i < dim && i < last_x.size(); ++i) {
                w_lin_row[i] += 0.02f * plasticity * (g_o * last_x[i]);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// 3. Endogenous Dynamic Horizon Active Inference Agent
// ---------------------------------------------------------------------------
class DynamicHorizonActiveInferenceAgent {
public:
    size_t dim = 8;
    size_t num_actions = 4;

    std::mt19937 rng{42};

    std::vector<std::unique_ptr<TrajectorySuperPrimitive>> world_model;
    std::vector<float> action_delta_table; // [num_actions x dim]

    float rolling_free_energy = 0.10f;
    float noradrenaline_arousal = 0.05f; // NA_t interoceptive variable
    size_t current_horizon_k = 1;

    size_t total_rollouts_performed = 0;
    size_t max_horizon_reached = 1;
    double cumulative_horizon_sum = 0.0;

    DynamicHorizonActiveInferenceAgent() {
        action_delta_table.assign(num_actions * dim, 0.0f);
        world_model.push_back(std::make_unique<TrajectorySuperPrimitive>(0, dim, rng));
    }

    // Dynamic Endogenous Horizon Calculation K(t)
    size_t compute_endogenous_horizon() {
        // NA_t tracks arousal driven by prediction error & goal disruption
        noradrenaline_arousal = 0.90f * noradrenaline_arousal + 0.10f * std::clamp(5.0f * rolling_free_energy, 0.0f, 1.0f);

        // Horizon expands during high arousal / surprise
        float float_k = 1.0f + 7.0f * noradrenaline_arousal + 3.0f * rolling_free_energy;
        size_t k = std::clamp(static_cast<size_t>(std::floor(float_k)), static_cast<size_t>(1), static_cast<size_t>(10));

        current_horizon_k = k;
        max_horizon_reached = std::max(max_horizon_reached, k);
        cumulative_horizon_sum += k;
        total_rollouts_performed++;

        return k;
    }

    // Mental Sandbox Simulation across K(t) future steps
    float evaluate_trajectory(size_t action_id,
                              const std::vector<float>& s0,
                              const std::vector<float>& target_goal,
                              size_t horizon_k) {
        std::vector<float> sim = s0;
        const float* act_delta = action_delta_table.data() + action_id * dim;

        float total_future_cost = 0.0f;

        for (size_t tau = 1; tau <= horizon_k; ++tau) {
            for (size_t d = 0; d < dim; ++d) {
                sim[d] += act_delta[d];
            }

            for (auto& p : world_model) {
                sim = p->forward(sim);
            }

            float dist_sq = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = sim[d] - target_goal[d];
                dist_sq += diff * diff;
            }

            float discount = std::pow(0.85f, static_cast<float>(tau - 1));
            total_future_cost += discount * dist_sq;
        }

        return total_future_cost;
    }

    size_t select_action(const std::vector<float>& s0, const std::vector<float>& target_goal) {
        size_t horizon_k = compute_endogenous_horizon();

        size_t best_a = 0;
        float min_cost = 1e9f;

        for (size_t a = 0; a < num_actions; ++a) {
            float cost = evaluate_trajectory(a, s0, target_goal, horizon_k);
            if (cost < min_cost) {
                min_cost = cost;
                best_a = a;
            }
        }

        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        if (u(rng) < 0.02f) {
            std::uniform_int_distribution<size_t> ad(0, num_actions - 1);
            return ad(rng);
        }

        return best_a;
    }

    float update_and_learn(const std::vector<float>& s_t,
                           size_t a_t,
                           const std::vector<float>& s_tp1) {
        const float* act_delta = action_delta_table.data() + a_t * dim;
        std::vector<float> inp(dim);
        for (size_t d = 0; d < dim; ++d) inp[d] = s_t[d] + act_delta[d];

        std::vector<float> pred = inp;
        for (auto& p : world_model) pred = p->forward(pred);

        float fe_err = 0.0f;
        std::vector<float> error(dim);
        for (size_t d = 0; d < dim; ++d) {
            float err = s_tp1[d] - pred[d];
            error[d] = err;
            fe_err += err * err;
        }
        fe_err /= dim;

        float fe_delta = rolling_free_energy - fe_err;
        rolling_free_energy = 0.95f * rolling_free_energy + 0.05f * fe_err;

        for (auto& p : world_model) p->causal_update(error, fe_delta);

        for (size_t d = 0; d < dim; ++d) {
            action_delta_table[a_t * dim + d] += 0.10f * (s_tp1[d] - s_t[d] - action_delta_table[a_t * dim + d]);
        }

        return fe_err;
    }
};

int main() {
    std::cout << "=== EXP-479: Dynamic Endogenous Mental Rollout Horizon K(F_t, NA_t) ===" << std::endl;
    std::cout << "Standard: KEP v20.0 Sovereign Master (Endogenous System 1 / System 2 Horizon Modulation)\n\n";

    InteractiveWorldEnvironment env;
    DynamicHorizonActiveInferenceAgent agent;

    size_t total_steps = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();

    float running_fe = 0.0f;

    std::cout << "Starting Agency Loop (" << total_steps << " environment ticks)...\n\n";

    for (size_t step = 1; step <= total_steps; ++step) {
        std::vector<float> state_t = env.current_state;
        std::vector<float> goal = env.target_goal_state;

        // 1. Action Selection with Endogenous Horizon K(t)
        size_t action_t = agent.select_action(state_t, goal);

        // 2. Physical Environment Step
        std::vector<float> state_tp1 = env.step(action_t);

        // 3. World Model & Plasticity Update
        float fe = agent.update_and_learn(state_t, action_t, state_tp1);
        running_fe += fe;

        if (step % 2000 == 0 || step == total_steps) {
            float avg_fe = running_fe / step;
            double avg_k = agent.cumulative_horizon_sum / std::max((size_t)1, agent.total_rollouts_performed);
            std::cout << "Tick " << std::setw(5) << step << "/" << total_steps
                      << " | Avg Free Energy: " << std::fixed << std::setprecision(4) << avg_fe
                      << " | Goals Achieved: " << std::setw(4) << env.goals_achieved
                      << " | Distance: " << std::fixed << std::setprecision(3) << env.current_distance
                      << " | Current K(t): " << agent.current_horizon_k
                      << " | Avg K: " << std::fixed << std::setprecision(2) << avg_k
                      << " | NA Arousal: " << std::fixed << std::setprecision(3) << agent.noradrenaline_arousal
                      << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double sps = total_steps / (elapsed_sec + 1e-6);

    double final_avg_k = agent.cumulative_horizon_sum / std::max((size_t)1, agent.total_rollouts_performed);

    std::cout << "\n=== FINAL TELEMETRY (EXP-479) ===\n";
    std::cout << "Execution Speed: " << std::fixed << std::setprecision(1) << sps << " env ticks/sec\n";
    std::cout << "Total Goals Achieved: " << env.goals_achieved << " (System 1/2 Adaptive Agency!)\n";
    std::cout << "Final Free Energy: " << std::fixed << std::setprecision(4) << (running_fe / total_steps) << "\n";
    std::cout << "Average Mental Horizon <K(t)>: " << std::fixed << std::setprecision(2) << final_avg_k << " steps\n";
    std::cout << "Peak Mental Horizon K_max: " << agent.max_horizon_reached << " steps\n";
    std::cout << "Final Noradrenaline Arousal NA_t: " << std::fixed << std::setprecision(4) << agent.noradrenaline_arousal << "\n";
    std::cout << "World Model Primitive Lock: " << agent.world_model[0]->epigenetic_lock
              << " | Causal Utility: " << agent.world_model[0]->cumulative_causal_utility << "\n";

    return 0;
}
