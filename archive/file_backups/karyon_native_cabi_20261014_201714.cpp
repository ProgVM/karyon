// karyon_native_cabi.cpp
// Standard: KEP v20.0 Sovereign Master (Principle 1: C++20 Zero-Allocation Engine)

#include "karyon_native_core.h"
#include <vector>
#include <cstring>
#include <memory>
#include <span>

struct KaryonCABIHandle {
    karyon::CoreConfig config;
    std::unique_ptr<karyon::KaryonSovereignCore> core;
};

extern "C" {

KaryonCABIHandle* karyon_cabi_create(size_t sensory_dim,
                                     size_t latent_dim,
                                     size_t action_dim,
                                     size_t initial_primitives,
                                     size_t max_primitives,
                                     float plasticity_rate) {
    auto handle = new KaryonCABIHandle();
    handle->config.sensory_dim = sensory_dim;
    handle->config.latent_dim = latent_dim;
    handle->config.action_dim = action_dim;
    handle->config.initial_primitives = initial_primitives;
    handle->config.max_primitives = max_primitives;
    handle->config.plasticity_rate = plasticity_rate;

    handle->core = std::make_unique<karyon::KaryonSovereignCore>(handle->config);
    return handle;
}

void karyon_cabi_destroy(KaryonCABIHandle* handle) {
    if (handle) {
        delete handle;
    }
}

void karyon_cabi_decide_action(KaryonCABIHandle* handle,
                               const float* sensory_curr,
                               const float* sensory_target,
                               float* action_out) {
    if (!handle || !handle->core) return;

    std::span<const float> s_curr(sensory_curr, handle->config.sensory_dim);
    std::span<const float> s_tgt(sensory_target, handle->config.sensory_dim);
    std::span<float> act_out(action_out, handle->config.action_dim);

    handle->core->decide_continuous_action(s_curr, s_tgt, act_out);
}

void karyon_cabi_step(KaryonCABIHandle* handle,
                      const float* sensory_curr,
                      const float* sensory_target,
                      const float* sensory_next,
                      float* action_inout,
                      float* out_free_energy,
                      float* out_primary_hormone,
                      size_t* out_primitives_count) {
    if (!handle || !handle->core) return;

    std::span<const float> s_curr(sensory_curr, handle->config.sensory_dim);
    std::span<const float> s_tgt(sensory_target, handle->config.sensory_dim);
    std::span<const float> s_nxt(sensory_next, handle->config.sensory_dim);
    std::span<float> act(action_inout, handle->config.action_dim);

    // 1. Decide continuous action
    handle->core->decide_continuous_action(s_curr, s_tgt, act);

    // 2. Step environment
    auto telem = handle->core->step_environment(s_curr, s_tgt, s_nxt, act);

    if (out_free_energy) *out_free_energy = telem.free_energy;
    if (out_primary_hormone) *out_primary_hormone = telem.primary_hormone;
    if (out_primitives_count) *out_primitives_count = telem.active_primitives_count;
}

void karyon_cabi_step_with_valence(KaryonCABIHandle* handle,
                                   const float* sensory_curr,
                                   const float* sensory_target,
                                   const float* sensory_next,
                                   float external_valence,
                                   float* action_inout,
                                   float* out_free_energy,
                                   float* out_primary_hormone,
                                   size_t* out_primitives_count) {
    if (!handle || !handle->core) return;

    std::span<const float> s_curr(sensory_curr, handle->config.sensory_dim);
    std::span<const float> s_tgt(sensory_target, handle->config.sensory_dim);
    std::span<const float> s_nxt(sensory_next, handle->config.sensory_dim);
    std::span<float> act(action_inout, handle->config.action_dim);

    // 1. Decide continuous action
    handle->core->decide_continuous_action(s_curr, s_tgt, act);

    // 2. Step environment with somatic valence
    auto telem = handle->core->step_environment(s_curr, s_tgt, s_nxt, act, external_valence);

    if (out_free_energy) *out_free_energy = telem.free_energy;
    if (out_primary_hormone) *out_primary_hormone = telem.primary_hormone;
    if (out_primitives_count) *out_primitives_count = telem.active_primitives_count;
}

void karyon_cabi_decide_action_discrete(KaryonCABIHandle* handle,
                                        const float* current_state,
                                        const float* target_goal,
                                        size_t* out_action) {
    if (!handle || !handle->core || !out_action) return;
    std::vector<float> s_curr(current_state, current_state + handle->config.sensory_dim);
    std::vector<float> s_tgt(target_goal, target_goal + handle->config.sensory_dim);
    *out_action = handle->core->decide_action(s_curr, s_tgt);
}

void karyon_cabi_step_discrete(KaryonCABIHandle* handle,
                               const float* current_state,
                               const float* target_goal,
                               const float* next_state,
                               size_t action_taken,
                               float external_valence,
                               float* out_free_energy,
                               float* out_primary_hormone,
                               size_t* out_primitives_count) {
    if (!handle || !handle->core) return;
    std::vector<float> s_curr(current_state, current_state + handle->config.sensory_dim);
    std::vector<float> s_tgt(target_goal, target_goal + handle->config.sensory_dim);
    std::vector<float> s_nxt(next_state, next_state + handle->config.sensory_dim);

    auto telem = handle->core->step_environment(s_curr, s_tgt, s_nxt, action_taken, external_valence);

    if (out_free_energy) *out_free_energy = telem.free_energy;
    if (out_primary_hormone) *out_primary_hormone = telem.primary_hormone;
    if (out_primitives_count) *out_primitives_count = telem.active_primitives_count;
}

} // extern "C"
