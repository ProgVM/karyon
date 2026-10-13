// karyon_native_cabi.cpp — Sovereign C-ABI Gateway for Karyon Native Core v7.7
// Standard: KEP v20.0 Sovereign Master (Principles 2 to 27 Compliant)

#include "karyon_native_core.h"
#include <vector>
#include <memory>
#include <cstring>

extern "C" {

struct KaryonCABIHandle {
    std::unique_ptr<karyon::KaryonSovereignCore> core;
    karyon::CoreConfig config;
};

KaryonCABIHandle* karyon_cabi_create(size_t sensory_dim, size_t latent_dim, size_t action_dim) {
    auto handle = new KaryonCABIHandle();
    handle->config.sensory_dim = sensory_dim;
    handle->config.latent_dim = latent_dim;
    handle->config.action_dim = action_dim;
    handle->core = std::make_unique<karyon::KaryonSovereignCore>(handle->config);
    return handle;
}

KaryonCABIHandle* karyon_cabi_create_custom(size_t sensory_dim, size_t latent_dim, size_t action_dim,
                                            size_t min_horizon_k, size_t max_horizon_k,
                                            float plasticity_rate, float aversive_weight, float peril_threshold) {
    auto handle = new KaryonCABIHandle();
    handle->config.sensory_dim = sensory_dim;
    handle->config.latent_dim = latent_dim;
    handle->config.action_dim = action_dim;
    handle->config.min_horizon_k = min_horizon_k;
    handle->config.max_horizon_k = max_horizon_k;
    handle->config.plasticity_rate = plasticity_rate;
    handle->config.aversive_weight = aversive_weight;
    handle->config.peril_threshold = peril_threshold;
    handle->core = std::make_unique<karyon::KaryonSovereignCore>(handle->config);
    return handle;
}

void karyon_cabi_destroy(KaryonCABIHandle* handle) {
    if (handle) {
        delete handle;
    }
}

void karyon_cabi_step(KaryonCABIHandle* handle,
                      const float* sensory_curr,
                      const float* sensory_target,
                      const float* sensory_next,
                      float* action_inout,
                      float* out_free_energy,
                      float* out_noradrenaline,
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
    if (out_noradrenaline) *out_noradrenaline = telem.noradrenaline_arousal;
    if (out_primitives_count) *out_primitives_count = telem.active_primitives_count;
}

void karyon_cabi_step_with_valence(KaryonCABIHandle* handle,
                                  const float* sensory_curr,
                                  const float* sensory_target,
                                  const float* sensory_next,
                                  float external_valence,
                                  float* action_inout,
                                  float* out_free_energy,
                                  float* out_noradrenaline,
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
    if (out_noradrenaline) *out_noradrenaline = telem.noradrenaline_arousal;
    if (out_primitives_count) *out_primitives_count = telem.active_primitives_count;
}

} // extern "C"
