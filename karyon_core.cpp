// ============================================================================
// Karyon-CoRE v28.0 Sovereign Architecture (Master C-ABI Shared Library)
// libkaryon_core.so Export Implementation
// ============================================================================

#include "karyon_core.h"
#include <memory>

static std::unique_ptr<karyon::SovereignContinuumCore> g_core_instance = nullptr;

extern "C" {

[[gnu::visibility("default")]]
void karyon_init() {
    if (!g_core_instance) {
        g_core_instance = std::make_unique<karyon::SovereignContinuumCore>();
    } else {
        g_core_instance->reset();
    }
}

[[gnu::visibility("default")]]
float karyon_step(float error, float error_rate, float dt) {
    if (!g_core_instance) {
        karyon_init();
    }
    return g_core_instance->perceive_and_act(error, error_rate, dt);
}

[[gnu::visibility("default")]]
float karyon_get_free_energy() {
    if (!g_core_instance) return 0.0f;
    return g_core_instance->get_free_energy();
}

[[gnu::visibility("default")]]
void karyon_reset() {
    if (g_core_instance) {
        g_core_instance->reset();
    }
}

} // extern "C"
