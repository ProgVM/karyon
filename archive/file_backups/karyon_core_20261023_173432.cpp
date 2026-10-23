// ============================================================================
// Karyon-CoRE v29.0 Sovereign Architecture (Master C-ABI Shared Library)
// libkaryon_core.so Export Implementation with Instance & Telemetry Support
// ============================================================================

#include "karyon_core.h"
#include <memory>

static std::unique_ptr<karyon::SovereignContinuumCore> g_default_core = nullptr;

extern "C" {

// Instance-based API
[[gnu::visibility("default")]]
void* karyon_create() {
    return new karyon::SovereignContinuumCore();
}

[[gnu::visibility("default")]]
void karyon_destroy(void* handle) {
    if (handle) {
        delete static_cast<karyon::SovereignContinuumCore*>(handle);
    }
}

[[gnu::visibility("default")]]
float karyon_step_instance(void* handle, float error, float error_rate, float dt) {
    if (!handle) return 0.0f;
    return static_cast<karyon::SovereignContinuumCore*>(handle)->perceive_and_act(error, error_rate, dt);
}

[[gnu::visibility("default")]]
float karyon_get_free_energy_instance(void* handle) {
    if (!handle) return 0.0f;
    return static_cast<karyon::SovereignContinuumCore*>(handle)->get_free_energy();
}

[[gnu::visibility("default")]]
float karyon_get_expected_free_energy_instance(void* handle) {
    if (!handle) return 0.0f;
    return static_cast<karyon::SovereignContinuumCore*>(handle)->get_expected_free_energy();
}

[[gnu::visibility("default")]]
size_t karyon_get_active_modes_instance(void* handle) {
    if (!handle) return 0;
    return static_cast<karyon::SovereignContinuumCore*>(handle)->get_active_modes();
}

[[gnu::visibility("default")]]
void karyon_reset_instance(void* handle) {
    if (handle) {
        static_cast<karyon::SovereignContinuumCore*>(handle)->reset();
    }
}

// Global default instance fallback API
[[gnu::visibility("default")]]
void karyon_init() {
    if (!g_default_core) {
        g_default_core = std::make_unique<karyon::SovereignContinuumCore>();
    } else {
        g_default_core->reset();
    }
}

[[gnu::visibility("default")]]
float karyon_step(float error, float error_rate, float dt) {
    if (!g_default_core) {
        karyon_init();
    }
    return g_default_core->perceive_and_act(error, error_rate, dt);
}

[[gnu::visibility("default")]]
float karyon_get_free_energy() {
    if (!g_default_core) return 0.0f;
    return g_default_core->get_free_energy();
}

[[gnu::visibility("default")]]
float karyon_get_expected_free_energy() {
    if (!g_default_core) return 0.0f;
    return g_default_core->get_expected_free_energy();
}

[[gnu::visibility("default")]]
size_t karyon_get_active_modes() {
    if (!g_default_core) return 0;
    return g_default_core->get_active_modes();
}

[[gnu::visibility("default")]]
void karyon_reset() {
    if (g_default_core) {
        g_default_core->reset();
    }
}

} // extern "C"
