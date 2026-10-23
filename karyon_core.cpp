// ============================================================================
// Karyon-CoRE v45.0 Sovereign Architecture (Master C-ABI Shared Library)
// libkaryon_core.so Export Implementation with Instance & Telemetry Support
// KEP v30.0 Standard Compliant
// ============================================================================

#include "karyon_core.h"
#include <memory>
#include <cstring>

static std::unique_ptr<karyon::SovereignFullPlasticityOrganism> g_default_core = nullptr;

extern "C" {

// Instance-based Management API
[[gnu::visibility("default")]]
void* karyon_create() {
    return new karyon::SovereignFullPlasticityOrganism();
}

[[gnu::visibility("default")]]
void karyon_destroy(void* handle) {
    if (handle) {
        delete static_cast<karyon::SovereignFullPlasticityOrganism*>(handle);
    }
}

[[gnu::visibility("default")]]
uint8_t karyon_step_instance(void* handle, uint8_t input_channel, float external_dt) {
    if (!handle) return 0;
    return static_cast<karyon::SovereignFullPlasticityOrganism*>(handle)->perceive_and_decide(input_channel, external_dt);
}

[[gnu::visibility("default")]]
void karyon_assimilate_instance(void* handle, uint8_t input_channel, float valence, float magnitude, int target_hint, float lr) {
    if (handle) {
        static_cast<karyon::SovereignFullPlasticityOrganism*>(handle)->assimilate_experience(input_channel, valence, magnitude, target_hint, lr);
    }
}

[[gnu::visibility("default")]]
float karyon_get_free_energy_instance(void* handle) {
    if (!handle) return 0.0f;
    return static_cast<karyon::SovereignFullPlasticityOrganism*>(handle)->get_telemetry().free_energy;
}

[[gnu::visibility("default")]]
size_t karyon_get_active_modes_instance(void* handle) {
    if (!handle) return 0;
    return static_cast<karyon::SovereignFullPlasticityOrganism*>(handle)->get_telemetry().active_modes;
}

[[gnu::visibility("default")]]
void karyon_reset_instance(void* handle) {
    if (handle) {
        static_cast<karyon::SovereignFullPlasticityOrganism*>(handle)->reset();
    }
}

[[gnu::visibility("default")]]
void karyon_get_telemetry_instance(void* handle, karyon::PurePhysicsTelemetry* out_telemetry) {
    if (handle && out_telemetry) {
        *out_telemetry = static_cast<karyon::SovereignFullPlasticityOrganism*>(handle)->get_telemetry();
    }
}

// Global Singleton Fallback API for backwards compatibility
[[gnu::visibility("default")]]
void karyon_init() {
    if (!g_default_core) {
        g_default_core = std::make_unique<karyon::SovereignFullPlasticityOrganism>();
    }
}

[[gnu::visibility("default")]]
uint8_t karyon_step(uint8_t input_channel, float external_dt) {
    karyon_init();
    return g_default_core->perceive_and_decide(input_channel, external_dt);
}

[[gnu::visibility("default")]]
void karyon_assimilate(uint8_t input_channel, float valence, float magnitude, int target_hint, float lr) {
    karyon_init();
    g_default_core->assimilate_experience(input_channel, valence, magnitude, target_hint, lr);
}

[[gnu::visibility("default")]]
float karyon_get_free_energy() {
    karyon_init();
    return g_default_core->get_telemetry().free_energy;
}

[[gnu::visibility("default")]]
size_t karyon_get_active_modes() {
    karyon_init();
    return g_default_core->get_telemetry().active_modes;
}

[[gnu::visibility("default")]]
void karyon_reset() {
    if (g_default_core) {
        g_default_core->reset();
    }
}

} // extern "C"
