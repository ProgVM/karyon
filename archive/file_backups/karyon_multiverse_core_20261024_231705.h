// ============================================================================
// KARYON MULTIVERSE WAVEFIELD CORE HEADER (karyon_multiverse_core.h)
// Master Specification: KEP v30.0 Substrate Core API
// Author & Lead Architect: Bazilevs (ProgVM) & Lead Cyberneticist
// ============================================================================

#ifndef KARYON_MULTIVERSE_CORE_H
#define KARYON_MULTIVERSE_CORE_H

#include <cstdint>

#define MAX_UNIVERSES 128
#define D1 64
#define MAX_D2 16
#define TOTAL_FIELD_DIM (D1 * MAX_D2)
#define LAW_SPECTRAL_DIM (4 * MAX_D2)
#define METRIC_TENSOR_DIM 4
#define NUM_SYMBOLS 4

#define BASE_DT 0.02f
#define DX_HORIZON 1.5f
#define LOOKAHEAD_HORIZON 5.0f
#define GRAVITY -38.0f
#define JUMP_IMPULSE 18.0f
#define AGENT_RADIUS 0.8f

struct Obstacle {
    float x;
    float y;
    int32_t oid;
};

struct MultiverseSimulationResult {
    int32_t active_universes;
    double gpu_execution_time_ms;
    float mean_navigated_distance_x;
    float hazard_evasion_rate_pct;
    float mean_discovered_symbols;
    float closed_loop_free_energy_F;
    float intrinsic_order_integrity_O;
};

#ifdef __cplusplus
extern "C" {
#endif

MultiverseSimulationResult run_multiverse_simulation(
    const Obstacle* obstacles,
    int total_obstacles,
    int num_universes,
    int ticks
);

#ifdef __cplusplus
}
#endif

#endif // KARYON_MULTIVERSE_CORE_H
