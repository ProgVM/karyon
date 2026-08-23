// karyon_runtime.h
#ifndef KARYON_RUNTIME_H
#define KARYON_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KaryonEntity KaryonEntity;

typedef struct {
    const uint8_t* text_bytes;
    size_t         text_len;
    
    const float*   vision_pixels;
    size_t         vision_len;
    
    const float*   audio_samples;
    size_t         audio_len;
    
    float          dt;
} SensoryStream;

typedef struct {
    const uint8_t* text_bytes;
    size_t         text_len;
    
    const float*   motor_actions;
    size_t         action_dim;
    
    const float*   cog_actions;
    size_t         cog_dim;
} MotorStream;

KaryonEntity* karyon_load(const char* kcore_file_path, const char* device);

void karyon_perceive_text(KaryonEntity* entity, const char* text_utf8, float dt);
void karyon_perceive_stream(KaryonEntity* entity, const SensoryStream* stream);

void karyon_step(KaryonEntity* entity);

const char* karyon_express_text(KaryonEntity* entity);
void karyon_express_stream(KaryonEntity* entity, MotorStream* out_stream);

void karyon_adapt(KaryonEntity* entity, float feedback_signal, float learning_rate);
void karyon_get_somatic_state(KaryonEntity* entity, float* energy, float* health, float* arousal);

void karyon_save(KaryonEntity* entity, const char* kcore_file_path);
void karyon_free(KaryonEntity* entity);

#ifdef __cplusplus
}
#endif

#endif // KARYON_RUNTIME_H
