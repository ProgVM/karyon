// karyon_runtime.h
#ifndef KARYON_RUNTIME_H
#define KARYON_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KaryonEntity KaryonEntity;

// Multi-modal Input Sensory Stream
typedef struct {
    const uint8_t* text_bytes;
    size_t         text_len;
    
    const float*   vision_pixels; // Raw image float tensor buffer [C, H, W]
    size_t         vision_len;
    
    const float*   audio_samples; // Raw PCM audio float buffer [Samples]
    size_t         audio_len;
    
    float          dt;
} SensoryStream;

// Multi-modal Output Motor Stream (Efference Frame)
typedef struct {
    const uint8_t* text_bytes;    // Generated output text byte stream
    size_t         text_len;
    
    const float*   motor_actions; // Continuous motor action vector [ActionDim]
    size_t         action_dim;
    
    const float*   cog_actions;   // Cognitive gating / attention flags [CogDim]
    size_t         cog_dim;
} MotorStream;

// Core C-ABI Lifecycle & Execution Interface
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
