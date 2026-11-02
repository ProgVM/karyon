#ifndef KARYON_CORE_WAVEFIELD_HPP
#define KARYON_CORE_WAVEFIELD_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

struct WavefrontState {
    size_t current_pos;
    double coherence_weight;
};

class ContinuousWavefieldEngine {
private:
    std::vector<int8_t> physical_manifold;
    std::vector<size_t> frame_offsets;
    size_t frame_bytes;

    std::vector<int8_t> working_history;
    mutable std::vector<WavefrontState> active_wavefronts;

public:
    ContinuousWavefieldEngine() : frame_bytes(128 * 128 * 3) {}

    void ResetHistory() {
        working_history.clear();
        active_wavefronts.clear();
    }

    void ReserveCapacity(size_t history_spins) {
        working_history.reserve(history_spins);
    }

    size_t GetManifoldSpinSize() const { return physical_manifold.size(); }
    size_t GetFrameCount() const { return frame_offsets.size(); }

    void IngestDataset(const uint8_t* data, size_t total_bytes, size_t single_frame_bytes) {
        frame_bytes = single_frame_bytes;
        size_t num_frames = total_bytes / frame_bytes;
        
        frame_offsets.clear();
        physical_manifold.clear();
        physical_manifold.reserve(total_bytes * 8);

        for (size_t f = 0; f < num_frames; ++f) {
            frame_offsets.push_back(physical_manifold.size());
            const uint8_t* frame_ptr = data + f * frame_bytes;
            for (size_t i = 0; i < frame_bytes; ++i) {
                uint8_t b = frame_ptr[i];
                for (int bit = 7; bit >= 0; --bit) {
                    int8_t s_int = ((b >> bit) & 1) ? +1 : -1;
                    physical_manifold.push_back(s_int);
                }
            }
        }
    }

    // Synchronize $K$ active wavefronts to start of candidate frames
    void InitializeSuperpositionWavefronts(const std::vector<size_t>& candidate_frame_indices, size_t start_bit_offset = 0) {
        active_wavefronts.clear();
        for (size_t idx : candidate_frame_indices) {
            if (idx < frame_offsets.size()) {
                size_t start_pos = frame_offsets[idx] + start_bit_offset;
                active_wavefronts.push_back({start_pos, 1.0});
            }
        }
    }

    // High-Speed O(K) Continuous Feynman Path Superposition Step
    double StepSuperpositionWavefield(double temperature = 1.0) const {
        if (active_wavefronts.empty()) return 0.0;

        double total_field = 0.0;
        double total_weight = 0.0;

        for (auto& wf : active_wavefronts) {
            if (wf.current_pos < physical_manifold.size()) {
                int8_t spin_val = physical_manifold[wf.current_pos];
                total_field += wf.coherence_weight * spin_val;
                total_weight += wf.coherence_weight;
                wf.current_pos++; // Synchronous wave propagation
            }
        }

        if (total_weight > 0.0) {
            total_field /= total_weight;
        }

        if (temperature > 0.0) {
            total_field /= temperature;
        }

        return std::tanh(total_field);
    }

    void PushWorkingSpin(double target_spin) {
        int8_t s_int = (target_spin >= 0.0) ? 1 : -1;
        working_history.push_back(s_int);
    }
};

#endif // KARYON_CORE_WAVEFIELD_HPP
