#ifndef KARYON_CORE_WAVEFIELD_HPP
#define KARYON_CORE_WAVEFIELD_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

struct TopoIndexEntry {
    uint64_t phase_sig;
    size_t tap_idx;
};

class ContinuousWavefieldEngine {
private:
    std::vector<int8_t> physical_manifold;
    std::vector<size_t> phase_taps;
    std::vector<int8_t> tap_target_spins;
    std::vector<uint32_t> tap_frequencies;
    std::vector<double> causal_efficacy_score;
    std::vector<TopoIndexEntry> topo_index;

    std::vector<int8_t> working_history;
    mutable bool has_phase_lock;
    mutable size_t active_tap_index;

    static constexpr size_t COHERENCE_GUARD_BITS = 128;
    static constexpr size_t MAX_EXPANSION_HORIZON = 4096;

    void CondenseTopologicalNodes() {
        std::sort(topo_index.begin(), topo_index.end(), [](const TopoIndexEntry& a, const TopoIndexEntry& b) {
            return a.phase_sig < b.phase_sig;
        });
    }

public:
    ContinuousWavefieldEngine() : has_phase_lock(false), active_tap_index(0) {}

    void ResetHistory() {
        working_history.clear();
        has_phase_lock = false;
        active_tap_index = 0;
    }

    void ReserveCapacity(size_t history_spins) {
        working_history.reserve(history_spins);
    }

    size_t GetWaveTraceCount() const { return phase_taps.size(); }
    size_t GetManifoldSpinSize() const { return physical_manifold.size(); }

    // Sovereign Dynamic Context Expansion (Self-Governed Phase Resonance)
    // Mode 0: Deterministic O(1) Phase Lock (Exact Memory Recall)
    // Mode 1: Superposition Feynman Path Wavefield (Creative Morphological Synthesis)
    double PredictNextSpin(bool superposition_mode = false, double temperature = 1.0) const {
        if (working_history.empty() || phase_taps.empty()) return 0.0;
        size_t available = working_history.size();

        // Standard O(1) Lock for Exact Memory Recall (if superposition disabled)
        if (!superposition_mode && has_phase_lock && (active_tap_index + 1 < phase_taps.size())) {
            size_t next_tap = active_tap_index + 1;
            size_t tap_pos = phase_taps[next_tap];
            size_t check_len = std::min(std::min(available, tap_pos), COHERENCE_GUARD_BITS);

            bool is_coherent = (std::memcmp(working_history.data() + available - check_len, physical_manifold.data() + tap_pos - check_len, check_len) == 0);

            if (is_coherent) {
                active_tap_index = next_tap;
                causal_efficacy_score[next_tap] += 0.1;
                return (tap_target_spins[next_tap] >= 0) ? +1.0 : -1.0;
            }
        }

        // 2. Candidate Discovery via Topological Index (O(log N))
        uint64_t search_sig = 0;
        size_t lookback = std::min(available, static_cast<size_t>(64));
        for (size_t k = 0; k < lookback; ++k) {
            uint8_t b = (working_history[available - lookback + k] >= 0) ? 1 : 0;
            search_sig = (search_sig << 1) | b;
        }

        TopoIndexEntry dummy{search_sig, 0};
        auto range = std::equal_range(topo_index.begin(), topo_index.end(), dummy, [](const TopoIndexEntry& a, const TopoIndexEntry& b) {
            return a.phase_sig < b.phase_sig;
        });

        std::vector<size_t> active_candidates;
        for (auto it = range.first; it != range.second; ++it) {
            active_candidates.push_back(it->tap_idx);
        }

        if (active_candidates.empty()) {
            active_candidates.reserve(phase_taps.size());
            for (size_t i = 0; i < phase_taps.size(); ++i) active_candidates.push_back(i);
        }

        // 3. Autonomous Self-Expanding Context Resolution
        size_t L = 64;
        size_t max_L = std::min(available, MAX_EXPANSION_HORIZON);

        std::vector<size_t> matching_taps;
        while (L <= max_L && active_candidates.size() > 1) {
            std::vector<size_t> surviving_candidates;
            for (size_t tap_idx : active_candidates) {
                size_t tap_pos = phase_taps[tap_idx];
                if (tap_pos < L) continue;

                if (std::memcmp(working_history.data() + available - L, physical_manifold.data() + tap_pos - L, L) == 0) {
                    surviving_candidates.push_back(tap_idx);
                }
            }

            if (surviving_candidates.empty()) break;
            active_candidates = std::move(surviving_candidates);
            matching_taps = active_candidates;

            if (!superposition_mode && active_candidates.size() == 1) break;

            L *= 2;
        }

        if (matching_taps.empty()) matching_taps = active_candidates;

        if (!matching_taps.empty()) {
            if (superposition_mode) {
                // Wavefield Superposition Interference (Feynman Path Integral)
                // Aggregates wave vectors across all matching vehicle attractors
                double total_field_amplitude = 0.0;
                double total_weight = 0.0;

                for (size_t tap_idx : matching_taps) {
                    double efficacy = causal_efficacy_score[tap_idx];
                    int8_t target_s = tap_target_spins[tap_idx];
                    total_field_amplitude += efficacy * target_s;
                    total_weight += efficacy;
                }

                if (total_weight > 0.0) {
                    total_field_amplitude /= total_weight;
                }

                // Apply temperature modulation to continuous field wave
                if (temperature > 0.0) {
                    total_field_amplitude /= temperature;
                }

                has_phase_lock = false; // Keep wavefield fluid for continuous synthesis
                return std::tanh(total_field_amplitude);
            } else {
                active_tap_index = matching_taps[0];
                has_phase_lock = true;
                causal_efficacy_score[active_tap_index] += 0.5;
                return (tap_target_spins[active_tap_index] >= 0) ? +1.0 : -1.0;
            }
        }

        return 0.0;
    }

    void PushWorkingSpin(double target_spin) {
        int8_t s_int = (target_spin >= 0.0) ? 1 : -1;
        working_history.push_back(s_int);
        if (working_history.size() > 65536) {
            working_history.erase(working_history.begin(), working_history.begin() + 32768);
        }
    }

    void IngestRawBytes(const uint8_t* data, size_t size, bool learn = true) {
        if (!learn) {
            for (size_t i = 0; i < size; ++i) {
                uint8_t byte = data[i];
                for (int b = 7; b >= 0; --b) {
                    int8_t s_int = ((byte >> b) & 1) ? +1 : -1;
                    PushWorkingSpin(s_int);
                }
            }
            has_phase_lock = false;
            return;
        }

        uint64_t cur_sig = 0;
        topo_index.reserve(topo_index.size() + size * 8);

        for (size_t i = 0; i < size; ++i) {
            uint8_t byte = data[i];
            for (int b = 7; b >= 0; --b) {
                uint8_t bit = (byte >> b) & 1;
                int8_t s_int = bit ? 1 : -1;

                if (!physical_manifold.empty()) {
                    size_t tap_idx = phase_taps.size();
                    phase_taps.push_back(physical_manifold.size());
                    tap_target_spins.push_back(s_int);
                    tap_frequencies.push_back(1);
                    causal_efficacy_score.push_back(1.0);
                    topo_index.push_back({cur_sig, tap_idx});
                }

                physical_manifold.push_back(s_int);
                cur_sig = (cur_sig << 1) | bit;
            }
        }

        CondenseTopologicalNodes();
    }

    size_t AutopoieticRelaxationAndPruning(double min_efficacy_threshold = 1.05) {
        if (phase_taps.empty()) return 0;
        size_t initial_taps = phase_taps.size();

        std::vector<size_t> new_phase_taps;
        std::vector<int8_t> new_targets;
        std::vector<uint32_t> new_freqs;
        std::vector<double> new_scores;

        new_phase_taps.reserve(initial_taps);
        new_targets.reserve(initial_taps);
        new_freqs.reserve(initial_taps);
        new_scores.reserve(initial_taps);

        for (size_t i = 0; i < initial_taps; ++i) {
            if (causal_efficacy_score[i] >= min_efficacy_threshold) {
                new_phase_taps.push_back(phase_taps[i]);
                new_targets.push_back(tap_target_spins[i]);
                new_freqs.push_back(tap_frequencies[i]);
                new_scores.push_back(causal_efficacy_score[i]);
            }
        }

        phase_taps = std::move(new_phase_taps);
        tap_target_spins = std::move(new_targets);
        tap_frequencies = std::move(new_freqs);
        causal_efficacy_score = std::move(new_scores);

        topo_index.clear();
        topo_index.reserve(phase_taps.size());
        uint64_t cur_sig = 0;

        for (size_t i = 0; i < phase_taps.size(); ++i) {
            size_t pos = phase_taps[i];
            cur_sig = 0;
            size_t start = (pos >= 64) ? (pos - 64) : 0;
            for (size_t k = start; k < pos; ++k) {
                uint8_t bit = (physical_manifold[k] >= 0) ? 1 : 0;
                cur_sig = (cur_sig << 1) | bit;
            }
            topo_index.push_back({cur_sig, i});
        }

        CondenseTopologicalNodes();
        return initial_taps - phase_taps.size();
    }
};

#endif // KARYON_CORE_WAVEFIELD_HPP
