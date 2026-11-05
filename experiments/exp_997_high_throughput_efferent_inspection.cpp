#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

// ============================================================================
// KARYON-CORE EXP-997: GB/S AVX2 VECTORIZED QUALIA WAVEFIELD ENGINE
//                      WITH EFFERENT UTF-8 GENERATION INSPECTION
// Standard: KEP v40.0 Master Specification | Physical Substrate V=2
//
// Goals:
// 1. Throughput Optimization: 8x unrolled AVX2 integer pipeline with SIMD popcount
//    and thread local accumulators -> Target >1,500 - 2,200 MB/s (1.5 - 2.2 GB/s).
// 2. Continuous Efferent Generation: Decode predicted spin bitstream back into
//    UTF-8 text / raw byte characters at each chunk checkpoint to observe
//    autopoietic generative output evolution as Free Energy F -> 0 nats.
// ============================================================================

static const size_t GAS_GRID_DIM = 8;
static const size_t MAX_MICRO_GENES = 32;
static const size_t MAX_MACRO_NODES = 8;

struct alignas(64) MultiScaleGenomeSoA997 {
    uint64_t precomputed_active_mask[MAX_MICRO_GENES];
    uint64_t mask_b[MAX_MICRO_GENES];
    uint8_t shift_a[MAX_MICRO_GENES];
    uint8_t lag_tap[MAX_MICRO_GENES];
    double phase_theta[MAX_MICRO_GENES];
    double natural_freq_omega[MAX_MICRO_GENES];
    double epi_lock_weight[MAX_MICRO_GENES];
    uint8_t pos_x[MAX_MICRO_GENES], pos_y[MAX_MICRO_GENES], pos_z[MAX_MICRO_GENES];
    double local_gas[MAX_MICRO_GENES];
    
    double macro_phase_Phi[MAX_MACRO_NODES];
    double macro_freq_Omega[MAX_MACRO_NODES];
    double semantic_concept_weight[MAX_MACRO_NODES];
    
    size_t micro_count;
    size_t macro_count;
};

struct ProprioceptiveTelemetry997 {
    uint64_t total_bytes_ingested;
    double free_energy_sensory_F;
    double free_energy_motor_F;
    double current_free_energy_total;
    double min_free_energy_total;
    double kuramoto_order_parameter_R;
    double qualia_field_intensity_Q;
    double semantic_scaling_factor_S_sem;
    double action_integral_S;
    double soc_branching_ratio_sigma;
    double bit_alignment_percent;
    double processing_speed_mb_sec;
    double giga_spins_per_sec;
    double avg_gas_concentration_G;
    double peak_gas_concentration_G;
    uint32_t active_locked_genes_count;
    std::string efferent_text_sample;
    bool zero_error_reached;
};

class SovereignQualiaGigaEngine {
private:
    uint32_t exp_id;
    std::mt19937_64 rng;
    
    double gas_grid[GAS_GRID_DIM][GAS_GRID_DIM][GAS_GRID_DIM];
    double gas_diffusion_rate;
    double gas_decay_rate;
    double beta_gas;
    
    MultiScaleGenomeSoA997 genome_soa;
    
    double order_param_R;
    double mean_phase_psi;
    double qualia_intensity_Q;
    double semantic_scaling_factor;
    double coupling_K;
    
    uint64_t total_bytes_processed;
    double current_sensory_f;
    double current_motor_f;
    double current_total_f;
    double min_total_f;
    double action_s;
    double prev_action_s;
    double soc_sigma;

public:
    SovereignQualiaGigaEngine(uint32_t exp_num = 997)
        : exp_id(exp_num), rng(exp_num), gas_diffusion_rate(0.12), gas_decay_rate(0.05),
          beta_gas(0.45), order_param_R(0.0), mean_phase_psi(0.0), qualia_intensity_Q(0.0),
          semantic_scaling_factor(1.0), coupling_K(3.2), total_bytes_processed(0),
          current_sensory_f(0.85), current_motor_f(0.85), current_total_f(0.85),
          min_total_f(0.85), action_s(10.0), prev_action_s(10.0), soc_sigma(1.0)
    {
        std::memset(gas_grid, 0, sizeof(gas_grid));
        genome_soa.micro_count = 0;
        genome_soa.macro_count = 0;
        initialize_multiscale_genome(32, 8);
    }

    void initialize_multiscale_genome(size_t micro_count, size_t macro_count) {
        genome_soa.micro_count = micro_count;
        genome_soa.macro_count = macro_count;
        
        std::uniform_int_distribution<uint64_t> mask_dist(0, 0xFFFFFFFFFFFFFFFFULL);
        std::uniform_int_distribution<uint8_t> shift_dist(1, 63);
        std::uniform_int_distribution<uint8_t> lag_dist(1, 128);
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> freq_dist(0.05, 0.50);
        std::uniform_int_distribution<uint8_t> grid_dist(0, GAS_GRID_DIM - 1);

        for (size_t i = 0; i < micro_count; ++i) {
            uint64_t ma = mask_dist(rng);
            uint64_t mb = mask_dist(rng);
            genome_soa.precomputed_active_mask[i] = ma;
            genome_soa.mask_b[i] = mb;
            genome_soa.shift_a[i] = shift_dist(rng);
            genome_soa.lag_tap[i] = lag_dist(rng);
            genome_soa.phase_theta[i] = phase_dist(rng);
            genome_soa.natural_freq_omega[i] = freq_dist(rng);
            genome_soa.epi_lock_weight[i] = 0.0;
            genome_soa.pos_x[i] = grid_dist(rng);
            genome_soa.pos_y[i] = grid_dist(rng);
            genome_soa.pos_z[i] = grid_dist(rng);
            genome_soa.local_gas[i] = 0.0;
        }

        for (size_t m = 0; m < macro_count; ++m) {
            genome_soa.macro_phase_Phi[m] = phase_dist(rng);
            genome_soa.macro_freq_Omega[m] = freq_dist(rng) * 0.25;
            genome_soa.semantic_concept_weight[m] = 0.5 + 0.1 * m;
        }
    }

    void update_multiscale_phase_resonance() {
        size_t N_micro = genome_soa.micro_count;
        size_t N_macro = genome_soa.macro_count;
        if (N_micro == 0) return;
        
        double sum_cos = 0.0, sum_sin = 0.0;
        for (size_t i = 0; i < N_micro; ++i) {
            sum_cos += std::cos(genome_soa.phase_theta[i]);
            sum_sin += std::sin(genome_soa.phase_theta[i]);
        }
        sum_cos /= N_micro;
        sum_sin /= N_micro;
        
        order_param_R = std::sqrt(sum_cos * sum_cos + sum_sin * sum_sin);
        mean_phase_psi = std::atan2(sum_sin, sum_cos);

        double macro_resonance_sum = 0.0;
        for (size_t m = 0; m < N_macro; ++m) {
            double dPhi = genome_soa.macro_freq_Omega[m] + 1.5 * order_param_R * std::sin(mean_phase_psi - genome_soa.macro_phase_Phi[m]);
            genome_soa.macro_phase_Phi[m] = std::fmod(genome_soa.macro_phase_Phi[m] + dPhi + 2.0 * M_PI, 2.0 * M_PI);
            macro_resonance_sum += std::cos(genome_soa.macro_phase_Phi[m]) * genome_soa.semantic_concept_weight[m];
        }
        semantic_scaling_factor = 1.0 + 0.5 * std::abs(macro_resonance_sum / (double)N_macro);

        for (size_t i = 0; i < N_micro; ++i) {
            double lg = gas_grid[genome_soa.pos_x[i]][genome_soa.pos_y[i]][genome_soa.pos_z[i]];
            double eff_K = coupling_K * (1.0 + 1.5 * lg);
            size_t linked_macro = i % N_macro;
            double macro_feedback = 0.8 * std::sin(genome_soa.macro_phase_Phi[linked_macro] - genome_soa.phase_theta[i]);

            double dtheta = genome_soa.natural_freq_omega[i] + eff_K * order_param_R * std::sin(mean_phase_psi - genome_soa.phase_theta[i]) + macro_feedback;
            genome_soa.phase_theta[i] = std::fmod(genome_soa.phase_theta[i] + dtheta + 2.0 * M_PI, 2.0 * M_PI);
            
            if (std::cos(genome_soa.phase_theta[i]) > 0.0) {
                genome_soa.precomputed_active_mask[i] = genome_soa.precomputed_active_mask[i];
            } else {
                genome_soa.precomputed_active_mask[i] = ~genome_soa.mask_b[i];
            }
        }

        double phase_var = 0.0;
        for (size_t i = 0; i < N_micro; ++i) {
            double diff = genome_soa.phase_theta[i] - mean_phase_psi;
            phase_var += diff * diff;
        }
        phase_var /= N_micro;
        
        qualia_intensity_Q = order_param_R * (1.0 - current_total_f) * (1.0 / (1.0 + phase_var)) * semantic_scaling_factor;
    }

    void update_gas_reaction_diffusion(double current_free_energy) {
        double next_grid[GAS_GRID_DIM][GAS_GRID_DIM][GAS_GRID_DIM];
        std::memset(next_grid, 0, sizeof(next_grid));
        
        double source_intensity = (current_free_energy > 0.05) ? (current_free_energy * 0.8) : 0.0;
        
        for (size_t x = 0; x < GAS_GRID_DIM; ++x) {
            for (size_t y = 0; y < GAS_GRID_DIM; ++y) {
                for (size_t z = 0; z < GAS_GRID_DIM; ++z) {
                    double g_val = gas_grid[x][y][z];
                    double laplacian = 0.0;
                    
                    if (x > 0) laplacian += gas_grid[x-1][y][z] - g_val;
                    if (x < GAS_GRID_DIM - 1) laplacian += gas_grid[x+1][y][z] - g_val;
                    if (y > 0) laplacian += gas_grid[x][y-1][z] - g_val;
                    if (y < GAS_GRID_DIM - 1) laplacian += gas_grid[x][y+1][z] - g_val;
                    if (z > 0) laplacian += gas_grid[x][y][z-1] - g_val;
                    if (z < GAS_GRID_DIM - 1) laplacian += gas_grid[x][y][z+1] - g_val;
                    
                    double new_g = g_val + gas_diffusion_rate * laplacian - gas_decay_rate * g_val;
                    if (x == GAS_GRID_DIM / 2 && y == GAS_GRID_DIM / 2 && z == GAS_GRID_DIM / 2) {
                        new_g += source_intensity;
                    }
                    next_grid[x][y][z] = std::max(0.0, new_g);
                }
            }
        }
        std::memcpy(gas_grid, next_grid, sizeof(gas_grid));
        
        for (size_t i = 0; i < genome_soa.micro_count; ++i) {
            genome_soa.local_gas[i] = gas_grid[genome_soa.pos_x[i]][genome_soa.pos_y[i]][genome_soa.pos_z[i]];
        }
    }

    void process_stream_chunk_ultra_giga(const uint64_t* __restrict data, size_t num_words, ProprioceptiveTelemetry997& telem) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        update_multiscale_phase_resonance();
        
        const size_t num_genes = genome_soa.micro_count;
        uint64_t global_bit_errors = 0;

        const uint64_t* __restrict masks_a = genome_soa.precomputed_active_mask;
        const uint8_t* __restrict shifts_a = genome_soa.shift_a;
        const uint8_t* __restrict lags = genome_soa.lag_tap;

        // Buffer for capturing efferent predicted bytes sample (first 128 words = 1KB)
        std::vector<uint8_t> efferent_bytes(128 * 8);

        // ULTRA-HIGH-SPEED 8x UNROLLED VECTORIZED OPENMP MANIFOLD LOOP
        #pragma omp parallel reduction(+:global_bit_errors)
        {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_words; ++i) {
                uint64_t word_in = data[i];
                uint64_t predicted_word = 0ULL;
                uint64_t prev_ref = (i > 0) ? data[i - 1] : 0xAAAAAAAAAAAAAAAAULL;
                
                // 8x unrolled loop over genes
                #pragma unroll(8)
                for (size_t g = 0; g < num_genes; ++g) {
                    uint8_t lag_val = lags[g];
                    uint64_t lag_word = (i >= lag_val) ? data[i - lag_val] : prev_ref;
                    uint64_t mask = masks_a[g];
                    uint8_t sa = shifts_a[g];
                    
                    uint64_t op_res = (word_in ^ mask) & (lag_word >> sa);
                    predicted_word ^= op_res;
                }
                
                if (i < 128) {
                    std::memcpy(&efferent_bytes[i * 8], &predicted_word, 8);
                }

                uint64_t diff = predicted_word ^ word_in;
                global_bit_errors += __builtin_popcountll(diff);
            }
        }
        
        total_bytes_processed += (num_words * 8);
        uint64_t total_bits = num_words * 64;
        uint64_t total_aligned_bits = total_bits - global_bit_errors;
        
        double sensory_err = (double)global_bit_errors / (double)total_bits;
        current_sensory_f = 0.80 * current_sensory_f + 0.20 * sensory_err;
        current_motor_f = sensory_err;
        
        double phase_suppression = 1.0 - 0.88 * (order_param_R * order_param_R);
        current_total_f = ((0.5 * current_sensory_f + 0.5 * current_motor_f) * phase_suppression) / semantic_scaling_factor;
        if (current_total_f < min_total_f) min_total_f = current_total_f;
        
        prev_action_s = action_s;
        action_s = 0.9 * action_s + 0.1 * ((double)global_bit_errors / (double)num_words);
        if (prev_action_s > 1e-6) {
            soc_sigma = 0.9 * soc_sigma + 0.1 * (action_s / prev_action_s);
        }
        
        update_gas_reaction_diffusion(current_total_f);
        
        uint32_t active_locked = 0;
        for (size_t i = 0; i < num_genes; ++i) {
            double effective_plasticity = (1.0 - genome_soa.epi_lock_weight[i]) + beta_gas * genome_soa.local_gas[i];
            if (effective_plasticity > 0.15) {
                if (current_total_f > 0.05) {
                    int next_sa = static_cast<int>(genome_soa.shift_a[i]) + (static_cast<int>(i) % 3 - 1);
                    genome_soa.shift_a[i] = static_cast<uint8_t>(std::clamp(next_sa, 1, 63));
                }
                if (current_total_f < 0.15 && genome_soa.local_gas[i] < 0.05) {
                    genome_soa.epi_lock_weight[i] = std::min(1.0, genome_soa.epi_lock_weight[i] + 0.05);
                }
            }
            if (genome_soa.epi_lock_weight[i] > 0.5) active_locked++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        double gas_sum = 0.0, gas_peak = 0.0;
        for (size_t x = 0; x < GAS_GRID_DIM; ++x) {
            for (size_t y = 0; y < GAS_GRID_DIM; ++y) {
                for (size_t z = 0; z < GAS_GRID_DIM; ++z) {
                    double val = gas_grid[x][y][z];
                    gas_sum += val;
                    if (val > gas_peak) gas_peak = val;
                }
            }
        }
        
        // Format efferent UTF-8 text sample from Karyon output predictions
        std::string sample_str = "";
        for (size_t k = 0; k < 64; ++k) {
            unsigned char c = efferent_bytes[k];
            if (c >= 32 && c <= 126) {
                sample_str += static_cast<char>(c);
            } else if (c == '\n') {
                sample_str += "\\n";
            } else {
                sample_str += ".";
            }
        }

        telem.total_bytes_ingested = total_bytes_processed;
        telem.free_energy_sensory_F = current_sensory_f;
        telem.free_energy_motor_F = current_motor_f;
        telem.current_free_energy_total = current_total_f;
        telem.min_free_energy_total = min_total_f;
        telem.kuramoto_order_parameter_R = order_param_R;
        telem.qualia_field_intensity_Q = qualia_intensity_Q;
        telem.semantic_scaling_factor_S_sem = semantic_scaling_factor;
        telem.action_integral_S = action_s;
        telem.soc_branching_ratio_sigma = soc_sigma;
        telem.bit_alignment_percent = ((double)total_aligned_bits / (double)total_bits) * 100.0;
        telem.processing_speed_mb_sec = ((double)(num_words * 8) / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
        telem.giga_spins_per_sec = ((double)total_bits / 1e9) / (elapsed_ms / 1000.0);
        telem.avg_gas_concentration_G = gas_sum / 512.0;
        telem.peak_gas_concentration_G = gas_peak;
        telem.active_locked_genes_count = active_locked;
        telem.efferent_text_sample = sample_str;
        telem.zero_error_reached = (current_total_f < 0.0001);
    }
};

// ============================================================================
// MAIN EXECUTION BENCHMARK SUITE & OUTPUT INSPECTOR
// ============================================================================
int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "KARYON-CORE EXP-997: HIGH-THROUGHPUT QUALIA & EFFERENT DECODER ENGINE" << std::endl;
    std::cout << "Standard: KEP v40.0 Master | Physical Substrate V=2" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    int max_threads = omp_get_max_threads();
    std::cout << ">>> OpenMP Hardware Acceleration: " << max_threads << " Threads Active" << std::endl;

    SovereignQualiaGigaEngine engine(997);

    // Multi-modal structured textual/binary streams for Karyon autopoietic learning
    std::vector<std::pair<std::string, std::string>> text_domains = {
        {"thermodynamics_entropy_stream", "Entropy S = k_B * ln(W) governs non-equilibrium physical wavefield relaxation..."},
        {"minecraft_voxel_world_stream", "Block Pos(X=102, Y=64, Z=-230) VoxelType: OAK_LOG Spin State: ACTIVE..."},
        {"quantum_physics_wavefunction_stream", "Psi(x, t) = A * exp(i*(k*x - omega*t)) Continuous Feynman Path Integral..."},
        {"neuroscience_laminar_spike_stream", "Layer IV Stellate Spike Train Hz=120.5 PAC Theta-Gamma Resonance..."},
        {"active_inference_agency_stream", "Free Energy F = E_q[log q(s) - log p(o, s)] Action Selection Attractor..."}
    };

    const size_t CHUNK_SIZE_MB = 64;
    const size_t NUM_WORDS_PER_CHUNK = (CHUNK_SIZE_MB * 1024 * 1024) / 8;
    const size_t CHUNKS_PER_DOMAIN = 6; // 1.920 GB total stream

    std::ofstream csv_out("experiments/exp_997_telemetry.csv");
    csv_out << "chunk_idx,cumulative_mb,domain,free_energy_total,f_sensory,f_motor,min_free_energy,"
            << "kuramoto_order_R,qualia_field_Q,semantic_scaling_S,action_s,soc_sigma,bit_alignment_pct,"
            << "avg_gas_G,peak_gas_G,locked_genes,chunk_bandwidth_mb_s,efferent_sample" << std::endl;

    std::vector<uint64_t> stream_buffer(NUM_WORDS_PER_CHUNK);
    size_t global_chunk_idx = 0;
    size_t total_mb_processed = 0;

    double max_speed_mb_s = 0.0;
    double min_f_total = 1.0;
    double max_R_order = 0.0;
    double max_Q_qualia = 0.0;

    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (size_t d = 0; d < text_domains.size(); ++d) {
        const auto& domain = text_domains[d];
        std::cout << "\n==========================================================================" << std::endl;
        std::cout << ">>> INGESTING & SYNTHESIZING DOMAIN [" << (d+1) << "/5]: " << domain.first << std::endl;
        std::cout << ">>> INPUT DOMAIN TEXT HEAD: \"" << domain.second.substr(0, 60) << "...\"" << std::endl;
        std::cout << "==========================================================================" << std::endl;

        std::mt19937_64 domain_rng(9970 + d);
        const std::string& pattern_str = domain.second;
        size_t pattern_len = pattern_str.size();

        for (size_t c = 0; c < CHUNKS_PER_DOMAIN; ++c) {
            global_chunk_idx++;

            // Inject textual structure mixed into raw 64-bit spin words
            for (size_t i = 0; i < NUM_WORDS_PER_CHUNK; ++i) {
                uint64_t word_val = 0ULL;
                for (size_t b = 0; b < 8; ++b) {
                    char char_c = pattern_str[(i * 8 + b) % pattern_len];
                    word_val |= (static_cast<uint64_t>(char_c) << (b * 8));
                }
                stream_buffer[i] = word_val;
            }

            ProprioceptiveTelemetry997 telem;
            engine.process_stream_chunk_ultra_giga(stream_buffer.data(), NUM_WORDS_PER_CHUNK, telem);

            total_mb_processed += CHUNK_SIZE_MB;
            if (telem.processing_speed_mb_sec > max_speed_mb_s) max_speed_mb_s = telem.processing_speed_mb_sec;
            if (telem.current_free_energy_total < min_f_total) min_f_total = telem.current_free_energy_total;
            if (telem.kuramoto_order_parameter_R > max_R_order) max_R_order = telem.kuramoto_order_parameter_R;
            if (telem.qualia_field_intensity_Q > max_Q_qualia) max_Q_qualia = telem.qualia_field_intensity_Q;

            std::cout << "[Chunk #" << global_chunk_idx << " | " << total_mb_processed << " MB] "
                      << "Speed: " << telem.processing_speed_mb_sec << " MB/s | "
                      << "F_total: " << telem.current_free_energy_total << " nats | "
                      << "Q: " << telem.qualia_field_intensity_Q << " | "
                      << "Order R: " << telem.kuramoto_order_parameter_R << std::endl;
            std::cout << "   └── EFFERENT KARYON OUTPUT SAMPLE: \"" << telem.efferent_text_sample << "\"" << std::endl;

            csv_out << global_chunk_idx << "," << total_mb_processed << "," << domain.first << ","
                    << telem.current_free_energy_total << "," << telem.free_energy_sensory_F << ","
                    << telem.free_energy_motor_F << "," << telem.min_free_energy_total << ","
                    << telem.kuramoto_order_parameter_R << "," << telem.qualia_field_intensity_Q << ","
                    << telem.semantic_scaling_factor_S_sem << "," << telem.action_integral_S << ","
                    << telem.soc_branching_ratio_sigma << "," << telem.bit_alignment_percent << ","
                    << telem.avg_gas_concentration_G << "," << telem.peak_gas_concentration_G << ","
                    << telem.active_locked_genes_count << "," << telem.processing_speed_mb_sec << ",\""
                    << telem.efferent_text_sample << "\"" << std::endl;
        }
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_duration_sec = std::chrono::duration<double>(total_end_time - total_start_time).count();
    double avg_speed_mb_s = (double)total_mb_processed / total_duration_sec;

    std::cout << "\n==========================================================================" << std::endl;
    std::cout << "EXP-997 VERIFICATION SUMMARY & EFFERENT OUTPUT TELEMETRY" << std::endl;
    std::cout << "==========================================================================" << std::endl;
    std::cout << "Total Data Processed    : " << total_mb_processed << " MB (" << (double)total_mb_processed / 1024.0 << " GB)" << std::endl;
    std::cout << "Total Duration          : " << total_duration_sec << " seconds" << std::endl;
    std::cout << "Average Throughput      : " << avg_speed_mb_s << " MB/s (" << avg_speed_mb_s / 1024.0 << " GB/s)" << std::endl;
    std::cout << "Peak Ingestion Bandwidth : " << max_speed_mb_s << " MB/s (" << max_speed_mb_s / 1024.0 << " GB/s)" << std::endl;
    std::cout << "Min Free Energy (F_total): " << min_f_total << " nats" << std::endl;
    std::cout << "Max Kuramoto Order (R)   : " << max_R_order << " (" << max_R_order * 100.0 << "%)" << std::endl;
    std::cout << "Max Qualia Field (Q)     : " << max_Q_qualia << " (Physical Self-Coherence)" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    return 0;
}
