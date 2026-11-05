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

// ============================================================================
// KARYON-CORE EXP-994: MULTI-THREADED OPENMP SOLITON RESONANCE ENGINE (2 GB/s PARITY)
// Standard: KEP v40.0 Master Specification
// Hypothesis:
// OpenMP parallel thread partitioning overPacked Bit-Manifold combined with
// Harmonic Phase-Frequency Resonant Locking (Kuramoto + 3D Gas) will restore
// streaming ingestion throughput to >1,500 - 2,000 MB/s (2 GB/s domain)
// while simultaneously accelerating Free Energy dissipation down to F < 0.050 nats.
// ============================================================================

static const size_t GAS_GRID_DIM = 8; // 8x8x8 = 512 cells
static const size_t MANIFOLD_HISTORY_CAP = 256;

struct KuramotoGeneNode994 {
    uint64_t mask_a;
    uint64_t mask_b;
    uint64_t precomputed_active_mask;
    uint8_t shift_a;
    uint8_t shift_b;
    uint8_t lag_tap;
    uint8_t op_type; // 0: XOR-AND, 1: XOR-OR, 2: AND-NOT, 3: XNOR-ADD
    
    // Kuramoto phase parameters
    double phase_theta;
    double natural_freq_omega;
    
    // Epigenetic & Gas coupling
    double epi_lock_weight; // 0.0 (plastic) to 1.0 (epigenetically locked)
    double causal_efficacy;
    
    // 3D Gas spatial coordinates
    uint8_t pos_x, pos_y, pos_z;
    double local_gas_concentration;
};

struct ProprioceptiveTelemetry994 {
    uint64_t total_bytes_ingested;
    double free_energy_sensory_F;
    double free_energy_motor_F;
    double current_free_energy_total;
    double min_free_energy_total;
    double kuramoto_order_parameter_R;
    double action_integral_S;
    double soc_branching_ratio_sigma;
    double bit_alignment_percent;
    double processing_speed_mb_sec;
    double giga_spins_per_sec;
    double avg_gas_concentration_G;
    double peak_gas_concentration_G;
    uint32_t active_sprouted_lags_count;
    uint32_t active_locked_genes_count;
    bool zero_error_reached;
};

class SovereignOpenMPSolitonEngine {
private:
    uint32_t exp_id;
    std::mt19937_64 rng;
    
    // 3D Gas Reaction-Diffusion Lattice
    double gas_grid[GAS_GRID_DIM][GAS_GRID_DIM][GAS_GRID_DIM];
    double gas_diffusion_rate;
    double gas_decay_rate;
    double beta_gas;
    
    // Genome & Wavefield Manifold
    std::vector<KuramotoGeneNode994> genome;
    uint64_t working_state_history[MANIFOLD_HISTORY_CAP];
    size_t history_head;
    
    // Kuramoto Global Phase Metrics
    double order_param_R;
    double mean_phase_psi;
    double coupling_K;
    
    // Telemetry & Homeostasis
    uint64_t total_bytes_processed;
    double current_sensory_f;
    double current_motor_f;
    double current_total_f;
    double min_total_f;
    double action_s;
    double prev_action_s;
    double soc_sigma;

public:
    SovereignOpenMPSolitonEngine(uint32_t exp_num = 994)
        : exp_id(exp_num), rng(exp_num), gas_diffusion_rate(0.12), gas_decay_rate(0.05),
          beta_gas(0.45), history_head(0), order_param_R(0.0), mean_phase_psi(0.0),
          coupling_K(2.8), total_bytes_processed(0), current_sensory_f(0.85),
          current_motor_f(0.85), current_total_f(0.85), min_total_f(0.85),
          action_s(10.0), prev_action_s(10.0), soc_sigma(1.0)
    {
        std::memset(gas_grid, 0, sizeof(gas_grid));
        std::memset(working_state_history, 0, sizeof(working_state_history));
        
        initialize_genome_seeds(24);
    }

    void initialize_genome_seeds(size_t seed_count) {
        genome.clear();
        std::uniform_int_distribution<uint64_t> mask_dist(0, 0xFFFFFFFFFFFFFFFFULL);
        std::uniform_int_distribution<uint8_t> shift_dist(1, 63);
        std::uniform_int_distribution<uint8_t> lag_dist(1, 128);
        std::uniform_int_distribution<uint8_t> op_dist(0, 3);
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> freq_dist(0.05, 0.50);
        std::uniform_int_distribution<uint8_t> grid_dist(0, GAS_GRID_DIM - 1);

        for (size_t i = 0; i < seed_count; ++i) {
            KuramotoGeneNode994 gene;
            gene.mask_a = mask_dist(rng);
            gene.mask_b = mask_dist(rng);
            gene.precomputed_active_mask = gene.mask_a;
            gene.shift_a = shift_dist(rng);
            gene.shift_b = shift_dist(rng);
            gene.lag_tap = lag_dist(rng);
            gene.op_type = op_dist(rng);
            gene.phase_theta = phase_dist(rng);
            gene.natural_freq_omega = freq_dist(rng);
            gene.epi_lock_weight = 0.0;
            gene.causal_efficacy = 0.5;
            gene.pos_x = grid_dist(rng);
            gene.pos_y = grid_dist(rng);
            gene.pos_z = grid_dist(rng);
            gene.local_gas_concentration = 0.0;
            genome.push_back(gene);
        }
    }

    void update_kuramoto_phase_dynamics_and_hoist_masks() {
        if (genome.empty()) return;
        
        double sum_cos = 0.0, sum_sin = 0.0;
        for (const auto& gene : genome) {
            sum_cos += std::cos(gene.phase_theta);
            sum_sin += std::sin(gene.phase_theta);
        }
        
        size_t N = genome.size();
        sum_cos /= N;
        sum_sin /= N;
        
        order_param_R = std::sqrt(sum_cos * sum_cos + sum_sin * sum_sin);
        mean_phase_psi = std::atan2(sum_sin, sum_cos);
        
        // Harmonic frequency locking & phase updates
        for (auto& gene : genome) {
            double local_gas = gas_grid[gene.pos_x][gene.pos_y][gene.pos_z];
            double eff_K = coupling_K * (1.0 + 1.5 * local_gas);
            
            // Phase adjustment toward global resonant wavefield
            double dtheta = gene.natural_freq_omega + eff_K * order_param_R * std::sin(mean_phase_psi - gene.phase_theta);
            gene.phase_theta = std::fmod(gene.phase_theta + dtheta + 2.0 * M_PI, 2.0 * M_PI);
            
            // Hoist active bitmask
            if (std::cos(gene.phase_theta) > 0.0) {
                gene.precomputed_active_mask = gene.mask_a;
            } else {
                gene.precomputed_active_mask = ~gene.mask_b;
            }
        }
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
                    size_t neighbors = 0;
                    
                    if (x > 0) { laplacian += gas_grid[x-1][y][z] - g_val; neighbors++; }
                    if (x < GAS_GRID_DIM - 1) { laplacian += gas_grid[x+1][y][z] - g_val; neighbors++; }
                    if (y > 0) { laplacian += gas_grid[x][y-1][z] - g_val; neighbors++; }
                    if (y < GAS_GRID_DIM - 1) { laplacian += gas_grid[x][y+1][z] - g_val; neighbors++; }
                    if (z > 0) { laplacian += gas_grid[x][y][z-1] - g_val; neighbors++; }
                    if (z < GAS_GRID_DIM - 1) { laplacian += gas_grid[x][y][z+1] - g_val; neighbors++; }
                    
                    double new_g = g_val + gas_diffusion_rate * laplacian - gas_decay_rate * g_val;
                    if (x == GAS_GRID_DIM / 2 && y == GAS_GRID_DIM / 2 && z == GAS_GRID_DIM / 2) {
                        new_g += source_intensity;
                    }
                    next_grid[x][y][z] = std::max(0.0, new_g);
                }
            }
        }
        std::memcpy(gas_grid, next_grid, sizeof(gas_grid));
        
        for (auto& gene : genome) {
            gene.local_gas_concentration = gas_grid[gene.pos_x][gene.pos_y][gene.pos_z];
        }
    }

    void process_stream_chunk_openmp(const uint64_t* data, size_t num_words, ProprioceptiveTelemetry994& telem) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 1. Pre-chunk Kuramoto Phase Alignment & Mask Hoisting
        update_kuramoto_phase_dynamics_and_hoist_masks();
        
        uint64_t total_bits = num_words * 64;
        uint64_t total_aligned_bits = 0;
        double chunk_action_accumulator = 0.0;
        
        const size_t num_genes = genome.size();

        // Snapshot genome array into a local cache for thread-safe OpenMP read
        std::vector<KuramotoGeneNode994> local_genome = genome;
        
        // HIGH-SPEED MULTI-THREADED OPENMP PARALLEL MANIFOLD LOOP
        #pragma omp parallel reduction(+:total_aligned_bits, chunk_action_accumulator)
        {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_words; ++i) {
                uint64_t word_in = data[i];
                uint64_t predicted_word = 0;
                
                // Fetch lag reference from lookback
                uint64_t prev_ref = (i > 0) ? data[i - 1] : 0xAAAAAAAAAAAAAAAAULL;
                
                for (size_t g = 0; g < num_genes; ++g) {
                    const auto& gene = local_genome[g];
                    uint64_t lag_word = (i >= gene.lag_tap) ? data[i - gene.lag_tap] : prev_ref;
                    uint64_t op_res = 0;
                    
                    switch (gene.op_type) {
                        case 0: op_res = (word_in ^ gene.precomputed_active_mask) & (lag_word >> gene.shift_a); break;
                        case 1: op_res = (word_in ^ gene.mask_b) | (lag_word << gene.shift_b); break;
                        case 2: op_res = (word_in & gene.precomputed_active_mask) & ~(lag_word >> gene.shift_b); break;
                        case 3: op_res = ~(word_in ^ lag_word) + gene.mask_b; break;
                    }
                    
                    predicted_word ^= op_res;
                }
                
                uint64_t diff = predicted_word ^ word_in;
                uint32_t bit_errors = __builtin_popcountll(diff);
                total_aligned_bits += (64 - bit_errors);
                chunk_action_accumulator += ((double)bit_errors / 64.0);
            }
        }
        
        // Update history tail with last words
        size_t words_to_copy = std::min(num_words, MANIFOLD_HISTORY_CAP);
        for (size_t k = 0; k < words_to_copy; ++k) {
            working_state_history[k] = data[num_words - words_to_copy + k];
        }
        
        total_bytes_processed += (num_words * 8);
        
        // Free Energy Calculations
        double sensory_err = 1.0 - ((double)total_aligned_bits / (double)total_bits);
        current_sensory_f = 0.80 * current_sensory_f + 0.20 * sensory_err;
        
        double motor_err = 0.80 * current_motor_f + 0.20 * (chunk_action_accumulator / num_words);
        current_motor_f = motor_err;
        
        // Phase Order R^2 non-linear suppression of residual Free Energy
        double phase_suppression = 1.0 - 0.85 * (order_param_R * order_param_R);
        current_total_f = (0.5 * current_sensory_f + 0.5 * current_motor_f) * phase_suppression;
        if (current_total_f < min_total_f) min_total_f = current_total_f;
        
        // Action & SOC
        prev_action_s = action_s;
        action_s = 0.9 * action_s + 0.1 * chunk_action_accumulator;
        if (prev_action_s > 1e-6) {
            soc_sigma = 0.9 * soc_sigma + 0.1 * (action_s / prev_action_s);
        }
        
        // 2. Gas Reaction-Diffusion
        update_gas_reaction_diffusion(current_total_f);
        
        // 3. Genome Adaptation & Epigenetics
        uint32_t active_locked = 0;
        for (auto& gene : genome) {
            double effective_plasticity = (1.0 - gene.epi_lock_weight) + beta_gas * gene.local_gas_concentration;
            if (effective_plasticity > 0.15) {
                if (current_total_f > 0.05) {
                    std::uniform_int_distribution<int> delta_dist(-1, 1);
                    gene.shift_a = std::clamp(gene.shift_a + delta_dist(rng), 1, 63);
                    gene.shift_b = std::clamp(gene.shift_b + delta_dist(rng), 1, 63);
                }
                if (current_total_f < 0.15 && gene.local_gas_concentration < 0.05) {
                    gene.epi_lock_weight = std::min(1.0, gene.epi_lock_weight + 0.05);
                }
            }
            if (gene.epi_lock_weight > 0.5) active_locked++;
        }
        
        // SOC Sprouting
        if (std::abs(soc_sigma - 1.0) > 0.02 && genome.size() < 64) {
            KuramotoGeneNode994 new_gene;
            std::uniform_int_distribution<uint64_t> mask_dist(0, 0xFFFFFFFFFFFFFFFFULL);
            std::uniform_int_distribution<uint8_t> shift_dist(1, 63);
            std::uniform_int_distribution<uint8_t> lag_dist(1, 128);
            std::uniform_int_distribution<uint8_t> op_dist(0, 3);
            std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
            std::uniform_real_distribution<double> freq_dist(0.05, 0.50);
            std::uniform_int_distribution<uint8_t> grid_dist(0, GAS_GRID_DIM - 1);
            
            new_gene.mask_a = mask_dist(rng);
            new_gene.mask_b = mask_dist(rng);
            new_gene.precomputed_active_mask = new_gene.mask_a;
            new_gene.shift_a = shift_dist(rng);
            new_gene.shift_b = shift_dist(rng);
            new_gene.lag_tap = lag_dist(rng);
            new_gene.op_type = op_dist(rng);
            new_gene.phase_theta = phase_dist(rng);
            new_gene.natural_freq_omega = freq_dist(rng);
            new_gene.epi_lock_weight = 0.0;
            new_gene.causal_efficacy = 0.45;
            new_gene.pos_x = grid_dist(rng);
            new_gene.pos_y = grid_dist(rng);
            new_gene.pos_z = grid_dist(rng);
            new_gene.local_gas_concentration = 0.0;
            genome.push_back(new_gene);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        double gas_sum = 0.0;
        double gas_peak = 0.0;
        for (size_t x = 0; x < GAS_GRID_DIM; ++x) {
            for (size_t y = 0; y < GAS_GRID_DIM; ++y) {
                for (size_t z = 0; z < GAS_GRID_DIM; ++z) {
                    double val = gas_grid[x][y][z];
                    gas_sum += val;
                    if (val > gas_peak) gas_peak = val;
                }
            }
        }
        
        telem.total_bytes_ingested = total_bytes_processed;
        telem.free_energy_sensory_F = current_sensory_f;
        telem.free_energy_motor_F = current_motor_f;
        telem.current_free_energy_total = current_total_f;
        telem.min_free_energy_total = min_total_f;
        telem.kuramoto_order_parameter_R = order_param_R;
        telem.action_integral_S = action_s;
        telem.soc_branching_ratio_sigma = soc_sigma;
        telem.bit_alignment_percent = ((double)total_aligned_bits / (double)total_bits) * 100.0;
        telem.processing_speed_mb_sec = ((double)(num_words * 8) / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
        telem.giga_spins_per_sec = ((double)total_bits / 1e9) / (elapsed_ms / 1000.0);
        telem.avg_gas_concentration_G = gas_sum / (double)(GAS_GRID_DIM * GAS_GRID_DIM * GAS_GRID_DIM);
        telem.peak_gas_concentration_G = gas_peak;
        telem.active_sprouted_lags_count = genome.size();
        telem.active_locked_genes_count = active_locked;
        telem.zero_error_reached = (current_total_f < 0.0001);
    }
};

// ============================================================================
// MAIN EXECUTION BENCHMARK SUITE
// ============================================================================
int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "KARYON-CORE EXP-994: MULTI-THREADED OPENMP SOLITON RESONANCE ENGINE" << std::endl;
    std::cout << "Standard: KEP v40.0 Master | Physical Substrate V=2" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    int max_threads = omp_get_max_threads();
    std::cout << ">>> OpenMP Hardware Acceleration Active: " << max_threads << " Parallel Threads" << std::endl;

    SovereignOpenMPSolitonEngine engine(994);

    std::vector<std::pair<std::string, uint32_t>> domains = {
        {"thermodynamics_entropy_stream", 9941},
        {"minecraft_voxel_world_stream", 9942},
        {"quantum_physics_wavefunction_stream", 9943},
        {"neuroscience_laminar_spike_stream", 9944},
        {"active_inference_agency_stream", 9945}
    };

    const size_t CHUNK_SIZE_MB = 64;
    const size_t NUM_WORDS_PER_CHUNK = (CHUNK_SIZE_MB * 1024 * 1024) / 8;
    const size_t CHUNKS_PER_DOMAIN = 6; // 1.920 GB total stream

    std::ofstream csv_out("experiments/exp_994_telemetry.csv");
    csv_out << "chunk_idx,cumulative_mb,domain,free_energy_total,f_sensory,f_motor,min_free_energy,"
            << "kuramoto_order_R,action_s,soc_sigma,bit_alignment_pct,avg_gas_G,peak_gas_G,sprouted_lags,"
            << "locked_genes,chunk_bandwidth_mb_s" << std::endl;

    std::vector<uint64_t> stream_buffer(NUM_WORDS_PER_CHUNK);
    size_t global_chunk_idx = 0;
    size_t total_mb_processed = 0;

    double max_speed_mb_s = 0.0;
    double min_f_total = 1.0;
    double max_R_order = 0.0;

    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (const auto& domain : domains) {
        std::cout << "\n>>> INGESTING MULTI-MODAL DOMAIN: [" << domain.first << "]" << std::endl;
        std::mt19937_64 domain_rng(domain.second);

        for (size_t c = 0; c < CHUNKS_PER_DOMAIN; ++c) {
            global_chunk_idx++;

            // Synthetic bit stream generation
            for (size_t i = 0; i < NUM_WORDS_PER_CHUNK; ++i) {
                uint64_t base = domain_rng();
                uint64_t pattern = (i > 0) ? (stream_buffer[i - 1] << 5) ^ (base & 0x0F0F0F0F0F0F0F0FULL) : base;
                stream_buffer[i] = pattern;
            }

            ProprioceptiveTelemetry994 telem;
            engine.process_stream_chunk_openmp(stream_buffer.data(), NUM_WORDS_PER_CHUNK, telem);

            total_mb_processed += CHUNK_SIZE_MB;
            if (telem.processing_speed_mb_sec > max_speed_mb_s) max_speed_mb_s = telem.processing_speed_mb_sec;
            if (telem.current_free_energy_total < min_f_total) min_f_total = telem.current_free_energy_total;
            if (telem.kuramoto_order_parameter_R > max_R_order) max_R_order = telem.kuramoto_order_parameter_R;

            std::cout << "Chunk #" << global_chunk_idx << " (" << total_mb_processed << " MB) | "
                      << "F_total: " << telem.current_free_energy_total << " nats | "
                      << "F_sens: " << telem.free_energy_sensory_F << " | "
                      << "Order R: " << telem.kuramoto_order_parameter_R << " | "
                      << "Gas <G>: " << telem.avg_gas_concentration_G << " | "
                      << "Speed: " << telem.processing_speed_mb_sec << " MB/s" << std::endl;

            csv_out << global_chunk_idx << "," << total_mb_processed << "," << domain.first << ","
                    << telem.current_free_energy_total << "," << telem.free_energy_sensory_F << ","
                    << telem.free_energy_motor_F << "," << telem.min_free_energy_total << ","
                    << telem.kuramoto_order_parameter_R << "," << telem.action_integral_S << ","
                    << telem.soc_branching_ratio_sigma << "," << telem.bit_alignment_percent << ","
                    << telem.avg_gas_concentration_G << "," << telem.peak_gas_concentration_G << ","
                    << telem.active_sprouted_lags_count << "," << telem.active_locked_genes_count << ","
                    << telem.processing_speed_mb_sec << std::endl;
        }
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_duration_sec = std::chrono::duration<double>(total_end_time - total_start_time).count();
    double avg_speed_mb_s = (double)total_mb_processed / total_duration_sec;

    std::cout << "\n==========================================================================" << std::endl;
    std::cout << "EXP-994 VERIFICATION SUMMARY & PHYSICAL TELEMETRY REPORT" << std::endl;
    std::cout << "==========================================================================" << std::endl;
    std::cout << "Total Data Processed    : " << total_mb_processed << " MB (" << (double)total_mb_processed / 1024.0 << " GB)" << std::endl;
    std::cout << "Total Duration          : " << total_duration_sec << " seconds" << std::endl;
    std::cout << "Average Throughput      : " << avg_speed_mb_s << " MB/s (" << avg_speed_mb_s / 1024.0 << " GB/s)" << std::endl;
    std::cout << "Peak Ingestion Bandwidth : " << max_speed_mb_s << " MB/s (" << max_speed_mb_s / 1024.0 << " GB/s)" << std::endl;
    std::cout << "Min Free Energy (F_total): " << min_f_total << " nats" << std::endl;
    std::cout << "Max Kuramoto Order (R)   : " << max_R_order << " (" << max_R_order * 100.0 << "%)" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    return 0;
}
