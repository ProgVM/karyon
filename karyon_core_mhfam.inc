// ============================================================================
// 18. MULTI-HEAD FAST-WEIGHT ASSOCIATIVE MEMORY & MARGIN RESONANCE (EXP-398 C++20)
// KEP Principle 27: Strict Deterministic Orthogonal Basis & Multi-Head Working Memory
// ============================================================================
class MultiHeadFastAssociativeMemoryImpl : public torch::nn::Module {
public:
    int64_t dim;
    int64_t num_heads;
    int64_t head_dim;
    float eta;

    torch::Tensor w_q, w_k, w_v, w_out;
    torch::Tensor decay_rates; // [num_heads]
    torch::Tensor m_fast;      // [num_heads, head_dim, head_dim]

    MultiHeadFastAssociativeMemoryImpl(int64_t dim = 258, int64_t num_heads = 6, float eta = 0.20f, std::string device_str = "cpu")
        : dim(dim), num_heads(num_heads), head_dim(dim / num_heads), eta(eta) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        float scale = 1.0f / std::sqrt((float)dim);

        w_q = register_parameter("w_q", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_k = register_parameter("w_k", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_v = register_parameter("w_v", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);
        w_out = register_parameter("w_out", torch::randn({dim, dim}, torch::TensorOptions().device(device)) * scale);

        std::vector<float> decays(num_heads);
        for (int64_t h = 0; h < num_heads; ++h) {
            decays[h] = 0.80f + 0.03f * h;
        }
        decay_rates = register_buffer("decay_rates", torch::tensor(decays, torch::TensorOptions().device(device)));
        m_fast = register_buffer("m_fast", torch::zeros({num_heads, head_dim, head_dim}, torch::TensorOptions().device(device)));
        this->to(device);
    }

    void reset_memory() {
        m_fast.zero_();
    }

    torch::Tensor read_memory(torch::Tensor x) {
        bool is_batched = x.dim() == 2;
        auto x_in = is_batched ? x : x.unsqueeze(0);
        auto B = x_in.size(0);

        auto q = torch::matmul(x_in, w_q.t()).view({B, num_heads, head_dim}); // [B, H, H_d]
        auto rec_out = torch::zeros_like(q);

        for (int64_t h = 0; h < num_heads; ++h) {
            auto q_h = q.select(1, h).unsqueeze(-1); // [B, H_d, 1]
            auto m_h = m_fast.select(0, h).unsqueeze(0); // [1, H_d, H_d]
            auto rec_h = torch::matmul(m_h, q_h).squeeze(-1); // [B, H_d]
            rec_out.select(1, h).copy_(rec_h);
        }

        auto stacked = rec_out.view({B, dim});
        auto out = torch::matmul(stacked, w_out.t());
        return is_batched ? out : out.squeeze(0);
    }

    void write_memory(torch::Tensor x, torch::Tensor target_v) {
        torch::NoGradGuard no_grad;
        bool is_batched = x.dim() == 2;
        auto x_in = is_batched ? x : x.unsqueeze(0);
        auto v_in = is_batched ? target_v : target_v.unsqueeze(0);
        auto B = x_in.size(0);

        auto k = torch::nn::functional::normalize(
            torch::matmul(x_in, w_k.t()).view({B, num_heads, head_dim}),
            torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1)
        );
        auto v = torch::matmul(v_in, w_v.t()).view({B, num_heads, head_dim});

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < num_heads; ++h) {
                float gamma_h = decay_rates[h].item<float>();
                auto k_hb = k.select(0, b).select(0, h); // [H_d]
                auto v_hb = v.select(0, b).select(0, h); // [H_d]
                auto outer_h = torch::outer(v_hb, k_hb); // [H_d, H_d]
                
                auto cur_m = m_fast.select(0, h);
                cur_m.copy_(gamma_h * cur_m + eta * outer_h);
            }
        }
    }
};
TORCH_MODULE(MultiHeadFastAssociativeMemory);

class StrictOrthogonalNexusImpl : public torch::nn::Module {
public:
    int64_t num_basins;
    torch::Tensor basis; // [258, 258] Identity

    StrictOrthogonalNexusImpl(int64_t num_basins = 258, std::string device_str = "cpu")
        : num_basins(num_basins) {
        auto device = device_str.find("cuda") != std::string::npos && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        basis = register_buffer("basis", torch::eye(num_basins, torch::TensorOptions().device(device)));
        this->to(device);
    }

    torch::Tensor perceive(int64_t byte_idx) {
        return basis.select(0, byte_idx);
    }

    std::tuple<torch::Tensor, int64_t, float, float> compute_margin_free_energy(
        torch::Tensor psi,
        int64_t target_idx,
        float margin = 0.40f
    ) {
        auto psi_norm = torch::nn::functional::normalize(
            psi, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1)
        );
        auto alignments = psi_norm; // since basis is I: <psi, e_i> = psi_i

        float target_score = alignments[target_idx].item<float>();

        auto competitor_scores = alignments.clone();
        competitor_scores[target_idx] = -1e9f;
        auto max_comp_tensor = torch::max(competitor_scores, -1);
        float max_comp_score = std::get<0>(max_comp_tensor).item<float>();

        auto margin_violation = torch::clamp(
            std::get<0>(max_comp_tensor) - alignments[target_idx] + margin,
            /*min=*/0.0f
        );

        auto alignment_strain = 1.0f - alignments[target_idx];
        auto total_energy = margin_violation + 0.5f * alignment_strain;

        int64_t pred_idx = torch::argmax(alignments).item<int64_t>();
        return std::make_tuple(total_energy, pred_idx, target_score, max_comp_score);
    }
};
TORCH_MODULE(StrictOrthogonalNexus);
