// karyon_core.cpp
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <memory>

namespace py = pybind11;

// ============================================================================
// 1. FAST BYTE-LEVEL TOKENIZER (UTF-8 DIRECT ENGINE)
// ============================================================================
class ByteTokenizer {
public:
    int64_t vocab_size;
    int64_t pad_id;
    int64_t eos_id;

    ByteTokenizer(int64_t vocab_size = 258) 
        : vocab_size(vocab_size), pad_id(256), eos_id(257) {}

    std::vector<int64_t> encode(const std::string& text) {
        std::vector<int64_t> bytes;
        bytes.reserve(text.size() + 1);
        for (unsigned char c : text) {
            bytes.push_back(static_cast<int64_t>(c));
        }
        bytes.push_back(eos_id);
        return bytes;
    }

    std::string decode(const std::vector<int64_t>& ids) {
        std::string result;
        result.reserve(ids.size());
        for (int64_t id : ids) {
            if (id >= 0 && id <= 255) {
                result.push_back(static_cast<char>(id));
            }
        }
        return result;
    }

    py::bytes decode_bytes(const std::vector<int64_t>& ids) {
        std::string result = decode(ids);
        return py::bytes(result);
    }
};

// ============================================================================
// 2. HOMEOSTATIC SOMATIC CONTROLLER & DYNAMIC ALLOSTASIS (ASHBY ULTRASTABILITY)
// ============================================================================
struct HomeostaticUnit {
    torch::Tensor state;
    torch::Tensor prev_pain;
    torch::Tensor consecutive_inactivity;
    std::string device;

    HomeostaticUnit(int64_t batch_size = 1, std::string device_str = "cpu") 
        : device(device_str) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            opts = opts.device(torch::kCUDA);
        } else {
            opts = opts.device(torch::kCPU);
        }

        state = torch::tensor({{0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}}, opts).repeat({batch_size, 1});
        prev_pain = torch::zeros({batch_size, 1}, opts);
        consecutive_inactivity = torch::zeros({batch_size, 1}, opts);
    }

    torch::Tensor update(torch::Tensor action_cost, torch::Tensor prediction_error, 
                         torch::Tensor epistemic_entropy, torch::Tensor cog_action) {
        auto curiosity = state.select(1, 0).unsqueeze(1);
        auto energy    = state.select(1, 1).unsqueeze(1);
        auto stability = state.select(1, 2).unsqueeze(1);
        auto health    = state.select(1, 3).unsqueeze(1);

        energy = torch::clamp(energy - action_cost + 0.0015f, 0.0f, 1.0f);
        curiosity = torch::clamp(curiosity + 0.2f * prediction_error - 0.02f, 0.0f, 1.0f);

        auto inactive_mask = (cog_action == 1) | (cog_action == 2);
        consecutive_inactivity = torch::where(inactive_mask, consecutive_inactivity + 1.0f, torch::zeros_like(consecutive_inactivity));

        auto dopamine = torch::clamp(prediction_error * 2.0f, 0.0f, 1.0f);
        auto curiosity_diff = torch::abs(curiosity - 0.8f) * (1.0f + 0.15f * consecutive_inactivity);
        auto current_pain = curiosity_diff + torch::abs(energy - 1.0f) + torch::abs(stability - 1.0f) + torch::abs(health - 1.0f);

        auto pain_jump = torch::clamp(current_pain - prev_pain, 0.0f, 1.0f);
        prev_pain = current_pain;

        auto noradrenaline = torch::clamp(0.6f * (1.0f - stability) + 0.85f * prediction_error + 0.35f * pain_jump, 0.0f, 1.0f);
        stability = torch::clamp(stability - (0.05f * prediction_error + 0.005f * epistemic_entropy) + 0.02f, 0.0f, 1.0f);

        state = torch::cat({curiosity, energy, stability, health, noradrenaline, dopamine}, 1);
        return state;
    }

    int64_t compute_allostatic_regime(float prediction_error = 0.0f) {
        float energy_val = state[0][1].item<float>();
        float na_val     = state[0][4].item<float>();

        if (energy_val < 0.20f) {
            return 2; // Regime 2: Deep Allostatic Sleep Consolidation
        } else if (prediction_error > 0.20f || na_val > 0.15f) {
            return 0; // Regime 0: Active High-Gain Perception
        } else {
            return 1; // Regime 1: Quiet Wakefulness & SWR Micro-Replay
        }
    }
};

// ============================================================================
// 3. MULTI-MODAL SENSORY GATEWAY (GLOBAL WORKSPACE INTEGRATION)
// ============================================================================
class SensoryGatewayImpl : public torch::nn::Module {
public:
    int64_t unified_dim;
    int64_t hidden_dim;
    int64_t homeo_dim;

    torch::nn::Linear text_proj{nullptr};
    torch::nn::Linear vision_proj{nullptr};
    torch::nn::Linear motor_proj{nullptr};
    torch::nn::Linear homeo_proj{nullptr};
    torch::nn::Linear mind_proj{nullptr};
    torch::nn::Linear attention_query_layer{nullptr};

    torch::nn::LayerNorm channel_norm{nullptr};
    torch::nn::LayerNorm query_norm{nullptr};

    SensoryGatewayImpl(int64_t unified_dim = 256, int64_t hidden_dim = 512, int64_t homeo_dim = 6,
                       int64_t text_dim = 256, int64_t vision_dim = 256, int64_t action_dim = 3,
                       std::string device_str = "cpu")
        : unified_dim(unified_dim), hidden_dim(hidden_dim), homeo_dim(homeo_dim) {

        text_proj = register_module("text_proj", torch::nn::Linear(text_dim, unified_dim));
        vision_proj = register_module("vision_proj", torch::nn::Linear(vision_dim, unified_dim));
        motor_proj = register_module("motor_proj", torch::nn::Linear(action_dim, unified_dim));
        
        homeo_proj = register_module("homeo_proj", torch::nn::Linear(homeo_dim, unified_dim));
        mind_proj = register_module("mind_proj", torch::nn::Linear(hidden_dim, unified_dim));
        attention_query_layer = register_module("attention_query_layer", torch::nn::Linear(hidden_dim, unified_dim));

        channel_norm = register_module("channel_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({unified_dim})));
        query_norm = register_module("query_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({unified_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, torch::Tensor> forward(
        torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor motor_input,
        torch::Tensor h_prev, torch::Tensor u_t) {

        int64_t batch_size = h_prev.size(0);
        std::vector<torch::Tensor> projected_channels;
        std::vector<std::string> channel_names;
        std::vector<torch::Tensor> channel_masks;

        auto text_max = std::get<0>(text_input.abs().max(-1, true));
        auto text_act = (text_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(text_proj->forward(text_input));
        channel_names.push_back("text");
        channel_masks.push_back((1.0f - text_act) * -1e9f);

        auto vis_max = std::get<0>(vision_input.abs().max(-1, true));
        auto vis_act = (vis_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(vision_proj->forward(vision_input));
        channel_names.push_back("vision");
        channel_masks.push_back((1.0f - vis_act) * -1e9f);

        auto mot_max = std::get<0>(motor_input.abs().max(-1, true));
        auto mot_act = (mot_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(motor_proj->forward(motor_input));
        channel_names.push_back("motor");
        channel_masks.push_back((1.0f - mot_act) * -1e9f);

        projected_channels.push_back(homeo_proj->forward(u_t));
        channel_names.push_back("body");
        channel_masks.push_back(torch::zeros({batch_size, 1}, h_prev.options()));

        projected_channels.push_back(mind_proj->forward(h_prev));
        channel_names.push_back("mind");
        channel_masks.push_back(torch::zeros({batch_size, 1}, h_prev.options()));

        auto stacked_channels = torch::stack(projected_channels, 1);
        auto norm_stacked = channel_norm->forward(stacked_channels);

        auto volition_query = attention_query_layer->forward(h_prev).unsqueeze(1);
        auto norm_query = query_norm->forward(volition_query);

        auto sim = (norm_query * norm_stacked).sum(-1) / std::sqrt(static_cast<float>(unified_dim));
        auto stacked_masks = torch::cat(channel_masks, 1);
        sim = sim + stacked_masks;

        sim.index_put_({torch::indexing::Slice(), 0}, sim.index({torch::indexing::Slice(), 0}) + 1.5f);

        auto attention_weights = torch::softmax(sim, -1);
        constexpr float eps = 1e-9f;
        auto epistemic_entropy = -torch::sum(attention_weights * torch::log(attention_weights + eps), -1, true);

        auto w_t = (attention_weights.unsqueeze(-1) * stacked_channels).sum(1);

        return std::make_tuple(w_t, attention_weights, channel_names, epistemic_entropy);
    }
};

// ============================================================================
// 4. MOTOR GATEWAY (EFFERENCE ACTIONS & COGNITIVE GATING)
// ============================================================================
class MotorGatewayImpl : public torch::nn::Module {
public:
    torch::nn::Linear motor_action{nullptr};
    torch::nn::Linear cognitive_gating{nullptr};
    torch::nn::Linear text_generation{nullptr};

    MotorGatewayImpl(int64_t hidden_dim = 512, int64_t action_dim = 3, int64_t cog_action_dim = 3, int64_t text_gen_dim = 258,
                     std::string device_str = "cpu") {
        motor_action = register_module("motor_action", torch::nn::Linear(hidden_dim, action_dim));
        cognitive_gating = register_module("cognitive_gating", torch::nn::Linear(hidden_dim, cog_action_dim));
        text_generation = register_module("text_generation", torch::nn::Linear(hidden_dim, text_gen_dim));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor h_t) {
        std::unordered_map<std::string, torch::Tensor> outputs;
        outputs["motor_action"] = motor_action->forward(h_t);
        outputs["cognitive_gating"] = cognitive_gating->forward(h_t);
        outputs["text_generation"] = text_generation->forward(h_t);
        return outputs;
    }
};

// ============================================================================
// 5. CAUSAL BYTE RECEPTIVE FIELD (UNSHACKLED 256D DEPTHWISE CONV1D + SILU)
// ============================================================================
class CausalByteReceptiveFieldImpl : public torch::nn::Module {
public:
    int64_t text_dim;
    int64_t kernel_size;
    torch::nn::Conv1d conv{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    CausalByteReceptiveFieldImpl(int64_t text_dim = 256, int64_t kernel_size = 4, std::string device_str = "cpu")
        : text_dim(text_dim), kernel_size(kernel_size) {
        
        auto conv_opts = torch::nn::Conv1dOptions(text_dim, text_dim, kernel_size)
            .groups(text_dim)
            .bias(false);
        conv = register_module("conv", torch::nn::Conv1d(conv_opts));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({text_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    torch::Tensor forward(torch::Tensor x_seq) {
        auto x_trans = x_seq.transpose(1, 2);
        auto x_padded = torch::nn::functional::pad(
            x_trans, 
            torch::nn::functional::PadFuncOptions({kernel_size - 1, 0}).mode(torch::kConstant).value(0.0)
        );
        auto conv_out = torch::silu(conv->forward(x_padded));
        auto out = norm->forward(conv_out.transpose(1, 2) + x_seq);
        return out;
    }
};

// ============================================================================
// 6. NATIVE C++20 SELECTIVE PARALLEL SSD CORE (DATA-DEPENDENT LATERAL INHIBITION)
// ============================================================================
class CalibratedParallelSSDCoreImpl : public torch::nn::Module {
public:
    int64_t text_dim;
    int64_t unified_dim;
    int64_t hidden_dim;
    int64_t num_heads;
    int64_t head_k;
    int64_t head_v;
    float inv_sqrt_k;

    torch::nn::Linear sensory_proj{nullptr};
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear delta_proj{nullptr};
    torch::Tensor decay_logits;
    torch::nn::Linear out_proj{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    CalibratedParallelSSDCoreImpl(int64_t text_dim = 256, int64_t unified_dim = 256, int64_t hidden_dim = 512,
                                 int64_t num_heads = 8, int64_t head_k = 32, int64_t head_v = 64,
                                 std::string device_str = "cpu")
        : text_dim(text_dim), unified_dim(unified_dim), hidden_dim(hidden_dim),
          num_heads(num_heads), head_k(head_k), head_v(head_v) {

        inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(head_k));

        sensory_proj = register_module("sensory_proj", torch::nn::Linear(text_dim, unified_dim));
        q_proj = register_module("q_proj", torch::nn::Linear(unified_dim, num_heads * head_k));
        k_proj = register_module("k_proj", torch::nn::Linear(unified_dim, num_heads * head_k));
        v_proj = register_module("v_proj", torch::nn::Linear(unified_dim, num_heads * head_v));
        
        // Native C++ Data-Dependent Selective Delta Projection (GABAergic Lateral Inhibition)
        delta_proj = register_module("delta_proj", torch::nn::Linear(unified_dim, num_heads));

        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) opts = opts.device(torch::kCUDA);
        
        // Multi-Timescale Decay Spectrum: alpha in [0.70, 0.9995]
        auto betas = torch::exp(torch::linspace(std::log(0.30f), std::log(0.0005f), num_heads, opts));
        auto alphas = 1.0f - betas;
        auto logit_init = torch::log(alphas / (1.0f - alphas)).view({1, num_heads, 1, 1});
        decay_logits = register_parameter("decay_logits", logit_init);

        out_proj = register_module("out_proj", torch::nn::Linear(hidden_dim, hidden_dim));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_chunk_parallel_ssd(
        torch::Tensor chunk_emb, torch::Tensor m_prev, torch::Tensor u_t, float dt = 1.0f) {

        int64_t batch_size = chunk_emb.size(0);
        int64_t chunk_len = chunk_emb.size(1);

        if (u_t.size(0) != batch_size) {
            u_t = u_t.expand({batch_size, -1});
        }

        auto curiosity = u_t.slice(1, 0, 1).view({batch_size, 1, 1, 1});
        auto na        = u_t.slice(1, 4, 5).view({batch_size, 1, 1, 1});
        auto da        = u_t.slice(1, 5, 6).view({batch_size, 1, 1, 1});
        auto eff_dt    = torch::clamp(dt * (1.0f - 0.4f * na + 0.4f * da), 0.30f, 2.00f);

        auto w_chunk = sensory_proj->forward(chunk_emb);

        auto q = (q_proj->forward(w_chunk).view({batch_size, chunk_len, num_heads, head_k}).transpose(1, 2)) * inv_sqrt_k;
        auto k = k_proj->forward(w_chunk).view({batch_size, chunk_len, num_heads, head_k}).transpose(1, 2);
        auto v = v_proj->forward(w_chunk).view({batch_size, chunk_len, num_heads, head_v}).transpose(1, 2);

        // Native C++ Data-Dependent Selective Delta Calculation
        auto selective_delta = torch::softplus(delta_proj->forward(w_chunk))
            .view({batch_size, chunk_len, num_heads, 1})
            .transpose(1, 2); // [B, H, chunk_len, 1]

        auto base_alpha = torch::sigmoid(decay_logits); // [1, H, 1, 1]
        auto alpha = torch::pow(base_alpha, (selective_delta * eff_dt).clamp(0.1f, 10.0f)); // [B, H, chunk_len, 1]
        auto beta = 1.0f - alpha;

        auto pos = torch::arange(chunk_len, chunk_emb.options().dtype(torch::kFloat32));
        auto diff = pos.unsqueeze(1) - pos.unsqueeze(0);
        auto causal_mask = (diff >= 0).to(torch::kFloat32).view({1, 1, chunk_len, chunk_len});

        auto mean_alpha = alpha.mean(2, true); // [B, H, 1, 1]
        auto decay_weights = torch::pow(mean_alpha, diff.clamp_min(0).view({1, 1, chunk_len, chunk_len})) * causal_mask * beta.mean(2, true);
        
        auto s_matrix = torch::matmul(q, k.transpose(-1, -2)) * decay_weights;
        auto y_intra = torch::matmul(s_matrix, v);

        auto decay_to_start = torch::pow(mean_alpha, (pos + 1.0f).view({1, 1, chunk_len, 1}));
        auto y_inter = torch::matmul(q * decay_to_start, m_prev);

        auto y_total = (y_intra + y_inter).transpose(1, 2).reshape({batch_size * chunk_len, hidden_dim});
        auto h_chunk = norm->forward(out_proj->forward(y_total) + y_total);

        auto decay_to_end = torch::pow(mean_alpha, (static_cast<float>(chunk_len) - 1.0f - pos).view({1, 1, chunk_len, 1}));
        auto k_decayed = k * decay_to_end;
        auto kv_chunk_update = torch::matmul(k_decayed.transpose(-1, -2), v);

        auto alpha_chunk = torch::pow(mean_alpha, static_cast<float>(chunk_len));
        
        // Biological Somatic-Modulated Langevin Dynamics (Non-deterministic Exploratory Noise)
        auto sigma_somatic = 1e-3f * (0.8f * curiosity + 0.4f * na + 0.1f);
        auto dW = torch::randn_like(m_prev) * torch::sqrt(eff_dt) * sigma_somatic;
        auto m_next = alpha_chunk * m_prev + beta.mean(2, true) * kv_chunk_update + dW;

        return std::make_tuple(h_chunk, m_next, eff_dt.view({batch_size, 1}));
    }
};

// ============================================================================
// 7. PARALLEL SWIGLU CHANNEL-MIXING BLOCK (2048 DIM)
// ============================================================================
class ParallelSwiGLUBlockImpl : public torch::nn::Module {
public:
    int64_t hidden_dim;
    int64_t expand_dim;

    torch::nn::Linear w_gate{nullptr};
    torch::nn::Linear w_up{nullptr};
    torch::nn::Linear w_down{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    ParallelSwiGLUBlockImpl(int64_t hidden_dim = 512, int64_t expand_dim = 2048, std::string device_str = "cpu")
        : hidden_dim(hidden_dim), expand_dim(expand_dim) {

        w_gate = register_module("w_gate", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, expand_dim).bias(false)));
        w_up = register_module("w_up", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, expand_dim).bias(false)));
        w_down = register_module("w_down", torch::nn::Linear(torch::nn::LinearOptions(expand_dim, hidden_dim).bias(false)));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    torch::Tensor forward(torch::Tensor x_flat) {
        auto gate = torch::silu(w_gate->forward(x_flat));
        auto up = w_up->forward(x_flat);
        auto ffn_out = w_down->forward(gate * up);
        return norm->forward(x_flat + ffn_out);
    }
};

// ============================================================================
// 8. DENSE MODERN HOPFIELD ATTRACTOR HEAD (COMMITMENT LOSS & PATTERN SEPARATION)
// ============================================================================
class DesaturatedHopfieldAttractorHeadImpl : public torch::nn::Module {
public:
    int64_t hidden_dim;
    int64_t num_attractors;
    float scale;
    torch::Tensor attractor_basins;
    torch::nn::LayerNorm norm{nullptr};

    DesaturatedHopfieldAttractorHeadImpl(int64_t hidden_dim = 512, int64_t vocab_size = 258, 
                                         int64_t num_attractors = 256, std::string device_str = "cpu")
        : hidden_dim(hidden_dim), num_attractors(num_attractors) {
        scale = 1.0f / std::sqrt(static_cast<float>(hidden_dim));
        
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            opts = opts.device(torch::kCUDA);
        }
        attractor_basins = register_parameter("attractor_basins", torch::randn({num_attractors, hidden_dim}, opts) * 0.05f);
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> relax_to_minima(torch::Tensor h_state, torch::Tensor u_t) {
        float da_val = 0.0f;
        if (u_t.defined() && u_t.numel() >= 6) {
            da_val = u_t[0][5].item<float>();
        }
        float beta = 1.0f + 1.5f * da_val;

        auto sim = torch::matmul(h_state, attractor_basins.transpose(0, 1)) * (scale * beta);
        auto attn_weights = torch::softmax(sim, -1);
        auto attractor_shift = torch::matmul(attn_weights, attractor_basins);
        auto h_relaxed = norm->forward(h_state + 0.25f * attractor_shift);
        
        // Native C++ Bounded Hopfield Commitment Loss
        auto commit_loss = torch::mse_loss(h_state, h_relaxed.detach()) + 
                           0.25f * torch::mse_loss(h_state.detach(), h_relaxed);
        
        return std::make_tuple(h_relaxed, commit_loss);
    }

    torch::Tensor compute_pattern_separation_loss() {
        auto norm_basins = torch::nn::functional::normalize(
            attractor_basins, 
            torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1)
        );
        auto cosine_matrix = torch::matmul(norm_basins, norm_basins.transpose(0, 1));
        auto eye = torch::eye(num_attractors, attractor_basins.options());
        return torch::mse_loss(cosine_matrix, eye);
    }
};

// ============================================================================
// 9. ACTIVE INFERENCE LATENT WORLD MODEL
// ============================================================================
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t hidden_dim;
    int64_t unified_dim;
    int64_t latent_dim;

    torch::nn::Linear prior_net{nullptr};
    torch::nn::Linear posterior_net{nullptr};
    torch::nn::Sequential decoder_net{nullptr};

    LatentPredictorImpl(int64_t hidden_dim = 512, int64_t unified_dim = 256, int64_t latent_dim = 128, std::string device_str = "cpu")
        : hidden_dim(hidden_dim), unified_dim(unified_dim), latent_dim(latent_dim) {
        
        prior_net = register_module("prior_net", torch::nn::Linear(hidden_dim, latent_dim * 2));
        posterior_net = register_module("posterior_net", torch::nn::Linear(hidden_dim + unified_dim, latent_dim * 2));
        
        decoder_net = register_module("decoder_net", torch::nn::Sequential(
            torch::nn::Linear(latent_dim + hidden_dim, unified_dim * 2),
            torch::nn::SiLU(),
            torch::nn::Linear(unified_dim * 2, unified_dim)
        ));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor h_fast_prev, 
        torch::Tensor h_slow_curr, 
        torch::Tensor w_t) {

        auto prior_out = prior_net->forward(h_fast_prev);
        auto prior_chunks = prior_out.chunk(2, -1);
        auto mu_prior = prior_chunks[0];
        auto logvar_prior = torch::clamp(prior_chunks[1], -10.0f, 10.0f);

        auto post_out = posterior_net->forward(torch::cat({h_fast_prev, w_t}, -1));
        auto post_chunks = post_out.chunk(2, -1);
        auto mu_post = post_chunks[0];
        auto logvar_post = torch::clamp(post_chunks[1], -10.0f, 10.0f);

        auto std_post = torch::exp(0.5f * logvar_post);
        auto eps = torch::randn_like(std_post);
        auto z_t = mu_post + eps * std_post;

        auto w_pred = decoder_net->forward(torch::cat({z_t, h_slow_curr}, -1));

        auto var_prior = torch::exp(logvar_prior) + 1e-7f;
        auto var_post = torch::exp(logvar_post) + 1e-7f;

        auto kl_div = 0.5f * torch::mean(
            logvar_prior - logvar_post + (var_post + torch::pow(mu_post - mu_prior, 2)) / var_prior - 1.0f,
            -1, true
        );

        auto rec_loss = torch::mean(torch::pow(w_t - w_pred, 2), -1, true);
        auto free_energy = kl_div + rec_loss;

        return std::make_tuple(w_pred, kl_div, free_energy, z_t);
    }
};

// ============================================================================
// 10. HIGH-VELOCITY BATCHED EPISODIC MEMORY (ZERO-SYNC HOST TRACKING)
// ============================================================================
class BatchedEpisodicMemoryImpl : public torch::nn::Module {
public:
    int64_t batch_size;
    int64_t memory_dim;
    int64_t max_capacity;
    int64_t max_active_cpu;

    torch::Tensor keys;
    torch::Tensor values;
    torch::Tensor pointer;
    torch::Tensor size;

    BatchedEpisodicMemoryImpl(int64_t batch_size = 1, int64_t memory_dim = 256, int64_t max_capacity = 1000, std::string device_str = "cpu")
        : batch_size(batch_size), memory_dim(memory_dim), max_capacity(max_capacity), max_active_cpu(0) {
        
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            opts = opts.device(torch::kCUDA);
        }
        keys = register_buffer("keys", torch::zeros({batch_size, max_capacity, memory_dim}, opts));
        values = register_buffer("values", torch::zeros({batch_size, max_capacity, memory_dim}, opts));
        pointer = register_buffer("pointer", torch::zeros({batch_size}, opts.dtype(torch::kInt64)));
        size = register_buffer("size", torch::zeros({batch_size}, opts.dtype(torch::kInt64)));
    }

    void write(torch::Tensor key, torch::Tensor value, int64_t protected_slots = 3) {
        if (keys.device() != key.device() || keys.dtype() != key.dtype()) {
            key = key.to(keys.device(), keys.dtype());
            value = value.to(values.device(), values.dtype());
        }

        int64_t curr_batch = key.size(0);
        auto batch_indices = torch::arange(curr_batch, pointer.options());

        keys.index_put_({batch_indices, pointer}, key);
        values.index_put_({batch_indices, pointer}, value);

        size = torch::clamp(size + 1, 0, max_capacity);
        max_active_cpu = std::min(max_active_cpu + 1, max_capacity);

        auto next_ptr = pointer + 1;
        auto wrap_mask = next_ptr >= max_capacity;
        pointer = torch::where(wrap_mask, torch::full_like(pointer, protected_slots), next_ptr);
    }

    std::tuple<torch::Tensor, torch::Tensor> read(torch::Tensor query, float temperature = 0.05f, float threshold = 0.5f, float sigmoid_beta = 15.0f) {
        if (keys.device() != query.device() || keys.dtype() != query.dtype()) {
            query = query.to(keys.device(), keys.dtype());
        }

        int64_t q_b = query.size(0);
        int64_t max_active = (max_active_cpu > 0) ? max_active_cpu : size.max().item<int64_t>();

        if (max_active == 0) {
            auto empty_val = torch::zeros({q_b, memory_dim}, query.options());
            auto empty_sim = torch::zeros({q_b, 1}, query.options());
            return std::make_tuple(empty_val, empty_sim);
        }

        auto active_keys = keys.slice(1, 0, max_active);
        auto active_values = values.slice(1, 0, max_active);
        auto active_size = size;

        if (q_b != keys.size(0)) {
            if (q_b == 1) {
                active_keys = active_keys.slice(0, 0, 1);
                active_values = active_values.slice(0, 0, 1);
                active_size = size.slice(0, 0, 1);
            } else {
                int64_t k_b = keys.size(0);
                int64_t repeats = (q_b + k_b - 1) / k_b;
                active_keys = active_keys.repeat({repeats, 1, 1}).slice(0, 0, q_b);
                active_values = active_values.repeat({repeats, 1, 1}).slice(0, 0, q_b);
                active_size = size.repeat({repeats}).slice(0, 0, q_b);
            }
        }

        auto q_norm = torch::nn::functional::normalize(query.unsqueeze(1), torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
        auto k_norm = torch::nn::functional::normalize(active_keys, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));

        auto sim = torch::bmm(q_norm, k_norm.transpose(1, 2));
        auto seq_range = torch::arange(max_active, query.options()).unsqueeze(0);
        auto invalid_mask = (seq_range >= active_size.unsqueeze(1)).unsqueeze(1);

        auto sim_masked = sim.masked_fill(invalid_mask, -1e9f);
        auto max_sim = std::get<0>(sim_masked.max(-1));
        auto max_sim_valid = torch::where(active_size.unsqueeze(-1) > 0, max_sim, torch::zeros_like(max_sim));

        auto gate = torch::sigmoid((max_sim_valid - threshold) * sigmoid_beta);
        auto attn_weights = torch::softmax(sim_masked / temperature, -1);

        auto retrieved_val = torch::bmm(attn_weights, active_values).squeeze(1);
        auto gated_retrieved = retrieved_val * gate;

        return std::make_tuple(gated_retrieved, max_sim_valid);
    }

    void consolidate_and_prune(float similarity_threshold = 0.95f, int64_t protected_slots = 3) {
        int64_t max_active = (max_active_cpu > 0) ? max_active_cpu : size.max().item<int64_t>();
        if (max_active <= protected_slots) return;

        auto active_k = keys.slice(1, 0, max_active);
        auto norm_keys = torch::nn::functional::normalize(active_k, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
        auto self_sim = torch::bmm(norm_keys, norm_keys.transpose(1, 2));

        auto triu_mask = torch::triu(torch::ones({max_active, max_active}, torch::kBool).to(keys.device()), 1);
        triu_mask.slice(0, 0, protected_slots) = false;

        auto redundant = (self_sim > similarity_threshold) & triu_mask.unsqueeze(0);
        auto dup_count = redundant.sum(-1);

        auto dup_mask = (dup_count > 0).unsqueeze(-1);
        active_k.masked_fill_(dup_mask, 0.0f);
    }
};

// ============================================================================
// 11. PYBIND11 MODULE BINDINGS
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ByteTokenizer>(m, "ByteTokenizer")
        .def(py::init<int64_t>(), py::arg("vocab_size") = 258)
        .def_readwrite("vocab_size", &ByteTokenizer::vocab_size)
        .def_readwrite("pad_id", &ByteTokenizer::pad_id)
        .def_readwrite("eos_id", &ByteTokenizer::eos_id)
        .def("encode", &ByteTokenizer::encode)
        .def("decode", &ByteTokenizer::decode)
        .def("decode_bytes", &ByteTokenizer::decode_bytes);

    py::class_<HomeostaticUnit>(m, "HomeostaticUnit")
        .def(py::init<int, std::string>(), py::arg("batch_size") = 1, py::arg("device") = "cpu")
        .def_readwrite("state", &HomeostaticUnit::state)
        .def_readwrite("prev_pain", &HomeostaticUnit::prev_pain)
        .def_readwrite("consecutive_inactivity", &HomeostaticUnit::consecutive_inactivity)
        .def_readwrite("device", &HomeostaticUnit::device)
        .def("update", &HomeostaticUnit::update)
        .def("compute_allostatic_regime", &HomeostaticUnit::compute_allostatic_regime, py::arg("prediction_error") = 0.0f);

    py::class_<SensoryGatewayImpl, torch::nn::Module, std::shared_ptr<SensoryGatewayImpl>>(m, "SensoryGateway")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("unified_dim") = 256, py::arg("hidden_dim") = 512, py::arg("homeo_dim") = 6,
             py::arg("text_dim") = 256, py::arg("vision_dim") = 256, py::arg("action_dim") = 3,
             py::arg("device") = "cpu")
        .def("forward", &SensoryGatewayImpl::forward)
        .def("parameters", [](std::shared_ptr<SensoryGatewayImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<SensoryGatewayImpl> m) { return m->named_parameters(); })
        .def("__call__", &SensoryGatewayImpl::forward);

    py::class_<MotorGatewayImpl, torch::nn::Module, std::shared_ptr<MotorGatewayImpl>>(m, "MotorGateway")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 512, py::arg("action_dim") = 3, py::arg("cog_action_dim") = 3, py::arg("text_gen_dim") = 258,
             py::arg("device") = "cpu")
        .def("forward", &MotorGatewayImpl::forward)
        .def("parameters", [](std::shared_ptr<MotorGatewayImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<MotorGatewayImpl> m) { return m->named_parameters(); })
        .def("__call__", &MotorGatewayImpl::forward);

    py::class_<CausalByteReceptiveFieldImpl, torch::nn::Module, std::shared_ptr<CausalByteReceptiveFieldImpl>>(m, "CausalByteReceptiveField")
        .def(py::init<int64_t, int64_t, std::string>(),
             py::arg("text_dim") = 256, py::arg("kernel_size") = 4, py::arg("device") = "cpu")
        .def("forward", &CausalByteReceptiveFieldImpl::forward)
        .def("parameters", [](std::shared_ptr<CausalByteReceptiveFieldImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CausalByteReceptiveFieldImpl> m) { return m->named_parameters(); })
        .def("__call__", &CausalByteReceptiveFieldImpl::forward);

    py::class_<CalibratedParallelSSDCoreImpl, torch::nn::Module, std::shared_ptr<CalibratedParallelSSDCoreImpl>>(m, "CalibratedParallelSSDCore")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("text_dim") = 256, py::arg("unified_dim") = 256, py::arg("hidden_dim") = 512,
             py::arg("num_heads") = 8, py::arg("head_k") = 32, py::arg("head_v") = 64, py::arg("device") = "cpu")
        .def("forward_chunk_parallel_ssd", &CalibratedParallelSSDCoreImpl::forward_chunk_parallel_ssd,
             py::arg("chunk_emb"), py::arg("m_prev"), py::arg("u_t"), py::arg("dt") = 1.0f)
        .def("parameters", [](std::shared_ptr<CalibratedParallelSSDCoreImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CalibratedParallelSSDCoreImpl> m) { return m->named_parameters(); })
        .def("__call__", &CalibratedParallelSSDCoreImpl::forward_chunk_parallel_ssd,
             py::arg("chunk_emb"), py::arg("m_prev"), py::arg("u_t"), py::arg("dt") = 1.0f);

    py::class_<ParallelSwiGLUBlockImpl, torch::nn::Module, std::shared_ptr<ParallelSwiGLUBlockImpl>>(m, "ParallelSwiGLUBlock")
        .def(py::init<int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 512, py::arg("expand_dim") = 2048, py::arg("device") = "cpu")
        .def("forward", &ParallelSwiGLUBlockImpl::forward)
        .def("parameters", [](std::shared_ptr<ParallelSwiGLUBlockImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<ParallelSwiGLUBlockImpl> m) { return m->named_parameters(); })
        .def("__call__", &ParallelSwiGLUBlockImpl::forward);

    py::class_<DesaturatedHopfieldAttractorHeadImpl, torch::nn::Module, std::shared_ptr<DesaturatedHopfieldAttractorHeadImpl>>(m, "DesaturatedHopfieldAttractorHead")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 512, py::arg("vocab_size") = 258, py::arg("num_attractors") = 256, py::arg("device") = "cpu")
        .def_readwrite("attractor_basins", &DesaturatedHopfieldAttractorHeadImpl::attractor_basins)
        .def("relax_to_minima", &DesaturatedHopfieldAttractorHeadImpl::relax_to_minima,
             py::arg("h_state"), py::arg("u_t") = torch::Tensor())
        .def("compute_pattern_separation_loss", &DesaturatedHopfieldAttractorHeadImpl::compute_pattern_separation_loss)
        .def("parameters", [](std::shared_ptr<DesaturatedHopfieldAttractorHeadImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<DesaturatedHopfieldAttractorHeadImpl> m) { return m->named_parameters(); })
        .def("__call__", &DesaturatedHopfieldAttractorHeadImpl::relax_to_minima,
             py::arg("h_state"), py::arg("u_t") = torch::Tensor());

    py::class_<LatentPredictorImpl, torch::nn::Module, std::shared_ptr<LatentPredictorImpl>>(m, "LatentPredictor")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 512, py::arg("unified_dim") = 256, py::arg("latent_dim") = 128, py::arg("device") = "cpu")
        .def("forward", &LatentPredictorImpl::forward)
        .def("parameters", [](std::shared_ptr<LatentPredictorImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<LatentPredictorImpl> m) { return m->named_parameters(); })
        .def("__call__", &LatentPredictorImpl::forward);

    py::class_<BatchedEpisodicMemoryImpl, torch::nn::Module, std::shared_ptr<BatchedEpisodicMemoryImpl>>(m, "BatchedEpisodicMemory")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(),
             py::arg("batch_size") = 1, py::arg("memory_dim") = 256, py::arg("max_capacity") = 1000, py::arg("device") = "cpu")
        .def_readwrite("batch_size", &BatchedEpisodicMemoryImpl::batch_size)
        .def_readwrite("memory_dim", &BatchedEpisodicMemoryImpl::memory_dim)
        .def_readwrite("max_capacity", &BatchedEpisodicMemoryImpl::max_capacity)
        .def_readwrite("max_active_cpu", &BatchedEpisodicMemoryImpl::max_active_cpu)
        .def_readwrite("keys", &BatchedEpisodicMemoryImpl::keys)
        .def_readwrite("values", &BatchedEpisodicMemoryImpl::values)
        .def_readwrite("pointer", &BatchedEpisodicMemoryImpl::pointer)
        .def_readwrite("size", &BatchedEpisodicMemoryImpl::size)
        .def("write", &BatchedEpisodicMemoryImpl::write,
             py::arg("key"), py::arg("value"), py::arg("protected_slots") = 3)
        .def("read", &BatchedEpisodicMemoryImpl::read,
             py::arg("query"), py::arg("temperature") = 0.05f, py::arg("threshold") = 0.5f, py::arg("sigmoid_beta") = 15.0f)
        .def("consolidate_and_prune", &BatchedEpisodicMemoryImpl::consolidate_and_prune,
             py::arg("similarity_threshold") = 0.95f, py::arg("protected_slots") = 3);
}
