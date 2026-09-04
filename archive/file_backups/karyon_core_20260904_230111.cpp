// karyon_core.cpp - Native C++20 Master Architecture Engine for Karyon-CoRE
// GROUNDED IN KEP PRINCIPLE 1 (C++20 as Engine, Python as Client) & PRINCIPLE 2 (Biological Realism)
#include <torch/extension.h>
#include <ATen/autocast_mode.h>
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
            return 2;
        } else if (prediction_error > 0.20f || na_val > 0.15f) {
            return 0;
        } else {
            return 1;
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
    torch::nn::Linear audio_proj{nullptr};
    torch::nn::Linear binary_proj{nullptr};
    torch::nn::Linear telepathic_proj{nullptr};
    torch::nn::Linear motor_proj{nullptr};
    torch::nn::Linear homeo_proj{nullptr};
    torch::nn::Linear mind_proj{nullptr};
    torch::nn::Linear attention_query_layer{nullptr};

    torch::nn::LayerNorm channel_norm{nullptr};
    torch::nn::LayerNorm query_norm{nullptr};

    SensoryGatewayImpl(int64_t unified_dim = 256, int64_t hidden_dim = 768, int64_t homeo_dim = 6,
                       int64_t text_dim = 256, int64_t vision_dim = 256, int64_t audio_dim = 256,
                       int64_t binary_dim = 256, int64_t telepathic_dim = 256, int64_t action_dim = 3,
                       std::string device_str = "cpu")
        : unified_dim(unified_dim), hidden_dim(hidden_dim), homeo_dim(homeo_dim) {

        text_proj = register_module("text_proj", torch::nn::Linear(text_dim, unified_dim));
        vision_proj = register_module("vision_proj", torch::nn::Linear(vision_dim, unified_dim));
        audio_proj = register_module("audio_proj", torch::nn::Linear(audio_dim, unified_dim));
        binary_proj = register_module("binary_proj", torch::nn::Linear(binary_dim, unified_dim));
        telepathic_proj = register_module("telepathic_proj", torch::nn::Linear(telepathic_dim, unified_dim));
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
        torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor audio_input,
        torch::Tensor binary_input, torch::Tensor telepathic_input, torch::Tensor motor_input,
        torch::Tensor h_prev, torch::Tensor u_t) {

        int64_t batch_size = h_prev.size(0);
        std::vector<torch::Tensor> projected_channels;
        std::vector<std::string> channel_names;
        std::vector<torch::Tensor> channel_masks;

        auto text_max = std::get<0>(text_input.abs().max(-1, true));
        auto text_act = (text_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(text_proj->forward(text_input));
        channel_names.push_back("text");
        channel_masks.push_back((1.0f - text_act) * -10000.0f);

        auto vis_max = std::get<0>(vision_input.abs().max(-1, true));
        auto vis_act = (vis_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(vision_proj->forward(vision_input));
        channel_names.push_back("vision");
        channel_masks.push_back((1.0f - vis_act) * -10000.0f);

        auto aud_max = std::get<0>(audio_input.abs().max(-1, true));
        auto aud_act = (aud_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(audio_proj->forward(audio_input));
        channel_names.push_back("audio");
        channel_masks.push_back((1.0f - aud_act) * -10000.0f);

        auto bin_max = std::get<0>(binary_input.abs().max(-1, true));
        auto bin_act = (bin_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(binary_proj->forward(binary_input));
        channel_names.push_back("binary");
        channel_masks.push_back((1.0f - bin_act) * -10000.0f);

        auto tel_max = std::get<0>(telepathic_input.abs().max(-1, true));
        auto tel_act = (tel_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(telepathic_proj->forward(telepathic_input));
        channel_names.push_back("telepathic");
        channel_masks.push_back((1.0f - tel_act) * -10000.0f);

        auto mot_max = std::get<0>(motor_input.abs().max(-1, true));
        auto mot_act = (mot_max > 1e-5f).to(torch::kFloat32);
        projected_channels.push_back(motor_proj->forward(motor_input));
        channel_names.push_back("motor");
        channel_masks.push_back((1.0f - mot_act) * -10000.0f);

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

        auto attention_weights = torch::softmax(sim, -1);
        constexpr float eps = 1e-9f;
        auto epistemic_entropy = -torch::sum(attention_weights * torch::log(attention_weights + eps), -1, true);

        auto w_t = (attention_weights.unsqueeze(-1) * stacked_channels).sum(1);

        return std::make_tuple(w_t, attention_weights, channel_names, epistemic_entropy);
    }

    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, torch::Tensor> forward(
        torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor motor_input,
        torch::Tensor h_prev, torch::Tensor u_t) {
        auto dummy_audio = torch::zeros({text_input.size(0), text_input.size(1)}, text_input.options());
        auto dummy_binary = torch::zeros({text_input.size(0), text_input.size(1)}, text_input.options());
        auto dummy_telepathic = torch::zeros({text_input.size(0), text_input.size(1)}, text_input.options());
        return forward(text_input, vision_input, dummy_audio, dummy_binary, dummy_telepathic, motor_input, h_prev, u_t);
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
    torch::nn::Sequential vision_generation{nullptr};
    torch::nn::Sequential audio_generation{nullptr};
    torch::nn::Sequential binary_generation{nullptr};
    torch::nn::Sequential telepathic_generation{nullptr};

    MotorGatewayImpl(int64_t hidden_dim = 768, int64_t action_dim = 3, int64_t cog_action_dim = 3, int64_t text_gen_dim = 258,
                     int64_t vision_dim = 256, int64_t audio_dim = 256, int64_t binary_dim = 256, int64_t telepathic_dim = 256,
                     std::string device_str = "cpu") {
        motor_action = register_module("motor_action", torch::nn::Linear(hidden_dim, action_dim));
        cognitive_gating = register_module("cognitive_gating", torch::nn::Linear(hidden_dim, cog_action_dim));
        text_generation = register_module("text_generation", torch::nn::Linear(hidden_dim, text_gen_dim));

        vision_generation = register_module("vision_generation", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, vision_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(vision_dim, vision_dim)
        ));
        audio_generation = register_module("audio_generation", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, audio_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(audio_dim, audio_dim)
        ));
        binary_generation = register_module("binary_generation", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, binary_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(binary_dim, binary_dim)
        ));
        telepathic_generation = register_module("telepathic_generation", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, telepathic_dim),
            torch::nn::SiLU(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({telepathic_dim}))
        ));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor h_t) {
        std::unordered_map<std::string, torch::Tensor> outputs;
        outputs["motor_action"] = motor_action->forward(h_t);
        outputs["cognitive_gating"] = cognitive_gating->forward(h_t);
        outputs["text_generation"] = text_generation->forward(h_t);
        outputs["vision_generation"] = vision_generation->forward(h_t);
        outputs["audio_generation"] = audio_generation->forward(h_t);
        outputs["binary_generation"] = binary_generation->forward(h_t);
        outputs["telepathic_generation"] = telepathic_generation->forward(h_t);
        return outputs;
    }
};

// ============================================================================
// 5. CAUSAL BYTE RECEPTIVE FIELD (1D DEPTHWISE CONV1D + SILU)
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
// 6. MULTI-SCALE MORPHOLOGICAL BYTE PYRAMID RECEPTIVE FIELD (EXP-70)
// ============================================================================
struct MultiScaleBytePyramidReceptiveFieldImpl : torch::nn::Module {
    int64_t text_dim;
    torch::nn::Conv1d conv_k2{nullptr};
    torch::nn::Conv1d conv_k4{nullptr};
    torch::nn::Conv1d conv_k8{nullptr};
    torch::nn::Linear scale_gate{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    MultiScaleBytePyramidReceptiveFieldImpl(int64_t text_dim = 256, std::string device_str = "cpu") : text_dim(text_dim) {
        auto opts_k2 = torch::nn::Conv1dOptions(text_dim, text_dim, 2).groups(text_dim).bias(false);
        auto opts_k4 = torch::nn::Conv1dOptions(text_dim, text_dim, 4).groups(text_dim).bias(false);
        auto opts_k8 = torch::nn::Conv1dOptions(text_dim, text_dim, 8).groups(text_dim).bias(false);

        conv_k2 = register_module("conv_k2", torch::nn::Conv1d(opts_k2));
        conv_k4 = register_module("conv_k4", torch::nn::Conv1d(opts_k4));
        conv_k8 = register_module("conv_k8", torch::nn::Conv1d(opts_k8));
        scale_gate = register_module("scale_gate", torch::nn::Linear(text_dim, 3));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({text_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    torch::Tensor forward(torch::Tensor x_seq) {
        auto x_trans = x_seq.transpose(1, 2);
        auto x_pad_k2 = torch::nn::functional::pad(x_trans, torch::nn::functional::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0.0));
        auto out_k2 = torch::silu(conv_k2->forward(x_pad_k2)).transpose(1, 2);

        auto x_pad_k4 = torch::nn::functional::pad(x_trans, torch::nn::functional::PadFuncOptions({3, 0}).mode(torch::kConstant).value(0.0));
        auto out_k4 = torch::silu(conv_k4->forward(x_pad_k4)).transpose(1, 2);

        auto x_pad_k8 = torch::nn::functional::pad(x_trans, torch::nn::functional::PadFuncOptions({7, 0}).mode(torch::kConstant).value(0.0));
        auto out_k8 = torch::silu(conv_k8->forward(x_pad_k8)).transpose(1, 2);

        auto gates = torch::softmax(scale_gate->forward(x_seq), -1);
        auto g2 = gates.slice(2, 0, 1);
        auto g4 = gates.slice(2, 1, 2);
        auto g8 = gates.slice(2, 2, 3);

        auto pyramid_out = g2 * out_k2 + g4 * out_k4 + g8 * out_k8;
        return norm->forward(x_seq + pyramid_out);
    }
};

// ============================================================================
// 7. ROPE UTILITIES FOR C++ STATE-SPACE DUALITY
// ============================================================================
static inline torch::Tensor rotate_half(const torch::Tensor& x) {
    int64_t half_d = x.size(-1) / 2;
    auto x1 = x.slice(-1, 0, half_d);
    auto x2 = x.slice(-1, half_d);
    return torch::cat({-x2, x1}, -1);
}

static inline torch::Tensor apply_rotary_pos_emb_cpp(
    const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin) {
    return (x * cos) + (rotate_half(x) * sin);
}

// ============================================================================
// 8. FUSED CHUNK-PARALLEL LOG-SPACE RETENTION DECAY SSD LAYER (EXP-82/83)
// ============================================================================
struct ParallelLogDecaySSDLayerImpl : torch::nn::Module {
    int64_t in_dim;
    int64_t out_dim;
    int64_t num_heads;
    int64_t head_k;
    int64_t head_v;
    int64_t chunk_size;
    float inv_sqrt_k;

    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear z_proj{nullptr};
    torch::nn::Linear delta_proj{nullptr};
    torch::Tensor decay_logits;

    torch::nn::GroupNorm head_norm{nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::Tensor inv_freq;

    torch::Tensor cos_cached;
    torch::Tensor sin_cached;
    torch::Tensor causal_mask_cached;

    ParallelLogDecaySSDLayerImpl(int64_t in_dim = 768, int64_t out_dim = 768, int64_t num_heads = 12,
                                int64_t head_k = 64, int64_t head_v = 128, float min_beta = 0.0005f, float max_beta = 0.08f,
                                int64_t chunk_size = 64, std::string device_str = "cpu")
        : in_dim(in_dim), out_dim(out_dim), num_heads(num_heads), head_k(head_k), head_v(head_v), chunk_size(chunk_size) {

        inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(head_k));

        q_proj = register_module("q_proj", torch::nn::Linear(in_dim, num_heads * head_k));
        k_proj = register_module("k_proj", torch::nn::Linear(in_dim, num_heads * head_k));
        v_proj = register_module("v_proj", torch::nn::Linear(in_dim, num_heads * head_v));
        z_proj = register_module("z_proj", torch::nn::Linear(in_dim, num_heads * head_v));
        delta_proj = register_module("delta_proj", torch::nn::Linear(in_dim, num_heads));

        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            opts = opts.device(torch::kCUDA);
        }

        auto betas = torch::exp(torch::linspace(std::log(max_beta), std::log(min_beta), num_heads, opts));
        auto alphas = 1.0f - betas;
        auto logit_init = torch::log(alphas / (1.0f - alphas)).view({1, 1, num_heads, 1});
        decay_logits = register_parameter("decay_logits", logit_init);

        head_norm = register_module("head_norm", torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_heads, num_heads * head_v)));
        out_proj = register_module("out_proj", torch::nn::Linear(num_heads * head_v, out_dim));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({out_dim})));

        auto steps = torch::arange(0, head_k, 2, opts);
        inv_freq = register_buffer("inv_freq", 1.0f / torch::pow(10000.0f, steps / static_cast<float>(head_k)));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> get_rope_cos_sin(int64_t chunk_len, torch::Device dev, torch::ScalarType dtype) {
        if (cos_cached.defined() && cos_cached.size(3) == chunk_len && cos_cached.device() == dev && cos_cached.scalar_type() == dtype) {
            return std::make_tuple(cos_cached, sin_cached);
        }
        auto t = torch::arange(chunk_len, torch::TensorOptions().device(dev).dtype(inv_freq.dtype()));
        auto freqs = torch::outer(t, inv_freq);
        auto emb = torch::cat({freqs, freqs}, -1);
        cos_cached = emb.cos().view({1, 1, 1, chunk_len, head_k}).to(dtype);
        sin_cached = emb.sin().view({1, 1, 1, chunk_len, head_k}).to(dtype);
        return std::make_tuple(cos_cached, sin_cached);
    }

    torch::Tensor get_causal_mask(int64_t Q, torch::Device dev, torch::ScalarType dtype) {
        if (causal_mask_cached.defined() && causal_mask_cached.size(3) == Q && causal_mask_cached.device() == dev && causal_mask_cached.scalar_type() == dtype) {
            return causal_mask_cached;
        }
        auto pos = torch::arange(Q, torch::TensorOptions().device(dev).dtype(dtype));
        causal_mask_cached = (pos.unsqueeze(1) >= pos.unsqueeze(0)).to(dtype).view({1, 1, 1, Q, Q});
        return causal_mask_cached;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t,
        torch::Tensor saliency_gate = torch::Tensor(), float dt = 1.0f) {

        auto orig_dtype = x_seq.scalar_type();
        m_prev = m_prev.to(orig_dtype);
        u_t = u_t.to(orig_dtype);
        if (saliency_gate.defined() && saliency_gate.numel() > 0) {
            saliency_gate = saliency_gate.to(orig_dtype);
        }

        int64_t batch_size = x_seq.size(0);
        int64_t seq_len = x_seq.size(1);
        int64_t Q = chunk_size;
        int64_t pad_len = 0;

        if (seq_len < Q) {
            Q = seq_len;
        } else {
            pad_len = (Q - (seq_len % Q)) % Q;
            if (pad_len > 0) {
                x_seq = torch::nn::functional::pad(x_seq.transpose(1, 2), torch::nn::functional::PadFuncOptions({pad_len, 0}).mode(torch::kConstant).value(0.0)).transpose(1, 2);
                if (saliency_gate.defined() && saliency_gate.numel() > 0) {
                    saliency_gate = torch::nn::functional::pad(saliency_gate, torch::nn::functional::PadFuncOptions({pad_len, 0}).mode(torch::kConstant).value(0.0));
                }
                seq_len = x_seq.size(1);
            }
        }
        int64_t num_chunks = seq_len / Q;

        if (u_t.size(0) != batch_size) {
            u_t = (u_t.size(0) > batch_size) ? u_t.slice(0, 0, batch_size) : u_t.expand({batch_size, -1});
        }

        auto curiosity = u_t.select(1, 0).view({batch_size, 1, 1, 1, 1});
        auto na = u_t.select(1, 4).view({batch_size, 1, 1, 1, 1});
        auto da = u_t.select(1, 5).view({batch_size, 1, 1, 1, 1});
        auto eff_dt = torch::clamp(dt * (1.0f - 0.4f * na + 0.4f * da), 0.30f, 2.00f);

        auto q_full = (q_proj->forward(x_seq).view({batch_size, num_chunks, Q, num_heads, head_k}).transpose(2, 3)) * inv_sqrt_k;
        auto k_full = k_proj->forward(x_seq).view({batch_size, num_chunks, Q, num_heads, head_k}).transpose(2, 3);
        auto v_full = v_proj->forward(x_seq).view({batch_size, num_chunks, Q, num_heads, head_v}).transpose(2, 3);
        auto z_full = torch::silu(z_proj->forward(x_seq)).view({batch_size * seq_len, num_heads * head_v});

        auto cos_sin = get_rope_cos_sin(Q, x_seq.device(), x_seq.scalar_type());
        q_full = apply_rotary_pos_emb_cpp(q_full, std::get<0>(cos_sin), std::get<1>(cos_sin));
        k_full = apply_rotary_pos_emb_cpp(k_full, std::get<0>(cos_sin), std::get<1>(cos_sin));

        auto delta_full = torch::softplus(delta_proj->forward(x_seq)).view({batch_size, num_chunks, Q, num_heads}).permute({0, 1, 3, 2});
        auto base_alpha = torch::sigmoid(decay_logits.view({1, 1, num_heads, 1}));
        
        // Optimize base_alpha^exponent to exp(exponent * log(base_alpha))
        auto log_base_alpha = torch::log(base_alpha);
        auto exponent = (delta_full * eff_dt.squeeze(-1)).clamp(0.1f, 10.0f);
        auto log_alpha = exponent * log_base_alpha;

        if (saliency_gate.defined() && saliency_gate.numel() > 0) {
            auto sal_chunk = saliency_gate.view({batch_size, num_chunks, 1, Q});
            log_alpha = log_alpha + torch::log(1.0f - 0.80f * sal_chunk);
        }

        log_alpha = torch::clamp(log_alpha, std::log(1e-4f), std::log(0.9999f));
        auto alpha = torch::exp(log_alpha);
        auto beta = 1.0f - alpha;

        auto lambda_t = torch::cumsum(log_alpha, -1);
        auto log_decay_matrix = lambda_t.unsqueeze(-1) - lambda_t.unsqueeze(-2);
        auto decay_matrix = torch::exp(torch::clamp(log_decay_matrix, -20.0f, 0.0f));

        auto causal_mask = get_causal_mask(Q, x_seq.device(), x_seq.scalar_type());
        auto decay_weights = (decay_matrix * causal_mask).to(x_seq.scalar_type());

        auto s_matrix = torch::matmul(q_full, k_full.transpose(-1, -2)) * decay_weights;
        auto y_intra = torch::matmul(s_matrix, v_full);

        auto decay_to_start = torch::exp(torch::clamp(lambda_t, -20.0f, 0.0f)).unsqueeze(-1);
        auto lambda_end = lambda_t.slice(3, -1).unsqueeze(-1);
        auto decay_to_end = torch::exp(torch::clamp(lambda_end - lambda_t.unsqueeze(-1), -20.0f, 0.0f));

        auto k_decayed = k_full * decay_to_end * beta.unsqueeze(-1);
        auto kv_chunk_updates = torch::matmul(k_decayed.transpose(-1, -2), v_full).to(torch::kFloat32);
        auto alpha_chunks = torch::exp(torch::clamp(lambda_t.slice(3, -1), -20.0f, 0.0f)).unsqueeze(-1);

        // =============================================================================
        // FUSED PARALLEL CHUNK ASSOCIATIVE SCAN (EXP-112)
        // =============================================================================
        auto log_alpha_chunks = lambda_t.slice(3, -1).clamp(-20.0f, 0.0f).unsqueeze(-1); // (B, num_chunks, num_heads, 1, 1)
        auto lambda_chunks = torch::cumsum(log_alpha_chunks, 1); // (B, num_chunks, num_heads, 1, 1)

        auto lambda_chunks_flat = lambda_chunks.squeeze(-1).squeeze(-1).permute({0, 2, 1}); // (B, num_heads, num_chunks)
        auto log_decay_matrix_chunks = lambda_chunks_flat.unsqueeze(-1) - lambda_chunks_flat.unsqueeze(-2);
        auto decay_matrix_chunks = torch::exp(torch::clamp(log_decay_matrix_chunks, -20.0f, 0.0f));

        auto causal_mask_chunks = torch::tril(torch::ones({num_chunks, num_chunks}, x_seq.options()), -1).view({1, 1, num_chunks, num_chunks});
        auto decay_weights_chunks = (decay_matrix_chunks * causal_mask_chunks).to(x_seq.scalar_type());

        auto kv_flat = kv_chunk_updates.permute({0, 2, 1, 3, 4}); // (B, num_heads, num_chunks, head_k, head_v)

        auto sigma_somatic = 1e-3f * (0.8f * curiosity.squeeze(1).to(torch::kFloat32) + 0.4f * na.squeeze(1).to(torch::kFloat32) + 0.1f);
        auto dW_scale = torch::sqrt(eff_dt.squeeze(1).to(torch::kFloat32)) * sigma_somatic;
        auto dW_all = torch::randn({num_chunks, batch_size, num_heads, head_k, head_v}, m_prev.options());
        auto dW_all_scaled = dW_all * dW_scale.view({1, batch_size, 1, 1, 1});

        auto dW_flat = dW_all_scaled.permute({1, 2, 0, 3, 4}); // (B, num_heads, num_chunks, head_k, head_v)
        auto U = (kv_flat + dW_flat).to(x_seq.scalar_type());

        auto U_reshaped = U.reshape({batch_size, num_heads, num_chunks, head_k * head_v});
        auto M_inter_all = torch::matmul(decay_weights_chunks, U_reshaped).reshape({batch_size, num_heads, num_chunks, head_k, head_v});

        // Initial State Decay
        auto ones_initial = torch::ones({batch_size, num_heads, 1}, x_seq.options());
        auto decay_initial_list = (num_chunks > 1) ? torch::cat({ones_initial, torch::exp(lambda_chunks_flat.slice(2, 0, -1))}, 2) : ones_initial;
        auto decay_initial = decay_initial_list.unsqueeze(-1).unsqueeze(-1); // (B, num_heads, num_chunks, 1, 1)

        auto M_initial_decayed = decay_initial * m_prev.unsqueeze(2);
        auto M_all = M_inter_all + M_initial_decayed;

        // Compute y_inter in parallel across all chunks
        auto q_decay = (q_full * decay_to_start).to(x_seq.scalar_type());
        auto q_decay_flat = q_decay.permute({0, 2, 1, 3, 4}); // (B, num_heads, num_chunks, Q, head_k)

        auto y_inter_all = torch::matmul(q_decay_flat, M_all); // (B, num_heads, num_chunks, Q, head_v)
        auto y_inter = y_inter_all.permute({0, 2, 1, 3, 4}); // (B, num_chunks, num_heads, Q, head_v)

        // Final state update for next sequence:
        // M_all contains state entering each chunk. For the last chunk (index num_chunks - 1),
        // state entering is M_all.slice(2, -1).
        // To compute the state LEAVING the last chunk, decay it across the last chunk and add last chunk update U.
        auto alpha_last_chunk = torch::exp(log_alpha_chunks.slice(1, -1)).squeeze(-1).squeeze(-1).permute({0, 2, 1}).unsqueeze(-1).unsqueeze(-1); // (B, num_heads, 1, 1, 1)
        auto m_enter_last = M_all.slice(2, -1); // (B, num_heads, 1, head_k, head_v)
        auto U_last = U.slice(2, -1); // (B, num_heads, 1, head_k, head_v)
        auto m_next = m_enter_last * alpha_last_chunk + U_last;
        auto m_curr = torch::clamp(m_next.squeeze(2), -10000.0f, 10000.0f);
        auto y_total = (y_intra + y_inter).permute({0, 1, 3, 2, 4}).reshape({batch_size * seq_len, num_heads * head_v});
        auto y_normed = head_norm->forward(y_total.to(torch::kFloat32)).to(orig_dtype); // Normalized in float32 for absolute numerical stability
        auto y_gated = y_normed * z_full;
        auto h_seq = norm->forward(out_proj->forward(y_gated)).view({batch_size, seq_len, out_dim});

        if (pad_len > 0) {
            h_seq = h_seq.slice(1, pad_len);
        }

        return std::make_tuple(h_seq.to(orig_dtype), m_curr.to(orig_dtype), eff_dt.mean());
    }
};

// ============================================================================
// 9. CAUSAL CONVSWIGLU CHANNEL-MIXING BLOCK (EXP-73/74)
// ============================================================================
struct CausalConvSwiGLUBlockImpl : torch::nn::Module {
    int64_t hidden_dim;
    int64_t expand_dim;
    int64_t kernel_size;
    int64_t pad_left;

    torch::nn::Linear w_gate{nullptr};
    torch::nn::Conv1d gate_conv{nullptr};
    torch::nn::Linear w_up{nullptr};
    torch::nn::Linear w_down{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    CausalConvSwiGLUBlockImpl(int64_t hidden_dim = 768, int64_t expand_dim = 3072, int64_t kernel_size = 3, std::string device_str = "cpu")
        : hidden_dim(hidden_dim), expand_dim(expand_dim), kernel_size(kernel_size), pad_left(kernel_size - 1) {

        w_gate = register_module("w_gate", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, expand_dim).bias(false)));
        auto conv_opts = torch::nn::Conv1dOptions(expand_dim, expand_dim, kernel_size).groups(expand_dim).bias(false);
        gate_conv = register_module("gate_conv", torch::nn::Conv1d(conv_opts));
        w_up = register_module("w_up", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, expand_dim).bias(false)));
        w_down = register_module("w_down", torch::nn::Linear(torch::nn::LinearOptions(expand_dim, hidden_dim).bias(false)));
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto raw_gate = w_gate->forward(x);
        auto gate_trans = raw_gate.transpose(1, 2);
        auto gate_pad = torch::nn::functional::pad(gate_trans, torch::nn::functional::PadFuncOptions({pad_left, 0}).mode(torch::kConstant).value(0.0));
        auto conv_gate = gate_conv->forward(gate_pad).transpose(1, 2);

        auto gate = torch::silu(conv_gate);
        auto up = w_up->forward(x);
        auto ffn_out = w_down->forward(gate * up);
        return norm->forward(x + ffn_out);
    }
};

// ============================================================================
// 10. ENTROPY-ADAPTIVE BOUNDARY DETECTOR (EABS)
// ============================================================================
struct EntropyAdaptiveBoundaryDetectorImpl : torch::nn::Module {
    int64_t hidden_dim;
    torch::nn::Sequential boundary_gate_net{nullptr};
    torch::Tensor boundary_bytes;

    EntropyAdaptiveBoundaryDetectorImpl(int64_t hidden_dim = 768, std::string device_str = "cpu") : hidden_dim(hidden_dim) {
        boundary_gate_net = register_module("boundary_gate_net", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, 128),
            torch::nn::SiLU(),
            torch::nn::Linear(128, 1),
            torch::nn::Sigmoid()
        ));

        auto opts = torch::TensorOptions().dtype(torch::kInt64);
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) opts = opts.device(torch::kCUDA);
        boundary_bytes = register_buffer("boundary_bytes", torch::tensor({32, 10, 44, 46, 58, 59, 63, 33, 34, 39}, opts));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    torch::Tensor forward(torch::Tensor h_stage1, torch::Tensor input_ids) {
        auto pred_boundary = boundary_gate_net->forward(h_stage1);
        auto is_token_boundary = torch::isin(input_ids, boundary_bytes).to(torch::kFloat32).unsqueeze(-1);
        auto saliency = torch::clamp(0.05f + 0.60f * pred_boundary + 0.35f * is_token_boundary, 0.0f, 1.0f);
        return saliency.squeeze(-1).unsqueeze(1);
    }
};

// ============================================================================
// 11. CORTICAL STAGE (SSD + CONVSWIGLU + PRE-LAYERNORM)
// ============================================================================
struct CorticalStageImpl : torch::nn::Module {
    torch::nn::LayerNorm pre_norm_ssd{nullptr};
    std::shared_ptr<ParallelLogDecaySSDLayerImpl> ssd{nullptr};
    torch::nn::LayerNorm pre_norm_swiglu{nullptr};
    std::shared_ptr<CausalConvSwiGLUBlockImpl> swiglu{nullptr};

    CorticalStageImpl(int64_t hidden_dim = 768, int64_t expand_dim = 3072, int64_t num_heads = 12,
                     int64_t head_k = 64, int64_t head_v = 128, float min_beta = 0.0005f, float max_beta = 0.08f,
                     int64_t swiglu_kernel_size = 3, int64_t chunk_size = 64, std::string device_str = "cpu") {

        pre_norm_ssd = register_module("pre_norm_ssd", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        ssd = register_module("ssd", std::make_shared<ParallelLogDecaySSDLayerImpl>(
            hidden_dim, hidden_dim, num_heads, head_k, head_v, min_beta, max_beta, chunk_size, device_str
        ));
        pre_norm_swiglu = register_module("pre_norm_swiglu", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        swiglu = register_module("swiglu", std::make_shared<CausalConvSwiGLUBlockImpl>(
            hidden_dim, expand_dim, swiglu_kernel_size, device_str
        ));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t,
        torch::Tensor saliency_gate = torch::Tensor(), float dt = 1.0f) {

        auto norm_x = pre_norm_ssd->forward(x);
        auto ssd_out = ssd->forward(norm_x, m_prev, u_t, saliency_gate, dt);
        auto h_ssd = std::get<0>(ssd_out);
        auto m_next = std::get<1>(ssd_out);
        auto eff_dt = std::get<2>(ssd_out);

        auto x_res1 = x + h_ssd.view_as(x);
        auto norm_res1 = pre_norm_swiglu->forward(x_res1);
        auto x_out = swiglu->forward(norm_res1);

        return std::make_tuple(x_out, m_next, eff_dt);
    }
};

// ============================================================================
// 12. PRECISION-WEIGHTED LAMINAR ERROR ROUTING (PW-LPER - EXP-75)
// ============================================================================
struct PrecisionWeightedLPERImpl : torch::nn::Module {
    int64_t hidden_dim;
    torch::nn::Sequential topdown_pred_net{nullptr};
    torch::nn::Sequential precision_net{nullptr};

    PrecisionWeightedLPERImpl(int64_t hidden_dim = 768, std::string device_str = "cpu") : hidden_dim(hidden_dim) {
        topdown_pred_net = register_module("topdown_pred_net", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, hidden_dim),
            torch::nn::SiLU(),
            torch::nn::Linear(hidden_dim, hidden_dim)
        ));
        precision_net = register_module("precision_net", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim * 2 + 1, 128),
            torch::nn::SiLU(),
            torch::nn::Linear(128, hidden_dim),
            torch::nn::Sigmoid()
        ));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor h_s1, torch::Tensor h1_prev_last, torch::Tensor u_t) {

        int64_t batch_size = h_s1.size(0);
        int64_t chunk_len = h_s1.size(1);

        auto h1_shifted = torch::cat({h1_prev_last, h_s1.slice(1, 0, -1)}, 1);
        auto h1_pred = topdown_pred_net->forward(h1_shifted);
        auto e1_raw = h_s1 - h1_pred;

        auto na_level = u_t.slice(1, 4, 5).unsqueeze(1).expand({batch_size, chunk_len, 1});
        auto prec_in = torch::cat({h_s1, h1_pred, na_level}, -1);
        auto pi_t = 2.0f * precision_net->forward(prec_in);

        auto e1_weighted = pi_t * e1_raw;
        return std::make_tuple(e1_weighted, h_s1.slice(1, -1).detach(), pi_t.mean());
    }
};

// ============================================================================
// 12.1 FUSED CASCADED LAMINAR CORTICAL STACK (ZERO-PYTHON PING-PONG - EXP-113)
// ============================================================================
struct FusedCascadedLaminarStackImpl : torch::nn::Module {
    int64_t hidden_dim;
    std::shared_ptr<CorticalStageImpl> stage1{nullptr};
    std::shared_ptr<EntropyAdaptiveBoundaryDetectorImpl> boundary_detector{nullptr};
    std::shared_ptr<PrecisionWeightedLPERImpl> pw_lper{nullptr};
    std::shared_ptr<CorticalStageImpl> stage2{nullptr};

    FusedCascadedLaminarStackImpl(int64_t hidden_dim = 768, int64_t expand_dim = 3072, int64_t num_heads = 12,
                                  int64_t head_k = 64, int64_t head_v = 128, int64_t chunk_size = 64,
                                  std::string device_str = "cpu") : hidden_dim(hidden_dim) {
        
        stage1 = register_module("stage1", std::make_shared<CorticalStageImpl>(
            hidden_dim, expand_dim, num_heads, head_k, head_v, 0.005f, 0.15f, 3, chunk_size, device_str
        ));
        boundary_detector = register_module("boundary_detector", std::make_shared<EntropyAdaptiveBoundaryDetectorImpl>(
            hidden_dim, device_str
        ));
        pw_lper = register_module("pw_lper", std::make_shared<PrecisionWeightedLPERImpl>(
            hidden_dim, device_str
        ));
        stage2 = register_module("stage2", std::make_shared<CorticalStageImpl>(
            hidden_dim, expand_dim, num_heads, head_k, head_v, 0.0001f, 0.05f, 7, chunk_size, device_str
        ));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor h_in, torch::Tensor m_s1_prev, torch::Tensor m_s2_prev, torch::Tensor u_t, torch::Tensor text_ids) {

        int64_t batch_size = h_in.size(0);

        // Stage 1 Morpho-Syntactic Execution
        auto s1_out = stage1->forward(h_in, m_s1_prev, u_t, torch::Tensor(), 1.0f);
        auto h_s1 = std::get<0>(s1_out);
        auto m_s1_next = std::get<1>(s1_out);

        // Dynamic Saliency & Entropy Boundary Detection
        auto sal_gate = boundary_detector->forward(h_s1, text_ids);

        // Precision-Weighted Laminar Error Routing
        auto h1_prev_proxy = m_s1_prev.view({batch_size, -1}).slice(1, 0, hidden_dim).unsqueeze(1);
        auto lper_out = pw_lper->forward(h_s1, h1_prev_proxy, u_t);
        auto e1_weighted = std::get<0>(lper_out);

        // Stage 2 Semantic-Discourse Execution
        auto s2_out = stage2->forward(e1_weighted, m_s2_prev, u_t, sal_gate, 1.0f);
        auto h_s2 = std::get<0>(s2_out);
        auto m_s2_next = std::get<1>(s2_out);

        return std::make_tuple(h_s1, h_s2, m_s1_next, m_s2_next, sal_gate);
    }
};

// ============================================================================
// 13. DENSE MODERN HOPFIELD ATTRACTOR HEAD (COMMITMENT LOSS)
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
        torch::Tensor beta;
        if (u_t.defined() && u_t.numel() >= 6) {
            auto da_val = u_t.select(1, 5).view({-1, 1});
            if (h_state.size(0) != u_t.size(0) && u_t.size(0) > 0 && h_state.size(0) % u_t.size(0) == 0) {
                int64_t factor = h_state.size(0) / u_t.size(0);
                da_val = da_val.unsqueeze(1).expand({u_t.size(0), factor, 1}).reshape({-1, 1});
            }
            beta = 1.0f + 1.5f * da_val;
        } else {
            beta = torch::ones({1, 1}, h_state.options());
        }

        auto sim = torch::matmul(h_state, attractor_basins.transpose(0, 1)) * (scale * beta);
        auto attn_weights = torch::softmax(sim, -1);
        auto attractor_shift = torch::matmul(attn_weights, attractor_basins);
        auto h_relaxed = norm->forward(h_state + 0.25f * attractor_shift);
        
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
// 14. ACTIVE INFERENCE LATENT WORLD MODEL
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
        auto logvar_prior = torch::clamp(prior_chunks[1], -4.0f, 4.0f);

        auto post_out = posterior_net->forward(torch::cat({h_fast_prev, w_t}, -1));
        auto post_chunks = post_out.chunk(2, -1);
        auto mu_post = post_chunks[0];
        auto logvar_post = torch::clamp(post_chunks[1], -4.0f, 4.0f);

        auto std_post = torch::exp(0.5f * logvar_post);
        auto eps = torch::randn_like(std_post);
        auto z_t = mu_post + eps * std_post;

        auto w_pred = decoder_net->forward(torch::cat({z_t, h_slow_curr}, -1));

        // Compute KL Divergence in float32 explicitly to prevent FP16 overflow under AMP
        auto mu_prior_f32 = mu_prior.to(torch::kFloat32);
        auto logvar_prior_f32 = logvar_prior.to(torch::kFloat32);
        auto mu_post_f32 = mu_post.to(torch::kFloat32);
        auto logvar_post_f32 = logvar_post.to(torch::kFloat32);

        auto var_prior_f32 = torch::exp(logvar_prior_f32) + 1e-6f;
        auto var_post_f32 = torch::exp(logvar_post_f32) + 1e-6f;

        auto kl_div_f32 = 0.5f * torch::mean(
            logvar_prior_f32 - logvar_post_f32 + (var_post_f32 + torch::pow(mu_post_f32 - mu_prior_f32, 2)) / var_prior_f32 - 1.0f,
            -1, true
        );
        auto kl_div_clamped = torch::clamp(kl_div_f32, 0.0f, 10.0f).to(w_t.scalar_type());

        auto rec_loss = torch::mean(torch::pow(w_t - w_pred, 2), -1, true);
        auto free_energy = kl_div_clamped + rec_loss;

        return std::make_tuple(w_pred, kl_div_clamped, free_energy, z_t);
    }

    std::tuple<torch::Tensor, float> evaluate_counterfactual_rollout(
        torch::Tensor h_prev, 
        torch::Tensor w_curr, 
        int64_t num_steps = 3) {
        
        auto h_sim = h_prev.clone();
        auto w_sim = w_curr.clone();
        float total_efe = 0.0f;

        for (int64_t step = 0; step < num_steps; ++step) {
            auto out = forward(h_sim, h_sim, w_sim);
            auto w_pred = std::get<0>(out);
            auto fe = std::get<2>(out);
            total_efe += fe.mean().item<float>();
            w_sim = w_pred;
        }
        return std::make_tuple(w_sim, total_efe);
    }
};

// ============================================================================
// 15. HIGH-VELOCITY BATCHED EPISODIC MEMORY (ZERO-SYNC HOST TRACKING)
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
        auto orig_dtype = query.scalar_type();
        if (keys.device() != query.device() || keys.dtype() != query.dtype()) {
            query = query.to(keys.device(), keys.dtype());
        }

        int64_t q_b = query.size(0);
        int64_t max_active = (max_active_cpu > 0) ? max_active_cpu : size.max().item<int64_t>();

        if (max_active == 0) {
            auto empty_val = torch::zeros({q_b, memory_dim}, query.options()).to(orig_dtype);
            auto empty_sim = torch::zeros({q_b, 1}, query.options()).to(orig_dtype);
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

        auto sim_masked = sim.masked_fill(invalid_mask, -10000.0f);
        auto max_sim = std::get<0>(sim_masked.max(-1));
        auto max_sim_valid = torch::where(active_size.unsqueeze(-1) > 0, max_sim, torch::zeros_like(max_sim));

        auto gate = torch::sigmoid((max_sim_valid - threshold) * sigmoid_beta);
        auto attn_weights = torch::softmax(sim_masked / temperature, -1);

        auto retrieved_val = torch::bmm(attn_weights, active_values).squeeze(1);
        auto gated_retrieved = (retrieved_val * gate).to(orig_dtype);
        auto max_sim_out = max_sim_valid.to(orig_dtype);

        return std::make_tuple(gated_retrieved, max_sim_out);
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
// 16. TEMPORAL-DIFFERENCE FREE ENERGY VALUE CRITIC (EXP-89/EXP-90)
// ============================================================================
class TDFreeEnergyCriticImpl : public torch::nn::Module {
public:
    torch::nn::Sequential net{nullptr};

    TDFreeEnergyCriticImpl(int64_t hidden_dim = 768, std::string device_str = "cpu") {
        net = register_module("net", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim, 256),
            torch::nn::SiLU(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})),
            torch::nn::Linear(256, 1)
        ));

        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    torch::Tensor forward(torch::Tensor h) {
        return net->forward(h);
    }
};
// ============================================================================
// 17. VOLITIONAL ACTION EVALUATOR & LOCAL PLASTICITY C++20 ENGINE (EXP-125)
// ============================================================================
class VolitionalActionEvaluatorImpl : public torch::nn::Module {
public:
    torch::nn::Linear action_head{nullptr};

    VolitionalActionEvaluatorImpl(int64_t hidden_dim = 512, std::string device_str = "cpu") {
        action_head = register_module("action_head", torch::nn::Linear(hidden_dim, 3));
        if (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) {
            this->to(torch::kCUDA);
        }
    }

    // Evaluates Expected Free Energy G and returns optimal policy index:
    // 0: EXPRESS_OUTPUT, 1: THINK_DEEPER_SANDBOX, 2: INITIATE_SLEEP_CONSOLIDATION
    int64_t select_volitional_action(torch::Tensor h_current, float curiosity, float energy) {
        torch::NoGradGuard no_grad;
        auto logits = action_head->forward(h_current);
        logits.select(1, 1).add_(1.5f * curiosity);
        logits.select(1, 2).add_(2.0f * std::max(0.0f, 0.40f - energy));
        return logits.argmax(-1).item<int64_t>();
    }
};

class LocalNeuromodulatedPlasticityImpl : public torch::nn::Module {
public:
    int64_t in_features;
    int64_t out_features;
    float lr;
    torch::Tensor W_base;
    torch::Tensor W_fast;

    LocalNeuromodulatedPlasticityImpl(int64_t in_features = 512, int64_t out_features = 512, float lr = 0.08f, std::string device_str = "cpu")
        : in_features(in_features), out_features(out_features), lr(lr) {
        auto dev = (device_str.find("cuda") != std::string::npos && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
        W_base = register_parameter("W_base", torch::randn({out_features, in_features}, dev) * (1.0f / std::sqrt(static_float_cast(in_features))));
        W_fast = register_buffer("W_fast", torch::zeros({out_features, in_features}, dev));
    }

    static float static_float_cast(int64_t val) { return static_cast<float>(val); }

    torch::Tensor forward(torch::Tensor x) {
        auto W_eff = W_base + W_fast;
        return torch::nn::functional::linear(x, W_eff);
    }

    void adapt_local_fast_weights(torch::Tensor pre_act, torch::Tensor post_err, float na_t, float da_t) {
        torch::NoGradGuard no_grad;
        float neuromodulation = 0.20f + 0.80f * na_t + 0.50f * da_t;
        auto dW = torch::bmm(post_err.unsqueeze(-1), pre_act.unsqueeze(1)).mean(0);
        W_fast.mul_(0.92f); // Passive decay
        W_fast.add_(dW * (lr * neuromodulation));
    }
};

// ============================================================================
// 16. PYBIND11 MODULE BINDINGS (ALL 15 NATIVE C++ COGNITIVE MODULES)
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
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("unified_dim") = 256, py::arg("hidden_dim") = 768, py::arg("homeo_dim") = 6,
             py::arg("text_dim") = 256, py::arg("vision_dim") = 256, py::arg("audio_dim") = 256,
             py::arg("binary_dim") = 256, py::arg("telepathic_dim") = 256, py::arg("action_dim") = 3,
             py::arg("device") = "cpu")
        .def("forward", [](SensoryGatewayImpl& self, torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor motor_input, torch::Tensor h_prev, torch::Tensor u_t) {
            return self.forward(text_input, vision_input, motor_input, h_prev, u_t);
        })
        .def("forward", [](SensoryGatewayImpl& self, torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor audio_input, torch::Tensor binary_input, torch::Tensor telepathic_input, torch::Tensor motor_input, torch::Tensor h_prev, torch::Tensor u_t) {
            return self.forward(text_input, vision_input, audio_input, binary_input, telepathic_input, motor_input, h_prev, u_t);
        })
        .def("parameters", [](std::shared_ptr<SensoryGatewayImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<SensoryGatewayImpl> m) { return m->named_parameters(); })
        .def("__call__", [](SensoryGatewayImpl& self, torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor motor_input, torch::Tensor h_prev, torch::Tensor u_t) {
            return self.forward(text_input, vision_input, motor_input, h_prev, u_t);
        })
        .def("__call__", [](SensoryGatewayImpl& self, torch::Tensor text_input, torch::Tensor vision_input, torch::Tensor audio_input, torch::Tensor binary_input, torch::Tensor telepathic_input, torch::Tensor motor_input, torch::Tensor h_prev, torch::Tensor u_t) {
            return self.forward(text_input, vision_input, audio_input, binary_input, telepathic_input, motor_input, h_prev, u_t);
        });

    py::class_<MotorGatewayImpl, torch::nn::Module, std::shared_ptr<MotorGatewayImpl>>(m, "MotorGateway")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 768, py::arg("action_dim") = 3, py::arg("cog_action_dim") = 3, py::arg("text_gen_dim") = 258,
             py::arg("vision_dim") = 256, py::arg("audio_dim") = 256, py::arg("binary_dim") = 256, py::arg("telepathic_dim") = 256,
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

    py::class_<MultiScaleBytePyramidReceptiveFieldImpl, torch::nn::Module, std::shared_ptr<MultiScaleBytePyramidReceptiveFieldImpl>>(m, "MultiScaleBytePyramidReceptiveField")
        .def(py::init<int64_t, std::string>(), py::arg("text_dim") = 256, py::arg("device") = "cpu")
        .def("forward", &MultiScaleBytePyramidReceptiveFieldImpl::forward)
        .def("parameters", [](std::shared_ptr<MultiScaleBytePyramidReceptiveFieldImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<MultiScaleBytePyramidReceptiveFieldImpl> m) { return m->named_parameters(); })
        .def("__call__", &MultiScaleBytePyramidReceptiveFieldImpl::forward);

    py::class_<ParallelLogDecaySSDLayerImpl, torch::nn::Module, std::shared_ptr<ParallelLogDecaySSDLayerImpl>>(m, "ParallelLogDecaySSDLayer")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, float, float, int64_t, std::string>(),
             py::arg("in_dim") = 768, py::arg("out_dim") = 768, py::arg("num_heads") = 12,
             py::arg("head_k") = 64, py::arg("head_v") = 128, py::arg("min_beta") = 0.0005f, py::arg("max_beta") = 0.08f,
             py::arg("chunk_size") = 64, py::arg("device") = "cpu")
        .def("forward", [](ParallelLogDecaySSDLayerImpl& self, torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t) {
            return self.forward(x_seq, m_prev, u_t, torch::Tensor(), 1.0f);
        })
        .def("forward", [](ParallelLogDecaySSDLayerImpl& self, torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate) {
            return self.forward(x_seq, m_prev, u_t, saliency_gate, 1.0f);
        })
        .def("forward", [](ParallelLogDecaySSDLayerImpl& self, torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate, float dt) {
            return self.forward(x_seq, m_prev, u_t, saliency_gate, dt);
        })
        .def("parameters", [](std::shared_ptr<ParallelLogDecaySSDLayerImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<ParallelLogDecaySSDLayerImpl> m) { return m->named_parameters(); })
        .def("__call__", [](ParallelLogDecaySSDLayerImpl& self, torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t) {
            return self.forward(x_seq, m_prev, u_t, torch::Tensor(), 1.0f);
        })
        .def("__call__", [](ParallelLogDecaySSDLayerImpl& self, torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate) {
            return self.forward(x_seq, m_prev, u_t, saliency_gate, 1.0f);
        })
        .def("__call__", [](ParallelLogDecaySSDLayerImpl& self, torch::Tensor x_seq, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate, float dt) {
            return self.forward(x_seq, m_prev, u_t, saliency_gate, dt);
        });

    py::class_<CausalConvSwiGLUBlockImpl, torch::nn::Module, std::shared_ptr<CausalConvSwiGLUBlockImpl>>(m, "CausalConvSwiGLUBlock")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 768, py::arg("expand_dim") = 3072, py::arg("kernel_size") = 3, py::arg("device") = "cpu")
        .def("forward", &CausalConvSwiGLUBlockImpl::forward)
        .def("parameters", [](std::shared_ptr<CausalConvSwiGLUBlockImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CausalConvSwiGLUBlockImpl> m) { return m->named_parameters(); })
        .def("__call__", &CausalConvSwiGLUBlockImpl::forward);

    py::class_<EntropyAdaptiveBoundaryDetectorImpl, torch::nn::Module, std::shared_ptr<EntropyAdaptiveBoundaryDetectorImpl>>(m, "EntropyAdaptiveBoundaryDetector")
        .def(py::init<int64_t, std::string>(), py::arg("hidden_dim") = 768, py::arg("device") = "cpu")
        .def("forward", &EntropyAdaptiveBoundaryDetectorImpl::forward)
        .def_readwrite("boundary_bytes", &EntropyAdaptiveBoundaryDetectorImpl::boundary_bytes)
        .def("parameters", [](std::shared_ptr<EntropyAdaptiveBoundaryDetectorImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<EntropyAdaptiveBoundaryDetectorImpl> m) { return m->named_parameters(); })
        .def("__call__", &EntropyAdaptiveBoundaryDetectorImpl::forward);

    py::class_<CorticalStageImpl, torch::nn::Module, std::shared_ptr<CorticalStageImpl>>(m, "CorticalStage")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, float, float, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 768, py::arg("expand_dim") = 3072, py::arg("num_heads") = 12,
             py::arg("head_k") = 64, py::arg("head_v") = 128, py::arg("min_beta") = 0.0005f, py::arg("max_beta") = 0.08f,
             py::arg("swiglu_kernel_size") = 3, py::arg("chunk_size") = 64, py::arg("device") = "cpu")
        .def("forward", [](CorticalStageImpl& self, torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t) {
            return self.forward(x, m_prev, u_t, torch::Tensor(), 1.0f);
        })
        .def("forward", [](CorticalStageImpl& self, torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate) {
            return self.forward(x, m_prev, u_t, saliency_gate, 1.0f);
        })
        .def("forward", [](CorticalStageImpl& self, torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate, float dt) {
            return self.forward(x, m_prev, u_t, saliency_gate, dt);
        })
        .def("parameters", [](std::shared_ptr<CorticalStageImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<CorticalStageImpl> m) { return m->named_parameters(); })
        .def("__call__", [](CorticalStageImpl& self, torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t) {
            return self.forward(x, m_prev, u_t, torch::Tensor(), 1.0f);
        })
        .def("__call__", [](CorticalStageImpl& self, torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate) {
            return self.forward(x, m_prev, u_t, saliency_gate, 1.0f);
        })
        .def("__call__", [](CorticalStageImpl& self, torch::Tensor x, torch::Tensor m_prev, torch::Tensor u_t, torch::Tensor saliency_gate, float dt) {
            return self.forward(x, m_prev, u_t, saliency_gate, dt);
        });

    py::class_<PrecisionWeightedLPERImpl, torch::nn::Module, std::shared_ptr<PrecisionWeightedLPERImpl>>(m, "PrecisionWeightedLPER")
        .def(py::init<int64_t, std::string>(), py::arg("hidden_dim") = 768, py::arg("device") = "cpu")
        .def("forward", &PrecisionWeightedLPERImpl::forward)
        .def("parameters", [](std::shared_ptr<PrecisionWeightedLPERImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<PrecisionWeightedLPERImpl> m) { return m->named_parameters(); })
        .def("__call__", &PrecisionWeightedLPERImpl::forward);

    py::class_<FusedCascadedLaminarStackImpl, torch::nn::Module, std::shared_ptr<FusedCascadedLaminarStackImpl>>(m, "FusedCascadedLaminarStack")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 768, py::arg("expand_dim") = 3072, py::arg("num_heads") = 12,
             py::arg("head_k") = 64, py::arg("head_v") = 128, py::arg("chunk_size") = 64, py::arg("device") = "cpu")
        .def("forward", &FusedCascadedLaminarStackImpl::forward,
             py::arg("h_in"), py::arg("m_s1_prev"), py::arg("m_s2_prev"), py::arg("u_t"), py::arg("text_ids"))
        .def("parameters", [](std::shared_ptr<FusedCascadedLaminarStackImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<FusedCascadedLaminarStackImpl> m) { return m->named_parameters(); })
        .def("__call__", &FusedCascadedLaminarStackImpl::forward,
             py::arg("h_in"), py::arg("m_s1_prev"), py::arg("m_s2_prev"), py::arg("u_t"), py::arg("text_ids"));

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
        .def("evaluate_counterfactual_rollout", &LatentPredictorImpl::evaluate_counterfactual_rollout,
             py::arg("h_prev"), py::arg("w_curr"), py::arg("num_steps") = 3)
        .def("parameters", [](std::shared_ptr<LatentPredictorImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<LatentPredictorImpl> m) { return m->named_parameters(); })
        .def("__call__", &LatentPredictorImpl::forward);

    py::class_<TDFreeEnergyCriticImpl, torch::nn::Module, std::shared_ptr<TDFreeEnergyCriticImpl>>(m, "TDFreeEnergyCritic")
        .def(py::init<int64_t, std::string>(), py::arg("hidden_dim") = 768, py::arg("device") = "cpu")
        .def("forward", &TDFreeEnergyCriticImpl::forward)
        .def("parameters", [](std::shared_ptr<TDFreeEnergyCriticImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<TDFreeEnergyCriticImpl> m) { return m->named_parameters(); })
        .def("__call__", &TDFreeEnergyCriticImpl::forward);

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
    py::class_<VolitionalActionEvaluatorImpl, torch::nn::Module, std::shared_ptr<VolitionalActionEvaluatorImpl>>(m, "VolitionalActionEvaluator")
        .def(py::init<int64_t, std::string>(), py::arg("hidden_dim") = 512, py::arg("device") = "cpu")
        .def("select_volitional_action", &VolitionalActionEvaluatorImpl::select_volitional_action,
             py::arg("h_current"), py::arg("curiosity"), py::arg("energy"));

    py::class_<LocalNeuromodulatedPlasticityImpl, torch::nn::Module, std::shared_ptr<LocalNeuromodulatedPlasticityImpl>>(m, "LocalNeuromodulatedPlasticity")
        .def(py::init<int64_t, int64_t, float, std::string>(),
             py::arg("in_features") = 512, py::arg("out_features") = 512, py::arg("lr") = 0.08f, py::arg("device") = "cpu")
        .def("forward", &LocalNeuromodulatedPlasticityImpl::forward)
        .def("__call__", &LocalNeuromodulatedPlasticityImpl::forward)
        .def("adapt_local_fast_weights", &LocalNeuromodulatedPlasticityImpl::adapt_local_fast_weights,
             py::arg("pre_act"), py::arg("post_err"), py::arg("na_t"), py::arg("da_t"))
        .def_readwrite("W_base", &LocalNeuromodulatedPlasticityImpl::W_base)
        .def_readwrite("W_fast", &LocalNeuromodulatedPlasticityImpl::W_fast);
}
