// karyon_core.cpp
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>
#include <algorithm>

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
// 2. HOMEOSTATIC SOMATIC CONTROLLER (ASHBY ULTRASTABILITY)
// ============================================================================
struct HomeostaticUnit {
    torch::Tensor state;                  // Shape: [batch_size, 6]
    torch::Tensor prev_pain;              // Shape: [batch_size, 1]
    torch::Tensor consecutive_inactivity; // Shape: [batch_size, 1]
    std::string device;

    HomeostaticUnit(int64_t batch_size = 1, std::string device_str = "cpu") 
        : device(device_str) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos) {
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

        energy = torch::clamp(energy - action_cost + 0.0012f, 0.0f, 1.0f);
        curiosity = torch::clamp(curiosity + 0.2f * prediction_error - 0.02f, 0.0f, 1.0f);

        auto inactive_mask = (cog_action == 1) | (cog_action == 2);
        consecutive_inactivity = torch::where(inactive_mask, consecutive_inactivity + 1.0f, torch::zeros_like(consecutive_inactivity));

        auto dopamine = torch::clamp(prediction_error * 2.0f, 0.0f, 1.0f);
        auto curiosity_diff = torch::abs(curiosity - 0.8f) * (1.0f + 0.15f * consecutive_inactivity);
        auto current_pain = curiosity_diff + torch::abs(energy - 1.0f) + torch::abs(stability - 1.0f) + torch::abs(health - 1.0f);

        auto pain_jump = torch::clamp(current_pain - prev_pain, 0.0f, 1.0f);
        prev_pain = current_pain;

        // Noradrenaline tracks surprise & pain gradient
        auto noradrenaline = torch::clamp(0.6f * (1.0f - stability) + 0.8f * prediction_error + 0.4f * pain_jump, 0.0f, 1.0f);
        stability = torch::clamp(stability - (0.05f * prediction_error + 0.005f * epistemic_entropy) + 0.02f, 0.0f, 1.0f);

        state = torch::cat({curiosity, energy, stability, health, noradrenaline, dopamine}, 1);
        return state;
    }
};

// ============================================================================
// 3. SENSORY GATEWAY (GLOBAL WORKSPACE INTEGRATION)
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

    SensoryGatewayImpl(int64_t unified_dim = 128, int64_t hidden_dim = 256, int64_t homeo_dim = 6,
                       int64_t text_dim = 128, int64_t vision_dim = 256, int64_t action_dim = 3,
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

        if (device_str.find("cuda") != std::string::npos) {
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

        // Prioritize text channel focus slightly
        sim.index_put_({torch::indexing::Slice(), 0}, sim.index({torch::indexing::Slice(), 0}) + 1.5f);

        auto attention_weights = torch::softmax(sim, -1);
        constexpr float eps = 1e-9f;
        auto epistemic_entropy = -torch::sum(attention_weights * torch::log(attention_weights + eps), -1, true);

        auto w_t = (attention_weights.unsqueeze(-1) * stacked_channels).sum(1);

        return std::make_tuple(w_t, attention_weights, channel_names, epistemic_entropy);
    }
};

// ============================================================================
// 4. MOTOR GATEWAY
// ============================================================================
class MotorGatewayImpl : public torch::nn::Module {
public:
    torch::nn::Linear motor_action{nullptr};
    torch::nn::Linear cognitive_gating{nullptr};
    torch::nn::Linear text_generation{nullptr};

    MotorGatewayImpl(int64_t hidden_dim = 256, int64_t action_dim = 3, int64_t cog_action_dim = 3, int64_t text_gen_dim = 258,
                     std::string device_str = "cpu") {
        motor_action = register_module("motor_action", torch::nn::Linear(hidden_dim, action_dim));
        cognitive_gating = register_module("cognitive_gating", torch::nn::Linear(hidden_dim, cog_action_dim));
        text_generation = register_module("text_generation", torch::nn::Linear(hidden_dim, text_gen_dim));

        if (device_str.find("cuda") != std::string::npos) {
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
// 5. STOCHASTIC HEUN 2ND-ORDER SDE RECURRENT CORE (REACTIVE DT)
// ============================================================================
class DynamicRecurrentCore : public torch::nn::Module {
public:
    int64_t hidden_dim;
    float gamma;
    torch::nn::Sequential slow_f{nullptr}, fast_f{nullptr};

    DynamicRecurrentCore(int64_t hidden_dim = 256, int64_t unified_dim = 128, int64_t homeo_dim = 6, float gamma = 0.1f,
                         std::string device_str = "cpu")
        : hidden_dim(hidden_dim), gamma(gamma) {
        
        slow_f = register_module("slow_f", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim + homeo_dim, hidden_dim),
            torch::nn::SiLU(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim}))
        ));

        fast_f = register_module("fast_f", torch::nn::Sequential(
            torch::nn::Linear(hidden_dim + unified_dim + hidden_dim, hidden_dim * 2),
            torch::nn::SiLU(),
            torch::nn::Linear(hidden_dim * 2, hidden_dim),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim}))
        ));

        if (device_str.find("cuda") != std::string::npos) {
            this->to(torch::kCUDA);
        }
    }

    inline torch::Tensor compute_slow_drift(const torch::Tensor& h_slow, const torch::Tensor& u_t, const torch::Tensor& eff_gamma) {
        return -eff_gamma * h_slow + slow_f->forward(torch::cat({h_slow, u_t}, -1));
    }

    inline torch::Tensor compute_fast_drift(const torch::Tensor& h_fast, const torch::Tensor& w_t_mod, const torch::Tensor& h_slow) {
        return -gamma * h_fast + fast_f->forward(torch::cat({h_fast, w_t_mod, h_slow}, -1));
    }

    std::vector<torch::Tensor> forward(
        torch::Tensor h_fast_prev, 
        torch::Tensor h_slow_prev, 
        torch::Tensor w_t, 
        torch::Tensor u_t, 
        float dt = 1.0f) {

        auto na = u_t.index({torch::indexing::Slice(), torch::indexing::Slice(4, 5)});
        auto da = u_t.index({torch::indexing::Slice(), torch::indexing::Slice(5, 6)});

        // Reactive dt: High arousal (NA) compresses step for precision, Dopamine (DA) expands step
        auto effective_dt = torch::clamp(dt * (1.0f - 0.7f * na + 0.8f * da), 0.20f, 2.50f);
        auto effective_gamma_slow = gamma * (1.0f - 0.3f * da);

        constexpr float sigma = 1e-3f;
        auto sqrt_dt = torch::sqrt(effective_dt);
        auto dW_slow = torch::randn_like(h_slow_prev) * sqrt_dt * sigma;
        auto dW_fast = torch::randn_like(h_fast_prev) * sqrt_dt * sigma;

        // Predictor step (Stratonovich-Heun SDE)
        auto k1_slow = compute_slow_drift(h_slow_prev, u_t, effective_gamma_slow);
        auto h_slow_pred = h_slow_prev + effective_dt * k1_slow + dW_slow;

        // Corrector step
        auto k2_slow = compute_slow_drift(h_slow_pred, u_t, effective_gamma_slow);
        auto h_slow_next = torch::tanh(h_slow_prev + 0.5f * effective_dt * (k1_slow + k2_slow) + dW_slow);

        auto w_t_mod = w_t * (1.0f + 0.5f * na);

        auto k1_fast = compute_fast_drift(h_fast_prev, w_t_mod, h_slow_next);
        auto h_fast_pred = h_fast_prev + effective_dt * k1_fast + dW_fast;

        auto k2_fast = compute_fast_drift(h_fast_pred, w_t_mod, h_slow_next);
        auto h_fast_next = torch::tanh(h_fast_prev + 0.5f * effective_dt * (k1_fast + k2_fast) + dW_fast);

        return {h_fast_next, h_slow_next, effective_dt};
    }
};

// ============================================================================
// 6. ACTIVE INFERENCE LATENT WORLD MODEL
// ============================================================================
class LatentPredictorImpl : public torch::nn::Module {
public:
    int64_t hidden_dim;
    int64_t unified_dim;
    int64_t latent_dim;

    torch::nn::Linear prior_net{nullptr};
    torch::nn::Linear posterior_net{nullptr};
    torch::nn::Sequential decoder_net{nullptr};

    LatentPredictorImpl(int64_t hidden_dim = 256, int64_t unified_dim = 128, int64_t latent_dim = 64, std::string device_str = "cpu")
        : hidden_dim(hidden_dim), unified_dim(unified_dim), latent_dim(latent_dim) {
        
        prior_net = register_module("prior_net", torch::nn::Linear(hidden_dim, latent_dim * 2));
        posterior_net = register_module("posterior_net", torch::nn::Linear(hidden_dim + unified_dim, latent_dim * 2));
        
        decoder_net = register_module("decoder_net", torch::nn::Sequential(
            torch::nn::Linear(latent_dim + hidden_dim, unified_dim * 2),
            torch::nn::SiLU(),
            torch::nn::Linear(unified_dim * 2, unified_dim)
        ));

        if (device_str.find("cuda") != std::string::npos) {
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
// 7. HIGH-VELOCITY BATCHED EPISODIC MEMORY (ACTIVE-CAPACITY DYNAMIC SLICING)
// ============================================================================
class BatchedEpisodicMemoryImpl : public torch::nn::Module {
public:
    int64_t batch_size;
    int64_t memory_dim;
    int64_t max_capacity;

    torch::Tensor keys;
    torch::Tensor values;
    torch::Tensor pointer;
    torch::Tensor size;

    BatchedEpisodicMemoryImpl(int64_t batch_size = 1, int64_t memory_dim = 128, int64_t max_capacity = 1000, std::string device_str = "cpu")
        : batch_size(batch_size), memory_dim(memory_dim), max_capacity(max_capacity) {
        
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (device_str.find("cuda") != std::string::npos) {
            opts = opts.device(torch::kCUDA);
        }
        keys = register_buffer("keys", torch::zeros({batch_size, max_capacity, memory_dim}, opts));
        values = register_buffer("values", torch::zeros({batch_size, max_capacity, memory_dim}, opts));
        pointer = register_buffer("pointer", torch::zeros({batch_size}, opts.dtype(torch::kInt64)));
        size = register_buffer("size", torch::zeros({batch_size}, opts.dtype(torch::kInt64)));
    }

    void write(torch::Tensor key, torch::Tensor value, int64_t protected_slots = 3) {
        if (keys.device() != key.device()) {
            keys = keys.to(key.device());
            values = values.to(value.device());
            pointer = pointer.to(key.device());
            size = size.to(key.device());
        }

        int64_t curr_batch = key.size(0);
        auto batch_indices = torch::arange(curr_batch, pointer.options());

        keys.index_put_({batch_indices, pointer}, key);
        values.index_put_({batch_indices, pointer}, value);

        size = torch::clamp(size + 1, 0, max_capacity);

        auto next_ptr = pointer + 1;
        auto wrap_mask = next_ptr >= max_capacity;
        pointer = torch::where(wrap_mask, torch::full_like(pointer, protected_slots), next_ptr);
    }

    std::tuple<torch::Tensor, torch::Tensor> read(torch::Tensor query, float temperature = 0.05f, float threshold = 0.5f, float sigmoid_beta = 15.0f) {
        if (keys.device() != query.device()) {
            keys = keys.to(query.device());
            values = values.to(query.device());
            pointer = pointer.to(query.device());
            size = size.to(query.device());
        }

        int64_t q_b = query.size(0);
        int64_t max_active = size.max().item<int64_t>();

        // Zero-cost return if memory is completely empty
        if (max_active == 0) {
            auto empty_val = torch::zeros({q_b, memory_dim}, query.options());
            auto empty_sim = torch::zeros({q_b, 1}, query.options());
            return std::make_tuple(empty_val, empty_sim);
        }

        // Active Dynamic Capacity Slicing (10x-30x CPU Acceleration)
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

        auto max_sim = std::get<0>(sim_masked.max(-1)).unsqueeze(-1);
        auto max_sim_valid = torch::where(active_size.unsqueeze(-1) > 0, max_sim, torch::zeros_like(max_sim));

        auto gate = torch::sigmoid((max_sim_valid - threshold) * sigmoid_beta);
        auto attn_weights = torch::softmax(sim_masked / temperature, -1);

        auto retrieved_val = torch::bmm(attn_weights, active_values).squeeze(1);
        auto gated_retrieved = retrieved_val * gate;

        return std::make_tuple(gated_retrieved, max_sim_valid);
    }

    void consolidate_and_prune(float similarity_threshold = 0.95f, int64_t protected_slots = 3) {
        int64_t max_active = size.max().item<int64_t>();
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
// 8. PYBIND11 MODULE BINDINGS
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
        .def("update", &HomeostaticUnit::update);

    py::class_<SensoryGatewayImpl, torch::nn::Module, std::shared_ptr<SensoryGatewayImpl>>(m, "SensoryGateway")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("unified_dim") = 128, py::arg("hidden_dim") = 256, py::arg("homeo_dim") = 6,
             py::arg("text_dim") = 128, py::arg("vision_dim") = 256, py::arg("action_dim") = 3,
             py::arg("device") = "cpu")
        .def("forward", &SensoryGatewayImpl::forward)
        .def("parameters", [](std::shared_ptr<SensoryGatewayImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<SensoryGatewayImpl> m) { return m->named_parameters(); })
        .def("__call__", &SensoryGatewayImpl::forward);

    py::class_<MotorGatewayImpl, torch::nn::Module, std::shared_ptr<MotorGatewayImpl>>(m, "MotorGateway")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 256, py::arg("action_dim") = 3, py::arg("cog_action_dim") = 3, py::arg("text_gen_dim") = 258,
             py::arg("device") = "cpu")
        .def("forward", &MotorGatewayImpl::forward)
        .def("parameters", [](std::shared_ptr<MotorGatewayImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<MotorGatewayImpl> m) { return m->named_parameters(); })
        .def("__call__", &MotorGatewayImpl::forward);

    py::class_<DynamicRecurrentCore, torch::nn::Module, std::shared_ptr<DynamicRecurrentCore>>(m, "DynamicRecurrentCore")
        .def(py::init<int64_t, int64_t, int64_t, float, std::string>(),
             py::arg("hidden_dim"), py::arg("unified_dim"), py::arg("homeo_dim") = 6, py::arg("gamma") = 0.1f,
             py::arg("device") = "cpu")
        .def("forward", &DynamicRecurrentCore::forward)
        .def("parameters", [](std::shared_ptr<DynamicRecurrentCore> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<DynamicRecurrentCore> m) { return m->named_parameters(); })
        .def("__call__", &DynamicRecurrentCore::forward);

    py::class_<LatentPredictorImpl, torch::nn::Module, std::shared_ptr<LatentPredictorImpl>>(m, "LatentPredictor")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(),
             py::arg("hidden_dim") = 256, py::arg("unified_dim") = 128, py::arg("latent_dim") = 64, py::arg("device") = "cpu")
        .def("forward", &LatentPredictorImpl::forward)
        .def("parameters", [](std::shared_ptr<LatentPredictorImpl> m) { return m->parameters(); })
        .def("named_parameters", [](std::shared_ptr<LatentPredictorImpl> m) { return m->named_parameters(); })
        .def("__call__", &LatentPredictorImpl::forward);

    py::class_<BatchedEpisodicMemoryImpl, torch::nn::Module, std::shared_ptr<BatchedEpisodicMemoryImpl>>(m, "BatchedEpisodicMemory")
        .def(py::init<int64_t, int64_t, int64_t, std::string>(),
             py::arg("batch_size") = 1, py::arg("memory_dim") = 128, py::arg("max_capacity") = 1000, py::arg("device") = "cpu")
        .def_readwrite("batch_size", &BatchedEpisodicMemoryImpl::batch_size)
        .def_readwrite("memory_dim", &BatchedEpisodicMemoryImpl::memory_dim)
        .def_readwrite("max_capacity", &BatchedEpisodicMemoryImpl::max_capacity)
        .def_readwrite("keys", &BatchedEpisodicMemoryImpl::keys)
        .def_readwrite("values", &BatchedEpisodicMemoryImpl::values)
        .def_readwrite("pointer", &BatchedEpisodicMemoryImpl::pointer)
        .def_readwrite("size", &BatchedEpisodicMemoryImpl::size)
        .def("write", &BatchedEpisodicMemoryImpl::write)
        .def("read", &BatchedEpisodicMemoryImpl::read)
        .def("consolidate_and_prune", &BatchedEpisodicMemoryImpl::consolidate_and_prune);
}
