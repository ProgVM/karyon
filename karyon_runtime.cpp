// karyon_runtime.cpp
#include "karyon_runtime.h"
#include "kcore_format.h"

#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

struct KaryonEntity {
    torch::Device device;
    
    torch::Tensor h_fast;
    torch::Tensor h_slow;
    torch::Tensor u_t;

    std::vector<uint8_t> generated_text_bytes;
    std::vector<float>   motor_action_vec;
    std::vector<float>   cog_action_vec;

    std::vector<uint8_t> manifest_buffer;
    std::vector<uint8_t> logic_buffer;
    std::vector<uint8_t> weights_buffer;
    std::vector<uint8_t> state_buffer;

    std::string response_buffer;

    KaryonEntity(const std::string& dev_str) 
        : device(dev_str.find("cuda") != std::string::npos ? torch::kCUDA : torch::kCPU) {
        motor_action_vec.resize(3, 0.0f);
        cog_action_vec.resize(3, 0.0f);
    }
};

extern "C" {

KaryonEntity* karyon_load(const char* kcore_file_path, const char* device) {
    std::ifstream file(kcore_file_path, std::ios::binary);
    if (!file.is_open()) return nullptr;

    KcoreHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(KcoreHeader));

    if (std::memcmp(header.magic, KCORE_MAGIC, 5) != 0) return nullptr;

    std::vector<KcoreSectionHeader> sections(header.num_sections);
    file.read(reinterpret_cast<char*>(sections.data()), sizeof(KcoreSectionHeader) * header.num_sections);

    auto entity = new KaryonEntity(device ? device : "cpu");

    for (const auto& sec : sections) {
        file.seekg(sec.offset, std::ios::beg);
        if (sec.type == static_cast<uint32_t>(KcoreSectionType::MANIFEST)) {
            entity->manifest_buffer.resize(sec.size);
            file.read(reinterpret_cast<char*>(entity->manifest_buffer.data()), sec.size);
        } else if (sec.type == static_cast<uint32_t>(KcoreSectionType::LOGIC_CPP_SOURCE) || 
                   sec.type == static_cast<uint32_t>(KcoreSectionType::LOGIC_LLVM_BITCODE)) {
            entity->logic_buffer.resize(sec.size);
            file.read(reinterpret_cast<char*>(entity->logic_buffer.data()), sec.size);
        } else if (sec.type == static_cast<uint32_t>(KcoreSectionType::WEIGHTS)) {
            entity->weights_buffer.resize(sec.size);
            file.read(reinterpret_cast<char*>(entity->weights_buffer.data()), sec.size);
        } else if (sec.type == static_cast<uint32_t>(KcoreSectionType::STATE)) {
            entity->state_buffer.resize(sec.size);
            file.read(reinterpret_cast<char*>(entity->state_buffer.data()), sec.size);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(entity->device);
    // Synced with v16.5 Cortical dimension (768)
    entity->h_fast = torch::zeros({1, 768}, opts);
    entity->h_slow = torch::zeros({1, 768}, opts);
    entity->u_t = torch::tensor({{0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}}, opts);

    return entity;
}

void karyon_perceive_text(KaryonEntity* entity, const char* text_utf8, float dt) {
    if (!entity || !text_utf8) return;
    std::string input_str(text_utf8);
    entity->generated_text_bytes.assign(input_str.begin(), input_str.end());
    entity->response_buffer = input_str;
}

void karyon_perceive_stream(KaryonEntity* entity, const SensoryStream* stream) {
    if (!entity || !stream) return;
    if (stream->text_bytes && stream->text_len > 0) {
        entity->generated_text_bytes.assign(stream->text_bytes, stream->text_bytes + stream->text_len);
    }
}

void karyon_step(KaryonEntity* entity) {
    if (!entity) return;
    auto curiosity = entity->u_t.select(1, 0).unsqueeze(1);
    auto energy    = entity->u_t.select(1, 1).unsqueeze(1);
    auto stability = entity->u_t.select(1, 2).unsqueeze(1);
    auto health    = entity->u_t.select(1, 3).unsqueeze(1);
    auto na        = entity->u_t.select(1, 4).unsqueeze(1);
    auto da        = entity->u_t.select(1, 5).unsqueeze(1);

    energy = torch::clamp(energy - 0.001f, 0.0f, 1.0f);
    entity->u_t = torch::cat({curiosity, energy, stability, health, na, da}, 1);
}

const char* karyon_express_text(KaryonEntity* entity) {
    return entity ? entity->response_buffer.c_str() : "";
}

void karyon_express_stream(KaryonEntity* entity, MotorStream* out_stream) {
    if (!entity || !out_stream) return;
    out_stream->text_bytes = entity->generated_text_bytes.data();
    out_stream->text_len = entity->generated_text_bytes.size();
    out_stream->motor_actions = entity->motor_action_vec.data();
    out_stream->action_dim = entity->motor_action_vec.size();
    out_stream->cog_actions = entity->cog_action_vec.data();
    out_stream->cog_dim = entity->cog_action_vec.size();
}

void karyon_adapt(KaryonEntity* entity, float feedback_signal, float learning_rate) {
    if (!entity) return;
    auto opts = entity->u_t.options();
    auto feedback_tensor = torch::tensor({{feedback_signal}}, opts);

    auto da = torch::clamp(feedback_tensor, 0.0f, 1.0f);
    auto na = torch::clamp(-feedback_tensor, 0.0f, 1.0f);

    auto curiosity = entity->u_t.select(1, 0).unsqueeze(1);
    auto energy    = entity->u_t.select(1, 1).unsqueeze(1);
    auto stability = entity->u_t.select(1, 2).unsqueeze(1);
    auto health    = entity->u_t.select(1, 3).unsqueeze(1);

    entity->u_t = torch::cat({curiosity, energy, stability, health, na, da}, 1);
}

void karyon_get_somatic_state(KaryonEntity* entity, float* energy, float* health, float* arousal) {
    if (!entity) return;
    auto u_cpu = entity->u_t.to(torch::kCPU);
    auto a = u_cpu.accessor<float, 2>();
    if (energy)  *energy  = a[0][1];
    if (health)  *health  = a[0][3];
    if (arousal) *arousal = a[0][4];
}

void karyon_save(KaryonEntity* entity, const char* kcore_file_path) {
    if (!entity || !kcore_file_path) return;

    std::ofstream file(kcore_file_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return;

    KcoreHeader header;
    std::memcpy(header.magic, KCORE_MAGIC, 8);
    header.header_size = sizeof(KcoreHeader);
    header.num_sections = 4;

    uint64_t offset_manifest = sizeof(KcoreHeader) + (4 * sizeof(KcoreSectionHeader));
    uint64_t size_manifest = entity->manifest_buffer.size();

    uint64_t offset_logic = offset_manifest + size_manifest;
    uint64_t size_logic = entity->logic_buffer.size();

    uint64_t offset_weights = offset_logic + size_logic;
    uint64_t size_weights = entity->weights_buffer.size();

    uint64_t offset_state = offset_weights + size_weights;
    uint64_t size_state = entity->state_buffer.size();

    header.total_file_size = offset_state + size_state;
    header.flags = 0;

    file.write(reinterpret_cast<char*>(&header), sizeof(KcoreHeader));

    auto write_sec = [&](uint32_t type, uint64_t offset, uint64_t size, const char* name) {
        KcoreSectionHeader sec;
        sec.type = type;
        sec.flags = 0;
        sec.offset = offset;
        sec.size = size;
        sec.alignment = 64;
        std::memset(sec.name, 0, 32);
        std::strncpy(sec.name, name, 31);
        file.write(reinterpret_cast<char*>(&sec), sizeof(KcoreSectionHeader));
    };

    write_sec(1, offset_manifest, size_manifest, "manifest");
    write_sec(5, offset_logic, size_logic, "logic_llvm_bitcode");
    write_sec(3, offset_weights, size_weights, "weights");
    write_sec(4, offset_state, size_state, "persistent_state");

    file.write(reinterpret_cast<char*>(entity->manifest_buffer.data()), size_manifest);
    file.write(reinterpret_cast<char*>(entity->logic_buffer.data()), size_logic);
    file.write(reinterpret_cast<char*>(entity->weights_buffer.data()), size_weights);
    file.write(reinterpret_cast<char*>(entity->state_buffer.data()), size_state);
}

void karyon_free(KaryonEntity* entity) {
    delete entity;
}

}
