// karyon_llvm_engine.h
#ifndef KARYON_LLVM_ENGINE_H
#define KARYON_LLVM_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>

class KaryonLLVMEngine {
public:
    KaryonLLVMEngine();
    ~KaryonLLVMEngine();

    // Loads and JIT-compiles LLVM Bitcode directly from in-memory byte buffer
    bool load_bitcode_from_memory(const uint8_t* bitcode_data, size_t size);

    // Looks up compiled symbol address in executable memory
    void* lookup_symbol(const std::string& symbol_name);
};

#endif // KARYON_LLVM_ENGINE_H
