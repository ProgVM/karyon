#ifndef KARYON_CORE_CROSSPLATFORM_HPP
#define KARYON_CORE_CROSSPLATFORM_HPP

// ============================================================================
// KARYON-CORE CROSS-PLATFORM & MULTI-DEVICE ABSTRACTION LAYER
// KEP v40.0 Sovereign Master | Windows / Linux / macOS & CPU SIMD / CUDA GPU
// ============================================================================

#include <cstdint>
#include <iostream>

// OS Platform Detection & Dynamic Library Exports
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    #define KARYON_OS_WINDOWS 1
    #ifdef KARYON_BUILD_DLL
        #define KARYON_API __declspec(dllexport)
    #else
        #define KARYON_API __declspec(dllimport)
    #endif
    #include <intrin.h>
#elif defined(__APPLE__) && defined(__MACH__)
    #define KARYON_OS_MACOS 1
    #define KARYON_API __attribute__((visibility("default")))
#else
    #define KARYON_OS_LINUX 1
    #define KARYON_API __attribute__((visibility("default")))
#endif

// Compiler Intrinsics Abstraction (MSVC vs GCC/Clang)
#if defined(KARYON_OS_WINDOWS) && defined(_MSC_VER)
    #include <intrin.h>
    inline int karyon_popcount64(uint64_t val) {
        return (int)__popcnt64(val);
    }
#else
    inline int karyon_popcount64(uint64_t val) {
        return __builtin_popcountll(val);
    }
#endif

// GPU CUDA Support Abstraction
#if defined(__CUDACC__) || defined(__NVCC__)
    #define KARYON_CUDA_SUPPORTED 1
    #define KARYON_CUDA_HOST_DEVICE __host__ __device__
#else
    #define KARYON_CUDA_HOST_DEVICE
#endif

#endif // KARYON_CORE_CROSSPLATFORM_HPP
