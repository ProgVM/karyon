# Makefile for Karyon-CoRE Sovereign Native Substrate
# Author: Bazilevs (ProgVM)

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -fPIC -mavx2 -mfma
LDFLAGS ?= -lm

.PHONY: all clean test-cxx test-rust cli

all: libkaryon_native.so karyon

# Compile native pure C++20 shared library (C-ABI)
libkaryon_native.so: karyon_native_core.cpp karyon_native_core.h
	$(CXX) $(CXXFLAGS) -shared karyon_native_core.cpp -o libkaryon_native.so $(LDFLAGS)

# Compile standalone CLI binary (Zero Python, Zero Torch)
karyon: karyon_cli.cpp karyon_native_core.cpp karyon_native_core.h
	$(CXX) $(CXXFLAGS) karyon_cli.cpp karyon_native_core.cpp -o karyon $(LDFLAGS)

# Check if rustc/cargo is available and compile Rust acceleration kernel
rust:
	@if command -v cargo >/dev/null 2>&1; then \
		echo "[INFO] Compiling Rust high-performance engine..."; \
		cd rust_engine && cargo build --release && cp target/release/libkaryon_rust_core.so ../libkaryon_rust_core.so 2>/dev/null || true; \
	else \
		echo "[NOTICE] Cargo/rustc not found in environment. Using high-speed C++20 native engine."; \
	fi

clean:
	rm -f libkaryon_native.so karyon *.kcore
