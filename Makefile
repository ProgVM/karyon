# Makefile for Karyon-CoRE Sovereign Native Substrate
# Author: Bazilevs (ProgVM)

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -fPIC -mavx2 -mfma -fopenmp
LDFLAGS ?= -lm

.PHONY: all clean test

all: libkaryon_core.so

# Compile native pure C++20 shared library (C-ABI)
libkaryon_core.so: karyon_core.cpp karyon_core.h
	$(CXX) $(CXXFLAGS) -shared karyon_core.cpp -o libkaryon_core.so $(LDFLAGS)

clean:
	rm -f libkaryon_core.so *.o
