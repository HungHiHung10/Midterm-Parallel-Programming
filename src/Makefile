# ============================================================================
# CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
# Makefile for compiling sequential baseline (V0) and CUDA versions
# ============================================================================

NVCC ?= nvcc
NVCCFLAGS = -O3 -std=c++14 -w
TARGET = stitch
SRCS = main.cu stb_impl.cpp v0_frontend.cpp v0_backend.cpp v0_stitching.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET) -lm

clean:
	rm -f $(TARGET) *.o *.obj *.exe

.PHONY: all clean
