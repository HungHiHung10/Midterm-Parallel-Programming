# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Portable V0 build; CUDA path remains available.
NVCC ?= nvcc
CXX = g++
TARGET ?= stitch
SRCS = main.cu stb_impl.cpp v0_frontend.cpp v0_backend.cpp v0_stitching.cpp
HEADERS = common.h versions.h logging.h stb_image.h stb_image_write.h
all: cpu
cpu: $(SRCS) $(HEADERS)
	$(CXX) -O3 -std=c++14 -x c++ $(SRCS) -o $(TARGET)
cuda: $(SRCS) $(HEADERS) cuda_utils.cuh
	$(NVCC) -O3 -std=c++14 $(SRCS) -o $(TARGET)
clean:
	rm -f $(TARGET) $(TARGET).exe *.o *.obj
.PHONY: all cpu cuda clean
