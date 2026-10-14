#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iostream>

// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Common Definitions, Data Structures & Timing Utilities
// Author: Group XX (Lead: Member 3)
// ============================================================================

// 1. Basic 2D Point structure
struct Point2f {
    float x;
    float y;
    Point2f(float _x = 0.0f, float _y = 0.0f) : x(_x), y(_y) {}
};

// 2. Keypoint structure
struct Keypoint {
    float x;
    float y;
    float response; // Corner score (FAST / Harris)
    int octave;     // Scale octave (if multiscale)
    Keypoint(float _x = 0.0f, float _y = 0.0f, float _resp = 0.0f, int _oct = 0)
        : x(_x), y(_y), response(_resp), octave(_oct) {}
};

// 3. 256-bit Binary Descriptor (32 bytes = 256 bits) for BRIEF / ORB
struct Descriptor256 {
    uint8_t data[32];
};

// 4. Feature Match structure
struct Match {
    int queryIdx; // index in Image A keypoints
    int trainIdx; // index in Image B keypoints
    int distance; // Hamming distance
    Match(int q = -1, int t = -1, int d = 256) : queryIdx(q), trainIdx(t), distance(d) {}
};

// 5. Homography 3x3 matrix (row-major: h[row * 3 + col])
struct Homography3x3 {
    double h[9];
    Homography3x3() {
        for (int i = 0; i < 9; ++i) h[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
};

// 6. Image wrappers for raw interleaved RGB / Grayscale
struct ImageRGB {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

struct ImageGray {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;
};

// 7. Timing metrics per stage (in milliseconds)
struct StageTimings {
    double stage1_preprocess_ms = 0.0;
    double stage2_keypoint_ms = 0.0;
    double stage3_descriptor_ms = 0.0;
    double stage4_matching_ms = 0.0;
    double stage5_ransac_ms = 0.0;
    double stage6_warp_blend_ms = 0.0;
    double total_pipeline_ms = 0.0;
};

// 8. High-resolution CPU Timer
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};
