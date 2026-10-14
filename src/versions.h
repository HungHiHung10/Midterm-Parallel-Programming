#pragma once
#include "common.h"

// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Version Interfaces & Stage Function Prototypes
// Author: Group XX (Lead: Member 3)
// ============================================================================

// ----------------------------------------------------------------------------
// Front-end Modules: Stages 1, 2, 3 (Lead: Member 2)
// ----------------------------------------------------------------------------

// Stage 1: Preprocessing
ImageGray v0_rgb_to_gray(const ImageRGB& img);
ImageGray v0_gaussian_blur(const ImageGray& gray, float sigma = 1.2f);

// Stage 2: Keypoint Detection (FAST / Harris + NMS + Top-K)
std::vector<Keypoint> v0_detect_keypoints(const ImageGray& gray, int max_keypoints = 1500);

// Stage 3: Feature Description (BRIEF 256-bit binary descriptor)
std::vector<Descriptor256> v0_compute_descriptors(const ImageGray& gray, const std::vector<Keypoint>& keypoints);


// ----------------------------------------------------------------------------
// Back-end Modules: Stages 4, 5, 6 (Lead: Member 1)
// ----------------------------------------------------------------------------

// Stage 4: Feature Matching (Brute-Force Hamming + Lowe's ratio test)
std::vector<Match> v0_match_descriptors(const std::vector<Descriptor256>& descA,
                                       const std::vector<Descriptor256>& descB,
                                       float ratio_threshold = 0.75f);

// Stage 5: Homography Estimation via RANSAC 4-point DLT & inlier refinement
bool v0_estimate_homography_ransac(const std::vector<Keypoint>& kptsA,
                                   const std::vector<Keypoint>& kptsB,
                                   const std::vector<Match>& matches,
                                   Homography3x3& bestH,
                                   std::vector<int>& inlier_indices,
                                   int max_iters = 1500,
                                   float inlier_thresh = 3.0f);

// Stage 6: Warping (Inverse Warping with Bilinear Interpolation) & Linear Feathering Blending
ImageRGB v0_warp_and_blend(const ImageRGB& imgA,
                           const ImageRGB& imgB,
                           const Homography3x3& H);


// ----------------------------------------------------------------------------
// Pipeline Execution Entrypoints (Lead: Member 3)
// ----------------------------------------------------------------------------

// V0: Sequential Baseline Execution
bool run_v0(const ImageRGB& imgA,
            const ImageRGB& imgB,
            ImageRGB& panorama,
            StageTimings& timings,
            Homography3x3* outH = nullptr);

// Future Version Prototypes (Midterm Plan / Final)
// bool run_v1(const ImageRGB& imgA, const ImageRGB& imgB, ImageRGB& panorama, StageTimings& timings);
// bool run_v2(const ImageRGB& imgA, const ImageRGB& imgB, ImageRGB& panorama, StageTimings& timings);
// bool run_v3(const ImageRGB& imgA, const ImageRGB& imgB, ImageRGB& panorama, StageTimings& timings);
