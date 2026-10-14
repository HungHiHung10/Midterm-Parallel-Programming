// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Module: v0_stitching.cpp (Sequential Pipeline Orchestrator)
// Author: [Điền tên Thành viên 3] (MSSV: [Điền MSSV])
// Lead Role: System, Data, Profiling & Notebook Lead
//
// Description:
// Integrates Front-end (Stages 1-3) and Back-end (Stages 4-6) into the V0 pipeline
// with isolated, accurate per-stage CPU timing measurements.
// ============================================================================

#include "versions.h"
#include <iostream>

bool run_v0(const ImageRGB& imgA,
            const ImageRGB& imgB,
            ImageRGB& panorama,
            StageTimings& timings,
            Homography3x3* outH) {
    CpuTimer timer;
    CpuTimer stage_timer;

    timer.start();

    // ------------------------------------------------------------------------
    // Stage 1: Preprocessing (RGB to Gray + Gaussian Blur)
    // ------------------------------------------------------------------------
    stage_timer.start();
    ImageGray grayA = v0_rgb_to_gray(imgA);
    ImageGray blurA = v0_gaussian_blur(grayA);

    ImageGray grayB = v0_rgb_to_gray(imgB);
    ImageGray blurB = v0_gaussian_blur(grayB);
    timings.stage1_preprocess_ms = stage_timer.stop();

    // ------------------------------------------------------------------------
    // Stage 2: Keypoint Detection (FAST / Harris + NMS + Top-K)
    // ------------------------------------------------------------------------
    stage_timer.start();
    std::vector<Keypoint> kptsA = v0_detect_keypoints(blurA, 1500);
    std::vector<Keypoint> kptsB = v0_detect_keypoints(blurB, 1500);
    timings.stage2_keypoint_ms = stage_timer.stop();

    // ------------------------------------------------------------------------
    // Stage 3: Feature Description (BRIEF 256-bit Binary Descriptor)
    // ------------------------------------------------------------------------
    stage_timer.start();
    std::vector<Descriptor256> descA = v0_compute_descriptors(blurA, kptsA);
    std::vector<Descriptor256> descB = v0_compute_descriptors(blurB, kptsB);
    timings.stage3_descriptor_ms = stage_timer.stop();

    // ------------------------------------------------------------------------
    // Stage 4: Feature Matching (Hamming Distance + Lowe's Ratio Test)
    // ------------------------------------------------------------------------
    stage_timer.start();
    std::vector<Match> matches = v0_match_descriptors(descA, descB, 0.75f);
    timings.stage4_matching_ms = stage_timer.stop();

    if (matches.size() < 4) {
        std::cerr << "[V0 Error] Insufficient feature matches found: " << matches.size() << " (need >= 4)\n";
        return false;
    }

    // ------------------------------------------------------------------------
    // Stage 5: Homography Estimation (RANSAC 4-point DLT)
    // ------------------------------------------------------------------------
    stage_timer.start();
    Homography3x3 H_BtoA;
    std::vector<int> inliers;
    bool ransac_ok = v0_estimate_homography_ransac(kptsA, kptsB, matches, H_BtoA, inliers, 1500, 3.0f);
    timings.stage5_ransac_ms = stage_timer.stop();

    if (!ransac_ok || inliers.size() < 4) {
        std::cerr << "[V0 Error] RANSAC failed to estimate valid homography matrix.\n";
        return false;
    }

    if (outH != nullptr) {
        *outH = H_BtoA;
    }

    // ------------------------------------------------------------------------
    // Stage 6: Warping & Blending (Inverse Warping + Bilinear + Feathering)
    // ------------------------------------------------------------------------
    stage_timer.start();
    panorama = v0_warp_and_blend(imgA, imgB, H_BtoA);
    timings.stage6_warp_blend_ms = stage_timer.stop();

    timings.total_pipeline_ms = timer.stop();
    return true;
}
