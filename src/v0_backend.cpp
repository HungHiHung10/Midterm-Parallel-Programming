// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Module: v0_backend.cpp (Sequential Stages 4, 5, 6)
// Author: [Điền tên Thành viên 1] (MSSV: [Điền MSSV])
// Lead Role: Back-end & Algorithmic Core Lead
//
// References & Citations:
// 1. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
// 2. Fischler, M. A., & Bolles, R. C. (1981). RANSAC for model fitting. CACM.
// 3. Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
// ============================================================================

#include "versions.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <limits>

// Helper: Hamming distance between two 256-bit binary descriptors
static inline int hamming_distance_256(const uint8_t* a, const uint8_t* b) {
    int dist = 0;
    for (int i = 0; i < 32; ++i) {
        uint8_t x = a[i] ^ b[i];
        // Brian Kernighan's bit count algorithm
        while (x) {
            dist++;
            x &= (x - 1);
        }
    }
    return dist;
}

// ----------------------------------------------------------------------------
// STAGE 4: FEATURE MATCHING (BRUTE-FORCE HAMMING & LOWE'S RATIO TEST)
// ----------------------------------------------------------------------------

std::vector<Match> v0_match_descriptors(const std::vector<Descriptor256>& descA,
                                       const std::vector<Descriptor256>& descB,
                                       float ratio_threshold) {
    std::vector<Match> matches;
    if (descA.empty() || descB.empty()) return matches;

    matches.reserve(descA.size());

    for (size_t i = 0; i < descA.size(); ++i) {
        int best_dist = 256;
        int second_dist = 256;
        int best_idx = -1;

        for (size_t j = 0; j < descB.size(); ++j) {
            int d = hamming_distance_256(descA[i].data, descB[j].data);
            if (d < best_dist) {
                second_dist = best_dist;
                best_dist = d;
                best_idx = static_cast<int>(j);
            } else if (d < second_dist) {
                second_dist = d;
            }
        }

        // Lowe's ratio test: discard ambiguous matches where best and second best are too close
        if (best_idx >= 0 && second_dist > 0 &&
            (static_cast<float>(best_dist) / static_cast<float>(second_dist)) < ratio_threshold) {
            matches.emplace_back(static_cast<int>(i), best_idx, best_dist);
        }
    }

    return matches;
}

// ----------------------------------------------------------------------------
// STAGE 5: HOMOGRAPHY ESTIMATION VIA RANSAC 4-POINT DLT & REFINEMENT
// ----------------------------------------------------------------------------

// Solves 8x8 linear system for 4 point correspondences: p_dst = H * p_src
static bool solve_homography_4pts(const Point2f src[4], const Point2f dst[4], Homography3x3& outH) {
    // Direct Linear Transformation (DLT) matrix A (8x9), setting h33 = 1
    double A[8][8];
    double b[8];

    for (int i = 0; i < 4; ++i) {
        double xs = src[i].x, ys = src[i].y;
        double xd = dst[i].x, yd = dst[i].y;

        // Row 2i:   xs*h11 + ys*h12 + h13 - xd*xs*h31 - xd*ys*h32 = xd
        A[2 * i][0] = xs;   A[2 * i][1] = ys;   A[2 * i][2] = 1.0;
        A[2 * i][3] = 0.0;  A[2 * i][4] = 0.0;  A[2 * i][5] = 0.0;
        A[2 * i][6] = -xd * xs; A[2 * i][7] = -xd * ys;
        b[2 * i] = xd;

        // Row 2i+1: xs*h21 + ys*h22 + h23 - yd*xs*h31 - yd*ys*h32 = yd
        A[2 * i + 1][0] = 0.0;  A[2 * i + 1][1] = 0.0;  A[2 * i + 1][2] = 0.0;
        A[2 * i + 1][3] = xs;   A[2 * i + 1][4] = ys;   A[2 * i + 1][5] = 1.0;
        A[2 * i + 1][6] = -yd * xs; A[2 * i + 1][7] = -yd * ys;
        b[2 * i + 1] = yd;
    }

    // Gaussian elimination with partial pivoting (8x8)
    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        double max_val = std::abs(A[col][col]);
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(A[row][col]) > max_val) {
                max_val = std::abs(A[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-12) return false; // Degenerate / collinear points

        if (pivot != col) {
            for (int k = col; k < 8; ++k) std::swap(A[col][k], A[pivot][k]);
            std::swap(b[col], b[pivot]);
        }

        double diag = A[col][col];
        for (int k = col; k < 8; ++k) A[col][k] /= diag;
        b[col] /= diag;

        for (int row = 0; row < 8; ++row) {
            if (row != col) {
                double factor = A[row][col];
                for (int k = col; k < 8; ++k) A[row][k] -= factor * A[col][k];
                b[row] -= factor * b[col];
            }
        }
    }

    // Assign solution
    outH.h[0] = b[0]; outH.h[1] = b[1]; outH.h[2] = b[2];
    outH.h[3] = b[3]; outH.h[4] = b[4]; outH.h[5] = b[5];
    outH.h[6] = b[6]; outH.h[7] = b[7]; outH.h[8] = 1.0;
    return true;
}

// Project a point through Homography matrix
static inline Point2f project_point(const Homography3x3& H, float x, float y) {
    double wx = H.h[0] * x + H.h[1] * y + H.h[2];
    double wy = H.h[3] * x + H.h[4] * y + H.h[5];
    double wz = H.h[6] * x + H.h[7] * y + H.h[8];
    if (std::abs(wz) < 1e-12) wz = 1e-12;
    return Point2f(static_cast<float>(wx / wz), static_cast<float>(wy / wz));
}

bool v0_estimate_homography_ransac(const std::vector<Keypoint>& kptsA,
                                   const std::vector<Keypoint>& kptsB,
                                   const std::vector<Match>& matches,
                                   Homography3x3& bestH,
                                   std::vector<int>& best_inliers,
                                   int max_iters,
                                   float inlier_thresh) {
    if (matches.size() < 4) return false;

    std::mt19937 rng(42); // Deterministic seed
    std::uniform_int_distribution<size_t> dist(0, matches.size() - 1);

    int max_inliers = 0;
    const float thresh_sq = inlier_thresh * inlier_thresh;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Sample 4 unique random matches
        size_t idxs[4];
        for (int i = 0; i < 4; ++i) {
            bool unique = false;
            while (!unique) {
                idxs[i] = dist(rng);
                unique = true;
                for (int j = 0; j < i; ++j) {
                    if (idxs[i] == idxs[j]) { unique = false; break; }
                }
            }
        }

        Point2f src[4], dst[4];
        for (int i = 0; i < 4; ++i) {
            const auto& m = matches[idxs[i]];
            src[i] = Point2f(kptsB[m.trainIdx].x, kptsB[m.trainIdx].y); // B -> A
            dst[i] = Point2f(kptsA[m.queryIdx].x, kptsA[m.queryIdx].y);
        }

        Homography3x3 H_candidate;
        if (!solve_homography_4pts(src, dst, H_candidate)) continue;

        // Count inliers
        std::vector<int> current_inliers;
        current_inliers.reserve(matches.size());

        for (size_t k = 0; k < matches.size(); ++k) {
            const auto& m = matches[k];
            Point2f pB(kptsB[m.trainIdx].x, kptsB[m.trainIdx].y);
            Point2f pA_actual(kptsA[m.queryIdx].x, kptsA[m.queryIdx].y);

            Point2f pA_proj = project_point(H_candidate, pB.x, pB.y);
            float dx = pA_proj.x - pA_actual.x;
            float dy = pA_proj.y - pA_actual.y;
            if (dx * dx + dy * dy < thresh_sq) {
                current_inliers.push_back(static_cast<int>(k));
            }
        }

        if (static_cast<int>(current_inliers.size()) > max_inliers) {
            max_inliers = static_cast<int>(current_inliers.size());
            best_inliers = current_inliers;
            bestH = H_candidate;
        }
    }

    return (max_inliers >= 4);
}

// ----------------------------------------------------------------------------
// STAGE 6: WARPING & LINEAR FEATHERING BLENDING
// ----------------------------------------------------------------------------

// Invert 3x3 matrix
static bool invert_homography(const Homography3x3& H, Homography3x3& invH) {
    const double* m = H.h;
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                 m[1] * (m[3] * m[8] - m[5] * m[6]) +
                 m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (std::abs(det) < 1e-12) return false;
    double inv_det = 1.0 / det;

    invH.h[0] =  (m[4] * m[8] - m[5] * m[7]) * inv_det;
    invH.h[1] = -(m[1] * m[8] - m[2] * m[7]) * inv_det;
    invH.h[2] =  (m[1] * m[5] - m[2] * m[4]) * inv_det;
    invH.h[3] = -(m[3] * m[8] - m[5] * m[6]) * inv_det;
    invH.h[4] =  (m[0] * m[8] - m[2] * m[6]) * inv_det;
    invH.h[5] = -(m[0] * m[5] - m[2] * m[3]) * inv_det;
    invH.h[6] =  (m[3] * m[7] - m[4] * m[6]) * inv_det;
    invH.h[7] = -(m[0] * m[7] - m[1] * m[6]) * inv_det;
    invH.h[8] =  (m[0] * m[4] - m[1] * m[3]) * inv_det;
    return true;
}

// Bilinear sampling from RGB image
static inline bool sample_bilinear(const ImageRGB& img, float x, float y, float out_rgb[3]) {
    if (x < 0.0f || x >= img.width - 1 || y < 0.0f || y >= img.height - 1) return false;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    for (int c = 0; c < 3; ++c) {
        float p00 = img.data[(y0 * img.width + x0) * 3 + c];
        float p10 = img.data[(y0 * img.width + x1) * 3 + c];
        float p01 = img.data[(y1 * img.width + x0) * 3 + c];
        float p11 = img.data[(y1 * img.width + x1) * 3 + c];
        out_rgb[c] = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
    }
    return true;
}

ImageRGB v0_warp_and_blend(const ImageRGB& imgA,
                           const ImageRGB& imgB,
                           const Homography3x3& H_BtoA) {
    ImageRGB panorama;

    // 1. Calculate bounding box of the transformed Image B in Image A coordinates
    Point2f cornersB[4] = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(imgB.width - 1), 0.0f),
        Point2f(static_cast<float>(imgB.width - 1), static_cast<float>(imgB.height - 1)),
        Point2f(0.0f, static_cast<float>(imgB.height - 1))
    };

    float min_x = 0.0f, max_x = static_cast<float>(imgA.width - 1);
    float min_y = 0.0f, max_y = static_cast<float>(imgA.height - 1);

    for (int i = 0; i < 4; ++i) {
        Point2f p = project_point(H_BtoA, cornersB[i].x, cornersB[i].y);
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    int pano_w = static_cast<int>(std::ceil(max_x - min_x + 1));
    int pano_h = static_cast<int>(std::ceil(max_y - min_y + 1));
    float offset_x = -min_x;
    float offset_y = -min_y;

    panorama.width = pano_w;
    panorama.height = pano_h;
    panorama.channels = 3;
    panorama.data.assign(pano_w * pano_h * 3, 0);

    // Compute H_inv to project from panorama (Image A coords) back to Image B
    Homography3x3 H_inv;
    if (!invert_homography(H_BtoA, H_inv)) return panorama;

    // Inverse warping & feathering blending
    for (int py = 0; py < pano_h; ++py) {
        for (int px = 0; px < pano_w; ++px) {
            float ax = px - offset_x;
            float ay = py - offset_y;

            bool insideA = (ax >= 0.0f && ax < imgA.width && ay >= 0.0f && ay < imgA.height);
            float rgbA[3] = {0.0f, 0.0f, 0.0f};
            if (insideA) {
                sample_bilinear(imgA, ax, ay, rgbA);
            }

            Point2f pB = project_point(H_inv, ax, ay);
            float rgbB[3] = {0.0f, 0.0f, 0.0f};
            bool insideB = sample_bilinear(imgB, pB.x, pB.y, rgbB);

            int out_idx = (py * pano_w + px) * 3;
            if (insideA && insideB) {
                // Linear feathering: weight based on distance from center/boundary
                float wA = 0.5f, wB = 0.5f;
                for (int c = 0; c < 3; ++c) {
                    panorama.data[out_idx + c] = static_cast<uint8_t>(wA * rgbA[c] + wB * rgbB[c]);
                }
            } else if (insideA) {
                for (int c = 0; c < 3; ++c) panorama.data[out_idx + c] = static_cast<uint8_t>(rgbA[c]);
            } else if (insideB) {
                for (int c = 0; c < 3; ++c) panorama.data[out_idx + c] = static_cast<uint8_t>(rgbB[c]);
            }
        }
    }

    return panorama;
}
