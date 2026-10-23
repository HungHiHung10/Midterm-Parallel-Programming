// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// Module: v0_backend.cpp (Sequential Stages 4, 5, 6)
// Author: Tran Nguyen Minh Quan - 23120342 (Member 1)
// Lead Role: Back-end & Algorithmic Core Lead
//
// References & Citations:
// 1. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
// 2. Fischler, M. A., & Bolles, R. C. (1981). RANSAC for model fitting. CACM.
// 3. Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
// ============================================================================

#include "versions.h"
#include "logging.h"
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
    std::vector<int> best_b(descA.size(), -1);
    std::vector<int> best_distance(descA.size(), 256);
    std::vector<int> second_distance(descA.size(), 256);
    for (size_t i = 0; i < descA.size(); ++i) {
        for (size_t j = 0; j < descB.size(); ++j) {
            int d = hamming_distance_256(descA[i].data, descB[j].data);
            if (d < best_distance[i]) {
                second_distance[i] = best_distance[i];
                best_distance[i] = d;
                best_b[i] = static_cast<int>(j);
            } else if (d < second_distance[i]) {
                second_distance[i] = d;
            }
        }
    }

    // Reverse nearest-neighbour lookup. Keeping only mutual pairs rejects many
    // accidental matches on repeated or weakly textured regions.
    std::vector<int> reverse_best_a(descB.size(), -1);
    std::vector<int> reverse_best_distance(descB.size(), 256);
    for (size_t j = 0; j < descB.size(); ++j) {
        for (size_t i = 0; i < descA.size(); ++i) {
            int d = hamming_distance_256(descB[j].data, descA[i].data);
            if (d < reverse_best_distance[j]) {
                reverse_best_distance[j] = d;
                reverse_best_a[j] = static_cast<int>(i);
            }
        }
    }

    for (size_t i = 0; i < descA.size(); ++i) {
        int j = best_b[i];
        bool passes_ratio = j >= 0 && second_distance[i] > 0 &&
            (static_cast<float>(best_distance[i]) /
             static_cast<float>(second_distance[i])) < ratio_threshold;
        if (passes_ratio && reverse_best_a[j] == static_cast<int>(i)) {
            matches.emplace_back(static_cast<int>(i), j, best_distance[i]);
        }
    }

    return matches;
}

// ----------------------------------------------------------------------------
// STAGE 5: HOMOGRAPHY ESTIMATION VIA RANSAC 4-POINT DLT & REFINEMENT
// ----------------------------------------------------------------------------

static bool solve_linear_8(double A[8][8], double b[8], double x[8]) {
    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        double max_val = std::abs(A[col][col]);
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(A[row][col]) > max_val) {
                max_val = std::abs(A[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-12) return false;

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
    for (int i = 0; i < 8; ++i) x[i] = b[i];
    return true;
}

static void multiply_3x3(const double a[9], const double b[9], double out[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[3*r+c] = 0.0;
            for (int k = 0; k < 3; ++k) out[3*r+c] += a[3*r+k] * b[3*k+c];
        }
    }
}

// Normalized least-squares DLT with h33 fixed to one. Normalization keeps the
// 8x8 system well-conditioned at HD/4K pixel coordinates and also supports
// the final refit from all RANSAC inliers.
static bool solve_homography_points(const std::vector<Point2f>& src,
                                    const std::vector<Point2f>& dst,
                                    Homography3x3& outH) {
    if (src.size() != dst.size() || src.size() < 4) return false;
    double csx=0, csy=0, cdx=0, cdy=0;
    for (size_t i=0; i<src.size(); ++i) {
        csx += src[i].x; csy += src[i].y;
        cdx += dst[i].x; cdy += dst[i].y;
    }
    double n = static_cast<double>(src.size());
    csx/=n; csy/=n; cdx/=n; cdy/=n;
    double src_mean=0, dst_mean=0;
    for (size_t i=0; i<src.size(); ++i) {
        src_mean += std::hypot(src[i].x-csx, src[i].y-csy);
        dst_mean += std::hypot(dst[i].x-cdx, dst[i].y-cdy);
    }
    src_mean/=n; dst_mean/=n;
    if (src_mean < 1e-9 || dst_mean < 1e-9) return false;
    const double ss = std::sqrt(2.0)/src_mean;
    const double sd = std::sqrt(2.0)/dst_mean;

    double ata[8][8] = {};
    double atb[8] = {};
    for (size_t i=0; i<src.size(); ++i) {
        double xs=ss*(src[i].x-csx), ys=ss*(src[i].y-csy);
        double xd=sd*(dst[i].x-cdx), yd=sd*(dst[i].y-cdy);
        double rows[2][8] = {
            {xs,ys,1, 0,0,0, -xd*xs,-xd*ys},
            {0,0,0, xs,ys,1, -yd*xs,-yd*ys}
        };
        double rhs[2] = {xd, yd};
        for (int q=0; q<2; ++q) {
            for (int r=0; r<8; ++r) {
                atb[r] += rows[q][r]*rhs[q];
                for (int c=0; c<8; ++c) ata[r][c] += rows[q][r]*rows[q][c];
            }
        }
    }
    double x[8];
    if (!solve_linear_8(ata, atb, x)) return false;
    double hn[9] = {x[0],x[1],x[2], x[3],x[4],x[5], x[6],x[7],1.0};
    double ts[9] = {ss,0,-ss*csx, 0,ss,-ss*csy, 0,0,1};
    double td_inv[9] = {1.0/sd,0,cdx, 0,1.0/sd,cdy, 0,0,1};
    double temp[9], h[9];
    multiply_3x3(hn, ts, temp);
    multiply_3x3(td_inv, temp, h);
    if (!std::isfinite(h[8]) || std::abs(h[8]) < 1e-12) return false;
    for (int i=0; i<9; ++i) outH.h[i] = h[i]/h[8];
    return true;
}

static bool invert_homography(const Homography3x3& H, Homography3x3& invH);

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

    auto collect_inliers = [&](const Homography3x3& H) {
        std::vector<int> result;
        Homography3x3 H_inv;
        if (!invert_homography(H, H_inv)) return result;
        result.reserve(matches.size());
        for (size_t k = 0; k < matches.size(); ++k) {
            const auto& m = matches[k];
            Point2f pB(kptsB[m.trainIdx].x, kptsB[m.trainIdx].y);
            Point2f pA(kptsA[m.queryIdx].x, kptsA[m.queryIdx].y);
            Point2f pA_proj = project_point(H, pB.x, pB.y);
            Point2f pB_proj = project_point(H_inv, pA.x, pA.y);
            double fdx=pA_proj.x-pA.x, fdy=pA_proj.y-pA.y;
            double bdx=pB_proj.x-pB.x, bdy=pB_proj.y-pB.y;
            double symmetric_error = 0.5*(fdx*fdx+fdy*fdy+bdx*bdx+bdy*bdy);
            if (std::isfinite(symmetric_error) && symmetric_error < thresh_sq)
                result.push_back(static_cast<int>(k));
        }
        return result;
    };

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

        std::vector<Point2f> src(4), dst(4);
        for (int i = 0; i < 4; ++i) {
            const auto& m = matches[idxs[i]];
            src[i] = Point2f(kptsB[m.trainIdx].x, kptsB[m.trainIdx].y); // B -> A
            dst[i] = Point2f(kptsA[m.queryIdx].x, kptsA[m.queryIdx].y);
        }

        Homography3x3 H_candidate;
        if (!solve_homography_points(src, dst, H_candidate)) continue;
        std::vector<int> current_inliers = collect_inliers(H_candidate);

        if (static_cast<int>(current_inliers.size()) > max_inliers) {
            max_inliers = static_cast<int>(current_inliers.size());
            best_inliers = current_inliers;
            bestH = H_candidate;
        }
    }
    if (max_inliers < 4) return false;

    // Refit with every inlier instead of retaining the four-point hypothesis.
    // A second collection updates the final consensus after least squares.
    for (int pass=0; pass<2; ++pass) {
        std::vector<Point2f> src, dst;
        src.reserve(best_inliers.size()); dst.reserve(best_inliers.size());
        for (int index : best_inliers) {
            const auto& m = matches[index];
            src.emplace_back(kptsB[m.trainIdx].x, kptsB[m.trainIdx].y);
            dst.emplace_back(kptsA[m.queryIdx].x, kptsA[m.queryIdx].y);
        }
        Homography3x3 refined;
        if (!solve_homography_points(src, dst, refined)) break;
        std::vector<int> refined_inliers = collect_inliers(refined);
        if (refined_inliers.size() < 4) break;
        bestH = refined;
        best_inliers.swap(refined_inliers);
    }
    return best_inliers.size() >= 4;
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

    // Estimate a conservative per-channel affine exposure transform A ~= gB+b
    // from non-dark, non-saturated overlap pixels.  It is accepted only when
    // it improves overlap MAE by at least 10%, so parallax cannot accidentally
    // trigger a destructive colour correction.
    double sx[3] = {}, sy[3] = {}, sxx[3] = {}, sxy[3] = {};
    size_t exposure_samples = 0;
    for (int py = 0; py < pano_h; py += 2) {
        for (int px = 0; px < pano_w; px += 2) {
            float ax = px - offset_x, ay = py - offset_y;
            float a[3], b[3];
            Point2f pB = project_point(H_inv, ax, ay);
            if (!sample_bilinear(imgA, ax, ay, a) ||
                !sample_bilinear(imgB, pB.x, pB.y, b)) continue;
            float mean_a=(a[0]+a[1]+a[2])/3.0f, mean_b=(b[0]+b[1]+b[2])/3.0f;
            float max_a=std::max(a[0],std::max(a[1],a[2]));
            float max_b=std::max(b[0],std::max(b[1],b[2]));
            if (mean_a <= 20 || mean_b <= 20 || max_a >= 245 || max_b >= 245) continue;
            ++exposure_samples;
            for (int c=0;c<3;++c) {
                sx[c]+=b[c]; sy[c]+=a[c]; sxx[c]+=b[c]*b[c]; sxy[c]+=b[c]*a[c];
            }
        }
    }
    double gain[3]={1,1,1}, bias[3]={0,0,0};
    if (exposure_samples >= 1000) {
        double n=static_cast<double>(exposure_samples);
        for (int c=0;c<3;++c) {
            double denominator=n*sxx[c]-sx[c]*sx[c];
            if (std::abs(denominator)>1e-9) {
                gain[c]=std::max(0.75,std::min(1.33,(n*sxy[c]-sx[c]*sy[c])/denominator));
                bias[c]=std::max(-64.0,std::min(64.0,(sy[c]-gain[c]*sx[c])/n));
            }
        }
    }
    double raw_error=0, corrected_error=0;
    size_t error_samples=0;
    for (int py = 0; py < pano_h; py += 2) {
        for (int px = 0; px < pano_w; px += 2) {
            float ax=px-offset_x, ay=py-offset_y, a[3], b[3];
            Point2f pB=project_point(H_inv,ax,ay);
            if (!sample_bilinear(imgA,ax,ay,a) || !sample_bilinear(imgB,pB.x,pB.y,b)) continue;
            for (int c=0;c<3;++c) {
                raw_error+=std::abs(a[c]-b[c]);
                corrected_error+=std::abs(a[c]-std::max(0.0,std::min(255.0,gain[c]*b[c]+bias[c])));
            }
            error_samples+=3;
        }
    }
    bool compensate_exposure=exposure_samples>=1000 && error_samples>0 &&
        corrected_error < raw_error*0.90;
    if (!compensate_exposure) {
        for (int c=0;c<3;++c) { gain[c]=1.0; bias[c]=0.0; }
        corrected_error=raw_error;
    }
    double overlap_mae=error_samples ? corrected_error/error_samples : 0.0;

    auto adjust_b = [&](float rgb[3]) {
        for (int c=0;c<3;++c)
            rgb[c]=static_cast<float>(std::max(0.0,std::min(255.0,gain[c]*rgb[c]+bias[c])));
    };

    // A high residual after exposure compensation indicates parallax or moving
    // content.  In that case find a minimum-error top-to-bottom seam, then use
    // only a narrow feather band. This keeps foliage sharper than blending the
    // complete overlap. Parent directions need one signed byte per canvas pixel.
    const bool use_seam = error_samples>3000 && overlap_mae>18.0;
    log_message(LogLevel::Process) << "Blend: exposure="
        << (compensate_exposure ? "affine" : "unchanged")
        << ", overlap MAE=" << overlap_mae
        << ", seam=" << (use_seam ? "optimal" : "feather") << "\n";
    std::vector<int> seam_x(pano_h,-1);
    if (use_seam) {
        const double inf=std::numeric_limits<double>::infinity();
        std::vector<double> previous(pano_w,inf), current(pano_w,inf);
        std::vector<int8_t> parent(size_t(pano_w)*pano_h,0);
        int first_y=-1,last_y=-1;
        for (int py=0;py<pano_h;++py) {
            bool any=false;
            std::fill(current.begin(),current.end(),inf);
            for (int px=0;px<pano_w;++px) {
                float ax=px-offset_x,ay=py-offset_y,a[3],b[3];
                Point2f pB=project_point(H_inv,ax,ay);
                if (!sample_bilinear(imgA,ax,ay,a) || !sample_bilinear(imgB,pB.x,pB.y,b)) continue;
                adjust_b(b); any=true;
                double pixel_cost=(std::abs(a[0]-b[0])+std::abs(a[1]-b[1])+std::abs(a[2]-b[2]))/3.0+1.0;
                if (first_y<0) current[px]=pixel_cost;
                else {
                    double best=inf; int direction=0;
                    for (int d=-2;d<=2;++d) {
                        int q=px+d;
                        if (q>=0 && q<pano_w && previous[q]<best) { best=previous[q]; direction=d; }
                    }
                    if (std::isfinite(best)) {
                        current[px]=best+pixel_cost;
                        parent[size_t(py)*pano_w+px]=static_cast<int8_t>(direction);
                    }
                }
            }
            if (any) {
                if (first_y<0) first_y=py;
                last_y=py;
                previous.swap(current);
            }
        }
        if (last_y>=first_y) {
            int x=static_cast<int>(std::min_element(previous.begin(),previous.end())-previous.begin());
            for (int py=last_y;py>=first_y;--py) {
                seam_x[py]=x;
                x+=parent[size_t(py)*pano_w+x];
                x=std::max(0,std::min(pano_w-1,x));
            }
        }
    }

    // Inverse warping & feathering blending
    for (int py = 0; py < pano_h; ++py) {
        for (int px = 0; px < pano_w; ++px) {
            float ax = px - offset_x;
            float ay = py - offset_y;

            float rgbA[3] = {0.0f, 0.0f, 0.0f};
            bool insideA = sample_bilinear(imgA, ax, ay, rgbA);

            Point2f pB = project_point(H_inv, ax, ay);
            float rgbB[3] = {0.0f, 0.0f, 0.0f};
            bool insideB = sample_bilinear(imgB, pB.x, pB.y, rgbB);
            if (insideB) adjust_b(rgbB);

            int out_idx = (py * pano_w + px) * 3;
            if (insideA && insideB) {
                // Linear feathering: pixels near an image boundary receive a
                // lower weight. Coordinates are evaluated in each image's own
                // frame, so this also works for a projectively warped image B.
                float wA, wB;
                if (use_seam && seam_x[py]>=0) {
                    const float band=24.0f;
                    float alpha=std::max(0.0f,std::min(1.0f,
                        (seam_x[py]+band-px)/(2.0f*band)));
                    wA=alpha; wB=1.0f-alpha;
                } else {
                    wA = std::max(1.0f, std::min(std::min(ax, imgA.width-1.0f-ax),
                                                 std::min(ay, imgA.height-1.0f-ay)));
                    wB = std::max(1.0f, std::min(std::min(pB.x, imgB.width-1.0f-pB.x),
                                                 std::min(pB.y, imgB.height-1.0f-pB.y)));
                }
                float inv_sum = 1.0f / (wA + wB);
                for (int c = 0; c < 3; ++c) {
                    panorama.data[out_idx + c] = static_cast<uint8_t>(
                        (wA * rgbA[c] + wB * rgbB[c]) * inv_sum);
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
