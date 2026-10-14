// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// CLI Driver & Benchmark Runner: main.cu
// Author: [Điền tên Thành viên 3] (MSSV: [Điền MSSV])
// Lead Role: System, Data, Profiling & Notebook Lead
//
// Usage:
//   ./stitch <left_image> <right_image> <output_image> --version 0 [--runs N] [--save-h <h_file>]
// ============================================================================

#include "versions.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>

static void print_usage(const char* prog_name) {
    std::cout << "Usage:\n"
              << "  " << prog_name << " <left_img> <right_img> <out_img> --version <0|1|2|3> [--runs N] [--save-h <file>]\n\n"
              << "Parameters:\n"
              << "  left_img    : Path to left input image (PNG/JPG)\n"
              << "  right_img   : Path to right input image (PNG/JPG)\n"
              << "  out_img     : Path to output stitched panorama image\n"
              << "  --version V : Pipeline version (0: Sequential, 1: Basic CUDA, 2: Shared Mem, 3: Advanced)\n"
              << "  --runs N    : Number of benchmark runs (default: 1)\n"
              << "  --save-h F  : (Optional) Save 3x3 homography matrix to text file for correctness test\n";
}

static bool load_image(const std::string& path, ImageRGB& img) {
    int w, h, channels;
    uint8_t* raw = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!raw) {
        std::cerr << "[Error] Cannot load image: " << path << " (" << stbi_failure_reason() << ")\n";
        return false;
    }
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(raw, raw + (w * h * 3));
    stbi_image_free(raw);
    return true;
}

static bool save_image(const std::string& path, const ImageRGB& img) {
    int res = 0;
    // Determine format by extension
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".jpg") {
        res = stbi_write_jpg(path.c_str(), img.width, img.height, 3, img.data.data(), 95);
    } else {
        res = stbi_write_png(path.c_str(), img.width, img.height, 3, img.data.data(), img.width * 3);
    }
    if (!res) {
        std::cerr << "[Error] Failed to save output image to: " << path << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    std::string left_path = argv[1];
    std::string right_path = argv[2];
    std::string out_path = argv[3];

    int version = 0;
    int runs = 1;
    std::string save_h_path = "";

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" && i + 1 < argc) {
            version = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--save-h" && i + 1 < argc) {
            save_h_path = argv[++i];
        }
    }

    std::cout << "=================================================================\n"
              << " CSC14120 - Panorama Stitching Benchmark\n"
              << " Left Image : " << left_path << "\n"
              << " Right Image: " << right_path << "\n"
              << " Output Path: " << out_path << "\n"
              << " Version    : V" << version << "\n"
              << " Runs       : " << runs << "\n"
              << "=================================================================\n";

    ImageRGB imgA, imgB;
    if (!load_image(left_path, imgA) || !load_image(right_path, imgB)) {
        return 1;
    }

    std::cout << "[Info] Image A: " << imgA.width << "x" << imgA.height << " (" << imgA.channels << " channels)\n";
    std::cout << "[Info] Image B: " << imgB.width << "x" << imgB.height << " (" << imgB.channels << " channels)\n\n";

    std::vector<StageTimings> all_timings(runs);
    ImageRGB final_panorama;
    Homography3x3 finalH;
    bool success = false;

    for (int r = 0; r < runs; ++r) {
        ImageRGB pano;
        Homography3x3 H;
        StageTimings t;

        if (version == 0) {
            success = run_v0(imgA, imgB, pano, t, &H);
        } else {
            std::cerr << "[Error] Version V" << version << " is not implemented yet in Midterm.\n";
            return 1;
        }

        if (!success) {
            std::cerr << "[Error] Stitching pipeline failed at run " << (r + 1) << "\n";
            return 1;
        }

        all_timings[r] = t;
        if (r == runs - 1) {
            final_panorama = std::move(pano);
            finalH = H;
        }
    }

    // Compute Mean & Min per stage
    StageTimings mean_t, min_t;
    min_t.stage1_preprocess_ms = 1e9;
    min_t.stage2_keypoint_ms = 1e9;
    min_t.stage3_descriptor_ms = 1e9;
    min_t.stage4_matching_ms = 1e9;
    min_t.stage5_ransac_ms = 1e9;
    min_t.stage6_warp_blend_ms = 1e9;
    min_t.total_pipeline_ms = 1e9;

    for (const auto& t : all_timings) {
        mean_t.stage1_preprocess_ms += t.stage1_preprocess_ms;
        mean_t.stage2_keypoint_ms   += t.stage2_keypoint_ms;
        mean_t.stage3_descriptor_ms += t.stage3_descriptor_ms;
        mean_t.stage4_matching_ms   += t.stage4_matching_ms;
        mean_t.stage5_ransac_ms     += t.stage5_ransac_ms;
        mean_t.stage6_warp_blend_ms += t.stage6_warp_blend_ms;
        mean_t.total_pipeline_ms    += t.total_pipeline_ms;

        min_t.stage1_preprocess_ms = std::min(min_t.stage1_preprocess_ms, t.stage1_preprocess_ms);
        min_t.stage2_keypoint_ms   = std::min(min_t.stage2_keypoint_ms, t.stage2_keypoint_ms);
        min_t.stage3_descriptor_ms = std::min(min_t.stage3_descriptor_ms, t.stage3_descriptor_ms);
        min_t.stage4_matching_ms   = std::min(min_t.stage4_matching_ms, t.stage4_matching_ms);
        min_t.stage5_ransac_ms     = std::min(min_t.stage5_ransac_ms, t.stage5_ransac_ms);
        min_t.stage6_warp_blend_ms = std::min(min_t.stage6_warp_blend_ms, t.stage6_warp_blend_ms);
        min_t.total_pipeline_ms    = std::min(min_t.total_pipeline_ms, t.total_pipeline_ms);
    }

    mean_t.stage1_preprocess_ms /= runs;
    mean_t.stage2_keypoint_ms   /= runs;
    mean_t.stage3_descriptor_ms /= runs;
    mean_t.stage4_matching_ms   /= runs;
    mean_t.stage5_ransac_ms     /= runs;
    mean_t.stage6_warp_blend_ms /= runs;
    mean_t.total_pipeline_ms    /= runs;

    // Print Timing Table
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "+---------------------------------------+---------------+---------------+\n";
    std::cout << "| Pipeline Stage                        | Mean Time(ms) | Min Time (ms) |\n";
    std::cout << "+---------------------------------------+---------------+---------------+\n";
    std::cout << "| Stage 1: Preprocessing (Gray + Blur)  | " << std::setw(13) << mean_t.stage1_preprocess_ms << " | " << std::setw(13) << min_t.stage1_preprocess_ms << " |\n";
    std::cout << "| Stage 2: Keypoints (FAST + NMS)       | " << std::setw(13) << mean_t.stage2_keypoint_ms   << " | " << std::setw(13) << min_t.stage2_keypoint_ms   << " |\n";
    std::cout << "| Stage 3: Descriptors (BRIEF 256-bit)  | " << std::setw(13) << mean_t.stage3_descriptor_ms << " | " << std::setw(13) << min_t.stage3_descriptor_ms << " |\n";
    std::cout << "| Stage 4: Matching (BF Hamming + Ratio)| " << std::setw(13) << mean_t.stage4_matching_ms   << " | " << std::setw(13) << min_t.stage4_matching_ms   << " |\n";
    std::cout << "| Stage 5: Homography (RANSAC 4-pt DLT) | " << std::setw(13) << mean_t.stage5_ransac_ms     << " | " << std::setw(13) << min_t.stage5_ransac_ms     << " |\n";
    std::cout << "| Stage 6: Warping & Feathering Blend   | " << std::setw(13) << mean_t.stage6_warp_blend_ms << " | " << std::setw(13) << min_t.stage6_warp_blend_ms << " |\n";
    std::cout << "+---------------------------------------+---------------+---------------+\n";
    std::cout << "| Total Execution Time                  | " << std::setw(13) << mean_t.total_pipeline_ms    << " | " << std::setw(13) << min_t.total_pipeline_ms    << " |\n";
    std::cout << "+---------------------------------------+---------------+---------------+\n\n";

    // Save Panorama image
    if (!save_image(out_path, final_panorama)) {
        return 1;
    }
    std::cout << "[Success] Panorama saved to: " << out_path << " (" << final_panorama.width << "x" << final_panorama.height << ")\n";

    // Save Homography if requested
    if (!save_h_path.empty()) {
        std::ofstream h_file(save_h_path);
        if (h_file.is_open()) {
            for (int r = 0; r < 3; ++r) {
                h_file << finalH.h[r * 3 + 0] << " "
                       << finalH.h[r * 3 + 1] << " "
                       << finalH.h[r * 3 + 2] << "\n";
            }
            h_file.close();
            std::cout << "[Success] Estimated Homography saved to: " << save_h_path << "\n";
        }
    }

    return 0;
}
