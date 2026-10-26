# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original dataset preparation; Pillow/NumPy APIs.
"""Run from the project root: python tools/prepare_dataset.py."""
from console import log
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image, ImageOps, ImageDraw

SITES = ['site1_indoor', 'site2_outdoor', 'site3_custom', 'site4_challenging']
RAW_STEMS = {
    'site1_indoor': 'indoor',
    'site2_outdoor': 'outdoor',
    'site3_custom': 'custom',
    'site4_challenging': 'challenging',
}
SIZES = {'small': (1280, 720), 'medium': (1920, 1080), 'large': (3840, 2160)}

def prepare(root=ROOT):
    from pillow_heif import register_heif_opener
    register_heif_opener()
    records = []
    for site in SITES:
        raw_dir = root / 'dataset/raw'
        stem = RAW_STEMS[site]
        sources = {}
        for side in ['left', 'right']:
            files = sorted(raw_dir.glob(f'{stem}_{side}.*'))
            if len(files) != 1:
                raise ValueError(
                    f'{site}/{side}: expected exactly one raw file named '
                    f'{stem}_{side}.<extension>, found {len(files)}')
            sources[side] = files[0]
        for side, source in sources.items():
            with Image.open(source) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
            w, h = image.size
            # Integer 16:9 crop, centered; identical rule on both images.
            unit = min(w // 16, h // 9)
            cw, ch = unit * 16, unit * 9
            box = ((w-cw)//2, (h-ch)//2, (w-cw)//2+cw, (h-ch)//2+ch)
            cropped = image.crop(box)
            for resolution, size in SIZES.items():
                target = root / f'dataset/prepared/{resolution}/{site}_{side}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                cropped.resize(size, Image.Resampling.LANCZOS).save(target)
                records.append(dict(site=site, side=side, resolution=resolution,
                    source=str(source.relative_to(root)).replace('\\', '/'),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    original_size=[w, h], crop_box=list(box), size=list(size),
                    upscaled=size[0] > cw or size[1] > ch,
                    output=str(target.relative_to(root)).replace('\\', '/')))
    manifest = root / 'dataset/prepared/manifest.json'
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    log('SUCCESS', f'Prepared {len(records)} images; manifest: {manifest}')
    return records

import math
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

def synthetic(root=ROOT):
    """
    Đọc ảnh Ground Truth sẵn có (synthetic_GT.HEIC), tự động tính toán 
    và cắt thành 2 nửa Left - Right với độ đè (overlap) 40%.
    """
    target = root / 'dataset/synthetic'
    gt_path = target / 'synthetic_GT.HEIC'
    
    if not gt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh Ground Truth tại: {gt_path}")
        
    # 1. Mở file ảnh HEIC và chuyển sang định dạng RGB
    scene = Image.open(gt_path).convert('RGB')
    SCENE_W, SCENE_H = scene.size
    
    # 2. Tính toán CROP_W và OFFSET dựa trên tỷ lệ đè 40%
    # Quy tắc: SCENE_W = CROP_W + OFFSET 
    # Mối liên hệ với Overlap: OFFSET = CROP_W * (1 - OVERLAP_RATIO)
    # => SCENE_W = CROP_W * (2 - OVERLAP_RATIO) = CROP_W * 1.6
    OVERLAP_RATIO = 0.40
    CROP_W = int(SCENE_W / (2.0 - OVERLAP_RATIO))
    CROP_H = SCENE_H
    OFFSET = SCENE_W - CROP_W
    
    # 3. Crop thành 2 nửa Left (0 -> CROP_W) và Right (OFFSET -> SCENE_W)
    left_crop = scene.crop((0, 0, CROP_W, CROP_H))
    right_crop = scene.crop((OFFSET, 0, SCENE_W, CROP_H))
    
    # 4. Lưu ảnh crop và chuyển đổi ảnh GT sang PNG để đồng bộ với pipeline show_correctness
    left_crop.save(target / 'synthetic_left.png')
    right_crop.save(target / 'synthetic_right.png')
    scene.save(target / 'synthetic_scene.png')
    
    # 5. Lưu ma trận Homography tịnh tiến chuẩn (OFFSET theo chiều X)
    np.savetxt(target / 'H_ground_truth.txt', [[1, 0, OFFSET], [0, 1, 0], [0, 0, 1]], fmt='%.17g')
    
    # 6. Ghi thông tin nguồn dữ liệu vào provenance.json
    (target / 'provenance.json').write_text(json.dumps(dict(
        kind='local_heic_translation_fixture',
        source_file=gt_path.name,
        direction='B_right_to_A_left',
        official_course_pair=False,
        crop_width=CROP_W,
        crop_height=CROP_H,
        offset_px=OFFSET,
        overlap_ratio=OVERLAP_RATIO,
        description=f'Two crops generated from {gt_path.name} with {int(OVERLAP_RATIO*100)}% overlap (offset: {OFFSET}px).'
    ), indent=2), encoding='utf-8')
    
    log('SUCCESS', f'Cropped {gt_path.name} into left/right pair (Crop size: {CROP_W}x{CROP_H}, Offset: {OFFSET}px)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    if not args.synthetic_only:
        prepare()
    synthetic()
