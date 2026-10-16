# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original dataset preparation; Pillow/NumPy APIs.
"""Run from the project root: python tools/prepare_dataset.py."""
from console import log
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools'))  # Optional project-local HEIC dependency.
import numpy as np
from PIL import Image, ImageOps, ImageDraw

SITES = ['site1_indoor', 'site2_outdoor', 'site3_custom', 'site4_challenging']
SIZES = {'small': (1280, 720), 'medium': (1920, 1080), 'large': (3840, 2160)}

def prepare(root=ROOT):
    from pillow_heif import register_heif_opener
    register_heif_opener()
    records = []
    for site in SITES:
        files = sorted(p for p in (root / 'dataset/raw' / site).iterdir()
                       if p.is_file() and not p.name.startswith('.'))
        if len(files) != 2:
            raise ValueError(f'{site}: expected exactly 2 raw images, found {len(files)}')
        for side, source in zip(['left', 'right'], files):
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

def synthetic(root=ROOT):
    """Local smoke-test fixture, NOT the official course synthetic pair.
    Two crops of the same procedural scene; H maps right B to left A.
    Exact integer translation exercises all six pipeline stages reproducibly.
    """
    rng = np.random.default_rng(14120)
    scene = Image.new('RGB', (1040, 480), (185, 190, 195))
    draw = ImageDraw.Draw(scene)
    for _ in range(650):
        x, y = (int(rng.integers(0, 1020)), int(rng.integers(0, 460)))
        w, h = map(int, rng.integers(5, 32, 2))
        color = tuple(map(int, rng.integers(0, 256, 3)))
        draw.rectangle((x, y, x+w, y+h), fill=color)
    target = root / 'dataset/synthetic'
    target.mkdir(parents=True, exist_ok=True)
    existing = [(target / name).exists() for name in
                ['synthetic_left.png', 'synthetic_right.png', 'H_ground_truth.txt']]
    if all(existing):
        log('WARN', 'Existing synthetic pair retained; no files overwritten')
        return
    if any(existing):
        raise FileExistsError('Incomplete existing synthetic pair; restore it before generating a fixture')
    scene.crop((0, 0, 640, 480)).save(target / 'synthetic_left.png')
    scene.crop((320, 0, 960, 480)).save(target / 'synthetic_right.png')
    np.savetxt(target / 'H_ground_truth.txt', [[1, 0, 320], [0, 1, 0], [0, 0, 1]], fmt='%.17g')
    (target / 'provenance.json').write_text(json.dumps(dict(
        kind='local_generated_translation_fixture', seed=14120,
        direction='B_right_to_A_left', official_course_pair=False,
        description='Two 640x480 crops of a procedural scene, offset by 320px. '
                    'Replace with the course synthetic pair for final course evaluation.'
    ), indent=2), encoding='utf-8')
    log('SUCCESS', 'Generated local synthetic fixture (not the course-provided pair)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    if not args.synthetic_only:
        prepare()
    synthetic()
