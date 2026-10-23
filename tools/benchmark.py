# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original benchmark and reporting automation.
from console import log
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITES = ['site1_indoor', 'site2_outdoor', 'site3_custom', 'site4_challenging']
STAGES = ['preprocess_ms', 'keypoint_ms', 'descriptor_ms', 'matching_ms', 'ransac_ms', 'warp_blend_ms']

def save_largest_valid_crop(source, destination):
    """Save the largest all-image rectangle; the warp canvas background is exact black."""
    with Image.open(source) as image:
        rgb = np.asarray(image.convert('RGB'))
    non_black = np.any(rgb != 0, axis=2)
    # The union of two projected image quadrilaterals is contiguous on each
    # scanline for this left/right panorama. Fill between its first and last
    # non-black pixels so naturally black objects are not mistaken for canvas.
    valid = np.zeros_like(non_black)
    for y, row in enumerate(non_black):
        columns = np.flatnonzero(row)
        if columns.size:
            valid[y, columns[0]:columns[-1] + 1] = True
    heights = np.zeros(valid.shape[1], dtype=np.int32)
    best_area, best_box = 0, None
    for y, row in enumerate(valid):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                left, previous_height = stack.pop()
                area = previous_height * (x - left)
                if area > best_area:
                    best_area = area
                    best_box = (left, y - previous_height + 1, x, y + 1)
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    if best_box is None:
        raise ValueError(f'Panorama has no valid pixels: {source}')
    cropped = Image.fromarray(rgb[best_box[1]:best_box[3], best_box[0]:best_box[2]])
    cropped.save(destination)
    return list(best_box), best_area / valid.size

def run(binary, left, right, output, runs, warmup, prefix):
    command = [str(binary), str(left), str(right), str(output), '--version', '0',
               '--runs', str(runs), '--warmup', str(warmup), '--save-h', str(prefix.with_suffix('.H.txt')),
               '--timings-csv', str(prefix.with_suffix('.csv'))]
    # Remove old products so failed reruns cannot reuse stale results.
    for path in [output, prefix.with_suffix('.H.txt'), prefix.with_suffix('.csv')]:
        path.unlink(missing_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    prefix.with_suffix('.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        print(result.stdout + result.stderr, flush=True)
        return dict(success=False, error=result.stderr.strip(), command=command)
    with prefix.with_suffix('.csv').open(newline='') as file:
        samples = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(file)]
    if len(samples) != runs:
        raise ValueError('Incorrect measured run count')
    for row in samples:
        if not all(np.isfinite(row[k]) and row[k] >= 0 for k in STAGES + ['total_pipeline_ms']):
            raise ValueError('Invalid timing sample')
    blend_rows = re.findall(r'Blend: exposure=(\w+), overlap MAE=([\d.]+), seam=(\w+)', result.stdout)
    blend = (dict(exposure=blend_rows[-1][0], overlap_mae=float(blend_rows[-1][1]),
                  seam=blend_rows[-1][2]) if blend_rows else None)
    return dict(success=True, runs=runs, warmup=warmup, command=command, blend=blend,
        mean={k: float(np.mean([r[k] for r in samples])) for k in STAGES + ['total_pipeline_ms']},
        min={k: float(min(r[k] for r in samples)) for k in STAGES + ['total_pipeline_ms']})

def correctness(root, binary, out):
    data = root / 'dataset/synthetic'
    result = run(binary, data/'synthetic_left.png', data/'synthetic_right.png',
                 out/'synthetic_panorama.png', 5, 1, out/'synthetic')
    if result['success']:
        gt = np.loadtxt(data/'H_ground_truth.txt')
        estimated = np.loadtxt(out/'synthetic.H.txt')
        gt, estimated = gt/gt[2, 2], estimated/estimated[2, 2]
        w, h = Image.open(data/'synthetic_right.png').size
        points = np.array([[x, y, 1] for y in np.linspace(20, h-21, 12)
                           for x in np.linspace(20, w-21, 16)]).T
        actual, predicted = gt @ points, estimated @ points
        actual, predicted = actual[:2]/actual[2], predicted[:2]/predicted[2]
        error = np.linalg.norm(actual-predicted, axis=0)
        result.update(frobenius_error=float(np.linalg.norm(gt-estimated)),
                      projection_mean_px=float(error.mean()), projection_max_px=float(error.max()),
                      threshold_px=1.0, passed=bool(error.mean() <= 1.0),
                      H_estimated=estimated.tolist(), H_ground_truth=gt.tolist(),
                      provenance=json.loads((data/'provenance.json').read_text(encoding='utf-8'))
                      if (data/'provenance.json').exists() else {'kind': 'user_provided'})
    return result

def benchmark(root=ROOT, runs=5, warmup=1):
    if runs < 5 or warmup < 0:
        raise ValueError('Report requires runs >= 5 and warmup >= 0')
    binary = root / ('stitch.exe' if os.name == 'nt' else 'stitch')
    out = root / 'outputs_v0'
    out.mkdir(exist_ok=True)
    summary = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   machine=platform.platform(), processor=platform.processor(),
                   runs=runs, warmup=warmup,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'src').iterdir() if p.suffix in ['.cu', '.cpp', '.h', '.cuh']},
                   selected_gpu_stages=STAGES, results=[])
    summary['correctness'] = correctness(root, binary, out)
    for resolution in ['small', 'medium', 'large']:
        for site in SITES:
            log('PROCESS', f'Benchmark {resolution}/{site}')
            data = root / 'dataset/prepared' / resolution
            prefix = out / f'{resolution}_{site}'
            # Only retain medium panoramas; temporary other sizes are removed after timing.
            output = out / (f'panorama_{site}.png' if resolution == 'medium' else f'bench_{resolution}_{site}.png')
            result = run(binary, data/f'{site}_left.png', data/f'{site}_right.png',
                         output, runs, warmup, prefix)
            if result['success'] and resolution == 'medium':
                cropped = out / f'panorama_{site}_cropped.png'
                crop_box, retained = save_largest_valid_crop(output, cropped)
                result.update(cropped_output=str(cropped.relative_to(root)).replace('\\', '/'),
                              crop_box=crop_box, retained_canvas_fraction=retained)
            result.update(site=site, resolution=resolution)
            summary['results'].append(result)
            (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            if resolution != 'medium':
                output.unlink(missing_ok=True)
    aggregate = []
    for resolution in ['small', 'medium', 'large']:
        rows = [r for r in summary['results'] if r['resolution'] == resolution and r['success']]
        if not rows:
            continue
        # Equal weight to each site, each of which has the same measured run count.
        mean = {k: float(np.mean([r['mean'][k] for r in rows])) for k in STAGES+['total_pipeline_ms']}
        # Mean of per-site minima: this is NOT the global minimum run.
        minimum = {k: float(np.mean([r['min'][k] for r in rows])) for k in STAGES+['total_pipeline_ms']}
        p = sum(mean[k] for k in summary['selected_gpu_stages']) / mean['total_pipeline_ms']
        aggregate.append(dict(resolution=resolution, sites=len(rows), mean=mean,
            mean_of_site_min=minimum, hot_stage=max(STAGES, key=mean.get),
            p=p, amdahl_limit=1/(1-p) if p < 1 else None))
    summary['aggregate'] = aggregate
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out/'timings_summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['resolution', 'site', 'stage', 'mean_ms', 'min_ms'])
        for row in summary['results']:
            if row['success']:
                for stage in STAGES+['total_pipeline_ms']:
                    writer.writerow([row['resolution'], row['site'], stage, row['mean'][stage], row['min'][stage]])
    failures = [r for r in summary['results'] if not r['success']]
    log('ERROR' if failures else 'SUCCESS', f"{len(summary['results'])-len(failures)}/12 successful real pairs")
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args()
    report = benchmark(runs=args.runs, warmup=args.warmup)
    sys.exit(0 if all(r['success'] for r in report['results']) and report['correctness'].get('passed') else 1)
