# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original notebook output helpers.
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown
from benchmark import SITES, STAGES

LABELS = ['1 Preprocess', '2 Keypoints', '3 BRIEF', '4 Matching', '5 RANSAC', '6 Warp/blend']

def table(headers, rows):
    display(Markdown('| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n'
                     + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)))

def show_correctness(summary):
    c = summary['correctness']
    if not c['success']:
        raise RuntimeError(c['error'])
    display(Markdown('**Synthetic provenance:** ' + c['provenance'].get('kind', 'unknown')))
    print('H estimated (B -> A):\n', np.array(c['H_estimated']))
    print('H ground truth (B -> A):\n', np.array(c['H_ground_truth']))
    table(['Metric', 'Measured value'], [
        ['Frobenius norm (H normalized by H[2,2])', f"{c['frobenius_error']:.6e}"],
        ['Mean projection error (px)', f"{c['projection_mean_px']:.6e}"],
        ['Max projection error (px)', f"{c['projection_max_px']:.6e}"],
        ['Pass threshold (mean px)', c['threshold_px']], ['Passed', c['passed']]])
    assert c['passed'], 'Synthetic error exceeds the declared threshold'

def show_panoramas(root, summary):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, site in zip(axes.flat, SITES):
        path = root/'outputs_v0'/f'panorama_{site}_cropped.png'
        with Image.open(path) as image:
            preview = image.copy()
            preview.thumbnail((1400, 800))
            ax.imshow(preview)
        ax.set_title(site)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(root/'outputs_v0/panoramas.png', dpi=120)
    plt.show()
    medium = [r for r in summary['results'] if r['resolution'] == 'medium']
    table(['Site', 'Exposure', 'Overlap MAE', 'Blend', 'Canvas retained'], [[
        r['site'], r['blend']['exposure'], f"{r['blend']['overlap_mae']:.3f}",
        r['blend']['seam'], f"{r['retained_canvas_fraction']:.1%}"
    ] for r in medium])
    display(Markdown('Overlap MAE dùng để chọn chiến lược trong từng cặp, không phải thước đo '
                     'so sánh chất lượng giữa các cảnh có nội dung khác nhau.'))
    display(Markdown('Bản trên được crop theo hình chữ nhật lớn nhất không chứa nền canvas đen. '
                     'Các file `panorama_<site>.png` vẫn giữ toàn bộ footprint để kiểm tra thuật toán.'))

def show_timings(summary, root):
    # Full per-site mean/min table, not a manually maintained Markdown copy.
    rows = []
    for result in summary['results']:
        if not result['success']:
            raise RuntimeError(f"Failed: {result['resolution']}/{result['site']}")
        for key, label in zip(STAGES+['total_pipeline_ms'], LABELS+['Total']):
            rows.append([result['resolution'], result['site'], label,
                         f"{result['mean'][key]:.3f}", f"{result['min'][key]:.3f}"])
    table(['Resolution', 'Site', 'Stage', 'Mean ms', 'Min ms'], rows)
    aggregate = summary['aggregate']
    table(['Stage']+[r['resolution']+' mean / mean(site min), ms' for r in aggregate],
          [[label]+[f"{r['mean'][stage]:.3f} / {r['mean_of_site_min'][stage]:.3f}" for r in aggregate]
           for stage, label in zip(STAGES+['total_pipeline_ms'], LABELS+['Total'])])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(aggregate))
    bottom = np.zeros(len(x))
    for stage, label in zip(STAGES, LABELS):
        values = np.array([r['mean'][stage] for r in aggregate])
        axes[0].bar(x, values, bottom=bottom, label=label)
        axes[1].plot([.9216, 2.0736, 8.2944], values, marker='o', label=label)
        bottom += values
    axes[0].set_xticks(x, [r['resolution'] for r in aggregate])
    axes[0].set_ylabel('Mean across four sites (ms)')
    axes[0].set_title('V0 per-stage timing')
    axes[1].set_xlabel('Input megapixels per image')
    axes[1].set_ylabel('Mean across four sites (ms)')
    axes[1].set_title('Scaling by resolution')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'outputs_v0/scaling.png', dpi=140)
    plt.show()

def show_analysis(summary):
    rows = []
    for result in summary['aggregate']:
        mean, total = result['mean'], result['mean']['total_pipeline_ms']
        ranked = sorted(STAGES, key=mean.get, reverse=True)
        hot = ', '.join(f'{s}: {mean[s]/total:.1%}' for s in ranked[:2])
        limit = 'unbounded' if result['amdahl_limit'] is None else f"{result['amdahl_limit']:.2f}x"
        rows.append([result['resolution'], result['sites'], hot, f"{result['p']:.8f}", limit])
    table(['Resolution', 'Successful sites', 'Two hottest stages (% total)', 'p (planned six stages)',
           '1/(1-p), compute-only bound'], rows)
    display(Markdown('Kế hoạch V1 chuyển cả sáu stage lên GPU. Các stage đứng đầu bảng được ưu tiên '
        'profiling và tối ưu. Tổng thời gian bao gồm điều phối/cấp phát trong pipeline, loại trừ đọc/ghi '
        'ảnh và log. Khi p gần 1, giới hạn Amdahl rất nhạy với phần thời gian điều phối nhỏ; đây là '
        'giới hạn lý thuyết cho **compute-only**, không phải dự báo speedup hay số đo GPU. '
        'Chi phí truyền H2D/D2H, launch và đồng bộ phải được đo ở V1.'))

def show_dataset(root):
    manifest = json.loads((root/'dataset/prepared/manifest.json').read_text(encoding='utf-8'))
    rows = []
    for resolution in ['small', 'medium', 'large']:
        selected = [r for r in manifest if r['resolution'] == resolution]
        for row in selected:
            with Image.open(root/row['output']) as image:
                assert list(image.size) == row['size']
        rows.append([resolution, len(selected), str(selected[0]['size']), sum(r['upscaled'] for r in selected)])
    assert len(manifest) == 24
    table(['Resolution', 'Images verified', 'Width, height', 'Upscaled images'], rows)
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for i, site in enumerate(SITES):
        for j, side in enumerate(['left', 'right']):
            # Display the normalized PNG used by V0. This keeps report-only
            # runs independent of an HEIC decoder; pillow-heif is needed only
            # when prepare_dataset must regenerate these files from raw data.
            record = next(r for r in manifest if r['resolution'] == 'medium'
                          and r['site'] == site and r['side'] == side)
            path = root/record['output']
            with Image.open(path) as image:
                preview = image.convert('RGB')
                prepared_size = preview.size
                preview.thumbnail((1000, 650))
                axes[i, j].imshow(preview)
            axes[i, j].set_title(f'{site} / {side} / medium {prepared_size}')
            axes[i, j].axis('off')
    fig.tight_layout()
    plt.show()
