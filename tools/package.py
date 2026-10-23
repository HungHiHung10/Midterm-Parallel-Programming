# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3).
# Original packaging/validation utility. Does not upload or submit files.
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def package(root=ROOT):
    notebook = root/'Group22_Midterm.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code_cells), 'Notebook has unexecuted cells'
    assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs), 'Notebook contains errors'
    summary = json.loads((root/'outputs_v0/summary.json').read_text(encoding='utf-8'))
    assert len(summary['results']) == 12 and all(r['success'] for r in summary['results'])
    assert len(summary['aggregate']) == 3 and all(r['sites'] == 4 for r in summary['aggregate'])
    assert summary['correctness']['passed']
    assert summary['runs'] >= 5
    for name, digest in summary['source_sha256'].items():
        assert hashlib.sha256((root/'src'/name).read_bytes()).hexdigest() == digest, f'Stale results: {name}'
    for result in summary['results']:
        assert result['blend']['exposure'] in ['affine', 'unchanged']
        assert result['blend']['seam'] in ['optimal', 'feather']
        assert math.isfinite(result['blend']['overlap_mae']) and result['blend']['overlap_mae'] >= 0
        prefix = root/'outputs_v0'/f"{result['resolution']}_{result['site']}"
        samples = list(csv.DictReader(prefix.with_suffix('.csv').open()))
        assert len(samples) == summary['runs']
        for stage in result['mean']:
            values = [float(r[stage]) for r in samples]
            assert abs(sum(values)/len(values)-result['mean'][stage]) < 1e-8
            assert abs(min(values)-result['min'][stage]) < 1e-8
    expected_sizes = {'small':(1280,720), 'medium':(1920,1080), 'large':(3840,2160)}
    files = list((root/'dataset/prepared').glob('*/*.png'))
    assert len(files) == 24
    for path in files:
        with Image.open(path) as image:
            assert image.size == expected_sizes[path.parent.name]
    for site in ['site1_indoor','site2_outdoor','site3_custom','site4_challenging']:
        assert (root/'outputs_v0'/f'panorama_{site}.png').is_file()
        assert (root/'outputs_v0'/f'panorama_{site}_cropped.png').is_file()
    dist = root/'dist'
    dist.mkdir(exist_ok=True)
    source_files = [notebook] + [root/name for name in
        ['README.md','MEMBER3_ANALYSIS.md','requirements.txt','team.json']]
    source_files += [p for folder in ['src','tools'] for p in (root/folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.name != '.gitkeep']
    source_files += [p for p in (root/'outputs_v0').iterdir()
                     if p.is_file() and (p.suffix in ['.json','.csv','.log','.txt'] or
                     p.name in ['panorama_site1_indoor.png','panorama_site2_outdoor.png',
                                'panorama_site3_custom.png','panorama_site4_challenging.png',
                                'panorama_site1_indoor_cropped.png','panorama_site2_outdoor_cropped.png',
                                'panorama_site3_custom_cropped.png','panorama_site4_challenging_cropped.png',
                                'synthetic_panorama.png','scaling.png','panoramas.png'])]
    dataset_files = [
        p for folder in ['raw', 'prepared', 'synthetic']
        for p in (root/'dataset'/folder).rglob('*')
        if p.is_file() and p.name != '.gitkeep'
    ]
    for name, paths in [('Group22_Midterm.zip', source_files), ('Group22_Dataset.zip', dataset_files)]:
        target = dist/name
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for path in sorted(paths):
                archive.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == len(set(archive.namelist()))
        print(f'[SUCCESS]: {target} ({target.stat().st_size/1e6:.1f} MB)')
    print('[SUCCESS]: Notebook outputs, source hashes, 12 benchmark pairs, CSV statistics, 24 dimensions and ZIP CRC verified')

if __name__ == '__main__':
    package()
