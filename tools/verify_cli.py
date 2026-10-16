# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). End-to-end CLI contract verification.
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from build import ROOT

binary = ROOT/('stitch.exe' if os.name == 'nt' else 'stitch')
left = ROOT/'dataset/synthetic/synthetic_left.png'
right = ROOT/'dataset/synthetic/synthetic_right.png'
env = dict(os.environ)
env.pop('NO_COLOR', None)
with tempfile.TemporaryDirectory(dir=ROOT/'outputs_v0') as directory:
    folder = Path(directory)
    output, csv_path, h_path = folder/'panorama.PNG', folder/'timing.csv', folder/'H.txt'
    base = [str(binary), str(left), str(right), str(output), '--version', '0']
    def call(args, success, overrides=None):
        result = subprocess.run(args, capture_output=True, text=True, env=overrides or env)
        assert (result.returncode == 0) == success, result.stdout+result.stderr
        return result.stdout+result.stderr
    text = call(base+['--runs','2','--warmup','1','--save-h',str(h_path),'--timings-csv',str(csv_path)], True)
    assert '\033[32m[SUCCESS]:' in text and '\033[37m[PROCESS]:' in text and '\033[33m[WARN]:' in text
    with Image.open(output) as image:
        assert image.format == 'PNG' and image.width >= 960 and image.height >= 480
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2 and len(rows[0]) == 8
    for key in rows[0]:
        if key == 'run': continue
        values = [float(row[key]) for row in rows]
        match = re.search(r'^'+key+r'\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE)
        assert match and abs(float(match[1])-sum(values)/len(values)) < 1e-5
        assert abs(float(match[2])-min(values)) < 1e-5
    for row in rows:
        stages = sum(float(value) for key,value in row.items() if key not in ['run','total_pipeline_ms'])
        assert stages <= float(row['total_pipeline_ms']) + 0.001
    for options in [['--runs','0'],['--runs','-1'],['--runs','abc'],['--runs','1x'],
                    ['--runs'],['--unknown','1'],['--version','2']]:
        text = call(base+options, False)
        assert '\033[31m[ERROR]:' in text
    text = call(base+['--runs','1'], True, dict(env, NO_COLOR='1'))
    assert '\033[' not in text
    jpg = folder/'result.JPEG'
    call([str(binary),str(left),str(right),str(jpg),'--version','0'], True)
    assert Image.open(jpg).format == 'JPEG'
    blank = folder/'blank.png'
    Image.new('RGB',(100,100),(127,127,127)).save(blank)
    call([str(binary),str(blank),str(blank),str(output),'--version','0'], False)
    call([str(binary),'missing.png',str(right),str(output),'--version','0'], False)
    call(base+['--save-h',str(folder/'absent'/'H.txt')], False)
    call(base+['--timings-csv',str(folder/'absent'/'stats.csv')], False)
    call(base+['--save-h',str(output)], False)
    call([str(binary),str(left),str(right),str(left),'--version','0'], False)
    print('[SUCCESS]: CLI, 4 ANSI colors, NO_COLOR, CSV mean/min, PNG/JPEG, failed I/O and failed pipeline verified')
