# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original portable build helper.
from console import log
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build(root=ROOT):
    sources = ['main.cu', 'stb_impl.cpp', 'v0_frontend.cpp', 'v0_backend.cpp', 'v0_stitching.cpp']
    binary = root / ('stitch.exe' if platform.system() == 'Windows' else 'stitch')
    compiler = shutil.which('nvcc') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('Install g++ or CUDA nvcc to build V0')
    command = [compiler, '-O3', '-std=c++14']
    if Path(compiler).stem != 'nvcc':
        command += ['-x', 'c++']
    command += [str(root / 'src' / name) for name in sources] + ['-o', str(binary)]
    log('PROCESS', ' '.join(command))
    subprocess.run(command, check=True, cwd=root)
    return binary, command

if __name__ == '__main__':
    build()
