# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original colored console helper.
import os

def log(level, message):
    colors = {'SUCCESS': '32', 'ERROR': '31', 'WARN': '33', 'PROCESS': '37'}
    label = f'[{level}]:'
    if 'NO_COLOR' not in os.environ:
        label = f'\033[{colors[level]}m{label}\033[0m'
    print(label, message, flush=True)
