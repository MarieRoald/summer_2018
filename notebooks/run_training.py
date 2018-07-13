from subprocess import run
from pathlib import Path
import json
from pprint import pprint

cfgs = Path('./architectures')

for p in cfgs.glob('*.json'):
    run(['python', 'autoencode.py', p])
    run(['python', 'multimodal_autoencode.py', p])