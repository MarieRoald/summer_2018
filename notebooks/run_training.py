from subprocess import run
from pathlib import Path
import json
from pprint import pprint

cfgs = Path('./recsys_architectures')

for p in cfgs.glob('*.json'):
    #run(['python', 'autoencode.py', p])
    #run(['python', 'multimodal_autoencode.py', p])
    
    #run(['python', 'variational_autoencode.py', p])

    run(['python', 'recsys_autoencode.py', p])
    run(['python', 'recsys_multimodal.py', p])
    run(['python', 'recsys_sharedembedding.py', p])
    run(['python', 'recsys_sound_autoencode.py', p])