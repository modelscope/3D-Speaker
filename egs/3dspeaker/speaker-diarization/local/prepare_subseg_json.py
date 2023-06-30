
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import re
import pathlib
import numpy as np
import argparse
from copy import deepcopy

import torch
import torchaudio

parser = argparse.ArgumentParser(description='Cut out the sub-segments.')
parser.add_argument('--vad', default='', type=str, help='Input vad json')
parser.add_argument('--out_file', default='', type=str, help='output file')
parser.add_argument('--dur', default=1.5, type=float, help='Duration of sub-segments')
parser.add_argument('--shift', default=0.75, type=float, help='Shift of sub-segments')

def main():
    args = parser.parse_args()
    with open(args.vad, 'r') as f:
        vad_json = json.load(f)
    subseg_json = {}
    print(f'[INFO]: Generate sub-segmetns...')
    for segid in vad_json:
        wavid = segid.rsplit('_', 2)[0]
        st = vad_json[segid]['start']
        ed = vad_json[segid]['stop']
        subseg_st = st
        subseg_dur = args.dur
        while subseg_st + subseg_dur < ed:
            subseg_ed = subseg_st+subseg_dur
            item = deepcopy(vad_json[segid])
            item.update({
                'start': round(subseg_st, 2),
                'stop': round(subseg_ed, 2)
            })
            subsegid = wavid+'_'+str(round(subseg_st, 2))+\
                '_'+str(round(subseg_ed, 2))
            subseg_json[subsegid] = item
            subseg_st += args.shift
        if subseg_st < ed:
            subseg_st = max(ed-subseg_dur, subseg_st)
            item = deepcopy(vad_json[segid])
            item.update({
                'start': round(subseg_st, 2),
                'stop': round(ed, 2)
            })
            subsegid = wavid+'_'+str(round(subseg_st, 2))+\
                '_'+str(round(ed, 2))
            subseg_json[subsegid] = item

    dirname = os.path.dirname(args.out_file)
    os.makedirs(dirname, exist_ok=True)
    with open(args.out_file, mode='w') as f:
        json.dump(subseg_json, f, indent=2)

    msg = "Subsegments json is prepared in %s" % (args.out_file)
    print(f'[INFO]: {msg}')

if __name__ == '__main__':
    main()
