# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download the pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and perform the voice activity detection given the audio. 
Please pre-install "modelscope" and "funasr".
Usage:
    1. do vad from one wav file.
        `python voice_activity_detection.py --wavs $wav_path --out_file $vad_json `
    2. do vad the wav list.
        `python voice_activity_detection.py --wavs $wav_list --out_file $vad_json `
"""

import os
import json
import pickle
import argparse
import torch

try:
    import modelscope
except ImportError:
    raise ImportError("Package \"modelscope\" not found. Please install them first.")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

try:
    from speakerlab.utils.utils import merge_vad
except:
    pass

parser = argparse.ArgumentParser(description='Voice activity detection')
parser.add_argument('--wavs', default='', type=str, help='Wavs')
parser.add_argument('--out_file', default='', type=str, help='output file')

# Use pretrained model from modelscope. So "model_id" and "model_revision" are necessary.
VAD_PRETRAINED = {
    'model_id': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'model_revision': 'v2.0.4',
}

def main():
    args = parser.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    segmentation_file = os.path.join(out_dir, 'segmentation.pkl')
    if os.path.exists(segmentation_file):
        consider_segmentation = True
        with open(segmentation_file, 'rb') as f:
            segmentations = pickle.load(f)
    else:
        consider_segmentation = False

    wavs = []
    if args.wavs.endswith('.wav'):
        # input is a wav path
        wavs.append(args.wavs)
    else:
        try:
            # input is wav list
            with open(args.wavs,'r') as f:
                wav_list = f.readlines()
        except:
            raise Exception('Input should be a wav file or a wav list.')
        for wav_path in wav_list:
            wav_path = wav_path.strip()
            wavs.append(wav_path)

    vad_pipeline = pipeline(
        task=Tasks.voice_activity_detection, 
        model=VAD_PRETRAINED['model_id'], 
        model_revision=VAD_PRETRAINED['model_revision'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        )

    json_dict = {}
    print(f'[INFO]: Start computing VAD...')
    for wpath in wavs:        
        vad_time = vad_pipeline(wpath)[0]
        vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in vad_time['value']]
        if consider_segmentation:
            basename = os.path.basename(wpath).rsplit('.', 1)[0]
            vad_time = merge_vad(vad_time, segmentations[basename]['valid_field'])
        vad_time = [[round(vad_t[0], 3), round(vad_t[1], 3)] for vad_t in vad_time]

        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        for strt, end in vad_time:
            subsegmentid = wid + '_' + str(strt) + '_' + str(end)
            json_dict[subsegmentid] = {
                        'file': wpath,
                        'start': strt,
                        'stop': end,
                }
    dirname = os.path.dirname(args.out_file)
    os.makedirs(dirname, exist_ok=True)
    with open(args.out_file, mode='w') as f:
        json.dump(json_dict, f, indent=2)

    msg = "VAD json is prepared in %s" % (args.out_file)
    print(f'[INFO]: {msg}')

if __name__ == '__main__':
    main()
