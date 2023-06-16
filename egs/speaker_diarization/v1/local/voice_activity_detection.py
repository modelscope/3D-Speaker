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
import sys
import json
import argparse
import torchaudio

try:
    import modelscope
    import funasr
except ImportError:
    raise ImportError("Package \"modelscope\" or \"funasr\" not found. Please install them first.")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

parser = argparse.ArgumentParser(description='Voice activity detection')
parser.add_argument('--model_id', default=None, help='VAD Model id in modelscope')
parser.add_argument('--wavs', default='', type=str, help='Wavs')
parser.add_argument('--out_file', default='', type=str, help='output file')

VAD_PRETRAINED = {
    'model_id': 'damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'sample_rate': 16000,
}

def main():
    args = parser.parse_args()
    if args.model_id is not None:
        VAD_PRETRAINED['model_id'] = args.model_id

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

    vad_pipeline = pipeline(Tasks.voice_activity_detection, VAD_PRETRAINED['model_id'])
    json_dict = {}
    print(f'[INFO]: Start computing VAD...')
    for wpath in wavs:
        _, fs = torchaudio.load(wpath)
        assert fs == VAD_PRETRAINED['sample_rate'], \
            "The sample rate of %s is not %d, please resample it first." % (
                wpath, VAD_PRETRAINED['sample_rate'])
        vad_time = vad_pipeline(wpath)
        wid = os.path.basename(wpath).rsplit('.', 1)[0]
        for vad_t in vad_time['text']:
            strt = round(vad_t[0]/1000, 2)
            end = round(vad_t[1]/1000, 2)
            subsegmentid = wid + '_' + str(strt) + '_' + str(end)
            json_dict[subsegmentid] = {
                        'file': wpath,
                        'start': strt,
                        'stop': end,
                        'sample_rate': fs,
                }
    dirname = os.path.dirname(args.out_file)
    os.makedirs(dirname, exist_ok=True)
    with open(args.out_file, mode='w') as f:
        json.dump(json_dict, f, indent=2)

    msg = "VAD json is prepared in %s" % (args.out_file)
    print(f'[INFO]: {msg}')

if __name__ == '__main__':
    main()
