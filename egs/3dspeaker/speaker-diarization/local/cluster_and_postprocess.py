# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script is designed to cluster speaker embeddings and generate RTTM result files as output.
"""

import os
import sys
import argparse
import pickle
import pathlib
import numpy as np

from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build

parser = argparse.ArgumentParser(description='Cluster embeddings and output rttm files')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--wavs', default=None, help='Wav list file')
parser.add_argument('--audio_embs_dir', default=None, type=str, help='Embedding dir')
parser.add_argument('--rttm_dir', default=None, type=str, help='Rttm dir')
parser.add_argument('--visual_embs_dir', default=None, type=str, help='Visual embedding dir')

def make_rttms(seg_list, out_rttm, rec_id):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed = seg[0]
        seg_st = float(seg_st)
        seg_ed = float(seg_ed)
        cluster_id = seg[1] + 1
        if i == 0:
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][3]:
            if seg_st > new_seg_list[-1][2]:
                new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][2] = seg_ed
        else:
            if seg_st < new_seg_list[-1][2]:
                p = (new_seg_list[-1][2]+seg_st) / 2
                new_seg_list[-1][2] = p
                seg_st = p
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])

    line_str ="SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
    with open(out_rttm,'w') as f:
        for seg in new_seg_list:
            seg_id, seg_st, seg_ed, cluster_id = seg
            f.write(line_str.format(seg_id, seg_st, seg_ed-seg_st, cluster_id))

def audio_only_func(local_wav_list, audio_embs_dir, rttm_dir, config):
    cluster = build('cluster', config)
    for wav_file in local_wav_list:
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        if not os.path.exists(embs_file):
            print("[WARNING]: %s does not exist, it is possible that vad model did not detect valid speech in file %s, please check it."%(embs_file, wav_file))
            continue
        with open(embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            embeddings = stat_obj['embeddings']
            times = stat_obj['times']
        # cluster
        labels = cluster(embeddings)
        # output rttm
        new_labels = np.zeros(len(labels), dtype=int)
        uniq = np.unique(labels)
        for i in range(len(uniq)):
            new_labels[labels==uniq[i]] = i 
        seg_list = [(i,j) for i, j in zip(times, new_labels)]
        out_rttm = os.path.join(rttm_dir, rec_id+'.rttm')
        make_rttms(seg_list, out_rttm, rec_id)

def audio_vision_func(local_wav_list, audio_embs_dir, visual_embs_dir, rttm_dir, config):
    cluster = build('cluster', config)
    for wav_file in local_wav_list:
        wav_name = os.path.basename(wav_file)
        rec_id = wav_name.rsplit('.', 1)[0]
        audio_embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')
        visual_embs_file = os.path.join(visual_embs_dir, rec_id + '.pkl')
        with open(audio_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            audio_embeddings = stat_obj['embeddings']
            audio_times = stat_obj['times']
        with open(visual_embs_file, 'rb') as f:
            stat_obj = pickle.load(f)
            visual_embeddings = stat_obj['embeddings']
            visual_times = stat_obj['times']

        # cluster
        labels = cluster(audio_embeddings, visual_embeddings, audio_times, visual_times, config)
        # output rttm
        new_labels = np.zeros(len(labels), dtype=int)
        uniq = np.unique(labels)
        for i in range(len(uniq)):
            new_labels[labels==uniq[i]] = i 
        seg_list = [(i,j) for i, j in zip(audio_times, new_labels)]
        out_rttm = os.path.join(rttm_dir, rec_id+'.rttm')
        make_rttms(seg_list, out_rttm, rec_id)

def main():
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    args = parser.parse_args()
    with open(args.wavs,'r') as f:
        wav_list = [i.strip() for i in f.readlines()]
    wav_list.sort()
    if len(wav_list) <= rank:
        print("[WARNING]: The number of threads exceeds the number of files")
        sys.exit()

    os.makedirs(args.rttm_dir, exist_ok=True)
    print("[INFO] Start clustering...")
    local_wav_list = wav_list[rank::threads_num]
    config = build_config(args.conf)
    if args.visual_embs_dir is None or args.visual_embs_dir == '':
        audio_only_func(local_wav_list, args.audio_embs_dir, args.rttm_dir, config)
    else:
        audio_vision_func(local_wav_list, args.audio_embs_dir, args.visual_embs_dir, args.rttm_dir, config)


if __name__ == "__main__":
    main()
