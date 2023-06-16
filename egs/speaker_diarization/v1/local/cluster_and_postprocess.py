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
parser.add_argument('--embs_dir', default='', type=str, help='Embedding dir')
parser.add_argument('--rttm_dir', default='', type=str, help='Rttm dir')

def make_rttms(seg_list, out_rttm):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_id, seg_st, seg_ed = seg[0].rsplit('_', 2)
        seg_st = float(seg_st)
        seg_ed = float(seg_ed)
        cluster_id = seg[1]
        if i == 0:
            new_seg_list.append([seg_id, seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][3]:
            if seg_st > new_seg_list[-1][2]:
                new_seg_list.append([seg_id, seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][2] = seg_ed
        else:
            if seg_st < new_seg_list[-1][2]:
                p = (new_seg_list[-1][2]+seg_st) / 2
                new_seg_list[-1][2] = p
                seg_st = p
            new_seg_list.append([seg_id, seg_st, seg_ed, cluster_id])

    line_str ="SPEAKER {} 0 {:.2f} {:.2f} <NA> <NA> {:d} <NA> <NA>\n"
    with open(out_rttm,'w') as f:
        for seg in new_seg_list:
            seg_id, seg_st, seg_ed, cluster_id = seg
            f.write(line_str.format(seg_id, seg_st, seg_ed-seg_st, cluster_id))    

def main():
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    args = parser.parse_args()
    embs_dir =  pathlib.Path(args.embs_dir)
    embs_files = list(embs_dir.glob('*.pkl'))
    embs_files.sort()
    if len(embs_files) <= rank:
        print("WARNING: The number of threads exceeds the number of files")
        sys.exit()

    os.makedirs(args.rttm_dir, exist_ok=True)
    print("[INFO] Start clustering...")
    local_embs_files = embs_files[rank::threads_num]
    config = build_config(args.conf)
    cluster = build('cluster', config)
    for embs_file in local_embs_files:
        with open(embs_file,'rb') as f:
            stat_obj = pickle.load(f)
        embeddings = stat_obj['embeddings']
        segids = stat_obj['segids']
        # cluster
        labels = cluster(embeddings)
        # output rttm
        new_labels = np.zeros(len(labels),dtype=int)
        uniq = np.unique(labels)
        for i in range(len(uniq)):
            new_labels[labels==uniq[i]] = i 
        seg_list = [(i,j) for i,j in zip(segids, new_labels)]
        rec_id = embs_file.name.rsplit('.', 1)[0]
        out_rttm = os.path.join(args.rttm_dir, rec_id+'.rttm')
        make_rttms(seg_list, out_rttm)

if __name__ == "__main__":
    main()
