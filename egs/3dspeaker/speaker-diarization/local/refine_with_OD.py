# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will refine the cluster results with overlap detection results.
"""

import os
import numpy as np
import pickle
import argparse
from scipy import optimize

try:
    import pyannote
except ImportError:
    raise ImportError("Package \"pyannote\" not found. Please install them first.")

from pyannote.audio import Model
from pyannote.core import SlidingWindow, SlidingWindowFeature

parser = argparse.ArgumentParser(description='Refine with overlap detection')
parser.add_argument('--init_rttm_dir', default='', type=str, help='init rttm dir')
parser.add_argument('--rttm_dir', default='', type=str, help='rttm dir')
parser.add_argument('--segmentation_dir', default='', type=str, help='segmentation dir')
parser.add_argument('--hf_access_token', default='', type=str, help='hf_access_token for pyannote/segmentation-3.0')

def main():
    args = parser.parse_args()

    segmentation_file = os.path.join(args.segmentation_dir, 'segmentation.pkl')
    with open(segmentation_file, 'rb') as f:
        pkl_data= pickle.load(f)

    segmentation_params = {
        'segmentation':'pyannote/segmentation-3.0',
        'segmentation_batch_size':32,
        'use_auth_token':args.hf_access_token,
    }

    model = Model.from_pretrained(
        segmentation_params['segmentation'], 
        use_auth_token=segmentation_params['use_auth_token'], 
        strict=False,
    )
    frame_windows = model.receptive_field # used to locate frame index
    chunks = SlidingWindow(
        start=0.0, 
        duration=model.specifications.duration, 
        step=0.1 * model.specifications.duration,
    )
    
    init_rttm_files = [i for i in os.listdir(args.init_rttm_dir) if i.endswith('.rttm')]
    for r in init_rttm_files:
        init_rttm_path = os.path.join(args.init_rttm_dir, r)
        out_rttm_path = os.path.join(args.rttm_dir, r)
        
        basename = r.rsplit('.', 1)[0]
        segmentations_data = pkl_data[basename]['segmentations']
        segmentations = SlidingWindowFeature(segmentations_data, chunks) # align segmentations data and chunk time
        count_data = pkl_data[basename]['count']

        output_field_labels = []
        spk_dict = {}
        with open(init_rttm_path, 'r') as f:
            for i in f.readlines():
                i = i.strip().split()
                st = float(i[3])
                ed = st + float(i[4])
                if i[7] not in spk_dict:
                    spk_dict[i[7]] = len(spk_dict)
                label = spk_dict[i[7]]
                output_field_labels.append([st, ed, label])
        
        num_frames = len(count_data)
        max_speaker = len(spk_dict)
        cluster_frames = np.zeros((num_frames, max_speaker))
        for i in output_field_labels:
            cluster_frames[frame_windows.closest_frame(i[0]+frame_windows.duration/2)\
                :frame_windows.closest_frame(i[1]+frame_windows.duration/2)\
                    , i[2]] = 1.0

        activations = np.zeros((num_frames, max_speaker))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            # data: [num_frames_per_chunk, num_classes]
            # chunk_cluster_frames: [num_frames_per_chunk, max_speaker]
            start_frame = frame_windows.closest_frame(c.start+frame_windows.duration/2)
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, max_speaker))

            # assign label to each dimension of "data" according to number of overlap frames between "data" and "chunk_cluster_frames"
            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j])>0:
                    num_of_overlap_frames = [(data[:, j].astype('int') & d.astype('int')).sum() for d in chunk_cluster_frames.T]
                else:
                    num_of_overlap_frames = [-1]*max_speaker
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix) # (num_classes, max_speaker)
            row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
            for j in range(len(row_index)):
                r = row_index[j]
                c = col_index[j]
                if cost_matrix[r, c] > 0:
                    align_chunk_cluster_frames[:, c] = np.maximum(
                            data[:, r], align_chunk_cluster_frames[:, c]
                            )
            activations[start_frame:end_frame] += align_chunk_cluster_frames

        # correct activations according to count_data
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t, (c, speakers) in enumerate(zip(count_data, sorted_speakers)):
            cur_max_spk_num = min(max_speaker, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1)==0) & (cluster_frames.sum(-1)!=0)
        binary[supplement_field] = cluster_frames[supplement_field]

        def binary_to_segs(binary, timestamps, threshold=0.5):
            output_field_labels = []
            # binary: [num_frames, num_classes]
            # timestamps: [T_1, ..., T_num_frames]        
            for k, k_scores in enumerate(binary.T):
                start = timestamps[0]
                is_active = k_scores[0] > threshold

                for t, y in zip(timestamps[1:], k_scores[1:]):
                    if is_active:
                        if y < threshold:
                            output_field_labels.append([round(start, 3), round(t, 3), k])
                            start = t
                            is_active = False
                    else:
                        if y > threshold:
                            start = t
                            is_active = True

                if is_active:
                    output_field_labels.append([round(start, 3), round(t, 3), k])
            return sorted(output_field_labels, key=lambda x : x[0])

        timestamps = [frame_windows[i].middle for i in range(binary.shape[0])]
        output_field_labels = binary_to_segs(binary, timestamps)

        line_str ="SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
        with open(out_rttm_path, 'w') as f:
            for seg in output_field_labels:
                seg_st, seg_ed, cluster_id = seg
                f.write(line_str.format(basename, seg_st, seg_ed-seg_st, cluster_id))


if __name__ == '__main__':
    main()
