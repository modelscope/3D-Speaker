# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This is a speaker diarization inference script based on pretrained models.
Usages:
    1. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir]
    2. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token]
    3. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token] --nprocs [n]
"""

import os
import sys
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from scipy import optimize
import json

import torch
import torch.multiprocessing as mp

try:
    from speakerlab.utils.config import Config
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(os.path.abspath(__file__)))
    from speakerlab.utils.config import Config

from speakerlab.utils.builder import build
from speakerlab.utils.utils import merge_vad, silent_print, download_model_from_modelscope, circle_pad
from speakerlab.utils.fileio import load_audio

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from pyannote.audio import Inference, Model

parser = argparse.ArgumentParser(description='Speaker diarization inference.')
parser.add_argument('--wav', type=str, required=True, help='Input wavs')
parser.add_argument('--out_dir', type=str, required=True, help='Out results dir')
parser.add_argument('--out_type', choices=['rttm', 'json'], default='rttm', type=str, help='Results format, rttm or json')
parser.add_argument('--include_overlap', action='store_true', help='Include overlapping region')
parser.add_argument('--hf_access_token', type=str, help='hf_access_token for pyannote/segmentation-3.0 model. It\'s required if --include_overlap is specified')
parser.add_argument('--diable_progress_bar', action='store_true', help='Close the progress bar')
parser.add_argument('--nprocs', default=None, type=int, help='Num of procs')
parser.add_argument('--speaker_num', default=None, type=int, help='Oracle num of speaker')


def get_speaker_embedding_model(device:torch.device = None, cache_dir:str = None):
    conf = {
        'model_id': 'iic/speech_campplus_sv_zh_en_16k-common_advanced',
        'revision': 'v1.0.0',
        'model_ckpt': 'campplus_cn_en_common.pt',
        'embedding_model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {
                'n_mels': 80,
                'sample_rate': 16000,
                'mean_nor': True,
                },
        }
    }

    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    pretrained_model_path = os.path.join(cache_dir, conf['model_ckpt'])
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model,  feature_extractor

def get_cluster_backend():
    conf = {
        'cluster':{
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args':{
                'cluster_type': 'spectral',
                'mer_cos': 0.8,
                'min_num_spks': 1,
                'max_num_spks': 15,
                'min_cluster_size': 4,
                'oracle_num': None,
                'pval': 0.012,
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)

def get_voice_activity_detection_model(device: torch.device=None, cache_dir:str = None):
    conf = {
        'model_id': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        'revision': 'v2.0.4',
    }
    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    with silent_print():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection, 
            model=cache_dir, 
            device = 'cpu' if device is None else '%s:%s'%(device.type, device.index) if device.index else device.type,
            disable_pbar=True,
            disable_update=True,
            )
    return vad_pipeline

def get_segmentation_model(use_auth_token, device: torch.device=None):
    segmentation_params = {
        'segmentation':'pyannote/segmentation-3.0',
        'segmentation_batch_size':32,
        'use_auth_token':use_auth_token,
        }
    model = Model.from_pretrained(
        segmentation_params['segmentation'], 
        use_auth_token=segmentation_params['use_auth_token'], 
        strict=False,
        )
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=segmentation_params['segmentation_batch_size'],
        device = device,
        )
    return segmentation


class Diarization3Dspeaker():
    """
    This class is designed to handle the speaker diarization process, 
    which involves identifying and segmenting audio by speaker identities. 
    Args:
        device (str, default=None): The device on which models will run. 
        include_overlap (bool, default=False): Indicates whether to include overlapping 
            speech segments in the diarization output. Overlapping speech occurs when multiple 
            speakers are talking simultaneously.
        hf_access_token (str, default=None): Access token for Hugging Face, required if 
            include_overlap is True. This token allows access to pynnote segmentation models 
            available on the Hugging Face that handles overlapping speech.
        speaker_num (int, default=None): Specify number of speakers.
        model_cache_dir (str, default=None): If specified, the pretrained model will be downloaded 
            to this directory; only pretrained from modelscope are supported.
    Usage:
        diarization_pipeline = Diarization3Dspeaker(device, include_overlap, hf_access_token)
        output = diarization_pipeline(input_audio) # input_audio can be a path to a WAV file, a NumPy array, or a PyTorch tensor
        print(output) # output: [[1.1, 2.2, 0], [3.1, 4.1, 1], ..., [st_n, ed_n, speaker_id]]
        diarization_pipeline.save_diar_output('audio.rttm') # or audio.json
    """
    def __init__(self, device=None, include_overlap=False, hf_access_token=None, speaker_num=None, model_cache_dir=None):
        if include_overlap and hf_access_token is None:
            raise ValueError("hf_access_token is required when include_overlap is True.")

        self.device = self.normalize_device(device)
        self.include_overlap = include_overlap

        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(self.device, model_cache_dir)
        self.vad_model = get_voice_activity_detection_model(self.device, model_cache_dir)
        self.cluster = get_cluster_backend()

        if include_overlap:
            self.segmentation_model = get_segmentation_model(hf_access_token, self.device)
        
        self.batchsize = 64
        self.fs = self.feature_extractor.sample_rate
        self.output_field_labels = None
        self.speaker_num = speaker_num

    def __call__(self, wav, wav_fs=None, speaker_num=None):
        wav_data = load_audio(wav, wav_fs, self.fs)

        # stage 1-1: do vad
        vad_time = self.do_vad(wav_data)
        if self.include_overlap:
            # stage 1-2: do segmentation
            segmentations, count = self.do_segmentation(wav_data)
            valid_field = get_valid_field(count)
            vad_time = merge_vad(vad_time, valid_field)

        # stage 2: prepare subseg
        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed)]

        # stage 3: extract embeddings
        embeddings = self.do_emb_extraction(chunks, wav_data)

        # stage 4: clustering
        speaker_num, output_field_labels = self.do_clustering(chunks, embeddings, speaker_num)

        if self.include_overlap:
            # stage 5: include overlap results
            binary = self.post_process(output_field_labels, speaker_num, segmentations, count)
            timestamps = [count.sliding_window[i].middle for i in range(binary.shape[0])]
            output_field_labels = self.binary_to_segs(binary, timestamps)

        self.output_field_labels = output_field_labels
        return output_field_labels

    def do_vad(self, wav):
        # wav: [1, T]
        vad_results = self.vad_model(wav[0])[0]
        vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in vad_results['value']]
        return vad_time

    def do_segmentation(self, wav):
        segmentations = self.segmentation_model({'waveform':wav, 'sample_rate': self.fs})
        frame_windows = self.segmentation_model.model.receptive_field

        count = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return segmentations, count

    def chunk(self, st, ed, dur=1.5, step=0.75):
        chunks = []
        subseg_st = st
        while subseg_st + dur < ed + step:
            subseg_ed = min(subseg_st + dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

    def do_emb_extraction(self, chunks, wav):
        # chunks: [[st1, ed1]...]
        # wav: [1, T]
        wavs = [wav[0, int(st*self.fs):int(ed*self.fs)] for st, ed in chunks]
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st: batch_st+self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                embeddings_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(embeddings_batch)
                batch_st += self.batchsize
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings

    def do_clustering(self, chunks, embeddings, speaker_num=None):
        cluster_labels = self.cluster(
            embeddings, 
            speaker_num = speaker_num if speaker_num is not None else self.speaker_num
        )
        speaker_num = cluster_labels.max()+1
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels

    def post_process(self, output_field_labels, speaker_num, segmentations, count):
        num_frames = len(count)
        cluster_frames = np.zeros((num_frames, speaker_num))
        frame_windows = count.sliding_window
        for i in output_field_labels:
            cluster_frames[frame_windows.closest_frame(i[0]+frame_windows.duration/2)\
                :frame_windows.closest_frame(i[1]+frame_windows.duration/2)\
                    , i[2]] = 1.0

        activations = np.zeros((num_frames, speaker_num))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            # data: [num_frames_per_chunk, num_classes]
            # chunk_cluster_frames: [num_frames_per_chunk, speaker_num]
            start_frame = frame_windows.closest_frame(c.start+frame_windows.duration/2)
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, speaker_num))

            # assign label to each dimension of "data" according to number of 
            # overlap frames between "data" and "chunk_cluster_frames"
            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j])>0:
                    num_of_overlap_frames = [(data[:, j].astype('int') & d.astype('int')).sum() \
                        for d in chunk_cluster_frames.T]
                else:
                    num_of_overlap_frames = [-1]*speaker_num
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix) # (num_classes, speaker_num)
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
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            cur_max_spk_num = min(speaker_num, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1)==0) & (cluster_frames.sum(-1)!=0)
        binary[supplement_field] = cluster_frames[supplement_field]
        return binary

    def binary_to_segs(self, binary, timestamps, threshold=0.5):
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

    def save_diar_output(self, out_file, wav_id=None, output_field_labels=None):
        if output_field_labels is None and self.output_field_labels is None:
            raise ValueError('No results can be saved.')
        if output_field_labels is None:
            output_field_labels = self.output_field_labels

        wav_id = 'default' if wav_id is None else wav_id
        if out_file.endswith('rttm'):
            line_str ="SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
            with open(out_file, 'w') as f:
                for seg in output_field_labels:
                    seg_st, seg_ed, cluster_id = seg
                    f.write(line_str.format(wav_id, seg_st, seg_ed-seg_st, cluster_id))
        elif out_file.endswith('json'):
            out_json = {}
            for seg in output_field_labels:
                seg_st, seg_ed, cluster_id = seg
                item = {
                    'start': seg_st,
                    'stop': seg_ed,
                    'speaker': cluster_id,
                }
                segid = wav_id+'_'+str(round(seg_st, 3))+\
                    '_'+str(round(seg_ed, 3))
                out_json[segid] = item
            with open(out_file, mode='w') as f:
                json.dump(out_json, f, indent=2)
        else:
            raise ValueError('The supported output file formats are currently limited to RTTM and JSON.')

    def normalize_device(self, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device

def get_valid_field(count):
    valid_field = []
    start = None
    for i, (c, data) in enumerate(count):
        if data.item()==0 or i==len(count)-1:
            if start is not None:
                end = c.middle
                valid_field.append([start, end])
                start = None
        else:
            if start is None:
                start = c.middle
    return valid_field

def compressed_seg(seg_list):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1]+seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list

def main_process(rank, nprocs, args, wav_list):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        ngpus = torch.cuda.device_count()
        device = torch.device('cuda:%d'%(rank%ngpus))
    diarization = Diarization3Dspeaker(device, args.include_overlap, args.hf_access_token, args.speaker_num)
    
    wav_list = wav_list[rank::nprocs]
    if rank == 0 and (not args.diable_progress_bar):
        wav_list = tqdm(wav_list, desc=f"Rank 0 processing")
    for wav_path in wav_list:
        ouput = diarization(wav_path)
        # write to file
        wav_id = os.path.basename(wav_path).rsplit('.', 1)[0]
        if args.out_dir is not None:
            out_file = os.path.join(args.out_dir, wav_id+'.%s'%args.out_type)
        else:
            out_file = '%s.%s'%(wav_path.rsplit('.', 1)[0], args.out_type)
        diarization.save_diar_output(out_file, wav_id)

def main():
    args = parser.parse_args()
    if args.include_overlap and args.hf_access_token is None:
        parser.error("--hf_access_token is required when --include_overlap is specified.")
    
    get_speaker_embedding_model()
    get_voice_activity_detection_model()
    get_cluster_backend()
    if args.include_overlap:
        get_segmentation_model(args.hf_access_token)
    print(f'[INFO]: Model downloaded successfully.')

    if args.wav.endswith('.wav'):
        # input is a wav file
        wav_list = [args.wav]
    else:
        try:
            # input should be a wav list
            with open(args.wav,'r') as f:
                wav_list = [i.strip() for i in f.readlines()]
        except:
            raise Exception('[ERROR]: Input should be a wav file or a wav list.')
    assert len(wav_list) > 0

    if args.nprocs is None:
        ngpus = torch.cuda.device_count()
        if ngpus > 0:
            print(f'[INFO]: Detected {ngpus} GPUs.')
            args.nprocs = ngpus
        else:
            print('[INFO]: No GPUs detected.')
            args.nprocs = 1

    args.nprocs = min(len(wav_list), args.nprocs)
    print(f'[INFO]: Set {args.nprocs} processes to extract embeddings.')

    # output dir
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    mp.spawn(main_process, nprocs=args.nprocs, args=(args.nprocs, args, wav_list))

if __name__ == '__main__':
    main()
