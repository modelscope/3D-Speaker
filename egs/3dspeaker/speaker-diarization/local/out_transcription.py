# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import json
import argparse
import re
import torch
import torch.distributed as dist

from speakerlab.utils.utils import silent_print

import jieba
import logging
logger = logging.getLogger('jieba')
logger.setLevel(logging.CRITICAL)
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'

try:
    import modelscope
except ImportError:
    raise ImportError("Package \"modelscope\" not found. Please install them first.")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Use pretrained model from modelscope. So "model_id" and "model_revision" are necessary.
ASR_MODEL = {
    'model_id': 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    'model_revision': 'v2.0.4',
}

def get_trans_sentence(wav_path, asr_pipeline):
    sentence_info = [[]]
    punc_pattern = r'[,.!?;:"\-—…、，。！？；：“”‘’]'
    asr_result = asr_pipeline(wav_path, return_raw_text=True)[0]
    raw_text = asr_result['raw_text']
    text = asr_result['text']
    timestamp = asr_result['timestamp']
    timestamp = [[i[0]/1000, i[1]/1000] for i in timestamp]
    raw_text_list = raw_text.split()
    assert len(timestamp) == len(raw_text_list)
    text_pt = 0
    for i, wd in enumerate(raw_text_list):
        cache_text = ''
        while text_pt < len(text) and cache_text.lower().replace(' ','') != wd.lower():
            cache_text+=text[text_pt]
            text_pt+=1
        if text_pt > len(text):
            print('[ERROR]: The ASR results may have wrong format, skip processing %s'%wav_path)
            return []
        while text_pt < len(text) and (text[text_pt] == ' ' or text[text_pt] in punc_pattern):
            cache_text += text[text_pt]
            text_pt+=1
        sentence_info[-1].append([cache_text, timestamp[i]])
        if cache_text[-1] in punc_pattern and text_pt < len(text):
            sentence_info.append([])
    return sentence_info

def match_spk(sentence, output_field_labels):
    # sentence: [[word1, timestamps1], [word2, timestamps2], ...]
    # output_field_labels: [[st, ed, spkid], ...]
    if len(sentence)==0:
        return []
    st_sent = sentence[0][1][0]
    ed_sent = sentence[-1][1][1]
    max_overlap = 0
    overlap_per_spk = {}
    for st_spk, ed_spk, spk in output_field_labels:
        overlap_dur = min(ed_sent, ed_spk) - max(st_sent, st_spk)
        if spk not in overlap_per_spk:
            overlap_per_spk[spk]=0
        if overlap_dur > 0:
            overlap_per_spk[spk]+=overlap_dur
    overlap_per_spk_list = [[spk, overlap_per_spk[spk]] for spk in overlap_per_spk if overlap_per_spk[spk]>0]
    # sort by duration
    overlap_per_spk_list = sorted(overlap_per_spk_list, key=lambda x:x[1], reverse=True)
    overlap_per_spk_list = [i[0] for i in overlap_per_spk_list]
    return overlap_per_spk_list

def distribute_spk(sentence_info, output_field_labels):
    #  sentence_info: [[[word1, timestamps1], [word2, timestamps2], ...], ...]
    last_spk = 0
    for sentence in sentence_info:
        main_spks = match_spk(sentence, output_field_labels)
        main_spk = main_spks[0] if len(main_spks) > 0 else last_spk
        for i, wd in enumerate(sentence):
            wd_spks = match_spk([wd], output_field_labels)
            if main_spk in wd_spks:
                sentence[i].append(main_spk)
            elif len(wd_spks) > 0:
                sentence[i].append(wd_spks[0])
            else:
                sentence[i].append(last_spk)
        last_spk = sentence[-1][2]
    if len(sentence_info) == 0:
        return []
    #  sentence_info_with_spk: [text_string, timeinterval, spk]
    sentence_info = [j for i in sentence_info for j in i]
    sentence_info_with_spk_merge = [sentence_info[0]]
    punc_pattern = r'[,.!?;:"\-—…、，。！？；：“”‘’]'
    for i in sentence_info[1:]:
        if i[2] == sentence_info_with_spk_merge[-1][2] and \
            i[1][0] < sentence_info_with_spk_merge[-1][1][1] + 2:
            sentence_info_with_spk_merge[-1][0] += i[0]
            sentence_info_with_spk_merge[-1][1][1] = i[1][1]
        else:
            sentence_info_with_spk_merge.append(i)
    return sentence_info_with_spk_merge

def main(args):
    sys_rttm_dir = os.path.join(args.exp_dir, 'rttm')
    result_dir = os.path.join(args.exp_dir, 'transcripts')
    os.makedirs(result_dir, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])

    meta_file = os.path.join(args.exp_dir, 'json/subseg.json')
    with open(meta_file, "r") as f:
        full_meta = json.load(f)
    all_keys = full_meta.keys()
    A = {'_'.join(word.rstrip().split("_")[:-2]):full_meta[word]['file'] for word in all_keys}
    rec_ids = list(A.keys())[rank::threads_num]

    if args.gpu is None or len(args.gpu)==0:
        args.gpu = ['0']

    gpu_id = int(args.gpu[rank%len(args.gpu)])
    if gpu_id < torch.cuda.device_count():
        device = 'cuda:%d'%gpu_id
    else:
        print("[WARNING]: Gpu %s is not available. Use cpu instead." % gpu_id)
        device = 'cpu'

    dist.init_process_group(backend='gloo')
    with silent_print():
        if rank == 0 and len(rec_ids) > 0:
            asr_pipeline = pipeline(
                task=Tasks.auto_speech_recognition, 
                model=ASR_MODEL['model_id'], 
                model_revision=ASR_MODEL['model_revision'],
                device=device,
                disable_pbar=True,
                disable_update=True,
            )
        dist.barrier()
        if rank != 0 and len(rec_ids) > 0:
            asr_pipeline = pipeline(
                task=Tasks.auto_speech_recognition, 
                model=ASR_MODEL['model_id'], 
                model_revision=ASR_MODEL['model_revision'],
                device=device,
                disable_pbar=True,
                disable_update=True,
            )

    for rec_id in rec_ids:
        rttm_path = os.path.join(sys_rttm_dir, rec_id+'.rttm')
        output_field_labels = []
        with open(rttm_path, 'r') as f:
            for i in f.readlines():
                i = i.strip().split()
                output_field_labels.append([float(i[3]), float(i[3])+float(i[4]), i[7]])
        sentence_info = get_trans_sentence(A[rec_id], asr_pipeline)
        sentence_info_with_spk = distribute_spk(sentence_info, output_field_labels)

        output_trans_path = os.path.join(result_dir, rec_id+'.txt')
        with open(output_trans_path, 'w') as f:
            for text_string, timeinterval, spk in sentence_info_with_spk:
                f.write('%s: [%.3f %.3f] %s\n'%(spk, timeinterval[0], timeinterval[1], text_string))
        msg = 'Transcripts of %s have been finished in %s'%(rec_id, result_dir)
        print(f'[INFO]: {msg}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',
                        type=str,
                        default="",
                        help="exp dir")
    parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
    args = parser.parse_args()
    main(args)