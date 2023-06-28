# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download the pretrained speaker embedding models from modelscope 
(https://www.modelscope.cn/models) based on the given model id, and extract speaker 
embeddings from subsegments of audio. Please pre-install "modelscope".
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np

import torch
import torchaudio
import torch.distributed as dist

from speakerlab.utils.config import yaml_config_loader, Config
from speakerlab.utils.builder import build

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract speaker embeddings for diarization.')
parser.add_argument('--model_id', default=None, help='Model id in modelscope')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--subseg_json', default='', type=str, help='Sub-segments info')
parser.add_argument('--embs_out', default='', type=str, help='Out embedding dir')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


FEATURE_COMMON = {
    'obj': 'speakerlab.process.processor.FBank',
    'args': {
        'n_mels': 80,
        'sample_rate': 16000,
        'mean_nor': True,
    },
}

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
}    

def main():
    args = parser.parse_args()
    conf = yaml_config_loader(args.conf)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='gloo')
    if args.model_id is not None:
        # use the model id pretrained model
        assert isinstance(args.model_id, str) and \
        is_official_hub_path(args.model_id), "Invalid modelscope model id."
        assert args.model_id in supports, "Model id not currently supported."
        model_config = supports[args.model_id]
        # download models from modelscope given model_id
        if rank == 0:
            cache_dir = snapshot_download(
                        args.model_id,
                        revision=model_config['revision'],
                        )
            obj_list = [cache_dir]
        else:
            obj_list = [None]
        dist.broadcast_object_list(obj_list, 0)
        cache_dir = obj_list[0]
        pretrained_model = os.path.join(cache_dir, model_config['model_pt'])
        conf['embedding_model'] = model_config['model']
        conf['pretrained_model'] = pretrained_model
        conf['feature_extractor'] = FEATURE_COMMON
    else:
        assert pretrained_model is not None, \
            "[ERROR] One of the params `model_id` and `pretrained_model` must be set."
        # use the local pretrained model
        conf['pretrained_model'] = pretrained_model
    
    os.makedirs(args.embs_out, exist_ok=True)
    with open(args.subseg_json, "r") as f:
        subseg_json = json.load(f)

    all_keys = subseg_json.keys()
    A = [i.rsplit('_', 2)[0] for i in all_keys]
    all_rec_ids = list(set(A))
    all_rec_ids.sort()
    if len(all_rec_ids) == 0:
        raise Exception(
            "No recording IDs found! Please check if json file is accuratly generated."
            )
    if len(all_rec_ids) <= rank:
        print("WARNING: The number of threads exceeds the number of files")
        sys.exit()

    metadata={}
    for rec_id in all_rec_ids:
        subset = {}
        for key in subseg_json:
            k = str(key)
            if k.rsplit('_',2)[0]==rec_id:
                subset[key] = subseg_json[key]
        metadata[rec_id]=subset

    print("[INFO] Start computing embeddings...")
    local_rec_ids = all_rec_ids[rank::threads_num]

    if args.use_gpu:
        gpu_id = int(args.gpu[rank%len(args.gpu)])
        if gpu_id < torch.cuda.device_count():
            device = torch.device('cuda:%d'%gpu_id)
        else:
            print("[WARNING]: Gpu %s is not available. Use cpu instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(config.pretrained_model, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    embedding_model.to(device)
    # compute embeddings of sub-segments
    for rec_id in local_rec_ids:
        meta = metadata[rec_id]
        emb_file_name = rec_id + ".pkl"
        stat_emb_file = os.path.join(args.embs_out, emb_file_name)
        if not os.path.isfile(stat_emb_file):
            embeddings = []
            wav_path = meta[list(meta.keys())[0]]['file']
            wav, fs = torchaudio.load(wav_path)
            for segid in meta:
                sample_start = int(meta[segid]['start']*fs)
                sample_stop = int(meta[segid]['stop']*fs)
                wav_seg = wav[:, sample_start:sample_stop]
                feat = feature_extractor(wav_seg).unsqueeze(0)
                feat = feat.to(device)
                with torch.no_grad():
                    emb = embedding_model(feat).cpu().numpy()
                embeddings.append(emb)
                
            embeddings = np.concatenate(embeddings, axis=0)
            stat_obj = {'embeddings': embeddings, 'segids': list(meta.keys())}
            pickle.dump(stat_obj, open(stat_emb_file,'wb'))
        else:
            print("[WARNING] Embeddings has been saved previously. Skip it.")

if __name__ == "__main__":
    main()
