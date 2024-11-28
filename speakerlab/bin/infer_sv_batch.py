# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input wav list, designed for large-scale 
embedding extraction. 
Please pre-install "modelscope".
Usage:
    `python infer_sv_batch.py --model_id $model_id --wavs $wav_list --feat_out_dir $feat_out_dir`
"""

import os
import sys
import re
import pathlib
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torchaudio
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset
from kaldiio import WriteHelper

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(os.path.abspath(__file__)))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract large-scale speaker embeddings.')
parser.add_argument('--model_id', default='', type=str, help='Model id in modelscope')
parser.add_argument('--wavs', default='', type=str, help='Wavs')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
parser.add_argument('--feat_out_dir', default='', type=str, help='Feat out dir')
parser.add_argument('--feat_out_format', choices=['npy', 'ark'], default='npy', type=str, help='Feat out format, npy or ark')
parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
parser.add_argument('--diable_progress_bar', action='store_true', help='Disable the progress bar')

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

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

ECAPA_CNCeleb = {
    'obj': 'speakerlab.models.ecapa_tdnn.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
        'batch_size': 64,
    },
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
        'batch_size': 16,
    },
    # ERes2NetV2 trained on 200k labeled speakers
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
        'batch_size': 16,
    },
    # ERes2Net_Base trained on 200k labeled speakers
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
        'batch_size': 16,
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
        'batch_size': 64,
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
        'batch_size': 64,
    },
    # ERes2Net trained on VoxCeleb
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
        'batch_size': 16,
    },
    # ERes2Net_Base trained on 3dspeaker
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1', 
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
        'batch_size': 16,
    },
    # ERes2Net_large trained on 3dspeaker
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
        'batch_size': 16,
    },
    # ECAPA-TDNN trained on CNCeleb
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
        'batch_size': 16,
    },
    # ECAPA-TDNN trained on 3dspeaker
    'iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
        'batch_size': 16,
    },
    # ECAPA-TDNN trained on VoxCeleb
    'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k': {
        'revision': 'v1.0.1', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa_tdnn.bin',
        'batch_size': 16,
    },
}

def main():
    args = parser.parse_args()
    assert isinstance(args.model_id, str) and \
        is_official_hub_path(args.model_id), "Invalid modelscope model id."
    if args.model_id.startswith('damo/'):
        args.model_id = args.model_id.replace('damo/','iic/', 1)
    assert args.model_id in supports, "Model id not currently supported."

    args.local_model_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
    args.local_model_dir = pathlib.Path(args.local_model_dir)
    args.local_model_dir.mkdir(exist_ok=True, parents=True)

    conf = supports[args.model_id]
    # download models from modelscope according to model_id
    cache_dir = snapshot_download(
                args.model_id,
                revision=conf['revision'],
                )
    cache_dir = pathlib.Path(cache_dir)

    # link
    download_files = ['examples', conf['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = args.local_model_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)
    try:
        # input should be wav list
        with open(args.wavs,'r') as f:
            wav_list = [i.strip() for i in f.readlines()]
        assert len(wav_list) > 0
    except:
        raise Exception('[ERROR]: Input should be wav list for batch inference.')

    # load model
    pretrained_model = args.local_model_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')
    model = conf['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    # set batch size
    if args.batch_size is None:
        args.batch_size = conf['batch_size']
    print(f'[INFO]: Set the batch size to {args.batch_size}.')

    # recommend using one GPU per process.
    ngpus = torch.cuda.device_count()
    if ngpus > 0:
        print(f'[INFO]: Detected {ngpus} GPUs.')
        nprocs = ngpus
    else:
        print('[INFO]: No GPUs detected.')
        nprocs = 1

    nprocs = min(len(wav_list), nprocs)
    print(f'[INFO]: Set {nprocs} processes to extract embeddings.')

    # output dir
    if args.feat_out_dir == '':
        args.feat_out_dir = args.local_model_dir / 'embeddings'
    else:
        args.feat_out_dir = pathlib.Path(args.feat_out_dir)
    args.feat_out_dir.mkdir(exist_ok=True, parents=True)
    print(f'[INFO]: Saving embedding dir is {args.feat_out_dir}')

    mp.spawn(main_process, nprocs=nprocs, args=(nprocs, args, wav_list, embedding_model))

def main_process(rank, nprocs, args, wav_list, embedding_model):
    if args.feat_out_format == 'ark':
        save_ark = args.feat_out_dir / f'embedding_{rank}.ark'
        save_scp = args.feat_out_dir / f'embedding_{rank}.scp'
        assert not save_ark.exists(), f'{save_ark} exists, please remove it manually.'
        writer = WriteHelper(f'ark,scp:{save_ark},{save_scp}')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        ngpus = torch.cuda.device_count()
        device = torch.device('cuda:%d'%(rank%ngpus))
    embedding_model.to(device)

    wav_dataset = IterWavList(wav_list, rank, nprocs, args.batch_size)
    data_mp_context = 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
    wav_loader = torch.utils.data.DataLoader(
        wav_dataset, 
        batch_size=1, 
        multiprocessing_context=data_mp_context,
        num_workers=16, pin_memory=True, prefetch_factor=2)
    
    if rank == 0 and (not args.diable_progress_bar):
        pbar = tqdm(total=len(wav_loader))
        pbar.set_description("Processing")

    with torch.no_grad():
        for wav_ids, feats, pos in wav_loader:
            feats = feats.squeeze(0).to(device)
            embeddings = embedding_model(feats).detach().cpu().numpy()
            for i in range(len(wav_ids)):
                wav_id = wav_ids[i][0]
                wav_embeddings = embeddings[pos[i]:pos[i+1]]
                wav_embedding = wav_embeddings.mean(0)

                if args.feat_out_format == 'npy':
                    save_path = args.feat_out_dir / f'{wav_id}.npy'
                    if os.path.exists(save_path):
                        print(f'[WARNING]: {save_path} already exists. Overwrite it.')
                    np.save(save_path, wav_embedding)
                elif args.feat_out_format == 'ark':
                    writer(wav_id, wav_embedding)

            if rank == 0 and (not args.diable_progress_bar):
                pbar.update(len(wav_ids))
    if rank == 0 and (not args.diable_progress_bar):
        pbar.close()


class IterWavList(IterableDataset):
    def __init__(self, wav_list, rank=0, world_size=1, batchsize=64):
        self.data = wav_list
        self.batchsize = batchsize
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        self.length = len(self.data) // world_size
        self.rank = rank
        self.world_size = world_size
    
    def initialize(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        indexes = list(range(len(self.data)))
        self.local_indexes = indexes[self.rank::self.world_size]
        self.indexes = self.local_indexes[worker_id::num_workers]

    def __len__(self):
        return self.length

    def __iter__(self):
        self.initialize()
        buf = []
        seg_in_buf_num = 0
        for index in self.indexes:
            data_path = self.data[index]
            try:
                wavs = self.load_wav(data_path)
            except:
                print(f'[WARNING]: Error reading {data_path}, please check.')
                continue
            wav_id = os.path.basename(data_path).rsplit('.', 1)[0]
            feats = []
            for wav in wavs:
                feats.append(self.feature_extractor(wav))
            feats = torch.stack(feats)
            seg_in_buf_num += len(feats)
            buf.append([wav_id, seg_in_buf_num, feats])

            if seg_in_buf_num >= self.batchsize:
                wav_ids = [i[0] for i in buf]
                pos = [0] + [i[1] for i in buf]
                feats = torch.cat([i[2] for i in buf], dim=0)
                yield wav_ids, feats, pos
                buf = []
                seg_in_buf_num = 0

        if len(buf) > 0:
            wav_ids = [i[0] for i in buf]
            pos = [0] + [i[1] for i in buf]
            feats = torch.cat([i[2] for i in buf], dim=0)
            yield wav_ids, feats, pos
    
    def chunk_wav(self, wav, chunk_sample_size):
        def circle_pad(wav, object_len):
            wav_len = wav.shape[0]
            n = int(np.ceil(object_len/wav_len))
            wav = [wav for i in range(n)]
            wav = torch.cat(wav)
            return wav[:object_len]

        n = int(np.ceil(wav.shape[0] / chunk_sample_size))
        wav = circle_pad(wav, n*chunk_sample_size)
        wavs = [wav[i*chunk_sample_size:(i+1)*chunk_sample_size] for i in range(n)]

        return wavs

    def load_wav(self, wav_path, obj_fs=16000, chunk_size=10, max_load_len=90):
        wav, fs = torchaudio.load(wav_path)
        if fs != obj_fs:
            print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        wav = wav.mean(dim=0)
        wav = wav[:int(max_load_len*obj_fs)]
        wavs = self.chunk_wav(wav, int(chunk_size*obj_fs))
        return wavs

if __name__ == '__main__':
    main()
