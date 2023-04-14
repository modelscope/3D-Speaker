# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download the pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id and model version, and extract the embeddings from the given audio file.
"""

import os
import json
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract embeddings for evaluation.')
parser.add_argument('--model_id', default='', type=str, help='Model id in modelscope')
parser.add_argument('--model_revision', default=None, type=str, help='Model revision in modelscope')
parser.add_argument('--wav_path', default='', type=str, help='Wav path')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')

CAMPPLUS = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': CAMPPLUS,
}

def main():
    args = parser.parse_args()
    assert isinstance(args.model_id, str) and \
        is_official_hub_path(args.model_id), "Invalid modelscope model id."
    assert args.model_id in supports, "Model id not currently supported."
    save_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
    save_dir =  pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # download models from modelscope according to model_id
    cache_dir = snapshot_download(
                args.model_id,
                revision=args.model_revision,
                )
    cache_dir = pathlib.Path(cache_dir)

    embedding_dir = save_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True, parents=True)

    # link
    download_files = ['examples', '.json', '.bin']
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass

            dst.symlink_to(src)

    # read config
    config_file = save_dir / 'configuration.json'
    conf = json.load(open(config_file))

    pretrained_model = save_dir / conf['model']['pretrained_model']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    model = supports[args.model_id]
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    # extract embeddings
    if len(args.wav_path) == 0:
        examples_dir = save_dir / 'examples'
        try:
            # use example wav
            args.wav_path = list(examples_dir.glob('*.wav'))[0]
            print(f'Wav_path is not specified, use {args.wav_path} instead.')
        except IndexError:
            assert FileNotFoundError('Invalid wav path.')

    wav, fs = torchaudio.load(args.wav_path)
    if 'sample_rate' in conf['model']['model_config']:
        assert fs == int(conf['model']['model_config']['sample_rate']), \
        f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."

    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)

    feature_extractor = FBank(80, fs, mean_nor=True)
    feat = feature_extractor(wav)
    feat = feat.unsqueeze(0)
    with torch.no_grad():
        embedding = embedding_model(feat).detach().cpu().numpy()

    # save embeddings
    save_path = embedding_dir / (
        '%s.npy' % (os.path.basename(args.wav_path).rsplit('.', 1)[0]))
    np.save(save_path, embedding)
    print(f'Extracted embedding from {args.wav_path} saved in {save_path}.')

if __name__ == '__main__':
    main()
