import os
import sys
import re
import json
import time
import argparse
import torch
import torchaudio
import pickle
import numpy as np
from speakerlab.utils.utils import parse_config, get_logger
from speakerlab.models.model import get_model
from speakerlab.dataset.features import Fbank


def main(args):
    logger = get_logger()
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    embedding_dir = args.exp_dir + '/embeddings'
    if rank==0:
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
    else:
        while not os.path.exists(embedding_dir):
            time.sleep(0.5)
    if args.use_gpu:
        if torch.cuda.is_available():
            gpu = int(args.gpu[rank % len(args.gpu)])
            device = torch.device('cuda', gpu)
        else:
            msg = 'No cuda device is detected. Using the cpu device.'
            logger.warning(msg)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    conf = parse_config(args.config)
    embedding_model = get_model(conf['model'], module='backbone')(**conf['model_args'])
    state_dict = torch.load(args.pretrain_path, map_location=device)
    new_state_dict = {}
    for k in state_dict:
        if k.startswith('module'):
            new_k=k[7:]
        else:
            new_k=k
        new_state_dict[new_k]=state_dict[k]
    embedding_model.load_state_dict(new_state_dict)
    embedding_model.to(device)
    embedding_model.eval()

    meta_file = os.path.join(args.exp_dir, 'metadata/subsegs.json')
    with open(meta_file, "r") as f:
        full_meta = json.load(f)

    all_keys = full_meta.keys()
    A = ['_'.join(word.rstrip().split("_")[:-2]) for word in all_keys]
    all_rec_ids = list(set(A))
    all_rec_ids.sort()
    if len(all_rec_ids) <= 0:
        msg = "No recording IDs found! Please check if meta_data json file is properly generated."
        raise ValueError(msg)
    if len(all_rec_ids) <= rank:
        msg = "Threads num is more than wav num. So close redundant threads"
        logger.info(msg)
        sys.exit()

    local_all_rec_ids = all_rec_ids[rank::threads_num]

    submeta={}
    for rec_id in local_all_rec_ids:
        subset = {}
        for key in full_meta:
            k = str(key)
            if k.rsplit('_',2)[0]==rec_id:
                subset[key] = full_meta[key]
        submeta[rec_id]=subset

    for i, rec_id in enumerate(submeta):
        tag = ("[" + str(i*threads_num+rank+1) + "/" + str(len(all_rec_ids)) + "]")
        # Log message.
        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.info(msg)

        meta = submeta[rec_id]
        emb_file_name = rec_id + ".emb_stat.pkl"
        stat_emb_file = os.path.join(embedding_dir, emb_file_name)

        if not os.path.isfile(stat_emb_file):
            logger.info("Extracting deep embeddings")
            embeddings = []
            compute_feature = Fbank(n_mels=conf['fbank_num_mel_bins'])
            wav_path = meta[list(meta.keys())[0]]['file']
            wav, fs = torchaudio.load(wav_path)
            for segid in meta:
                sample_start = int(meta[segid]['start'])
                sample_stop = int(meta[segid]['stop'])
                wav_seg = wav[:, sample_start:sample_stop]

                feat = compute_feature(wav_seg)
                feat = feat - feat.mean(dim=1, keepdim=True)
                feat = feat.to(device)
                with torch.no_grad():
                    emb = embedding_model(feat).cpu().numpy()
                embeddings.append(emb)

            embeddings = np.concatenate(embeddings, axis=0)
            stat_obj = {'embeddings': embeddings, 'segids': list(meta.keys())}
            logger.info("Saving Embeddings...")
            pickle.dump(stat_obj, open(stat_emb_file,'wb'))

        else:
            logger.info("Embeddings has been saved previously. Skipping it.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default="",
                        help="config file")
    parser.add_argument('--exp_dir',
                        type=str,
                        default="",
                        help="exp dir")
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=False,
                        help="use gpu")
    parser.add_argument('--pretrain_path',
                        type=str,
                        default="",
                        help="pretrained speaker model path")

    parser.add_argument('--gpu',
                        nargs='+',
                        help='GPU id to use.')
    args = parser.parse_args()
    main(args)
