import os
import torch
import torchaudio
import numpy as np
import torchaudio.compliance.kaldi as Kaldi
from speakerlab.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer,
                                           compute_c_norm)

def get_wavscp_dict(wavscp, suffix=''):
    temp_dict={}
    with open(wavscp, 'r') as wavscp_f:
        lines = wavscp_f.readlines()
    for i in lines:
        i=i.strip().split()
        if suffix == '' or suffix is None:
            key_i = i[0]
        else:
            key_i = i[0]+'_'+suffix
        value_path = i[1]
        if key_i in temp_dict:
            raise ValueError('The key must be unique.')
        temp_dict[key_i]=value_path
    return temp_dict

class CosineTestIntraEpoch():
    def __init__(self, data_dir='/home/admin/workspace/speechbrain/recipes/SV_kaldi/base/data', save_dir='', test_name='VoxCeleb1-test'):
        self.data_dir = data_dir + '/' + test_name
        self.save_dir = save_dir
        self.test_name = test_name

        wavscp = os.path.join(self.data_dir, 'wav.scp')
        trials = os.path.join(self.data_dir, 'trials')
        if not os.path.isfile(wavscp):
            raise ValueError('wav.scp must exists!')
        if not os.path.isfile(trials):
            raise ValueError('trials must exists!')

        self.data = get_wavscp_dict(wavscp)
        with open(trials) as f:
            self.veri_test = [line.rstrip() for line in f]

        self.enrol_id_list = set()
        self.test_id_list = set()
        for line in self.veri_test:
            self.enrol_id_list.add(line.split()[1])
            self.test_id_list.add(line.split()[2])

        self.f_path = '%s/%s_eer.log'%(self.save_dir, test_name)

    def test_once(self, model, epoch):
        self.embedding_model = model
        enrol_dict, test_dict = self.compute_embedding_loop(self.data)
        scores, labels = self.get_verification_scores(enrol_dict, test_dict)
        self.compute_eer(scores, labels, epoch)

    def compute_eer(self, scores, labels, epoch, p_target=0.01, c_miss=1, c_fa=1):
        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)
        min_dcf = compute_c_norm(fnr,
                                fpr,
                                p_target=p_target,
                                c_miss=c_miss,
                                c_fa=c_fa)

        with open(self.f_path, 'a') as f:
            f.write('Epoch: {}, EER = {:.3f}, minDCF(p_target:{} c_miss:{} c_fa:{}) = {:.3f}\n'.format(epoch, eer*100, p_target, c_miss, c_fa, min_dcf))

    def get_verification_scores(self, enrol_dict, test_dict):
        save_file = os.path.join(self.save_dir, "%s_scores.txt"%self.test_name)
        s_file = open(save_file, "w")

        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        enrol_id_list = list(enrol_dict.keys())
        test_id_list = list(test_dict.keys())
        test_embs = torch.cat(list(test_dict.values()))
        score_dict = {}
        for enrol_id in enrol_id_list:
            enrol = enrol_dict[enrol_id]
            enrol = enrol.repeat(test_embs.shape[0], 1)
            scores = similarity(enrol, test_embs)
            for ind, test_id in enumerate(test_id_list):
                score_dict[enrol_id+test_id]=scores[ind].item()

        scores = []
        labels = []

        for i, line in enumerate(self.veri_test):
            labe_pair = int(line.split()[0])
            enrol_id = line.split()[1]
            test_id = line.split()[2]
            score = score_dict[enrol_id+test_id]
            s_file.write("%s %s %i %f\n" % (enrol_id, test_id, labe_pair, score))
            scores.append(score)
            labels.append(labe_pair)

        s_file.close()
        scores = np.hstack(scores)
        labels = np.hstack(labels)

        return scores, labels

    def compute_embedding_loop(self, data):
        self.embedding_model.eval()
        enrol_data_dict = {}
        test_data_dict = {}
        with torch.no_grad():
            for num, seg_id in enumerate(data):
                data_path = data[seg_id]
                wav, fs = torchaudio.load(data_path)
                wav = wav.cuda()
                feat = Kaldi.fbank(wav, num_mel_bins=80)
                feat = feat - feat.mean(dim=0, keepdim=True)
                emb = self.embedding_model(feat.unsqueeze(0))
                if len(emb.shape)==3:
                    emb = emb.squeeze(0).detach().clone()
                else:
                    assert len(emb.shape)==2
                    emb = emb.detach().clone()
                if seg_id in self.enrol_id_list:
                    enrol_data_dict[seg_id] = emb
                if seg_id in self.test_id_list:
                    test_data_dict[seg_id] = emb
        return enrol_data_dict, test_data_dict
