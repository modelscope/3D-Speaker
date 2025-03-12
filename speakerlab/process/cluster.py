# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import scipy
import sklearn
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity

import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

try:
    import umap, hdbscan
except ImportError:
    raise ImportError(
        "Package \"umap\" or \"hdbscan\" not found. \
        Please install them first by \"pip install umap-learn hdbscan\"."
        )


class SpectralCluster:
    """A spectral clustering method using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=10, pval=0.02, min_pnum=6, oracle_num=None):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num

    def __call__(self, X, **kwargs):
        pval = kwargs.get('pval', None)
        oracle_num = kwargs.get('speaker_num', None)

        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval=None):
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0]-self.min_pnum)

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        if k_oracle is None:
            k_oracle = self.k

        lambdas, eig_vecs = scipy.sparse.linalg.eigsh(L, k=min(self.max_num_spks+1, L.shape[0]), which='SM')

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks + 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        # k-means
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    """
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With 
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(self, n_neighbors=20, n_components=60, min_samples=20, min_cluster_size=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X, **kwargs):
        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0]-2),
            metric=self.metric,
        ).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_samples=self.min_samples, min_cluster_size=self.min_cluster_size).fit_predict(umap_X)
        return labels

class AHCluster:
    """
    Agglomerative Hierarchical Clustering, a bottom-up approach which iteratively merges 
    the closest clusters until a termination condition is reached.
    This implementation is adapted from https://github.com/BUTSpeechFIT/VBx.
    """

    def __init__(self, fix_cos_thr=0.4):
        self.fix_cos_thr = fix_cos_thr

    def __call__(self, X, **kwargs):
        scr_mx = cosine_similarity(X)
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels = fcluster(lin_mat, -self.fix_cos_thr + adjust, criterion='distance') - 1
        return labels


class CommonClustering:
    """Perfom clustering for input embeddings and output the labels.
    """

    def __init__(self, cluster_type, cluster_line=40, mer_cos=None, min_cluster_size=4, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        if self.cluster_type == 'spectral':
            self.cluster = SpectralCluster(**kwargs)
        elif self.cluster_type == 'umap_hdbscan':
            kwargs['min_cluster_size'] = min_cluster_size
            self.cluster = UmapHdbscan(**kwargs)
        elif self.cluster_type == 'AHC':
            self.cluster = AHCluster(**kwargs)
        else:
            raise ValueError(
                '%s is not currently supported.' % self.cluster_type
            )
        if self.cluster_type != 'AHC':
            self.cluster_for_short = AHCluster()
        else:
            self.cluster_for_short = self.cluster

    def __call__(self, X, **kwargs):
        # clustering and return the labels
        assert len(X.shape) == 2, 'Shape of input should be [N, C]'
        if X.shape[0] <= 1:
            return np.zeros(X.shape[0], dtype=int)
        if X.shape[0] < self.cluster_line:
            labels = self.cluster_for_short(X)
        else:
            labels = self.cluster(X, **kwargs)

        # remove extremely minor cluster
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        # merge similar  speaker
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)

        return labels

    def filter_minor_cluster(self, labels, x, min_cluster_size):
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize <= self.min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels

        minor_cset = cset[minor_idx]
        major_idx = np.where(csize > self.min_cluster_size)[0]
        if len(major_idx) == 0:
            return np.zeros_like(labels)
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) \
            for i in major_cset])
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]

        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            centers = np.stack([x[labels == i].mean(0) \
                for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels==c2]=c1
        return labels


class JointClustering:
    """Perfom joint clustering for input audio and visual embeddings and output the labels.
    """

    def __init__(self, audio_cluster, vision_cluster):
        self.audio_cluster = audio_cluster
        self.vision_cluster = vision_cluster

    def __call__(self, audioX, visionX, audioT, visionT, conf):
        # audio-only and video-only clustering
        alabels = self.audio_cluster(audioX)
        vlabels = self.vision_cluster(visionX)

        alabels = self.arrange_labels(alabels)
        vlist, vspk_embs, vspk_dur = self.get_vlist_embs(audioX, alabels, vlabels, audioT, visionT, conf)

        # modify alabels according to vlabels
        aspk_num = alabels.max()+1
        for i in range(aspk_num):
            aspki_index = np.where(alabels==i)[0]
            aspki_embs = audioX[alabels==i]

            aspkiT_part = np.array(audioT)[alabels==i]
            overlap_vspk = self.overlap_spks(self.cast_overlap(aspkiT_part), vlist, vspk_dur)
            if len(overlap_vspk) > 1:
                centers = np.stack([vspk_embs[s] for s in overlap_vspk])
                distribute_labels = self.distribute_embs(aspki_embs, centers)
                for j in range(distribute_labels.max()+1):
                    for loc in aspki_index[distribute_labels==j]:
                        alabels[loc] = overlap_vspk[j]
            elif len(overlap_vspk) == 1:
                for loc in aspki_index:
                    alabels[loc] = overlap_vspk[0]

        alabels = self.arrange_labels(alabels)
        return alabels

    def overlap_spks(self, times, vlist, vspk_dur=None):
        # get the vspk that overlaps with times.
        overlap_dur = {}
        for [a_st, a_ed] in times:
            for [v_st, v_ed, v_id] in vlist:
                if a_ed > v_st and v_ed > a_st:
                    if v_id not in overlap_dur:
                        overlap_dur[v_id]=0
                    overlap_dur[v_id] += min(a_ed, v_ed) - max(a_st, v_st)
        vspk_list = []
        for v_id, dur in overlap_dur.items():
            # set the criteria for confirming overlap.
            if (vspk_dur is None and dur > 0.5) or (vspk_dur is not None and dur > min(vspk_dur[v_id]*0.5, 0.5)):
                vspk_list.append(v_id)
        return vspk_list

    def distribute_embs(self, embs, centers):
        # embs: [n, D]. centers: [k, D]
        norm_centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        similarity = np.matmul(norm_embs, norm_centers.T) # [n, k]
        argsort = np.argsort(similarity, axis=-1)
        return argsort[:, -1]

    def get_vlist_embs(self, audioX, alabels, vlabels, audioT, visionT, conf):
        assert len(vlabels) == len(visionT)
        vlist = []
        for i, ti in enumerate(visionT):
            if len(vlist)==0 or vlabels[i] != vlist[-1][2] or ti - visionT[i-1] > conf.face_det_stride*0.04 + 1e-4:
                if len(vlist) > 0 and vlist[-1][1] - vlist[-1][0] < 1e-4:
                    # remove too short intervals. 
                    vlist.pop()
                vlist.append([ti, ti, vlabels[i]])
            else:
                vlist[-1][1] = ti

        # adjust vision labels
        vlabels_arrange = self.arrange_labels([i[2] for i in vlist], a_st=alabels.max()+1)
        vlist = [[i[0], i[1], j] for i, j in zip(vlist, vlabels_arrange)]

        # get audio spk embs aligning with 'vlist'
        vspk_embs = {}
        for [v_st, v_ed, v_id] in vlist:
            for i, [a_st, a_ed] in enumerate(audioT):
                if a_ed >= v_st and v_ed >= a_st:
                    if min(a_ed, v_ed) - max(a_st, v_st) > 1:
                        if v_id not in vspk_embs:
                            vspk_embs[v_id] = []
                        vspk_embs[v_id].append(audioX[i])
        for k in vspk_embs:
            vspk_embs[k] = np.stack(vspk_embs[k]).mean(0)

        vlist_new = []
        for i in vlist:
            if i[2] in vspk_embs:
                vlist_new.append(i)
        # get duration of v_spk
        vspk_dur = {}
        for i in vlist_new:
            if i[2] not in vspk_dur:
                vspk_dur[i[2]]=0
            vspk_dur[i[2]] += i[1]-i[0]

        return vlist_new, vspk_embs, vspk_dur

    def cast_overlap(self, input_time):
        if len(input_time)==0:
            return input_time
        output_time = []
        for i in range(0, len(input_time)-1):
            if i == 0 or output_time[-1][1] < input_time[i][0]:
                output_time.append(input_time[i])
            else:
                output_time[-1][1] = input_time[i][1]
        return output_time

    def arrange_labels(self, labels, a_st=0):
        # arrange labels in order from 0.
        new_labels = []
        labels_dict = {}
        idx = a_st
        for i in labels:
            if i not in labels_dict:
                labels_dict[i] = idx
                idx += 1
            new_labels.append(labels_dict[i])
        return np.array(new_labels)
