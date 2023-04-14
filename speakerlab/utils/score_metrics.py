"""
This script computes the official performance metrics for the NIST 2016 SRE.
The metrics include EER and DCFs (min/act).
"""

__author__ = "Omid Sadjadi"
__email__ = "omid.sadjadi@nist.gov"
__version__ = "4.1"

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys


def compute_norm_counts(scores, edges, wghts=None):
    """ computes normalized (and optionally weighted) score counts for the
        bin edges.
    """

    if scores.size > 0:
        score_counts = np.histogram(scores, bins=edges,
                                    weights=wghts)[0].astype('f')
        norm_counts = np.cumsum(score_counts) / score_counts.sum()
    else:
        norm_counts = None
    return norm_counts


def compute_pmiss_pfa(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided to
    equalize the counts over score partitions (if there is such partitioning).
    """

    tgt_scores = scores[labels == 1]  # target trial scores
    imp_scores = scores[labels == 0]  # impostor trial scores

    resol = max(
        [np.count_nonzero(labels == 0),
         np.count_nonzero(labels == 1), 1.e6])
    edges = np.linspace(np.min(scores), np.max(scores), resol)

    if weights is not None:
        tgt_weights = weights[labels == 1]
        imp_weights = weights[labels == 0]
    else:
        tgt_weights = None
        imp_weights = None

    fnr = compute_norm_counts(tgt_scores, edges, tgt_weights)
    fpr = 1 - compute_norm_counts(imp_scores, edges, imp_weights)

    return fnr, fpr


def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided to
    equalize the counts over score partitions (if there is such partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr


def compute_eer(fnr, fpr, scores=None):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))

    if scores is not None:
        score_sort = np.sort(scores)
        return fnr[x1] + a * (fnr[x2] - fnr[x1]), score_sort[x1]

    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det / c_def


def compute_c_dcf(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))

    return c_det


def plot_det_curve(fnr, fpr, save_path=None):
    """ plots the detection error trade-off (DET) curve
    """

    p_miss = norm.ppf(fnr)
    p_fa = norm.ppf(fpr)

    xytick = [
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        0.2, 0.4
    ]
    xytick_labels = map(str, [x * 100 for x in xytick])

    plt.plot(p_fa, p_miss, 'r')
    plt.xticks(norm.ppf(xytick), xytick_labels)
    plt.yticks(norm.ppf(xytick), xytick_labels)
    plt.xlim(norm.ppf([0.00051, 0.5]))
    plt.ylim(norm.ppf([0.00051, 0.5]))
    plt.xlabel("false-alarm rate [%]", fontsize=12)
    plt.ylabel("false-reject rate [%]", fontsize=12)
    eer = compute_eer(fnr, fpr)
    plt.plot(norm.ppf(eer), norm.ppf(eer), 'o')
    plt.annotate(
        "EER = %.2f%%" % (eer * 100),
        xy=(norm.ppf(eer), norm.ppf(eer)),
        xycoords='data',
        xytext=(norm.ppf(eer + 0.05), norm.ppf(eer + 0.05)),
        textcoords='data',
        arrowprops=dict(arrowstyle="-|>",
                        connectionstyle="arc3, rad=+0.2",
                        fc="w"),
        size=12,
        va='center',
        ha='center',
        bbox=dict(boxstyle="round4", fc="w"),
    )
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def compute_equalized_scores(max_tar_imp_counts, sc, labs, masks):

    count_weights = []
    scores = []
    labels = []
    for ix in range(len(masks)):
        amask = masks[ix]
        alabs = labs[amask]
        num_targets = np.count_nonzero(alabs == 1)
        num_non_targets = alabs.size - num_targets
        labels.append(alabs)
        scores.append(sc[amask])
        tar_weight = max_tar_imp_counts[
            0] / num_targets if num_targets > 0 else 0
        imp_weight = max_tar_imp_counts[
            1] / num_non_targets if num_non_targets > 0 else 0

        acount_weights = np.empty(alabs.shape, dtype='f')
        acount_weights[alabs == 1] = np.array([tar_weight] * num_targets)
        acount_weights[alabs == 0] = np.array([imp_weight] * num_non_targets)
        count_weights.append(acount_weights)

    scores = np.hstack(scores)
    labels = np.hstack(labels)
    count_weights = np.hstack(count_weights)

    return scores, labels, count_weights
