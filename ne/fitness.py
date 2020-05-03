"""
ne.fitness
    Exposes some common fitness functions
"""

import ne.stats
from itertools import starmap

def _loss_to_metric(carry):
    return 1 / (1 + carry)

def _normalize(min_, max_, val):
    """
    Natural map from [min_, max_] -> [0, 1]
    """

    assert min_ < max_, '{} !< {}'.format(min_, max_)
    return (val - min_) / (max_ - min_)

def zero_sum(thresh):
    def helper(tp, tn, fp, fn):
        fc, tc = tn+fp, ne.stats.eps(tp+fn)
        r = fc/ne.stats.eps(tc)
        score = (tp-fn)*r + (tn-fp)
        return _normalize(-2*fc, +2*ne.stats.eps(fc), score) 
    return lambda true_ys, pred_ys: starmap(helper, ne.stats.score(thresh, true_ys, pred_ys))

def accuracy(thresh):
    return lambda true_ys, pred_ys: starmap(ne.stats.accuracy, ne.stats.score(thresh, true_ys, pred_ys))

def f1(thresh):
    return lambda true_ys, pred_ys: starmap(ne.stats.f1, ne.stats.score(thresh, true_ys, pred_ys))

def mcc(thresh):
    return lambda true_ys, pred_ys: starmap(ne.stats.mcc, ne.stats.score(thresh, true_ys, pred_ys))

def auroc(thresh):
    from sklearn.metrics import roc_auc_score
    # NOTE: We have to force pred_ys since it could be lazy; true_ys should never be
    return lambda true_ys, pred_ys: [ roc_auc_score(true_ys, list(pred_ys)) ]

def inverse_mean_square_error(thresh):
    def __inner(true_ys, pred_ys):
        clamp   = lambda x: min(max(x, 0), 1)
        count   = 0
        partial = 0 
        yield 0 
        for (true_y, pred_y) in zip(true_ys, pred_ys):
            true_y = clamp(true_y)
            pred_y = clamp(pred_y)
            partial += (true_y - pred_y) ** 2
            count   += 1
            yield _loss_to_metric(partial / count)
    return __inner

def inverse_binary_crossentropy(thresh):
    from math import log
    _eps = lambda e: 1e-10 if e == 0 else e

    def __inner(true_ys, pred_ys):
        clamp   = lambda x: min(max(x, 0), 1)
        count   = 0
        partial = 0
        yield 0 
        for (true_y, pred_y) in zip(true_ys, pred_ys):
            true_y = clamp(true_y)
            pred_y = clamp(pred_y)
            if thresh(true_y):
                partial += -log(_eps(pred_y))
            else:
                partial += -log(_eps(1-pred_y))
            count   += 1
            yield _loss_to_metric(partial / count)
    return __inner
