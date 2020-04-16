"""
ne.fitness
    Exposes some common fitness functions
"""

import ne.stats
from itertools import starmap

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
