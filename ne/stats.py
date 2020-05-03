"""
ne.stats
    Basic statistical measures derived via confusion matrices
    All functions take (tp, tn, fp, fn) as arguments in that order
"""

from math import sqrt
from sys  import float_info
from collections import namedtuple

Stats = namedtuple('Stats',
        [ 'total', 'total_true', 'total_false',
          'true_positive', 'true_negative', 'false_positive', 'false_negative',
          'tpr', 'tnr', 'ppv', 'npv',
          'accuracy', 'f1', 'mcc' ])
 
eps = lambda x: x if x else float_info.epsilon

total       = lambda tp, tn, fp, fn: tp+tn+fp+fn
total_true  = lambda tp, tn, fp, fn: fn+tp
total_false = lambda tp, tn, fp, fn: tn+fp

true_positive  = lambda tp, tn, fp, fn: tp
true_negative  = lambda tp, tn, fp, fn: tn
false_positive = lambda tp, tn, fp, fn: fp
false_negative = lambda tp, tn, fp, fn: fn

tpr  = lambda tp, tn, fp, fn: tp/eps(tp+fn)
tnr  = lambda tp, tn, fp, fn: tn/eps(tn+fp)
ppv  = lambda tp, tn, fp, fn: tp/eps(tp+fp)
npv  = lambda tp, tn, fp, fn: tn/eps(tn+fn)

accuracy = lambda tp, tn, fp, fn: (tp+tn)/eps(tp+tn+fp+fn)
f1       = lambda tp, tn, fp, fn: 2*tp/eps(2*tp+fp+fn)
mcc      = lambda tp, tn, fp, fn: ((tp*tn)-(fp*fn))/eps(sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

def score(thresh, true_ys, pred_ys):
    tp, tn, fp, fn = 0, 0, 0, 0
    yield (tp, tn, fp, fn)
    for true, pred in zip(true_ys, pred_ys):
        f_t, f_p = thresh(true), thresh(pred)
        if   f_t == True  and f_p == True:  tp += 1
        elif f_t == True  and f_p == False: fn += 1
        elif f_t == False and f_p == True:  fp += 1
        elif f_t == False and f_p == False: tn += 1
        yield (tp, tn, fp, fn) 

def compute_statistics(thresh, true_ys, pred_ys):
    *_, t = score(thresh, true_ys, pred_ys)
    return Stats(total=total(*t)
                ,total_true=total_true(*t)
                ,total_false=total_false(*t)
                ,true_positive=true_positive(*t)
                ,true_negative=true_negative(*t)
                ,false_positive=false_positive(*t)
                ,false_negative=false_negative(*t)
                ,tpr=tpr(*t)
                ,tnr=tnr(*t)
                ,ppv=ppv(*t)
                ,npv=npv(*t)
                ,accuracy=accuracy(*t)
                ,f1=f1(*t)
                ,mcc=mcc(*t))

def feature_selection_PCA(mod, thresh=lambda x:x>0.5):
    import ne
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes   import GaussianNB

    data = mod.load_data()
    for x in range(data.xs.shape[1]):
        data.xs[:, x] = _normalize(data.xs[:, x])
    split = ne.data.make_split(data)

    for n in range(1, data.xs.shape[1]):
        pca = PCA(n_components=n).fit(data.xs, data.ys)

        train_xs = pca.transform(split.train.xs)
        test_xs  = pca.transform(split.test.xs)

        model = GaussianNB(var_smoothing=1e-15)
        model.fit(train_xs, split.train.ys)
        yield (n, #feats.transform(cols), 
               compute_statistics(thresh, split.test.ys, model.predict(test_xs)))


def feature_selection_pearson(mod, thresh=lambda x:x>0.5):
    import ne
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif#, f_regression
    from sklearn.naive_bayes       import GaussianNB

    data = mod.load_data()
    cols = np.asarray(list(range(data.xs.shape[1]))).reshape((1, -1))
    for x in range(data.xs.shape[1]):
        data.xs[:, x] = _normalize(data.xs[:, x])
    split = ne.data.make_split(data)

    for k in range(1, data.xs.shape[1]):
        feats    = SelectKBest(score_func=f_classif, k=k).fit(data.xs, data.ys)

        train_xs = feats.transform(split.train.xs)
        test_xs  = feats.transform(split.test.xs)

        model = GaussianNB(var_smoothing=1e-15)
        model.fit(train_xs, split.train.ys)
        yield (feats.transform(cols), 
               compute_statistics(thresh, split.test.ys, model.predict(test_xs)))

def _normalize(arr):
    import numpy as np
    mu    = np.mean(arr)
    sigma = np.std(arr)
    return (arr - mu) / eps(sigma)

