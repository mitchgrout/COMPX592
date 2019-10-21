"""
ne.stats
    Compute rudimentary statistics for tagged data
"""

def compute_stats(true_list, pred_list, thresh):
    from collections import namedtuple
    Stats = namedtuple( 'Stats', 
        [ 'total', 'total_true', 'total_false',
          'true_positive', 'true_negative', 'false_positive', 'false_negative', 
          'recall', 'precision', 'accuracy', 'f1'])
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(true_list, pred_list):
        if thresh(true) and thresh(pred):       tp += 1 
        elif thresh(true) and not thresh(pred): fn += 1
        elif not thresh(true) and thresh(pred): fp += 1
        else:                                   tn += 1
    eps = lambda x: 1e-20 if x == 0 else x
    return Stats(total=tp+tn+fp+fn
                ,total_true=fn+tp
                ,total_false=tn+fp
                ,true_positive=tp
                ,true_negative=tn
                ,false_positive=fp
                ,false_negative=fn
                ,recall=tp/eps(tp+fn)
                ,precision=tp/eps(tp+fp)
                ,accuracy=(tp+tn)/eps(tp+tn+fp+fn)
                ,f1=2*tp/eps(2*tp+fp+fn)
                )
