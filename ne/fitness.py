"""
ne.fitness
    Exposes some common fitness functions
"""

def normalized(minimum, maximum, fitness_func):
    """
    Returns
    -------
    (int, int, ([x], [x]->bool))
    - min
    - max
    - fitness
    """

    assert minimum < maximum, '{} !< {}'.format(minimum, maximum)
    def __inner(true_list, pred_list):
        scores = fitness_func(true_list, pred_list)
        for s in scores:
            yield (s - minimum) / (maximum - minimum)
    return 0, 1, __inner

def zero_sum_builder(true_count, false_count, thresh):
    """
    Build a fitness function which is zero-sum in the case of constant
    predictions. Requires tagged data and approximate true/false counts.
    Assumes true_count < false_count (significantly)
    Parameters
    ----------
    thresh: x -> bool
        Typically something like lambda x: x > 0.0
        Classifies the output as a true or false rather than continuous
    Returns
    -------
    (int, int, ([x], [x] -> bool))
    - min
    - max
    - fitness
        
    """

    def __inner(true_list, pred_list):
        # Lookup is (thresh(true_val), thresh(pred_val))
        reward_table = {
            (True,  True):  +false_count/true_count,
            (True,  False): -false_count/true_count,
            (False, True):  -1.0,
            (False, False): +1.0,
        }
        score = 0.0
        yield score

        for true, pred in zip(true_list, pred_list):
            score += reward_table[thresh(true), thresh(pred)]
            yield score
    return -2*false_count, +2*false_count, __inner

def accuracy(thresh):
    """

    """

    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(true_ys, pred_ys):
        f_t, f_p = self.thresh(true), self.thresh(pred)
        if   f_t is True  and f_p is True:  tp += 1
        elif f_t is True  and f_p is False: fn += 1
        elif f_t is False and f_p is True: fp += 1
        elif f_t is False and f_p is False: tn += 1

    eps = lambda x: 1e-20 if x == 0 else x
    return (tp+tn)/eps(tp+tn+fp+fn)

def f1(thresh):
    """

    """

    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(true_ys, pred_ys):
        f_t, f_p = self.thresh(true), self.thresh(pred)
        if   f_t is True  and f_p is True:  tp += 1
        elif f_t is True  and f_p is False: fn += 1
        elif f_t is False and f_p is True: fp += 1
        elif f_t is False and f_p is False: tn += 1

    eps = lambda x: 1e-20 if x == 0 else x
    return 2*tp/eps(2*tp+fp+fn)

