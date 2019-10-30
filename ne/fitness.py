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

