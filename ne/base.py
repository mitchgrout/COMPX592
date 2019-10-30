"""
ne.base
    Provides general base classes and general functions for using models
    Allows for easier integration of other methods if needed 
"""

class Model(object):
    """
    Abstraction of a model. Primary use is for evaluating inputs, but also may
    allow for rendering, saving, and loading
    Should maintain information about:
    - Neurons
        - Biases
        - Activations
    - Connections
        - Weights
    - Fitness+thresholding function
    """

    def __init__(self, fitness, thresh):
        self.fitness = fitness
        self.thresh  = thresh

    def activate(self, x):
        raise NotImplementedError()

    def get_topology(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def predict(self, xs):
        return map(self.activate, xs)

    def evaluate(self, true_ys, pred_ys):
        return self.fitness(true_ys, pred_ys)

    def compute_statistics(self, true_ys, pred_ys):
        from collections import namedtuple
        Stats = namedtuple( 'Stats', 
            [ 'total', 'total_true', 'total_false',
              'true_positive', 'true_negative', 'false_positive', 'false_negative', 
              'tpr', 'tnr', 'ppv', 'npv',
              'accuracy', 'f1'])
        
        tp, tn, fp, fn = 0, 0, 0, 0
        for true, pred in zip(true_ys, pred_ys):
            f_t, f_p = self.thresh(true), self.thresh(pred)
            if   f_t is True  and f_p is True:  tp += 1
            elif f_t is True  and f_p is False: fn += 1
            elif f_t is False and f_p is True:  fp += 1
            elif f_t is False and f_p is False: tn += 1

        eps = lambda x: 1e-20 if x == 0 else x
        return Stats(total=tp+tn+fp+fn
                    ,total_true=fn+tp
                    ,total_false=tn+fp
                    ,true_positive=tp
                    ,true_negative=tn
                    ,false_positive=fp
                    ,false_negative=fn
                    ,tpr=tp/eps(tp+fn)
                    ,tnr=tn/eps(tn+fp)
                    ,ppv=tp/(tp+fp)
                    ,npv=tn/(tn+fn)
                    ,accuracy=(tp+tn)/eps(tp+tn+fp+fn)
                    ,f1=2*tp/eps(2*tp+fp+fn))

    def save(self, filename):
        from dill import dump
        with open(filename, 'wb') as fd:
            dump(self, fd)

    @classmethod
    def load(cls, filename):
        from dill import load
        with open(filename, 'rb') as fd:
            obj = load(fd)
        return obj


class Strategy(object):
    """
    Abstraction of an evolutionary strategy. Allows for training up to a certain
    number of epochs
    """

    def train(self, epochs):
        raise NotImplementedError()

