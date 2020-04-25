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
        import ne.stats
        return ne.stats.compute_statistics(self.thresh, true_ys, pred_ys)

    def save(self, filename):
        from dill import dump, HIGHEST_PROTOCOL
        with open(filename, 'wb') as fd:
            dump(self, fd, HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        from dill import load, HIGHEST_PROTOCOL
        with open(filename, 'rb') as fd:
            obj = load(fd, HIGHEST_PROTOCOL)
        return obj

