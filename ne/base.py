"""
ne.base
    Provides general base classes and general functions for using models
"""


class Model(object):
    """
    Abstraction of a model. Primary use is for evaluating inputs, but also may
    allow for rendering, saving, and loading
    """

    def activate(self, x):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


class Strategy(object):
    """
    Abstraction of an evolutionary strategy. Allows for training up to a certain
    number of epochs
    """

    def train(self, epochs=None):
        raise NotImplementedError()


def evaluate(model, xs, ys, fitness):
    """
    Parameters
    ----------
    model: Model
    xs: [x]
    ys: [y]
    fitness: 
    """

    return fitness(ys, predict(model, xs))


def predict(model, xs):
    """
    Parameters
    ----------
    model: Model
    xs: [x]
    """

    return map(model.activate, xs)


def run_parallel(pool_size, fn, arglist):
    """
    Parameters
    ----------
    pool_size: int
    fn: Callable
    arglist: [...]
    """

    from multiprocess import Pool
    pool = Pool(processes=pool_size)
    return pool.starmap(fn, arglist)
