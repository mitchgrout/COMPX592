"""
ne.neat
    Exposes an interface for using the NEAT algorithm (provided by neat-python)
"""

from ne.base        import Model, Strategy
from ne.execute     import Sequential
#from ne.visualize   import draw_net
from neat.reporting import BaseReporter


def _render_neat_model(base_model, filename):
    import graphviz
    dot = graphviz.Digraph(format='svg', 
                           node_attr={
                            'shape': 'circle',
                            'fontsize': '9',
                            'height': '0.2',
                            'width': '0.2' })

    for k in base_model.input_nodes:
        dot.node(str(k), _attributes={
            'style': 'filled', 
            'shape': 'box',
            'fillcolor': 'lightgray'
            })

    for t in base_model.node_evals:
        out = t[0] in base_model.output_nodes
        dot.node(name=str(t[0]),
                label='f={}\nb={}'.format(t[1].__name__, t[3]),
                _attributes={
                    'style': 'filled',
                    'shape': 'box',
                    'fillcolor': 'lightblue' if out else 'white'
                })

        for edge in t[5]:
            dot.edge(tail_name=str(edge[0]),
                     head_name=str(t[0]),
                     label='w={}'.format(edge[1]),
                     _attributes={
                         'style': 'solid',
                         'color': 'green' if edge[1]>0.0 else 'red',
                         'width': '0.5'
                     })
    dot.render(filename)


# Model wrappers
class FeedForward(Model):
    def __init__(self, fitness, thresh, genome, config):
        from neat.nn import FeedForwardNetwork
        super(FeedForward, self).__init__(fitness, thresh)
        self.base_model = FeedForwardNetwork.create(genome, config)

    def activate(self, x):
        return self.base_model.activate(x)[0]

    def render(self, filename):
        _render_neat_model(self.base_model, filename)


class Recurrent(Model):
    def __init__(self, fitness, thresh, genome, config):
        from neat.nn import RecurrentNetwork
        super(Recurrent, self).__init__(fitness, thresh)
        self.base_model = RecurrentNetwork.create(genome, config)

    def activate(self, x):
        return self.base_model.activate(x)[0]

    def render(self, filename):
        _render_neat_model(self.base_model, filename)


def _batch_old(train, model_type, fitness, thresh, executor, genomes, config):
    def flatten(xss): return [x for xs in xss for x in xs]
    def _inner(idx_start, idx_stop):
        results = []
        for genome_id, genome in genomes[idx_start : idx_stop]:
            m     = model_type(fitness, thresh, genome, config)
            *_, f = m.evaluate(train[1], m.predict(train[0]))
            results.append(f)
        return results
    from math import ceil
    n = ceil(len(genomes) / executor.num_workers())
    indices = []
    for idx in range(0, len(genomes), n):
        indices.append((idx, idx+n))
    results = flatten(executor.run(_inner, indices))
    for (_, genome), fitness in zip(genomes, results):
        genome.fitness = fitness
    return results
 
def _batch(train, model_type, fitness, thresh, executor, genomes, config):
    def flatten(xss): return [x for xs in xss for x in xs]
    def _inner(s_genomes):
        results = []
        for genome_id, genome in s_genomes:
            m     = model_type(fitness, thresh, genome, config)
            *_, f = m.evaluate(train[1], m.predict(train[0]))
            results.append(f)
        return results
    from math import ceil
    n = ceil(len(genomes) / executor.num_workers())
    splits = []
    for idx in range(0, len(genomes), n):
        splits.append((genomes[idx:idx+n],))
    results = flatten(executor.run(_inner, splits))
    for (_, genome), fitness in zip(genomes, results):
        genome.fitness = fitness
    return results


class StatsReporter(BaseReporter):
    def __init__(self, train, val, model_type, fitness, thresh):
        self.train      = train
        self.val        = val
        self.model_type = model_type
        self.fitness    = fitness
        self.thresh     = thresh

    def post_evaluate(self, config, _0, _1, best_genome):
        m = self.model_type(self.fitness, self.thresh, best_genome, config)
        print("Training statistics:",
                m.compute_statistics(self.train[1], m.predict(self.train[0])))
        print("Validation statistics:",
                m.compute_statistics(self.val[1], m.predict(self.val[0])))
       

class ModelSaver(BaseReporter):
    def __init__(self, log_dir, model_type, fitness, thresh):
        import os
        self.model_type = model_type
        self.fitness    = fitness
        self.thresh     = thresh
        self.dir        = '{}/models'.format(log_dir)
        os.makedirs(self.dir, exist_ok=True)
        
    def start_generation(self, gen):
        self.gen = gen

    def post_evaluate(self, config, _0, _1, best_genome):
        m = self.model_type(self.fitness, self.thresh, best_genome, config)
        m.save('{}/{}.pkl'.format(self.dir, self.gen))


# class PopulationSaver(BaseReporter):


class DataUpdater(BaseReporter):
    def __init__(self, data, train, val, num_splits, delay):
        from sklearn.model_selection import TimeSeriesSplit as TSS
        self.data    = data
        self.train   = train
        self.val     = val
        self.delay   = delay
        self.indices = list(TSS(n_splits=num_splits).split(data.xs))

    def start_generation(self, gen):
        if len(self.indices) > 1 and gen > 0 and (gen % self.delay) == 0:
            self.indices = self.indices[1:]
        train_index, test_index = self.indices[0]
        train_index = train_index[-1]
        test_index  = test_index[-1]
        print('Training on 0:{} items, validating on {}:{}'.format(
                train_index, train_index, test_index))
        self.train[0] = self.data.xs[:train_index]
        self.train[1] = self.data.ys[:train_index]
        self.val[0]   = self.data.xs[train_index:test_index]
        self.val[1]   = self.data.ys[train_index:test_index]


def run(epochs, data, model_type, fitness, config_file, log_dir,
        executor=Sequential(), thresh=lambda x: x>0.5, num_splits=None, delay=None):
    """
    """
    
    import neat
    from functools import partial

    if (num_splits, delay) == (None, None):
        num_splits = epochs
        delay      = 1
    assert num_splits != None
    assert delay      != None

    # Splits of the data object
    train     = [[], []]
    val       = [[], []]
    evaluator = partial(_batch, train, model_type, fitness, thresh, executor)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(StatsReporter(train, val, model_type, fitness, thresh))
    population.add_reporter(ModelSaver(log_dir, model_type, fitness, thresh))
    population.add_reporter(DataUpdater(data, train, val, num_splits, delay))
  
    winner = population.run(evaluator, epochs)

