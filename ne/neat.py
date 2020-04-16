"""
ne.neat
    Exposes an interface for using the NEAT algorithm (provided by neat-python)
"""

from ne.base        import Model
from ne.execute     import Sequential
#from ne.visualize   import draw_net
from neat.reporting import BaseReporter
from ne.util        import benchmark

def _render_neat_model(base_model, label_names, filename):
    import graphviz
    dot = graphviz.Digraph(format='svg', 
                           node_attr={
                            'shape': 'circle',
                            'fontsize': '9',
                            'height': '0.2',
                            'width': '0.2' })

    for k, l in zip(base_model.input_nodes, label_names):
        dot.node(l, _attributes={
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

# Hacky, but I guess it works
epoch = 0
def _batch(train, num_splits, model_type, fitness, thresh, executor, genomes, config):
    global epoch
    epoch   += 1
    l        = len(train.xs) / num_splits
    train_xs = train.xs#[int(l*(epoch-1)):int(l*epoch)] # NOTE: Small segs instead of growing in size
    train_ys = train.ys#[int(l*(epoch-1)):int(l*epoch)] # NOTE: 
    
    def flatten(xss): return [x for xs in xss for x in xs]
    def _inner(s_genomes):
        results = []
        for genome_id, genome in s_genomes:
            m     = model_type(fitness, thresh, genome, config)
            *_, f = m.evaluate(train_ys, m.predict(train_xs))
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
    def __init__(self, split_data, model_type, fitness, thresh):
        self.split_data = split_data
        self.model_type = model_type
        self.fitness    = fitness
        self.thresh     = thresh

    def post_evaluate(self, config, _0, _1, best_genome):
        m    = self.model_type(self.fitness, self.thresh, best_genome, config)
        #train= self.split_data.train
        val  = self.split_data.val
        test = self.split_data.test

        val_pred_ys  = list(m.predict(val.xs))
        test_pred_ys = list(m.predict(test.xs))

         #*_, train_fitness = m.evaluate(train.ys, m.predict(train.xs))
        *_, val_fitness  = m.evaluate(val.ys,  val_pred_ys)
        *_, test_fitness = m.evaluate(test.ys, test_pred_ys)

        # print("Train fitness", train_fitness)
        print("Validation fitness:", val_fitness)
        print("Test fitness      :", test_fitness)

        # print("Train      :", benchmark(lambda:m.compute_statistics(train.ys,m.predict(train.xs))))
        print("Validation stats:", m.compute_statistics(val.ys , val_pred_ys ))
        print("Test stats      :", m.compute_statistics(test.ys, test_pred_ys))
       

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


def run(epochs, split_data, model_type, fitness, config_file, log_dir,
        executor=Sequential(), thresh=lambda x: x>0.5, train_ratio=0.8,
        num_splits=None):
    """
    """
    
    import neat
    from functools import partial

    if num_splits == None: num_splits = epochs
    assert num_splits != None

    evaluator = partial(_batch, split_data.train, num_splits, model_type, fitness, thresh, executor) 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(False))
    population.add_reporter(StatsReporter(split_data, model_type, fitness, thresh))
    population.add_reporter(ModelSaver(log_dir, model_type, fitness, thresh))
  
    winner = population.run(evaluator, epochs)

