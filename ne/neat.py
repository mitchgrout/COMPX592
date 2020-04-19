"""
ne.neat
    Exposes an interface for using the NEAT algorithm (provided by neat-python)
"""

from ne.base        import Model
from ne.execute     import Sequential
from neat.reporting import BaseReporter
from ne.util        import benchmark

def create_config_file(**kwargs):
    import tempfile
    # neat has a bad API for doing this, so we're going to patch it
    default_config = {
        'NEAT': {
            'fitness_criterion':'max',
            'fitness_threshold': 1.1,
            'pop_size': 64,
            'reset_on_extinction': False,
        },

        'DefaultGenome': {
            'activation_default': 'identity',
            'activation_mutate_rate': 0.2,
            'activation_options': 'identity relu clamped tanh sigmoid',
            'aggregation_default': 'sum',
            'aggregation_mutate_rate': 0.2,
            'aggregation_options': 'sum',
            'bias_init_mean': 0.0,
            'bias_init_stdev': 1.0,
            'bias_max_value': 30.0,
            'bias_min_value': -30.0,
            'bias_mutate_power': 0.5,
            'bias_mutate_rate': 0.7,
            'bias_replace_rate': 0.1,
            'compatibility_disjoint_coefficient': 1.0,
            'compatibility_weight_coefficient': 0.5,
            'conn_add_prob': 0.5,
            'conn_delete_prob': 0.5,
            'enabled_default': True,
            'enabled_mutate_rate': 0.01,
            'feed_forward': True,
            'initial_connection': 'full_direct',
            'node_add_prob': 0.2,
            'node_delete_prob': 0.5,
            'num_hidden': 0,
            'num_inputs': 0,
            'num_outputs': 1,
            'response_init_mean': 1.0,
            'response_init_stdev': 0.0,
            'response_max_value': 30.0,
            'response_min_value': -30.0,
            'response_mutate_power': 0.0,
            'response_mutate_rate': 0.0,
            'response_replace_rate': 0.0,
            'weight_init_mean': 0.0,
            'weight_init_stdev': 1.0,
            'weight_max_value': 30.0,
            'weight_min_value': -30.0,
            'weight_mutate_power': 0.5,
            'weight_mutate_rate': 0.8,
            'weight_replace_rate': 0.1,
        },

        'DefaultSpeciesSet': {
            'compatibility_threshold': 3.0, 
        },

        'DefaultStagnation': {
            'species_fitness_func': 'max',
            'max_stagnation': 20,
            'species_elitism': 2,
        },

        'DefaultReproduction': {
            'elitism': 2,
            'survival_threshold': 0.2,
        },
    }

    config_file = tempfile.mktemp()
    with open(config_file, 'w') as fd:
        for section in default_config.keys():
            fd.write('[{}]\n'.format(section))
            for (key, value) in default_config[section].items():
                if key in kwargs:
                    value = kwargs[key]
                fd.write('{} = {}\n'.format(key, value))
    return config_file

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
def _batch(train, num_chunks, model_type, fitness, thresh, executor, genomes, config):
    from math import ceil
    global epoch
    l        = ceil(train.xs.shape[0] / num_chunks)
    offset   = (epoch % num_chunks) * l
    epoch   += 1
    
    train_xs = train.xs[offset : offset + l]
    train_ys = train.ys[offset : offset + l]
    
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
    def __init__(self, split_data, model_type, fitness, thresh, verbose):
        self.split_data   = split_data
        self.model_type   = model_type
        self.fitness      = fitness
        self.thresh       = thresh
        self.verbose      = verbose
        self.best_model   = None
        self.best_fitness = None

    def post_evaluate(self, config, _0, _1, best_genome):
        m    = self.model_type(self.fitness, self.thresh, best_genome, config)
        val  = self.split_data.val
        val_pred_ys  = list(m.predict(val.xs))
        *_, val_fitness  = m.evaluate(val.ys,  val_pred_ys)

        if self.best_fitness is None or val_fitness > self.best_fitness:
            self.best_model   = m
            self.best_fitness = val_fitness

        if self.verbose:
            print("Validation fitness:", val_fitness)
            # print("Validation stats:  ", stats)
       
def run(epochs, split_data, num_chunks, model_type, fitness, config_file, log_dir,
        executor=Sequential(), thresh=lambda x: x>0.5, verbose=True):

    import neat
    from functools import partial
    global epoch
    epoch = 0 # For re-running [if we ever do that]

    evaluator = partial(_batch, split_data.train, num_chunks, model_type, fitness, thresh, executor) 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    if verbose:
        population.add_reporter(neat.StdOutReporter(False))
    reporter = StatsReporter(split_data, model_type, fitness, thresh, verbose)
    population.add_reporter(reporter)
  
    # Overall winner != best training fitness
    population.run(evaluator, epochs)
    return reporter.best_model

