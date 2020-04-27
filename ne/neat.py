"""
ne.neat
    Exposes an interface for using the NEAT algorithm (provided by neat-python)
"""

from ne.base        import Model
from ne.execute     import Sequential
from neat.reporting import BaseReporter
from time           import time
from os.path        import join

def create_config_file(**kwargs):
    import tempfile
    # neat has a bad API for doing this, so we're going to patch it
    default_config = {
        'NEAT': {
            'fitness_criterion'         : 'max',
            'fitness_threshold'         : 1.1,
            'pop_size'                  : 256,
            'reset_on_extinction'       : True,
        },

        'DefaultGenome': {
            'activation_default'        : 'random',
            'activation_mutate_rate'    : 0.3,
            'activation_options'        : 'identity relu clamped tanh sigmoid',

            'aggregation_default'       : 'random',
            'aggregation_mutate_rate'   : 0.2,
            'aggregation_options'       : 'sum',

            'bias_init_mean'            : 0.0,
            'bias_init_stdev'           : 1.0,
            'bias_max_value'            : 30.0,
            'bias_min_value'            : -30.0,
            'bias_mutate_power'         : 0.5,
            'bias_mutate_rate'          : 0.7,
            'bias_replace_rate'         : 0.1,

            'compatibility_disjoint_coefficient': 1.0,
            'compatibility_weight_coefficient'  : 0.6,

            'conn_add_prob'             : 0.7,
            'conn_delete_prob'          : 0.3,
            'enabled_default'           : True,
            'enabled_mutate_rate'       : 0.01,
            'feed_forward'              : True,
            'initial_connection'        : 'full',

            'node_add_prob'             : 0.6,
            'node_delete_prob'          : 0.4,
            'num_hidden'                : 0,
            'num_inputs'                : 0,
            'num_outputs'               : 1,

            'response_init_mean'        : 1.0,
            'response_init_stdev'       : 0.0,
            'response_max_value'        : 30.0,
            'response_min_value'        : -30.0,
            'response_mutate_power'     : 0.0,
            'response_mutate_rate'      : 0.0,
            'response_replace_rate'     : 0.0,

            'weight_init_mean'          : 0.0,
            'weight_init_stdev'         : 1.0,
            'weight_max_value'          : 30.0,
            'weight_min_value'          : -30.0,
            'weight_mutate_power'       : 0.5,
            'weight_mutate_rate'        : 0.8,
            'weight_replace_rate'       : 0.1,
        },

        'DefaultSpeciesSet': {
            'compatibility_threshold'   : 2.5, 
        },

        'DefaultStagnation': {
            'species_fitness_func'      : 'mean',
            'max_stagnation'            : 15,
            'species_elitism'           : 3,
        },

        'DefaultReproduction': {
            'elitism'                   : 2,
            'survival_threshold'        : 0.2,
            'min_species_size'          : 3,
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
perm  = None
def _batch(train, batch_size, model_type, fitness, thresh, executor, genomes, config):
    import numpy
    from math import ceil
    global epoch
    global perm
    num_chunks = ceil(train.xs.shape[0] / batch_size)
    offset     = ((epoch % num_chunks) * batch_size)
    if (epoch % num_chunks) == 0:
        perm = numpy.random.permutation(train.xs.shape[0])

    train_xs = train.xs[perm][offset : offset + batch_size]
    train_ys = train.ys[perm][offset : offset + batch_size]
    epoch   += 1
   
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
    def __init__(self, log_dir, split_data, num_chunks, model_type, fitness, thresh, verbose):
        self.log_dir      = log_dir
        self.split_data   = split_data
        self.num_chunks   = num_chunks
        self.model_type   = model_type
        self.fitness      = fitness
        self.thresh       = thresh
        self.verbose      = verbose
        self.last_time    = time()
        self.best_model   = None
        self.best_fitness = None

    def post_evaluate(self, config, population, species, best_genome):
        global epoch

        _epoch = 1 + ((epoch-1) // self.num_chunks)
        _batch = 1 + ((epoch-1) %  self.num_chunks)

        # Check validation *every* batch
        m    = self.model_type(self.fitness, self.thresh, best_genome, config)
        val  = self.split_data.val
        val_pred_ys  = list(m.predict(val.xs))
        *_, val_fitness  = m.evaluate(val.ys,  val_pred_ys)

        indicator = ''
        if self.best_fitness is None or val_fitness > self.best_fitness:
            self.best_model   = m
            self.best_fitness = val_fitness
            m.save(join(self.log_dir, 'model.{}.{}.pkl'.format(_epoch, _batch)))
            indicator = '!!!'

        if self.verbose:
            print('*** Epoch {}, batch {}/{} ***'.format(_epoch, _batch, self.num_chunks))
            print('Training fitness:', best_genome.fitness)
            print('Model size:', best_genome.size())
            print('Validation fitness:', val_fitness, indicator)
            new_time = time()
            print('Elapsed:', new_time - self.last_time) 
            self.last_time = new_time
            print()

def run(epochs, split_data, batch_size, model_type, fitness, config_file, log_dir,
        executor=Sequential(), thresh=lambda x: x>0.5, verbose=True):

    import neat
    from functools import partial
    from math import ceil
    global epoch
    epoch = 0 # For re-running [if we ever do that]

    if batch_size is None:
        batch_size = split_data.train.xs.shape[0]
    num_chunks = ceil(split_data.train.xs.shape[0] / batch_size)
    epochs *= num_chunks
    evaluator = partial(_batch, split_data.train, batch_size, model_type, fitness, thresh, executor) 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    reporter = StatsReporter(log_dir, split_data, num_chunks, model_type, fitness, thresh, verbose)
    population.add_reporter(reporter)
  
    try:
        # Overall winner != best training fitness
        population.run(evaluator, epochs)
    except neat.CompleteExtinctionException:
        # This could happen depending on the config
        pass
    return reporter.best_model

