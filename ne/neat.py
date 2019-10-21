"""
ne.neat
    Exposes an interface for using the NEAT algorithm (provided by neat-python)
"""

import neat
from neat.nn import FeedForwardNetwork, RecurrentNetwork
#from neat.nn import FeedForwardNetwork
#from tf_neat.recurrent_net import RecurrentNet as RecurrentNetwork

from ne.base import Model, Strategy, evaluate, predict, run_parallel
from ne.stats import compute_stats
from ne.visualize import draw_net

class FeedForward(Model):
    def __init__(self, model):
        self.base_model = model

    @classmethod
    def create(cls, genome, config):
        return cls(FeedForwardNetwork.create(genome, config))

    def activate(self, x):
        return self.base_model.activate(x)[0]


class Recurrent(Model):
    def __init__(self, model):
        self.base_model = model

    @classmethod
    def create(cls, genome, config):
        return cls(RecurrentNetwork.create(genome, config))

    def activate(self, x):
        return self.base_model.activate(x)[0]

    # def save, load render

class ContinuousRecurrent(Model):
    def __init__(self, model):
        self.base_model = model

    @classmethod
    def create(cls, genome, config):
        raise NotImplementedError()

    def activate(self, x):
        raise NotImplementedError()

def create_fitness_func(executor, fitness_func, model_type, sdata, slabels):
    def flatten(xss): return [x for xs in xss for x in xs]

    def __individual(genomes, config):
        def __inner(genome_id, genome):
            model = model_type.create(genome, config)
            *_, genome.fitness = evaluate(model, sdata, slabels, fitness_func)
            return genome.fitness
        results = executor.run(__inner, genomes)
        for (_, genome), fitness in zip(genomes, results):
            genome.fitness = fitness
        return results
 
    def __batch(genomes, config):
        def __inner(idx_start, idx_stop):
            results = []
            for genome_id, genome in genomes[idx_start : idx_stop]:
                model = model_type.create(genome, config)
                *_, genome.fitness = evaluate(model, sdata, slabels, fitness_func)
                results.append(genome.fitness)
            return results
        from math import ceil
        n = ceil(len(genomes) / executor.num_workers())
        indices = []
        for idx in range(0, len(genomes), n):
            indices.append((idx, idx+n))
        results = flatten(executor.run(__inner, indices))
        for (_, genome), fitness in zip(genomes, results):
            genome.fitness = fitness
        return results
 
 
    def __precomp_batch(genomes, config):
        models = []
        def __inner(idx_start, idx_stop):
            results = []
            for model in models[idx_start : idx_stop]:
                *_, genome.fitness = evaluate(model, sdata, slabels, fitness_func)
                results.append(genome.fitness)
            return results
        from math import ceil
        n = ceil(len(genomes) / executor.num_workers())
        indices = []
        for idx in range(0, len(genomes), n):
            indices.append((idx, idx+n))
        for genome_id, genome in genomes:
            models.append(model_type.create(genome, config))
        results = flatten(executor.run(__inner, indices))
        for (_, genome), fitness in zip(genomes, results):
            genome.fitness = fitness
        return results
 
    # NOTE: Choose your technique here
    return __batch

class NEAT(Strategy):
    """
    Train a population of models using the NEAT technique
    """

    def __init__(self, config_file, model_type, fitness_func, data, labels,
                       executor, val_data=None, val_labels=None, val_thresh=None):
        """
        Parameters
        ----------
        config_file: str
            Config file specifically for the neat-python package
        model_Type: Model
            One of ne.neat.{FeedForward|Recurrent|ContinuousRecurrent}
        fitness_func: (x, x) -> float
            A fitness function which takes (true, pred) and provides a score
        data: [x]
            Data to predict on
        labels: [y]
            Associated labels
        executor:
            Runs a function with a certain parallelisation strategy
        """

        self.config = neat.Config(
                        neat.DefaultGenome,
                        neat.DefaultReproduction,
                        neat.DefaultSpeciesSet,
                        neat.DefaultStagnation,
                        config_file)
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.model_type = model_type
        self.data       = data
        self.labels     = labels
        self.eval       = create_fitness_func(executor, fitness_func, model_type, self.data, self.labels)

        class RenderReporter(neat.reporting.BaseReporter):
            def __init__(self_):
                self_.idx = 0
            def post_evaluate(self_, config_, population_, species_, best_genome_):
                self_.idx += 1
                draw_net(config_, best_genome_, filename='render/best_network_{}'.format(self_.idx))
        self.population.add_reporter(RenderReporter())

        if not any(x is None for x in [val_data, val_labels, val_thresh]):
            class StatsReporter(neat.reporting.BaseReporter):
                def post_evaluate(self_, config_, population_, species_, best_genome_):
                    print("Validation statistics:",
                        compute_stats(
                            val_labels,
                            predict(model_type.create(best_genome_, config_), val_data),
                            val_thresh))
            self.population.add_reporter(StatsReporter())

    def train(self, epochs=None):
        """
        epochs: int
            Maximum number of rounds to train for. May stop early depending on config
        """

        winner = self.population.run(self.eval, epochs)
        model = self.model_type.create(winner, self.config)
        return model

