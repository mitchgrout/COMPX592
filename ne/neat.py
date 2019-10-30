"""
ne.neat
    Exposes an interface for using the NEAT algorithm (provided by neat-python)
"""

from ne.base      import Model, Strategy
from ne.visualize import draw_net

# Model wrappers
class FeedForward(Model):
    def __init__(self, fitness, thresh, genome, config):
        from neat.nn import FeedForwardNetwork
        super(FeedForward, self).__init__(fitness, thresh)
        self.base_model = FeedForwardNetwork.create(genome, config)

    def activate(self, x):
        return self.base_model.activate(x)[0]


class Recurrent(Model):
    def __init__(self, fitness, thresh, genome, config):
        from neat.nn import RecurrentNetwork
        super(Recurrent, self).__init__(fitness, thresh)
        self.base_model = RecurrentNetwork.create(genome, config)

    def activate(self, x):
        return self.base_model.activate(x)[0]


class ContinuousRecurrent(Model):
    pass


class NEAT(Strategy):
    def __init__(self, log_dir, config_file, model_type, fitness_func, data, labels,
            executor, thresh, num_splits, delay):
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

        import neat
        from neat.reporting import BaseReporter
        self.model_type = model_type
        self.config     = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                      config_file)
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.thresh  = thresh
        self.fitness = fitness_func
 
        self.data         = data
        self.labels       = labels
        self.num_splits   = num_splits
        self.delay        = delay
        self.train_data   = []
        self.train_labels = []
        self.val_data     = []
        self.val_labels   = []


        def __batch(genomes, config):
            def flatten(xss): return [x for xs in xss for x in xs]
            def __inner(idx_start, idx_stop):
                results = []
                for genome_id, genome in genomes[idx_start : idx_stop]:
                    m = model_type(self.fitness, self.thresh, genome, config)
                    *_, genome.fitness = \
                            m.evaluate(self.train_labels,
                                       m.predict(self.train_data))
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
        self.eval = __batch

        # Reporter objects
        class StatsReporter(BaseReporter):
            """ Run statistics on validation data using the best model """
            def post_evaluate(self_, config_, population_, species_, best_genome_):
                m = model_type(self.fitness, self.thresh, best_genome_, config_)
                print("Validation statistics:",
                        m.compute_statistics(self.val_labels, m.predict(self.val_data)))
       
        class ModelSaver(BaseReporter):
            def __init__(self_):
                import os
                self_.dir = '{}/models'.format(log_dir)
                os.makedirs(self_.dir, exist_ok=True)

            def start_generation(self_, gen):
                self_.gen = gen

            def post_evaluate(self_, config_, population_, species_, best_genome_):
                m = self.model_type(self.fitness, self.thresh, best_genome_, config_)
                m.save('{}/{}.pkl'.format(self_.dir, self_.gen))

        class DataUpdater(BaseReporter):
            def __init__(self_):
                from sklearn.model_selection import TimeSeriesSplit
                self_.indices = list(TimeSeriesSplit(n_splits=self.num_splits or epochs+1).split(self.data))

            def start_generation(self_, generation):
                print('Updating data for generation {}'.format(generation))
                train_index, test_index = self_.indices[0]
                train_index = train_index[-1]
                test_index  = test_index[-1]
                print('Training on 0:{} items, validating on {}:{}'.format(train_index, train_index, test_index))
        
                if len(self_.indices) > 1 and generation % (self.delay or 1) == 0:
                    self_.indices = self_.indices[1:]
        
                self.train_data   = self.data[:train_index]
                self.train_labels = self.labels[:train_index]
                self.val_data     = self.data[train_index:test_index]
                self.val_labels   = self.labels[train_index:test_index]

        self.population.add_reporter(StatsReporter())
        self.population.add_reporter(ModelSaver())
        self.population.add_reporter(DataUpdater())

    def train(self, epochs):
        """
        epochs: int
            Maximum number of rounds to train for. May stop early depending on config
        """

        winner = self.population.run(self.eval, epochs)

