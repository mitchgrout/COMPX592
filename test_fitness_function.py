#!/usr/bin/env python3

import numpy
import ne
import os
import tee
import tempfile

if __name__ == '__main__':
    DATASET     = ne.data.nsl_kdd.dataset
    SELECTOR    = ne.data.PCA(12)
    FITNESS     = ne.fitness.accuracy
    NUM_CHUNKS  = 1
    EPOCHS      = 64 * NUM_CHUNKS
    CONFIG_FILE = ne.neat.create_config_file(num_inputs=SELECTOR.n_features)
    MODEL_TYPE  = ne.neat.FeedForward    
    EXECUTOR    = ne.execute.Sequential()
    THRESH      = lambda x: x > 0.5
        
    log_dir = os.path.join('results', 'fitness_functions', FITNESS.__name__, DATASET.name())
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        ne.util.dump(__file__)

        split = DATASET.data(selector=SELECTOR, save=False, cache=False)
        model = ne.neat.run(
                    epochs=EPOCHS,
                    split_data=split,
                    num_chunks=NUM_CHUNKS,
                    model_type=MODEL_TYPE,
                    fitness=FITNESS(THRESH),
                    config_file=CONFIG_FILE,
                    log_dir=log_dir,
                    executor=EXECUTOR,
                    thresh=THRESH,
                    verbose=True)
        model.save(os.path.join(log_dir, 'model.pkl'))
        stats = ne.stats.compute_statistics(THRESH, split.test.ys, model.predict(split.test.xs))
        print('Test statistics: ', stats)

