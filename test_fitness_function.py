#!/usr/bin/env python3

import numpy
import ne
import os
import tee
import tempfile

with open(__file__, 'r') as fd:
    FILE = fd.read()

def run_test(test_dataset, 
             test_selector,
             test_fitness,
             test_batch_size=128,
             test_epochs=16,
             test_pop_size=128):

    DATASET     = test_dataset
    SELECTOR    = test_selector
    FITNESS     = test_fitness
    BATCH_SIZE  = test_batch_size 
    EPOCHS      = test_epochs 
    CONFIG_FILE = ne.neat.create_config_file(num_inputs=SELECTOR.n_features, 
                                             pop_size=test_pop_size)
    MODEL_TYPE  = ne.neat.FeedForward    
    EXECUTOR    = ne.execute.Sequential()
    THRESH      = lambda x: x > 0.5
        
    log_dir = os.path.join('results', 'fitness_functions', FITNESS.__name__, DATASET.name())
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        print(FILE)
        print('log_dir:', log_dir)

        split = DATASET.data(selector=SELECTOR, save=False, cache=False)
        model = ne.neat.run(
                    epochs=EPOCHS,
                    split_data=split,
                    batch_size=BATCH_SIZE,
                    model_type=MODEL_TYPE,
                    fitness=FITNESS(THRESH),
                    config_file=CONFIG_FILE,
                    log_dir=log_dir, 
                    executor=EXECUTOR,
                    thresh=THRESH,
                    verbose=True)
        model.save(os.path.join(log_dir, 'model.pkl'))
        pred_ys = list(model.predict(split.test.xs))
        *_, f = model.evaluate(split.test.ys, pred_ys)
        stats = model.compute_statistics(split.test.ys, pred_ys)
        print('Test fitness:', f)
        print('Test statistics:', stats)

if __name__ == '__main__' and 0:
    dataset_pairs = [
        (ne.data.nsl_kdd.dataset,  ne.data.PCA(12)),
        (ne.data.unsw2015.dataset, ne.data.PCA(17)),
        (ne.data.ids2017.dataset,  ne.data.PCA(5)),
    ]

    fitness_funcs = [
        ne.fitness.accuracy,
        ne.fitness.f1,
        ne.fitness.mcc,
        ne.fitness.zero_sum,
        ne.fitness.inverse_mean_square_error,
        ne.fitness.inverse_binary_crossentropy,
    ]

    for (d, s) in dataset_pairs:
        for f in fitness_funcs:
            run_test(d, s, f)

