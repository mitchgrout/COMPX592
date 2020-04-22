#!/usr/bin/env python3

import numpy
import ne
import os
import tee
import tempfile

def run_test(test_dataset, test_selector, test_fitness):
    DATASET     = test_dataset
    SELECTOR    = test_selector
    FITNESS     = test_fitness
    BATCH_SIZE  = 128 
    EPOCHS      = 16
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

if __name__ == '__main__':
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

    for p in dataset_pairs:
        for f in fitness_funcs:
            run_test(*p, f)

