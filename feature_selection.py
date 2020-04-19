#!/usr/bin/env python3

import numpy
import ne
import os
import tee
import warnings
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings('ignore')

if __name__ ==  '__main__':
    DATASET  = ne.data.kdd99.dataset
    SELECTOR = ne.data.Pearsons
    THRESH   = lambda x: x > 0.5
    
    log_dir = os.path.join('results', 'feature_selection', SELECTOR.name, DATASET.name())
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        ne.util.dump(__file__)

        results = {}
        for n_features in range(1, DATASET.num_features()):
            split = DATASET.data(selector=SELECTOR(n_features), save=False, cache=True)
            model = GaussianNB(var_smoothing=1e-15)
            model.fit(*split.train)
            stats = ne.stats.compute_statistics(THRESH, split.test.ys, model.predict(split.test.xs))
            results[n_features] = stats
            print("{} - {}".format(n_features, stats.mcc))
        print(list(map(lambda t:t[0], sorted(results.items(), key=lambda t:t[1].mcc, reverse=True))))

