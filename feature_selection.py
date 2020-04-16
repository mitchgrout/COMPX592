#!/usr/bin/env python3

import numpy
import ne
import os
import warnings
warnings.filterwarnings('ignore')

if __name__ ==  '__main__':
    DATA    = ne.data.kdd99
    LOG_DIR = 'results/feature_selection/{}/'.format(DATA.name())

    os.makedirs(LOG_DIR, exist_ok=True)
    with tee.StdoutTee('{}/output.log'.format(LOG_DIR), buff=1):
        ne.util.dump(__file__)
        
        it = ne.stats.feature_selection_PCA(DATA)
        results = []
        for (n, stats) in it:
            print("{} - {}".format(stats.mcc, n))
            results.append((n, stats))
        print(list(map(lambda t: t[0], sorted(results, key=lambda t:t[1].mcc, reverse=True))))

