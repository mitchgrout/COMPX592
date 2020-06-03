import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd 
from parse import *
from glob import glob
from os.path import join

def _loader(name, selector):
    fname_t = join('results', 'comparison', name, selector, '{fold}', '{model}', 'output.log')

    cases = {
        'neat': {
            'name': 'NEAT',
            'loader': parse_neat,
        },
        'shallow_neural': {
            'name': 'Shallow',
            'loader': parse_keras,
        },
        'deep_neural': {
            'name': 'Deep',
            'loader': parse_keras,
        },
        'decision_tree': {
            'name': 'Tree',
            'loader': parse_sklearn,
        },
    }

    models = []
    folds  = []
    scores = []
    times  = []
    inftimes = []

    for model in cases.keys():
        for fold in range(10):
            import math
            r = cases[model]['loader'](fname_t.format(model=model, fold=fold))
            s = r['output']['test_stats'].mcc
            t = r['output']['train_time']

            models.append(cases[model]['name'])
            folds.append(fold)
            scores.append(s)
            times.append(math.log2(t))
            inftimes.append(math.log2( r['output']['test_time'] ))

    return {
        'model':       models,
        'fold':        folds,
        'score':       scores,
        'log2time':    times,
        'log2inftime': inftimes,
    }

nsl_kdd_pca = pd.DataFrame(_loader('nsl_kdd', 'pca'))
#unsw_nb15_pca = pd.DataFrame(_loader('unsw2015', 'pca'))
#ids_2017_pca = pd.DataFrame(_loader('ids2017', 'pca'))

sn.set(context='paper', style='darkgrid', palette='muted')

# for df in [ nsl_kdd_pca, unsw_nb15_pca, ids_2017_pca ]:
for name, df in [ ('NSL KDD', nsl_kdd_pca) ]:
    f, axes = plt.subplots(3, 1, figsize=(7,7), sharex=True)

    ax = sn.boxplot(x='model', y='score', data=df, ax=axes[0])
    ax.set(xlabel='Model',
           ylabel='Test MCC',
           title='Test MCC for {}'.format(name))

    ax = sn.boxplot(x='model', y='log2time', data=df, ax=axes[1])
    ax.set(xlabel='Model',
           ylabel='log2(Training Time)',
           title='Logarithmic Training Time for {}'.format(name))

    ax = sn.boxplot(x='model', y='log2inftime', data=df, ax=axes[2])
    ax.set(xlabel='Model',
           ylabel='log2(Inference Time)',
           title='Logarithmic Inference Time for {}'.format(name))

    plt.show()
