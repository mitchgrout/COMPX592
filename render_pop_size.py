import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd 
from parse import parse_neat
from os.path import join

def _loader(name, selector):
    fname_t = join('results', 'pop_size', name, selector, '{pop_size}', '{fold}', 'output.log')

    pop_sizes = []
    folds     = []
    scores    = []
    times     = []

    for pop_size in [16, 32, 64, 128, 256, 512, 1024]:
        for fold in range(10):
            import math
            r = parse_neat(fname_t.format(pop_size=pop_size, fold=fold))
            s = max(r['train']['val_fit'])
            t = r['output']['train_time']

            pop_sizes.append(pop_size)
            folds.append(fold)
            scores.append(s)
            times.append(math.log2(t))

    return {
        'pop_size': pop_sizes,
        'fold': folds,
        'score': scores,
        'log2time': times,
    }

nsl_kdd_pca = pd.DataFrame(_loader('nsl_kdd', 'pca'))

sn.set(context='paper', style='darkgrid', palette='muted')
f, axes = plt.subplots(2, 1, figsize=(7,7), sharex=True)

ax = sn.boxplot(x='pop_size', y='score', data=nsl_kdd_pca, ax=axes[0])
ax.set(xlabel='Population Size',
       ylabel='Validation Fitness',
       title='Population Size vs Validation Fitness')

ax = sn.boxplot(x='pop_size', y='log2time', data=nsl_kdd_pca, ax=axes[1])
ax.set(xlabel='Population Size',
       ylabel='log2(Training Time)',
       title='Population Size vs Logarithmic Training Time')

plt.show()
