import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd 
from parse import parse_neat
from os.path import join

def _loader(name, selector):
    fname_t = join('results', 'epochs', name, selector, '{fold}', 'output.log')
    
    epochs = []
    folds  = []
    scores = []
    times  = []

    # NOTE: We need to get best score for each individual epoch
    #       Also need to derive training times...
    for fold in range(10):
        import math
        import numpy
        r = parse_neat(fname_t.format(fold=fold))
        num_chunks = len(r['train']['val_fit']) // 32 # periods / epoch
        # chunks   = [ slice(idx * num_chunks, (idx+1) * num_chunks) for idx in range(32) ]
        chunks = [ slice(0, (idx+1)*num_chunks) for idx in range(32) ]
        e_scores = [ numpy.max(r['train']['val_fit'][chunks[idx]]) for idx in range(32) ]
        e_times  = [ numpy.max(numpy.cumsum(r['train']['elapsed'])[chunks[idx]]) for idx in range(32) ]
        
        for epoch in range(32):
            epochs.append(epoch+1)
            folds.append(fold)
            scores.append( e_scores[epoch] )
            times.append( math.log2(e_times[epoch]) )

    return {
        'epoch': epochs,
        'fold': folds,
        'score': scores,
        'log2time': times,
    }

nsl_kdd_pca = pd.DataFrame(_loader('nsl_kdd', 'pca'))

sn.set(context='paper', style='darkgrid', palette='muted')
f, axes = plt.subplots(2, 1, figsize=(7,7), sharex=True)

ax = sn.boxplot(x='epoch', y='score', data=nsl_kdd_pca, ax=axes[0])
ax.set(xlabel='Epochs',
       ylabel='Validation Fitness',
       title='Epochs versus Validation Fitness')

ax = sn.boxplot(x='epoch', y='log2time', data=nsl_kdd_pca, ax=axes[1])
ax.set(xlabel='Epochs',
       ylabel='log2(Training Time)',
       title='Epochs vs Logarithmic Training Time')

plt.show()
