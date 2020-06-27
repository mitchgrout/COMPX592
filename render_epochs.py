from render_preamble import *

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

f, axes = create_figure(2, 1, "Epochs Evaluated Against NSL-KDD")

ax = sn.boxplot(x='epoch', y='score', data=nsl_kdd_pca, ax=axes[0])
ax.set(xlabel='', ylabel='Validation Fitness')
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

ax = sn.boxplot(x='epoch', y='log2time', data=nsl_kdd_pca, ax=axes[1])
ax.set(xlabel='', ylabel='log2(Training Time)')
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

plt.savefig('renders/results/epochs_nsl_kdd.png')

