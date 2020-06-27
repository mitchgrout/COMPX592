from render_preamble import *

def _loader(name, selector):
    fname_t = join('results', 'batch_sizes', name, selector, '{batch_size}', '{fold}', 'output.log')

    batch = []
    folds = []
    score = []
    time  = []

    for batch_size in [4, 8, 16, 32, 64, 128, 256, 512]:
        for fold in range(10):
            import math
            r = parse_neat(fname_t.format(batch_size=batch_size, fold=fold))
            s = max(r['train']['val_fit'])
            t = r['output']['train_time']

            batch.append(batch_size)
            folds.append(fold)
            score.append(s)
            time.append(math.log2(t))
    
    return {
        'batch_size': batch,
        'fold': folds,
        'score': score,
        'log2time': time,
    }

nsl_kdd_pca = pd.DataFrame(_loader('nsl_kdd', 'pca'))

f, axes = create_figure(2, 1, "Batch Sizes Evaluated Against NSL-KDD")

ax = sn.boxplot(x='batch_size', y='score', data=nsl_kdd_pca, ax=axes[0])
ax.set(xlabel='', ylabel='Validation Fitness')

ax = sn.boxplot(x='batch_size', y='log2time', data=nsl_kdd_pca, ax=axes[1])
ax.set(xlabel='', ylabel='log2(Training Time)')

plt.savefig("renders/results/batch_size_nsl_kdd")

