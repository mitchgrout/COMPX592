from render_preamble import *

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

f, axes = create_figure(2, 1, "Population Sizes Evaluated Against NSL-KDD")
plt.tight_layout()

ax = sn.boxplot(x='pop_size', y='score', data=nsl_kdd_pca, ax=axes[0])
ax.set(xlabel='', ylabel='Validation Fitness')

ax = sn.boxplot(x='pop_size', y='log2time', data=nsl_kdd_pca, ax=axes[1])
ax.set(xlabel='', ylabel='log2(Training Time)')

plt.savefig('renders/results/pop_size_nsl_kdd.png')

