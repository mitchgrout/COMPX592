from render_preamble import *

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
        'conv_neural': {
            'name': 'Conv',
            'loader': parse_keras,
        },
        #'dbn': {
        #    'name': 'DBN',
        #    'loader': parse_sklearn,
        #},
        #'decision_tree': {
        #    'name': 'Tree',
        #    'loader': parse_sklearn,
        #},
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

for name, df in [ ('NSL KDD', nsl_kdd_pca), ]:#('UNSW NB-15', unsw_nb15_pca), ('CIC IDS 2017', ids_2017_pca) ]:
    f, axes = create_figure(1, 3, "Comparative Tests Against {}".format(name))
    plt.tight_layout()

    ax = sn.boxplot(x='model', y='score', data=df, ax=axes[0])
    ax.set(xlabel='', ylabel='Test MCC')
    rotate_axis(ax)

    ax = sn.boxplot(x='model', y='log2time', data=df, ax=axes[1])
    ax.set(xlabel='', ylabel='log2(Training Time)')
    rotate_axis(ax)

    ax = sn.boxplot(x='model', y='log2inftime', data=df, ax=axes[2])
    ax.set(xlabel='', ylabel='log2(Inference Time)')
    rotate_axis(ax)

    filenames = {
        'NSL KDD': 'nsl_kdd',
        'UNSW NB-15': 'unsw2015',
        'CIC IDS 2017': 'ids2017',
    }

    plt.savefig( 'renders/results/comparison_{}.png'.format(filenames[name]) )

