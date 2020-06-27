from render_preamble import *

def _loader(name, selector):
    fname_t = join('results', 'fitness_functions', name, selector, '{func}', '{fold}', 'output.log') 

    funcs = []
    folds = []
    tpr   = []
    tnr   = []
    ppv   = []
    npv   = []
    mcc   = []

    name_map = {
        'accuracy': 'ACC',
        'f1': 'F1',
        'mcc': 'MCC',
        'zero_sum': 'ZSUM',
        'auroc': 'AUROC',
        'inverse_binary_crossentropy': 'IBCE',
        'inverse_mean_square_error': 'IMSE',
    }

    for func in name_map.keys():
        for fold in range(10):
            r = parse_neat(fname_t.format(func=func, fold=fold))
            s = r['output']['test_stats']
            if abs(s.mcc) < 0.2: continue
            funcs.append(name_map[func])
            folds.append(fold)
            tpr.append(s.tpr)
            tnr.append(s.tnr)
            ppv.append(s.ppv)
            npv.append(s.npv)
            mcc.append(s.mcc)

    return {
        'metric': funcs,
        'fold': folds,
        'tpr': tpr,
        'tnr': tnr,
        'ppv': ppv,
        'npv': npv,
        'mcc': mcc,
    }

nsl_kdd_pca = pd.DataFrame(_loader('nsl_kdd', 'pca'))

f, axes = create_figure(2, 2, "Fitness Functions Evaluated Against NSL-KDD")

ax = sn.boxplot(x='metric', y='tpr', data=nsl_kdd_pca, ax=axes[0,0]) 
ax.set(xlabel='', ylabel='True Positive Rate')
rotate_axis(ax)

ax = sn.boxplot(x='metric', y='tnr', data=nsl_kdd_pca, ax=axes[0,1]) 
ax.set(xlabel='', ylabel='True Negative Rate')
rotate_axis(ax)

ax = sn.boxplot(x='metric', y='ppv', data=nsl_kdd_pca, ax=axes[1,0]) 
ax.set(xlabel='', ylabel='Positive Predictive Rate')
rotate_axis(ax)

ax = sn.boxplot(x='metric', y='npv', data=nsl_kdd_pca, ax=axes[1,1]) 
ax.set(xlabel='', ylabel='Negative Predictive Rate')
rotate_axis(ax)

plt.savefig('renders/results/fitness_functions_nsl_kdd.png')

