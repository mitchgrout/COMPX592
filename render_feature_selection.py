from render_preamble import *

# Load
def _loader(name, k_max, stat='mcc'):
    # File templates
    f_pca = 'results/feature_selection/{}/pca/{{}}/output.log'.format(name)
    f_prs = 'results/feature_selection/{}/pearsons/{{}}/output.log'.format(name)

    # All of the features tested [ 1 .. k_max ]
    labels = list(range(1, 1+k_max))

    # Load our data
    d_pca  = [ parse_sklearn(f_pca.format(l))['output']['test_stats'].__getattribute__(stat) for l in labels ]
    d_prs  = [ parse_sklearn(f_prs.format(l))['output']['test_stats'].__getattribute__(stat) for l in labels ]
    d_none = [ d_prs[-1] for l in labels ]

    return { 
        'xs': labels,
        'none': d_none,
        'pearsons': d_prs,
        'pca': d_pca,
    }

stat = 'mcc'

nsl_kdd  = _loader('nsl_kdd',  41, stat)
unsw2015 = _loader('unsw2015', 47, stat)
ids2017  = _loader('ids2017',  78, stat)

# Process
nsl_kdd  = pd.melt(pd.DataFrame(nsl_kdd,  columns=list(nsl_kdd.keys())),  ['xs'])
unsw2015 = pd.melt(pd.DataFrame(unsw2015, columns=list(unsw2015.keys())), ['xs'])
ids2017  = pd.melt(pd.DataFrame(ids2017,  columns=list(ids2017.keys())),  ['xs'])

# Render
for name, df in [ ("NSL KDD", nsl_kdd), ("UNSW NB-15", unsw2015), ("CIC IDS 2017", ids2017) ]:
    plt.tight_layout()
    sn.lineplot(x='xs', y='value', hue='variable', data=df)
    plt.xlabel("Number of features")
    plt.ylabel("Test {} score".format(stat.upper()))
    plt.title("Feature Selection for {}".format(name))

    filenames = {
        'NSL KDD': 'nsl_kdd',
        'UNSW NB-15': 'unsw2015',
        'CIC IDS 2017': 'ids2017',
    }
    plt.savefig( 'renders/results/feature_selection_{}.png'.format(filenames[name]) )

