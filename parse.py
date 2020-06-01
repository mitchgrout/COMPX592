"""
parse.py
    Parses supplied results/* files into approprate pandas dataframes
"""

import pandas as pd
from ne.stats import Stats

def read(fd, f, key):
    l = fd.readline().strip()
    if l[:len(key)] != key:
        raise Exception('malformed, expected "{}" but got "{}"'.format(key, l))
    return f(l[len(key)+2:])

def parse_neat(fname):
    with open(fname, 'r') as fd:
        config = { }
        train  = { 'train_fit': [], 'model_size': [], 'val_fit': [], 'elapsed': [], 'best': [] }
        output = { } 

        fd.readline() # Configuration:
        # dataset, selector, fitness, batch_size, epochs, config_args
        config['dataset'    ] = read(fd, str,  'Dataset')
        config['selector'   ] = read(fd, str,  'Selector')
        config['fold'       ] = read(fd, int,  'Fold')
        config['fitness'    ] = read(fd, str,  'Fitness')
        config['batch_size' ] = read(fd, int,  'Batch size')
        config['epochs'     ] = read(fd, int,  'Epochs')
        config['config_args'] = read(fd, eval, 'Config args') # dict-type

        while True:
            # Tricky since we don't have a loop condition
            if fd.read(1) != '*':
                break # NOTE: We overread 1 byte into the next id
            fd.readline() # Read the rest, Epoch %d, batch %d/%d

            train_fit  = read(fd, float, 'Training fitness')
            model_size = read(fd, eval,  'Model size') # tuple
            val_fit    = read(fd, str,   'Validation fitness') # conditional
            elapsed    = read(fd, float, 'Elapsed')
            fd.readline() # Whitespace

            if val_fit[-3:] == '!!!':
                val_fit = float(val_fit[:-4])
                best    = True
            else:
                val_fit = float(val_fit)
                best    = False

            train['train_fit' ].append(train_fit)
            train['model_size'].append(model_size)
            train['val_fit'   ].append(val_fit)
            train['best'      ].append(best)
            train['elapsed'   ].append(elapsed)
    
        # Now we have the output statistics
        # NOTE: We overread 1 byte into 'Total train time'
        output['train_time'] = read(fd, float, 'Total train time'[1:])
        output['test_time']  = read(fd, float, 'Test time')
        output['test_fit']   = read(fd, float, 'Test fitness')
        output['test_stats'] = read(fd, eval,  'Test statistics') # ne.stats.Stats

        # And now we should be EOF

        return {
            'config': config,
            'train' : pd.DataFrame(train, columns=list(train.keys())), 
            'output': output,
        }

def parse_keras(fname):
    with open(fname, 'r') as fd:
        config = {}
        train  = { 'train_loss': [], 'val_mcc': [] }
        output = {}

        fd.readline() # Configuration:
        config['dataset'   ] = read(fd, str, 'Dataset')
        config['selector'  ] = read(fd, str, 'Selector')
        config['fold'      ] = read(fd, int, 'Fold')
        config['batch_size'] = read(fd, int, 'Batch size')
        config['epochs'    ] = read(fd, int, 'Epochs')

        # Training
        for _ in range(config['epochs']):
            fd.readline()      # Epoch [x]/epochs
            l1 = fd.readline() # ' - [time] - loss: [loss]
            l2 = fd.readline() # ' - val_mcc: [val_mcc]

            parts1 = l1.split(' - ')
            parts2 = l2.split(' - ')

            loss    = float(parts1[2][len('loss: '):])
            val_mcc = float(parts2[1][len('val_mcc: '):])

            train['train_loss'].append(loss)
            train['val_mcc'   ].append(val_mcc)

        # Test
        output['train_time'] = read(fd, float, 'Total train time')
        output['test_time' ] = read(fd, float, 'Test time')
        output['test_stats'] = read(fd, eval,  'Test statistics')

        return {
            'config': config,
            'train' : pd.DataFrame(train),
            'output': output,
        }
    pass

def parse_sklearn(fname):
    with open(fname, 'r') as fd:
        config = {}
        train  = {} # we dont really need this info
        output = {}

        fd.readline() # Configuration:
        config['dataset'    ] = read(fd, str,  'Dataset')
        config['selector'   ] = read(fd, str,  'Selector')
        config['fold'       ] = read(fd, int,  'Fold')

        # No train step
        pass

        # Output
        output['train_time'] = read(fd, float, 'Total train time')
        output['test_time']  = read(fd, float, 'Test time')
        output['test_stats'] = read(fd, eval,  'Test statistics')

        return {
            'config': config,
            'train' : pd.DataFrame(train, columns=list(train.keys())),
            'output': output,
        }

