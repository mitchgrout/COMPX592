#!/usr/bin/env python3

import numpy
import ne
import tensorflow
import keras
import sklearn.naive_bayes
import sklearn.tree
import os
import tee
import tempfile

def neat(test_name,
         dataset=ne.data.nsl_kdd.dataset,
         selector=None,
         fold=0,
         fitness=ne.fitness.mcc,
         batch_size=256,
         epochs=1,
         config_args={}):

    return _make_task(_neat_task, test_name, dataset, selector, fold, fitness, batch_size, epochs, config_args)

def shallow_neural(test_name,
                   dataset=ne.data.nsl_kdd.dataset,
                   selector=None,
                   fold=0):

    model = keras.models.Sequential([
        keras.layers.InputLayer( ( (selector or dataset).num_features(),) ),
        keras.layers.Dense(7, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    return _make_task(_keras_task, test_name, model, dataset, selector, fold, 1, 10) # unspecified in ref doc

def deep_neural(test_name,
                dataset=ne.data.nsl_kdd.dataset,
                selector=None,
                fold=0):

    model = keras.models.Sequential([
        keras.layers.InputLayer( ( (selector or dataset).num_features(),) ),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return _make_task(_keras_task, test_name, model, dataset, selector, fold, 1, 10)

def conv_neural(test_name,
                dataset=ne.data.nsl_kdd.dataset,
                selector=None,
                fold=0):

    model = keras.models.Sequential([
        keras.layers.InputLayer( ( (selector or dataset).num_features(),) ),
        keras.layers.Reshape( (-1, 1) ),
        keras.layers.Conv1D(64, kernel_size=(3,), activation='relu'),
        keras.layers.MaxPooling1D(),
        keras.layers.Conv1D(64, kernel_size=(3,), activation='relu'),
        keras.layers.MaxPooling1D(),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    return _make_task(_keras_task, test_name, model, dataset, selector, fold, 1, 10) # specified to be 100 in ref doc

def dbn(test_name,
        dataset=ne.data.nsl_kdd.dataset,
        selector=None,
        fold=0):

    def _helper(_test_name, _dataset, _selector, _fold):
        import dbn.tensorflow as dbn_lib
        # NOTE: tensorflow/models.py updated to use only one thread
        #       See _keras_task for relevant code
        model = dbn_lib.SupervisedDBNClassification(\
                hidden_layers_structure=[256,256],
                activation_function='sigmoid',
                optimization_algorithm='sgd',
                learning_rate=1e-3,
                learning_rate_rbm=1e-3,
                n_iter_backprop=10,
                n_epochs_rbm=2,
                verbose=False)
        return _sklearn_task(_test_name, model, _dataset, _selector, _fold)

    return _make_task(_helper, test_name, dataset, selector, fold)
    # return _make_task(_sklearn_task, test_name, model, dataset, selector, fold)

def naive_bayes(test_name,
                dataset=ne.data.nsl_kdd.dataset,
                selector=None,
                fold=0,
                var_smoothing=1e-9):
    model = sklearn.naive_bayes.GaussianNB(\
                priors=None,
                var_smoothing=var_smoothing)
    return _make_task(_sklearn_task, test_name, model, dataset, selector, fold)

def decision_tree(test_name,
                  dataset=ne.data.nsl_kdd.dataset,
                  selector=None,
                  fold=0,
                  max_depth=None):

    model = sklearn.tree.DecisionTreeClassifier(\
                criterion='gini',
                splitter='best',
                max_depth=max_depth,
                max_features='sqrt')
    return _make_task(_sklearn_task, test_name, model, dataset, selector, fold) 

def _make_task(fn, *args):
    from multiprocessing import Process
    p = Process(target=fn, args=args)
    p.daemon = True
    p.start()
    return p

def _neat_task(test_name, dataset, selector, fold, fitness, batch_size, epochs, config_args):
    config_file = ne.neat.create_config_file(num_inputs=(selector or dataset).num_features(), **config_args)
    model_type  = ne.neat.FeedForward 
    executor    = ne.execute.Sequential()
    thresh      = lambda x: x > 0.5
        
    log_dir = os.path.join('results', test_name)
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        print('Configuration:')
        print('Dataset:', dataset.name())
        print('Selector:', selector.name)
        print('Fold:', fold)
        print('Fitness:', fitness.__name__)
        print('Batch size:', batch_size)
        print('Epochs:', epochs)
        print('Config args:' , config_args)

        split = dataset.data(selector=selector, save=False, cache=True, fold=fold)
        t, model = ne.util.benchmark(lambda:\
                ne.neat.run(
                    epochs=epochs,
                    split_data=split,
                    batch_size=batch_size,
                    model_type=model_type,
                    fitness=fitness(thresh),
                    config_file=config_file,
                    log_dir=log_dir, 
                    executor=executor,
                    thresh=thresh,
                    verbose=True))
        print('Total train time:', t)

        t, pred_ys = ne.util.benchmark(lambda: list(model.predict(split.test.xs)))
        *_, f = model.evaluate(split.test.ys, pred_ys)
        stats = model.compute_statistics(split.test.ys, pred_ys)
        print('Test time:', t)
        print('Test fitness:', f)
        print('Test statistics:', stats)

def _keras_task(test_name, model, dataset, selector, fold, batch_size, epochs):
    cfg = tensorflow.compat.v1.ConfigProto(\
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count={'CPU': 1})
    ses = tensorflow.compat.v1.Session(config=cfg)
    keras.backend.set_session(ses)

    thresh = lambda x: x > 0.5
    log_dir = os.path.join('results', test_name)
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        print('Configuration:')
        print('Dataset:', dataset.name())
        print('Selector:', selector.name)
        print('Fold:', fold)
        print('Batch size:', batch_size)
        print('Epochs:', epochs)
 
        split = dataset.data(selector=selector, save=False, cache=False, fold=fold)
        
        model.compile('adam', 'binary_crossentropy') # placeholder

        class MCC_Callback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                pred_ys = [t[0] for t in model.predict(split.val.xs)]
                stats = ne.stats.compute_statistics(thresh, split.val.ys, pred_ys)
                logs['val_mcc'] = stats.mcc
                # Stupid..
                print(' - val_mcc: {:04f}'.format( logs['val_mcc'] ))

        callbacks = [
            MCC_Callback(), # Infuriatingly we have to do it like this
            keras.callbacks.ModelCheckpoint(\
                filepath=os.path.join(log_dir, 'model.{epoch}.h5'),
                monitor='val_mcc',
                save_best_only=True,
                mode='max'),
        ]

        t, _ = ne.util.benchmark(lambda: model.fit(*split.train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2))
        print('Total train time:', t)

        t, r = ne.util.benchmark(lambda: ne.stats.compute_statistics(thresh, split.test.ys, [ t[0] for t in model.predict(split.test.xs) ]))
        print('Test time:', t)
        print('Test statistics:', r)

def _sklearn_task(test_name, model, dataset, selector, fold):
    thresh = lambda x: x > 0.5

    log_dir = os.path.join('results', test_name)
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        print('Configuration:')
        print('Dataset:', dataset.name())
        print('Selector:', selector.name)
        print('Fold:', fold)

        split = dataset.data(selector=selector, save=False, cache=False, fold=fold)
        t, _ = ne.util.benchmark(lambda: model.fit(*split.train))
        print('Total train time:', t)

        t, r = ne.util.benchmark(lambda: ne.stats.compute_statistics(thresh, split.test.ys, model.predict(split.test.xs)))
        print('Test time:', t)
        print('Test statistics:', r)

