#!/usr/bin/env python3

import numpy
import ne
import tensorflow
import keras
import sklearn.naive_bayes
import os
import tee
import tempfile

def neat(test_name,
         dataset=ne.data.nsl_kdd.dataset,
         selector=None,
         fitness=ne.fitness.mcc,
         batch_size=256,
         epochs=1,
         config_args={}):

    return _make_task(_neat_task, test_name, dataset, selector, fitness, batch_size, epochs, config_args)

def shallow_neural(test_name,
                   dataset=ne.data.nsl_kdd.dataset,
                   selector=None):

    model = keras.models.Sequential([
        keras.layers.InputLayer( ( (selector or dataset).num_features(),) ),
        keras.layers.Dense(7, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    return _make_task(_keras_task, test_name, model, dataset, selector, 1, 10) # unspecified in ref doc

def deep_neural(test_name,
                dataset=ne.data.nsl_kdd.dataset,
                selector=None):

    model = keras.models.Sequential([
        keras.layers.InputLayer( ( (selector or dataset).num_features(),) ),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.MaxoutDense(10),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return _make_task(_keras_task, test_name, model, dataset, selector, 1, 10)

def naive_bayes(test_name,
                dataset=ne.data.nsl_kdd.dataset,
                selector=None,
                var_smoothing=1e-9):
    model = sklearn.naive_bayes.GaussianNB(\
                priors=None,
                var_smoothing=var_smoothing)
    return _make_task(_sklearn_task, test_name, model, dataset, selector)

def decision_tree(test_name,
                  dataset=ne.data.nsl_kdd.dataset,
                  selector=None,
                  max_depth=None):

    model = sklearn.tree.DecisionTreeClassifier(\
                criterion='gini',
                splitter='best',
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='auto')
    return _make_task(_sklearn_task, test_name, model, dataset, selector) 

def random_forest(test_name,
        dataset=ne.data.nsl_kdd.dataset):

    model = sklearn.ensemble.RandomForestClassifier(\
                n_estimators=100,
                criterion='gini',
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='auto',
                n_jobs=1)
    return _make_task(_sklearn_task, test_name, model, dataset, selector)

def _make_task(fn, *args):
    from multiprocessing import Process
    p = Process(target=fn, args=args)
    p.start()
    return p

def _neat_task(test_name, dataset, selector, fitness, batch_size, epochs, config_args):
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
        print('Fitness:', fitness.__name__)
        print('Batch size:', batch_size)
        print('Epochs:', epochs)
        print('Config args:' , config_args)

        split = dataset.data(selector=selector, save=False, cache=False)
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

def _keras_task(test_name, model, dataset, selector, batch_size, epochs):
    cfg = tensorflow.compat.v1.ConfigProto(\
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count={'CPU': 1})
    ses = tensorflow.compat.v1.Session(config=cfg)
    keras.backend.set_session(ses)

    def mcc(y_true, y_pred):
        import tensorflow as tf
        from tensorflow.keras import backend as K

        # y_pred needs to be thresholded; TODO; Allow variable threshold value
        y_pred = K.cast( K.greater_equal(y_pred, 0.5), dtype=K.floatx() )
        y_true = K.cast( y_true, dtype=K.floatx() )

        # Tensor-friendly way to compute confusion matrix
        # Assumes only a single output [otherwise axis should be specified]
        tp = K.sum(     y_true  *      y_pred)
        tn = K.sum((1 - y_true) * (1 - y_pred))
        fp = K.sum((1 - y_true) * (    y_pred))
        fn = K.sum(     y_true  * (1 - y_pred))

        num = (tp * tn) - (fp * fn)
        den = K.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )
        return num / (den + K.epsilon())

    thresh = lambda x: x > 0.5
    log_dir = os.path.join('results', test_name)
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        print('Configuration:')
        print('Dataset:', dataset.name())
        print('Selector:', selector.name)
        print('Batch size:', batch_size)
        print('Epochs:', epochs)
 
        split = dataset.data(selector=selector, save=False, cache=False)
        
        model.compile('adam', 'binary_crossentropy', metrics=[mcc])
        callbacks = [
            keras.callbacks.ModelCheckpoint(\
                filepath=os.path.join(log_dir, 'model.{epoch}.h5'),
                monitor='val_mcc',
                save_best_only=True,
                mode='max'),
        ]
        t, _ = ne.util.benchmark(lambda: model.fit(*split.train, batch_size=batch_size, epochs=epochs, validation_data=split.val, callbacks=callbacks, verbose=2))
        print('Total train time:', t)

        t, r = ne.util.benchmark(lambda: ne.stats.compute_statistics(thresh, split.test.ys, [ t[0] for t in model.predict(split.test.xs) ]))
        print('Test time:', t)
        print('Test statistics:', r)

def _sklearn_task(test_name, model, dataset, selector):
    thresh = lambda x: x > 0.5

    log_dir = os.path.join('results', test_name)
    os.makedirs(log_dir, exist_ok=True)
    with tee.StdoutTee(os.path.join(log_dir, 'output.log'), buff=1):
        print('Configuration:')
        print('Dataset:', dataset.name())
        print('Selector:', selector.name)
    
        split = dataset.data(selector=selector, save=False, cache=False)
        t, _ = ne.util.benchmark(lambda: model.fit(*split.train))
        print('Total train time:', t)

        t, r = ne.util.benchmark(lambda: ne.stats.compute_statistics(thresh, split.test.ys, model.predict(split.test.xs)))
        print('Test time:', t)
        print('Test statistics:', r)

