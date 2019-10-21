#!/usr/bin/env python3
from keras.models    import *
from keras.layers    import *
from keras.callbacks import *
from ne.data.kdd99   import *
from ne.stats        import * 
import numpy as np
 
if __name__ == '__main__':
    if True:
        CELL    = Dense
        RESHAPE = lambda x:x
    else:
        CELL    = LSTM
        RESHAPE = Reshape((-1, 1))

    i = input_layer = Input(shape=(39,))
    input_layer = RESHAPE(input_layer)

    # Create n many forward-connected layers
    activation = lambda _: 'tanh'
    prev_layers = []
    for idx in range(5):
        layer = CELL(int(2**4 / (2**idx)))(input_layer if not idx else concatenate([ input_layer, *prev_layers ], axis=1))
        layer = Activation(activation(idx))(layer)
        layer = BatchNormalization()(layer)
        layer = RESHAPE(layer)
        prev_layers.append(layer)

    o = prev_layers[-1]
    o = Reshape((-1,))(o)

    model = Model(inputs=i, outputs=o)
    model.compile('adam', 'binary_crossentropy')
    model.summary()

    xs, ys = load_data()
    xs     = np.asarray(xs)
    ys     = np.asarray(ys)
    spl    = 20000 # len(xs) - 1024 # 2**15
    train  = (xs[:spl], ys[:spl])
    val    = (xs[spl:], ys[spl:])

    model.fit(xs, ys, validation_split=0.25,
    #model.fit(*train, validation_data=val,
              epochs=256, batch_size=2**7,
              callbacks=[EarlyStopping(patience=8, restore_best_weights=True)
                        ,ReduceLROnPlateau(monitor='loss', patience=4, verbose=1)
                        ,ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)])

    print(compute_stats(ys, model.predict(xs), lambda x: x > 0.0))
    #print(compute_stats(val[1], model.predict(val[0]), lambda x: x > 0.0))
