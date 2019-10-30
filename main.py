#!/usr/bin/env python3 

import ne
import tee
import os

if __name__ == '__main__':
    DATA        = ne.data.unsw2015.load_data()
    LOG_DIR     = './logs/unsw2015_testing'
    CONFIG_FILE = 'config/unsw2015'
    MODEL_TYPE  = ne.neat.Recurrent
    EXECUTOR    = ne.execute.Sequential()
    NUM_SPLITS  = 16
    DELAY       = 3
    EPOCHS      = (NUM_SPLITS + 1) * DELAY

    os.makedirs(LOG_DIR, exist_ok=True)
    with tee.StdoutTee('{}/output.log'.format(LOG_DIR), buff=1):
        with open(__file__, 'r') as fd: 
            for line in fd: 
                print(line, end='')

        thresh    = lambda x: x > 0.5
        _, _, fit = ne.fitness.normalized(
                        *ne.fitness.zero_sum_builder(
                            true_count=DATA.nt,
                            false_count=DATA.nf,
                            thresh=thresh))
        
        ne.neat.NEAT(log_dir=LOG_DIR,
                     config_file=CONFIG_FILE,
                     model_type=MODEL_TYPE,
                     fitness_func=fit,
                     data=DATA.xs,
                     labels=DATA.ys,
                     executor=EXECUTOR,
                     thresh=thresh,
                     num_splits=NUM_SPLITS,
                     delay=DELAY,).train(EPOCHS)

