#!/usr/bin/env python3 

import ne
if __name__ == '__main__':
    thresh = lambda x: x > 0.0

    # LO=0
    # HI=8
    # NT=128
    # NF=NT*(HI+LO)/2
    # xs, ys = ne.data.timestamp.load_data(NT, HI, LO)
    # fitness = ne.fitness.normalized(
    #             -2*NF, +2*NF,
    #             ne.fitness.zero_sum_builder(
    #                 true_count=NT,
    #                 false_count=NF,
    #                 thresh=thresh))

    txs, tys = ne.data.unsw2015.load_data()
    xs = txs[:2**15]
    ys = tys[:2**15]
    NT = sum(ys)
    NF = len(ys) - NT
    fitness = ne.fitness.normalized(
                -2*NF, +2*NF,
                ne.fitness.zero_sum_builder(
                        true_count=NT,
                        false_count=NF,
                        thresh=thresh))

    strategy = ne.neat.NEAT(
                'config/unsw2015',
                ne.neat.Recurrent,
                fitness,
                xs, ys,
                ne.execute.Parallel(4),
                txs[:],
                tys[:],
                thresh)
    model = strategy.train(epochs=256)

    xs = txs[-2**15:]
    ys = tys[-2**15:]
    pred_list = list(ne.base.predict(model, xs))
    print(ne.stats.compute_stats(ys, pred_list, thresh=thresh))

