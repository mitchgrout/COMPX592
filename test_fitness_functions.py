
if __name__ == '__main__':
    import ne
    from test_harness import neat
    from os.path      import join

    dataset_pairs = [
        (ne.data.nsl_kdd.dataset,  ne.data.PCA(12)),
        (ne.data.unsw2015.dataset, ne.data.PCA(17)),
        (ne.data.ids2017.dataset,  ne.data.PCA(16)),
    ]

    fitness_funcs = [
        ne.fitness.accuracy,
        ne.fitness.f1,
        ne.fitness.mcc,
        ne.fitness.zero_sum,
        ne.fitness.inverse_mean_square_error,
        ne.fitness.inverse_binary_crossentropy,
    ]

    procs = []
    for (d, s) in dataset_pairs:
        for f in fitness_funcs:
            test_name = join('fitness_functions', d.name(), s.name, f.__name__)
            p = run_test(\
                    test_name=test_name,
                    dataset=d
                    selector=s,
                    fitness=f)
            procs.append(p)
    for p in procs:
        p.join()

