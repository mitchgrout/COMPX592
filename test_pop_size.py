if __name__ == '__main__':
    import ne
    from test_harness import neat
    from os.path      import join

    dataset_pairs = [
        (ne.data.nsl_kdd.dataset,  ne.data.PCA(12)),
        (ne.data.unsw2015.dataset, ne.data.PCA(17)),
        (ne.data.ids2017.dataset,  ne.data.PCA(16)),
    ]

    pop_sizes = [
        32, 64, 128, 256, 512, 1024, 2048
    ]

    num_runs = 10

    for run in range(1, num_runs+1):
        procs = []
        for (d, s) in dataset_pairs:
            for p in pop_sizes:
                test_name = join('pop_size', d.name(), s.name, str(p), str(run))
                p = neat(\
                        test_name=test_name,
                        dataset=d,
                        selector=s,
                        fitness=ne.fitness.mcc,
                        batch_size=512,
                        epochs=4,
                        config_args={'pop_size': p})
                procs.append(p)
        for p in procs:
            p.join()

