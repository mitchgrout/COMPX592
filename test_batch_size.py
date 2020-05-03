if __name__ == '__main__':
    import ne
    from test_harness import neat
    from os.path      import join

    dataset_pairs = [
        (ne.data.nsl_kdd.dataset,  ne.data.PCA(12)),
        #(ne.data.unsw2015.dataset, ne.data.PCA(17)),
        #(ne.data.ids2017.dataset,  ne.data.PCA(16)),
    ]

    batch_sizes = [ 16, 32, 64, 128, 256, 512, 1024 ]

    procs = []
    for (d, s) in dataset_pairs:
        for b in batch_sizes:
            test_name = join('batch_sizes', d.name(), s.name, str(b))
            p = neat(\
                    test_name=test_name,
                    dataset=d,
                    selector=s,
                    fitness=ne.fitness.mcc,
                    batch_size=b)
            procs.append(p)
    for p in procs:
        p.join()

