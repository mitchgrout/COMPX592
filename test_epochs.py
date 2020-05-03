if __name__ == '__main__':
    import ne
    from test_harness import neat
    from os.path      import join

    dataset_pairs = [
        (ne.data.nsl_kdd.dataset,  ne.data.PCA(12)),
        #(ne.data.unsw2015.dataset, ne.data.PCA(17)),
        #(ne.data.ids2017.dataset,  ne.data.PCA(16)),
    ]

    # NOTE: Rather than run parallel tests we will
    #       instead take the best after each 2^n epochs
    max_epochs = 32

    procs = []
    for (d, s) in dataset_pairs:
        test_name = join('epochs', d.name(), s.name)
        p = neat(\
                test_name=test_name,
                dataset=d,
                selector=s,
                fitness=ne.fitness.mcc,
                batch_size=512,
                epochs=max_epochs)
        procs.append(p)
    for p in procs:
        p.join()

