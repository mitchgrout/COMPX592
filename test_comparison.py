if __name__ == '__main__':
    import ne
    from test_harness import neat, shallow_neural, deep_neural, decision_tree 
    from os.path      import join

    dataset_pairs = [
        (ne.data.nsl_kdd.dataset,  ne.data.PCA(12)),
        (ne.data.unsw2015.dataset, ne.data.PCA(17)),
        (ne.data.ids2017.dataset,  ne.data.PCA(16)),
    ]

    models = [
        lambda n,d,s,f: neat(\
                join(n, 'neat'),
                dataset=d,
                selector=s,
                fold=f,
                fitness=ne.fitness.mcc,
                batch_size=512,
                epochs=4,
                config_args={'pop_size':128}),

        lambda n,d,s,f: shallow_neural(\
                join(n, 'shallow_neural'),
                dataset=d,
                selector=s,
               fold=f),
        
       lambda n,d,s,f: deep_neural(\
               join(n, 'deep_neural'),
               dataset=d,
               selector=s,
               fold=f),

        lambda n,d,s,f: decision_tree(\
                join(n, 'decision_tree'),
                dataset=d,
                selector=s,
                fold=f,
                max_depth=32),
    ]

    for (d, s) in dataset_pairs:
        for m in models:
            procs = []
            for f in range(10):
                n = join('comparison', d.name(), s.name, str(f))
                procs.append( m(n, d, s, f) )
            for p in procs:
                p.join()

