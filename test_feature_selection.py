if __name__ == '__main__':
    import ne
    from test_harness import naive_bayes
    from os.path      import join

    datasets = [
        #ne.data.nsl_kdd.dataset,
        ne.data.unsw2015.dataset,
        #ne.data.ids2017.dataset,
    ]

    selectors = [
        #ne.data.Pearsons,
        ne.data.PCA,
    ]

    for d in datasets:
        for s in selectors:
            for n in range(1, d.num_features()):
                sel = s(n)
                test_name = join('feature_selection', 'unsw2015_reduced', s.name, str(n))
                naive_bayes(\
                    test_name=test_name,
                    dataset=d,
                    selector=sel).join()

