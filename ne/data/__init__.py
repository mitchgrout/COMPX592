"""
ne.data
    Functions for processing and splitting data
"""

from collections import namedtuple
Data  = namedtuple(typename='Data',  field_names=['xs', 'ys', 'nt', 'nf', 'label_names'])
Pair  = namedtuple(typename='Pair',  field_names=['xs', 'ys'])
Split = namedtuple(typename='Split', field_names=['train', 'test', 'val'])

def make_split(data, ratio=(0.8, 0.1, 0.1)):
    assert 0.0 <= sum(ratio)
    assert sum(ratio) <= 1.0
    a = round(len(data.xs) * ratio[0])
    b = round(len(data.xs) * ratio[1]) + a
    c = round(len(data.xs) * ratio[2]) + b
    return Split(train=Pair(xs=data.xs[ :a], ys=data.ys[ :a]),
                 test =Pair(xs=data.xs[a:b], ys=data.ys[a:b]),
                 val  =Pair(xs=data.xs[b:c], ys=data.ys[b:c]))

