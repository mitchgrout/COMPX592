"""
ne.util
    Convenience functions
"""

def benchmark(action):
    from time import time
    s = time()
    r = action()
    return (time()-s, r)

def dump(fname):
    with open(fname, 'r') as fd:
        for line in fd:
            print(line, end='')

