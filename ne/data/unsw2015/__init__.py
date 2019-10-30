"""
ne.data.unsw2015
    Actual UNSW2015 data
"""

from ne.data import Data

def load_data():
    xs, ys = [], []
    nt, nf = 0, 0

    with open('ne/data/unsw2015/UNSW-NB15.csv', 'r') as fd:
        fd.readline() # Header line
        for line in fd:
            parts = line.strip().split(',')
            x, y = None, None
            #if nt+nf > 10000: break
            try:
                x = [ float(parts[idx]) for idx in range(len(parts)-1) ]
                y = float(parts[-1])
            except Exception:
                print('Line failed: {}'.format(line))
                continue

            xs.append(x)
            ys.append(y)
            if y > 0.5: nt += 1
            else:       nf += 1
    return Data(xs=xs, ys=ys, nt=nt, nf=nf)

