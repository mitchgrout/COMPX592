"""
ne.data.kdd99
    Actual KDD99 data
"""

from ne.data import Data

def load_data():
    xs, ys = [], []
    nt, nf = 0, 0

    with open('ne/data/kdd99/kddcup.data', 'r') as fd:
        for line in fd:
            parts = line.strip().split(',')
            x, y = None, None
            try:
                parts[1]  = {'tcp':0, 'udp':1, 'icmp':2}[parts[1]]
                parts[-1] = 0 if parts[-1] == 'normal.' else 1
                x = [ float(parts[idx]) for idx in range(len(parts)-1) if idx not in (2,3) ]
                y = float(parts[-1])
            except Exception:
                print('Line {} failed: {}'.format(ctr, line))
                continue

            xs.append(x)
            ys.append(y)
            if y > 0.5: nt += 1
            else:       nf += 1

    return Data(xs=xs, ys=ys, nt=nt, nf=nf)

