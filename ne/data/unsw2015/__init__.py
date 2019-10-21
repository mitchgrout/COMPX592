"""
ne.data.unsw2015
    Actual UNSW2015 data
"""

def load_data():
    xs, ys = [], []
    with open('UNSW-NB15_1_python.csv', 'r') as fd:
        fd.readline() # Header line
        ctr = 1
        for line in fd:
            ctr += 1
            parts = line.strip().split(',')
            x, y = None, None

            try:
                if parts[-2] not in ('Normal', 'DoS'): 
                    continue 
                x = [ float(parts[idx]) for idx in range(len(parts)-2) if idx not in (2,3,4) ]
                y = float(parts[-1])
            except Exception:
                print('Line {} failed: {}'.format(ctr, line))
                continue

            xs.append(x)
            ys.append(y)
    return xs, ys

