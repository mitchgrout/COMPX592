"""
ne.data.kdd99
    Sample of the KDD99 data
"""

def load_data():
    xs, ys = [], []
    # with open('kddcup.data_10_percent_corrected', 'r') as fd:
    with open('ne/data/kdd99/kddcup.data.txt', 'r') as fd:
        fd.readline() # Header line
        ctr = 1
        for line in fd:
            ctr += 1
            if ctr == 3000000: break
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
    return xs, ys

