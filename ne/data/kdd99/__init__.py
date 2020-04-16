"""
ne.data.kdd99
    Actual KDD99 data
"""

from ne.data import Data

def name():
    return "kdd99"

def load_best():
    # Negate the set of columns we actually want
    columns  = [2, 3, 11, 22, 24, 25, 28, 31, 32, 33, 37, 38]
    excluded = list(filter(lambda x: x not in columns, range(41)))
    return load_data(exclude_cols=excluded)

def load_data(exclude_cols=[], dedupe=True):
    from os.path import exists
    import wget
    import numpy
    import gzip
    numpy.random.seed(20192020)

    exclude_cols = exclude_cols.copy()
    assert all(0 <= c and c < 41 for c in exclude_cols)
    exclude_cols.append(41)
    labels = [
        "duration", "proto", "service", "flag", "src_byte", "dst_byte", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_shells",
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]
    urlf  = 'http://kdd.ics.uci.edu/databases/kddcup99/{}'
    pathf = 'ne/data/kdd99/{}'
    fname = 'kddcup.data.gz'

    categorical_tables = {
        1: {},
        2: {},
        3: {},
    }
    seen_lines = set()
    xs, ys = [], []
    nt, nf = 0, 0

    if not exists(pathf.format(fname)):
        wget.download(url=urlf.format(fname), out=pathf.format(fname))
    with gzip.open(pathf.format(fname), 'rb') as fd:
        for line in fd:
            line = line.decode('ascii').replace(' ', '').strip()
            parts = line.split(',')
            if dedupe:
                if line in seen_lines:
                    continue
                seen_lines.add(line)
            for col in categorical_tables.keys():
                x = parts[col]
                d = categorical_tables[col]
                if x not in d:
                    d[x] = len(d)
                parts[col] = d[x]
            parts[41] = int(parts[41] != 'normal.')

            x = [ float(x) for (idx, x) in enumerate(parts) if idx not in exclude_cols ]
            y = float(parts[41])
            xs.append(x)
            ys.append(y)

            if y > 0.5: nt += 1
            else:       nf += 1
    xs = numpy.asarray(xs)
    ys = numpy.asarray(ys)
    
    p = numpy.random.permutation(len(xs))
    xs = xs[p]
    ys = ys[p]
    label_names = [ labels[idx] for idx in range(41) if idx not in exclude_cols ]
    return Data(xs=xs, ys=ys, nt=nt, nf=nf, label_names=label_names)

