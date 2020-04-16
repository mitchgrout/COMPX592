"""
ne.data.unsw2015
    Actual UNSW2015 data
"""

from ne.data import Data

def name():
    return "unsw2015"

def load_best():
    # Negate the set of columns we actually want
    columns  = [0, 4, 9, 10, 14, 28, 29, 32, 33, 34, 36, 38, 39, 44, 45, 46]
    excluded = list(filter(lambda x: x not in columns, range(47)))
    return load_data(exclude_cols=excluded)

def load_data(exclude_cols=[], dedupe=True):
    from os.path import exists
    import wget
    import numpy
    numpy.random.seed(20192020)

    exclude_cols = exclude_cols.copy()
    assert all(0 <= c and c < 47 for c in exclude_cols)
    exclude_cols.append(47)
    exclude_cols.append(48)
    labels = [ 
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
        "dbytes", "sttl", "dttl", " sloss", "dloss", "service", "Sload",
        "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", " smeansz",
        "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime",
        "Ltime", "Sintpkt", " Dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", " ct_flw_http_mthd", "is_ftp_login",
        "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", " ct_src_ ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
    ]
    urlf  = 'https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/{}' 
    pathf = 'ne/data/unsw2015/{}'
    files = [ 'UNSW-NB15_{}.csv'.format(idx) for idx in [1,2,3,4] ]

    categorical_tables = {
        4:  {},
        5:  {},
        13: {},
        37: {},
        38: {},
        39: {},
    }
    seen_lines = set()
    xs, ys     = [], []
    nt, nf     = 0, 0

    for fname in files:
        if not exists(pathf.format(fname)):
            wget.download(url=urlf.format(fname), out=pathf.format(fname)) 
        with open(pathf.format(fname), 'r') as fd:
            for line in fd:
                line  = line.encode('ascii', 'ignore').decode('ascii').replace(' ', '')
                parts = line.split(',')
                if dedupe:
                    if line in seen_lines:
                        continue
                    seen_lines.add(line)
                ipv4_to_int = lambda ip: sum( int(t[0]) << (8*t[1]) for t in zip(ip.split('.'), [3,2,1,0]))
                for col in [0, 2]:
                    parts[col] = ipv4_to_int(parts[col])
                for col in [1, 3]:
                    if parts[col][:2] == '0x':
                        parts[col] = str(int(parts[col], base=16))
                    elif parts[col] == '-':
                        parts[col] = '0'
                for col in categorical_tables.keys():
                    x = parts[col]
                    d = categorical_tables[col]
                    if x not in d:
                        d[x] = len(d)
                    parts[col] = d[x]
               
                x = [ float(x) for (idx, x) in enumerate(parts) if idx not in exclude_cols ]
                y = float(parts[48])
                xs.append(x)
                ys.append(y)
    
                if y > 0.5: nt += 1
                else:       nf += 1
    xs = numpy.asarray(xs)
    ys = numpy.asarray(ys)
        
    p = numpy.random.permutation(len(xs))
    xs = xs[p]
    ys = ys[p]
    label_names = [ labels[idx] for idx in range(47) if idx not in exclude_cols ]
    return Data(xs=xs, ys=ys, nt=nt, nf=nf, label_names=label_names)

