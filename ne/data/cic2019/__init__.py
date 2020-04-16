"""
ne.data.cic2019
    CIC2019 DDoS UDP data
"""

from ne.data import Data

def name():
    return "cic2019"

def load_best():
    # Negate the set of columns we actually want
    columns  = []
    excluded = list(filter(lambda x: x not in columns, range(88)))
    return load_data(exclude_cols=excluded)

def load_data(exclude_cols=[], dedupe=True):
    from os.path import exists
    import wget
    import numpy
    numpy.random.seed(20192020)

    exclude_cols = exclude_cols.copy()
    assert all(0 <= c and c < 88 for c in exclude_cols)
    exclude_cols.append(87)
    labels = [ "Unnamed", "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", 
        "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
        "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", 
        "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", 
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
        "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", 
        "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "SimillarHTTP", "Inbound", "Label" ]

    return Data(xs=[], ys=[], nt=0, nf=0, label_names=labels)

    url   = 'http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip'
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

