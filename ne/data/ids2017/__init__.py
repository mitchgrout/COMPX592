"""
ne.data.ids2017
    CIC IDS 2017 data 
"""

from ne.data import Data

def load_best():
    # Negate the set of columns we actually want
    columns  = [10, 12, 13, 22, 39, 40, 41, 42, 52, 54]
    excluded = list(filter(lambda x: x not in columns, range(78)))
    return load_data(exclude_cols=excluded)

def load_data(exclude_cols=[], dedupe=True):
    from os.path import exists
    import wget
    import numpy
    import zipfile
    numpy.random.seed(20192020)

    exclude_cols = exclude_cols.copy()
    assert all(0 <= c and c < 78 for c in exclude_cols)
    exclude_cols.append(78)
    labels = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", 
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", 
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", " Fwd IAT Min", 
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", 
        "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", " Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", 
        "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", 
        "Avg Bwd Segment Size", "Fwd Header Length", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", 
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", " act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label" 
    ]

    urlf  = 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/{}'
    pathf = 'ne/data/ids2017/{}'
    fname = 'MachineLearningCSV.zip'

    categorical_tables = {
    }
    seen_lines = set()
    xs, ys     = [], []
    nt, nf     = 0, 0

    if not exists(pathf.format(fname)):
        wget.download(url=urlf.format(fname), out=pathf.format(fname))
    with zipfile.ZipFile(pathf.format(fname)) as archive:
        for fd in map(archive.open, [ f for f in archive.namelist() if f[-4:] == '.csv' ]):
            fd.readline() # header line
            for line in fd:
                line = line.decode('utf8').replace('Infinity', '0').replace('NaN', '0').strip()
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
                parts[78] = int(parts[78] != 'BENIGN')

                def safe_float(s):
                    r = float(s)
                    if numpy.isinf(r) or numpy.isnan(r):
                        print(line)
                        return 0
                    return r
                x = [ float(x) for (idx, x) in enumerate(parts) if idx not in exclude_cols ]
                y = float(parts[78])
                xs.append(x)
                ys.append(y)

                if y > 0.5: nt += 1
                else:       nf += 1
    xs = numpy.asarray(xs).astype('float32')
    ys = numpy.asarray(ys)
        
    p = numpy.random.permutation(len(xs))
    xs = xs[p]
    ys = ys[p]
    label_names = [ labels[idx] for idx in range(78) if idx not in exclude_cols ]
    return Data(xs=xs, ys=ys, nt=nt, nf=nf, label_names=label_names)

