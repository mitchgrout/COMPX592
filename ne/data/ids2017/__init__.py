"""
ne.data.ids2017
    CIC IDS 2017 data 
"""

from ne.data import Dataset

def loader():
    from os.path import exists
    import wget
    import zipfile
    
    urlf  = 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/{}'
    pathf = 'ne/data/ids2017/{}'
    fname = 'MachineLearningCSV.zip'
    seen_lines = set()
    
    if not exists(pathf.format(fname)):
        wget.download(url=urlf.format(fname), out=pathf.format(fname))

    with zipfile.ZipFile(pathf.format(fname)) as archive:
        for fd in map(archive.open, [ f for f in archive.namelist() if f[-4:] == '.csv' ]):
            fd.readline() # header line
            for line in fd:
                line = line.decode('utf8').replace('Infinity', '0').replace('NaN', '0').strip()
                parts = line.split(',')
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                parts[78] = int(parts[78] != 'BENIGN')
                def safe_float(s):
                    r = float(s)
                    if numpy.isinf(r) or numpy.isnan(r):
                        print(line)
                        return 0
                    return r
                yield list(map(float, parts))
                
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

dataset = Dataset('ids2017', loader, labels)
