"""
ne.data.kdd99
    KDD99 data
"""

from ne.data import Dataset

def loader():
    from os.path import exists
    import wget
    import gzip

    urlf  = 'http://kdd.ics.uci.edu/databases/kddcup99/{}'
    pathf = 'ne/data/kdd99/{}'
    fname = 'kddcup.data.gz'
    seen_lines = set()
    categorical_tables = {
        1: {},
        2: {},
        3: {},
    }

    if not exists(pathf.format(fname)):
        wget.download(url=urlf.format(fname), out=pathf.format(fname))
    
    with gzip.open(pathf.format(fname), 'rb') as fd:
        for line in fd:
            line = line.decode('ascii').replace(' ', '').strip()
            parts = line.split(',')
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
            yield list(map(float, parts))
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

dataset = Dataset('kdd99', loader, labels)
