categorical_tables = {
    # 0-based indices
    4:  {},
    5:  {},
    13: {}        
}

if __name__ == '__main__':
    for filename in [ 'UNSW-NB15_{}.csv'.format(idx) for idx in [1,2,3,4] ]:
        with open(filename, 'r') as fd:
            ipv4_to_int = lambda ip: sum( int(t[0]) << (8*t[1]) \
                                     for t in zip(ip.split('.'), [3,2,1,0]) )
            for line in fd:
                # NOTE: Hack to deal with sometimes-UTF8 data
                parts = line.strip().encode('ascii', 'ignore').decode('ascii').split(',')
                parts[0] = ipv4_to_int(parts[0])
                parts[2] = ipv4_to_int(parts[2])
                # The ports are sometimes in hex, or missing entirely
                for col in [1,3]:
                    if parts[col][:2] == '0x':
                        parts[col] = str(int(parts[col], base=16))
                    elif parts[col] == '-':
                        parts[col] = '0'

                # Change nominal to categorical
                for col_id in categorical_tables.keys():
                    x = parts[col_id]
                    d = categorical_tables[col_id]
                    if x not in d:
                        d[x] = len(d)
                    parts[col_id] = d[x]

                # Coalesce all non-DoS traffic into the Normal category
                parts[47] = int(parts[47] == 'DoS')
                # No longer need the Label column now; also drop these three for missing data
                ignored_cols = [37, 38, 39, 48]
                b = False
                for idx, c in [ (idx, parts[idx]) for idx in range(len(parts)) if idx not in ignored_cols ]:
                    if b: print(',', end='')
                    print(c, end='')
                    b = True
                print('')

