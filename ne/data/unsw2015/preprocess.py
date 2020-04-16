# Al-Zewairi style preprocessing

categorical_tables = {
    # 0-based indices
    4:  {},
    5:  {},
    13: {},
    37: {}, #
    38: {}, # Currently excluded
    39: {}, #
}

if __name__ == '__main__':
    lno = 0
    seen_lines = set()

    for filename in [ 'UNSW-NB15_{}.csv'.format(idx) for idx in [1,2,3,4] ]:
        with open(filename, 'r') as fd:
            ipv4_to_int = lambda ip: sum( int(t[0]) << (8*t[1]) \
                                     for t in zip(ip.split('.'), [3,2,1,0]) )
            for line in fd:
                lno += 1
                # Weird absence of positive examples here, filter out
                # if lno > 186_800 and lno < (700_000+387_240):
                #    continue

                line = ''.join([ x for x in line if x not in (' ', '\t') ])
                if line in seen_lines: continue
                seen_lines.add(line)

                # NOTE: Hack to deal with sometimes-UTF8 data
                parts = line.strip().encode('ascii', 'ignore').decode('ascii').split(',')
                if parts[0] == '127.0.0.1': continue # Stray record
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

                if not True:
                    # Coalesce all non-DoS traffic into the Normal category
                    parts[47] = int(parts[47] == 'DoS')
                else:
                    # Coalesce all non-Normal into the DoS category
                    parts[47] = parts[48]


                # Columns to be removed
                ignored_cols = [0, 2, 48]
                b = False
                for idx, c in [ (idx, parts[idx]) for idx in range(len(parts)) if idx not in ignored_cols ]:
                    if b: print(',', end='')
                    print(c, end='')
                    b = True
                print('')

    import sys
    print(categorical_tables, file=sys.stderr)
