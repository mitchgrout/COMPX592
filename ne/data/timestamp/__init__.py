"""
ne.data.timestamp
    Phony 'packet' data which has two fields: timestamp, length
    Data consists of the following:
        - The first packet is a 'heartbeat' packet
        - A random number of packets follow, between `hi` and `lo`
        - The process repeats
"""

def load_data(num=64, hi=6, lo=2):
    """
    num = number of heartbeats sent in total
    hi = maximum number of packets between two heartbeats
    lo = minimum number ^
    returns an iterator
    """
    
    from collections import namedtuple
    from random import randint, uniform

    Packet = namedtuple('Packet', ['timestamp', 'length'])
    current_time = uniform(0,1)
    HEARTBEAT_LEN = 5 
    xs, ys = [], []
    for _ in range(num): 
        xs.append(Packet(timestamp=current_time, length=HEARTBEAT_LEN))
        ys.append(1.0) 
        new_time = current_time + uniform(0.9, 1.1)
        traffic_times = \
            sorted([ uniform(current_time+0.01, new_time-0.01) \
                     for _ in range(randint(lo,hi)) ])
        for time in traffic_times:
            xs.append(Packet(timestamp=time, length=randint(1,10)))
            ys.append(0.0)
        current_time = new_time
    return xs, ys
