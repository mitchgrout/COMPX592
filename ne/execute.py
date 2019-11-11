"""
ne.execute
    Exposes interfaces for controlling execution
    This includes serial, parallel and distribution execution schemes
"""

from multiprocess import cpu_count, Pool, Process, Queue

class Executor(object):
    def __init__(self): pass
    def run(self, func, arglist): raise NotImplementedError()
    def num_workers(self):        raise NotImplementedError()

class Sequential(Executor):
    def __init__(self, eager=True):
        self.eager = eager
    def run(self, func, arglist):
        res = map(lambda t: func(*t), arglist)
        if self.eager: return list(res)
        else:          return res
    def num_workers(self): return 1

class Parallel(Executor):
    def __init__(self, pool_size=cpu_count()-1):
        self.workers = pool_size
        self.pool    = Pool(processes=pool_size)
    def run(self, func, arglist): return self.pool.starmap(func, arglist)
    def num_workers(self):        return self.workers

class MultiProcess(Executor):
    def __init__(self, process_count=cpu_count()-1):
        self.workers = process_count
    
    def run(self, func, arglist):
        # TODO: Generalise this a bit maybe?
        #       We know len(arglist) == self.workers in pretty much all cases

        pool  = []
        queue = Queue()
        for idx, split in enumerate(arglist):
            proc = Process(target=self._eval,
                           args=(idx, func, split, queue))
            pool.append(proc)
            proc.start()
        for proc in pool:
            proc.join()

        yss = [[]]*len(arglist)
        while not queue.empty():
            idx, ys = queue.get()
            yss[idx] = ys
        ret = [y for ys in yss for y in ys]
        #from code import interact
        #interact(local=locals())
        return ret

    def _eval(self, idx, f, xs, queue):
        queue.put((idx, list(map(f, xs))))

    def num_workers(self): 
        return self.workers

class Distributed(Executor):
    def __init__(self):           raise NotImplementedError()
    def run(self, func, arglist): raise NotImplementedError()
    def num_workers(self):        raise NotImplementedError()
