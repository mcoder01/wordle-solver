import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Value

class ProcessPool:
    def __init__(self):
        self.executor = ProcessPoolExecutor()
        self.workers = os.cpu_count()

    def submit(self, size, target, reduce=None):
        index = Value('i')
        lock = Lock()

        global run
        def run(): 
            local_result = None
            while True:
                with lock:
                    if index.value == size:
                        break
                    i = index.value
                    index.value += 1
                local_result = target(i, *local_result) if local_result else target(i)
            return local_result

        futures = []
        for _ in range(min(size, self.workers)):
            futures.append(self.executor.submit(run))

        if reduce:
            while len(futures) > 1:
                results = len(futures)
                new_futures = []
                for i in range(0, results-1, 2):
                    res1 = futures[i].result()
                    res2 = futures[i+1].result()
                    new_futures.append(self.executor.submit(reduce, res1, res2))
                if results%2 == 1:
                    new_futures.append(futures[-1])
                futures = new_futures
            return futures[0].result()
        
        for future in futures:
            future.result()

    def close(self):
        self.executor.shutdown()