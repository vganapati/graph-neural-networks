import sys
import time
from threading import Thread
from multiprocessing import Pool

COUNT = 500000000

def countdown(n):
    while n>0:
        n -= 1

if __name__ == '__main__':

    a = []
    b = a
    print(sys.getrefcount(a))

    

    # Single threaded
    start = time.time()
    countdown(COUNT)
    end = time.time()

    print('Time taken in seconds -', end - start)

    # Multi-threaded, no benefit as there is only 1 GIL

    t1 = Thread(target=countdown, args=(COUNT//2,))
    t2 = Thread(target=countdown, args=(COUNT//2,))

    start = time.time()
    t1.start()
    t2.start()
    t1.join() # wait for thread to finish
    t2.join() # wait for thread to finish
    end = time.time()
    print('Multi-threaded time taken in seconds -', end - start)




    # multiprocessing, separate GILs for each process

    pool = Pool(processes=2)
    start = time.time()
    r1 = pool.apply_async(countdown, [COUNT//2])
    r2 = pool.apply_async(countdown, [COUNT//2])
    pool.close()
    pool.join()
    end = time.time()
    print('Multi-processing time taken in seconds -', end - start)