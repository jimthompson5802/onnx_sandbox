# source: https://stackoverflow.com/questions/43367138/multithreaded-wordcount-and-global-dictionary-update-in-python
# source: https://stackoverflow.com/questions/1312331/using-a-global-dictionary-with-threads-in-python

import time
from threading import Thread
from functools import reduce


def add_to_counter(ctr):
    for i in range(100000):
        ctr['ctr'] = ctr.get('ctr', 0) + 1

def add_unique_key(ctr, key_prefix):
    for i in range(100000):
        key = f'{key_prefix}_{i:06d}'
        ctr[key] = 1


# Do sequentially
ctr_seq = {}
t0 = time.perf_counter()
add_to_counter(ctr_seq)
add_to_counter(ctr_seq)
add_to_counter(ctr_seq)
print(f'sequential in {1000* (time.perf_counter() - t0)} msecs, ctr is {ctr_seq["ctr"]}')

# do multithreaded common key
ctr_mt = {}

t1 = Thread(target=add_to_counter, args=(ctr_mt,))
t2 = Thread(target=add_to_counter, args=(ctr_mt,))
t3 = Thread(target=add_to_counter, args=(ctr_mt,))


t0 = time.perf_counter()
t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()
print(f'multithreaded common key in {1000* (time.perf_counter() - t0)} msecs, ctr is {ctr_mt["ctr"]}')

# do multithreaded unique key
ctr_uniq = {}

t1 = Thread(target=add_unique_key, args=(ctr_uniq, 't1'))
t2 = Thread(target=add_unique_key, args=(ctr_uniq, 't2'))
t3 = Thread(target=add_unique_key, args=(ctr_uniq, 't3'))


t0 = time.perf_counter()
t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()
sum_= 0
for k,v in ctr_uniq.items():
    sum_ += ctr_uniq[k]
print(f'multithreaded unique key in {1000* (time.perf_counter() - t0)} msecs, ctr is {sum_}')

try:
    assert ctr_seq['ctr'] == ctr_mt['ctr']
    print('sequential and common key  equal')
except AssertionError:
    print('sequential and common key NOT equal')

try:
    assert ctr_seq['ctr'] == sum_
    print('sequential and unique keys equal')
except AssertionError:
    print('sequential and unique keys NOT equal')
