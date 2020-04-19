import numpy as np
import time
a = np.random.rand(100000)
b = np.random.rand(100000)

c = 0
start = time.time()
for i in range(100000):
    c += a[i]+b[i]
end = time.time()
print('for计算耗时%lf'%((end-start)*1000)+'ms')

start = time.time()
c = np.dot(a,b)
end = time.time()
print('向量化运算耗时%lf'%((end-start)*1000)+'ms')