from locality.helpers.rank import rank
import numpy as np

query_num = 5
num_points = 10
each = np.random.randint(2,10,num_points)
indptr = np.empty(num_points + 1, dtype=np.int)
indptr[0] = 0
indptr[1:] = np.cumsum(each)
dist = np.random.random(indptr[-1])
indices = np.random.randint(0,100,indptr[-1])

confirm = np.empty(query_num * num_points, dtype=np.int)
j = 0
for i in range(num_points):
    a, b = indptr[i], indptr[i+1]
    sort = np.argsort(dist[a:b])
    confirm[j:j + query_num] = indices[a:b][sort[:query_num]]
    j += query_num
    
test = rank(query_num, num_points, dist, indices, indptr)
print(test)
print(confirm)

