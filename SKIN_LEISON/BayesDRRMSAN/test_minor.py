import numpy as np

a = [[1,2,3],[4,5,6],[7,8,9]]
a = np.array(a)
print(a.shape)


b = [[11,21,31],[41,51,61],[71,81,91]]
b = np.array(b)
print(b.shape)


c = [[12,22,32],[42,52,62],[72,82,92]]
c = np.array(c)
print(c.shape)



d = np.array(list(zip(a,b,c)))
print(d.shape)

print(d[0])


e = [a[0], b[0], c[0]]
e = np.array(e)
print(e.shape)
print(e)

