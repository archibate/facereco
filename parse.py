import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('/tmp/233.pkl', 'rb') as f:
    res = pickle.load(f)


pos = []
for i in range(100):
     if f'x_{i}' not in res:
         break
     pos.append((res[f'x_{i}'], res[f'y_{i}']))
assert len(pos)
pos = np.array(pos)

plt.scatter(pos[:, 0], pos[:, 1])
plt.show()
