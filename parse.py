import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('/tmp/233.pkl', 'rb') as f:
    res = pickle.load(f)


pos = []
for i in range(100):
     if f'X_{i}' not in res:
         break
     pos.append((res[f'X_{i}'], res[f'Y_{i}'], res[f'Z_{i}']))
assert len(pos)
pos = np.array(pos)

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='^')
plt.show()
