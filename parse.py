import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_landmarks(res):
    pos = []
    for i in range(100):
         if f'X_{i}' not in res:
             break
         pos.append((res[f'X_{i}'], res[f'Y_{i}'], res[f'Z_{i}']))
    assert len(pos)
    return np.array(pos)


for i, x in enumerate('001 030 233 666'.split()):
    with open(f'/tmp/{x}.pkl', 'rb') as f:
        pos = get_landmarks(pickle.load(f))

    ax = plt.subplot(221 + i, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='^')

plt.show()
