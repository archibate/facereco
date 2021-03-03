import pickle
import matrix
import numpy as np
import matplotlib.pyplot as plt

def mse(x, y):
    return np.linalg.norm(x - y)

# 0
# xiaba
# 17
# eyebrow
# 28
# nose
# 36
# eyes
# 48
# mouse
# 68

def get_landmarks(res):
    if isinstance(res, str):
        with open(res, 'rb') as f:
            res = pickle.load(f)

    pos = []
    A = matrix.translate([res['pose_Tx'], res['pose_Ty'], res['pose_Tz']]) @ matrix.eularXYZ([res['pose_Rx'], res['pose_Ry'], res['pose_Rz']])
    A = np.linalg.inv(A)
    for i in range(68):
         if f'X_{i}' not in res:
             break
         p = np.array([res[f'X_{i}'], res[f'Y_{i}'], res[f'Z_{i}']])
         p = matrix.np43(A @ matrix.np34(p))
         pos.append(p)
    assert len(pos)
    pos = np.array(pos)
    bmax = np.max(pos, axis=0, keepdims=True)
    bmin = np.min(pos, axis=0, keepdims=True)
    pos = (pos - bmin) / (bmax - bmin)
    pos[:, 2] *= 0.5
    return pos
    #return pos[0:17]
    #return pos[17:28]
    #return pos[28:36]
    #return pos[36:48]
    #return pos[48:68]


pos = [get_landmarks(f'/tmp/{x}.pkl') for x in '001 030 233 666'.split()]
print(1, mse(pos[0], pos[1]))
print(0, mse(pos[0], pos[2]))
print(0, mse(pos[0], pos[3]))
print(0, mse(pos[1], pos[2]))
print(0, mse(pos[1], pos[3]))
print(1, mse(pos[2], pos[3]))

exit(1)
'''
for i, x in enumerate('001 030 233 666'.split()):
    pos = get_landmarks(f'/tmp/{x}.pkl')
    ax = plt.subplot(221 + i, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 2], -pos[:, 1], c='r', marker='.')

plt.show()
'''
