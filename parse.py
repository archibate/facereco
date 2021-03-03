import pickle
import matrix
import numpy as np
import matplotlib.pyplot as plt


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
    pos = []
    A = matrix.translate([res['pose_Tx'], res['pose_Ty'], res['pose_Tz']]) @ matrix.eularXYZ([res['pose_Rx'], res['pose_Ry'], res['pose_Rz']])
    A = np.linalg.inv(A)
    for i in range(68):
         if f'X_{i}' not in res:
             break
         p = (A @ [res[f'X_{i}'], res[f'Y_{i}'], res[f'Z_{i}'], 1.0])[:3]
         pos.append(p)
    assert len(pos)
    return np.array(pos)


for i, x in enumerate('001 030 233 666'.split()):
    with open(f'/tmp/{x}.pkl', 'rb') as f:
        pos = get_landmarks(pickle.load(f))

    ax = plt.subplot(221 + i, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 2], -pos[:, 1], c='r', marker='.')

plt.show()
