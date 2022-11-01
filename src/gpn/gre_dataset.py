import numpy as np
from scipy import ndimage
import torch.utils.data
import pandas as pd

from gpn.io import *
from gpn.perception import *
from gpn.utils.transform import Rotation, Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.df = pd.read_csv(root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        grasp_pose = self.df.loc[i, "x":"qw"].to_numpy(np.single)
        label = self.df.loc[i, "label"].astype(np.long)

        x, y = grasp_pose, label
        return x, y

## grasp reachability data transform:TODO
def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position
