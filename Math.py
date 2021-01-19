import numpy as np


# 获取旋转矩阵
def get_R(rotation_x, rotation_y, rotation_z=0):
    R_x = np.array([[1, 0, 0],
                    [0, +np.cos(rotation_x), -np.sin(rotation_x)],
                    [0, +np.sin(rotation_x), +np.cos(rotation_x)]],
                   dtype=np.float32)
    R_y = np.array([[+np.cos(rotation_y), 0, +np.sin(rotation_y)],
                    [0, 1, 0],
                    [-np.sin(rotation_y), 0, +np.cos(rotation_y)]],
                   dtype=np.float32)
    R_z = np.array([[+np.cos(rotation_z), -np.sin(rotation_z), 0],
                    [+np.sin(rotation_z), +np.cos(rotation_z), 0],
                    [0, 0, 1]],
                   dtype=np.float32)
    R = np.dot(R_z, np.dot(R_x, R_y))
    return R


# 世界坐标转相机坐标
def get_corners(dimensions, location, rotation_x, rotation_y, rotation_z):
    # 旋转矩阵
    R = get_R(rotation_x, rotation_y, rotation_z)
    # print('Truth R: %s' % R)
    # 建立世界坐标系
    h, w, l = dimensions
    # 世界坐标系的建立
    x_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]

    corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
    corners_3D += location.reshape((3, 1))
    return corners_3D
