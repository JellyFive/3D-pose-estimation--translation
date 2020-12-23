import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('QT5Agg')


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
    R = np.dot(R_x, R_y)
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


def draw_projection(corners, P2, ax, color):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    orders = [[0, 1, 2, 3, 0],
              [4, 5, 6, 7, 4],
              [2, 6], [3, 7],
              [1, 5], [0, 4]]
    for order in orders:
        ax.plot(projection[0, order], projection[1, order],
                color=color, linewidth=2)
    return


def draw_2dbbox(bbox, ax, color):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               color=color, fill=False, linewidth=2))
    ax.text(xmin, ymin, 'building', size='x-large',
            color='white', bbox={'facecolor': 'green', 'alpha': 1.0})
    return


def draw(image, bbox, proj_matrix, dimensions, gt_trans, est_trans, rotation_x, rotation_y, rotation_z=0):
    fig = plt.figure(figsize=(8, 8))

    # 绘制3DBBOX
    ax = fig.gca()
    ax.grid(False)
    ax.set_axis_off()
    ax.imshow(image)

    # 获取8个顶点的世界坐标
    truth_corners = get_corners(
        dimensions, gt_trans, rotation_x, rotation_y, rotation_z)

    est_corners = get_corners(
        dimensions, est_trans, rotation_x, rotation_y, rotation_z)

    draw_projection(truth_corners, proj_matrix, ax, 'orange')  # 真实3D框
    draw_projection(est_corners, proj_matrix, ax, 'red')  # 预测3D框

    draw_2dbbox(bbox, ax, 'green')
