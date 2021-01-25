import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from Math import get_corners
from PIL import Image
from Reader import Reader
mpl.use('QT5Agg')


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


def draw(image, bbox, proj_matrix, dimensions, gt_trans, est_trans, rotation_x, rotation_y, rotation_z, patch, yaw, roll):
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
        dimensions, est_trans, patch, yaw, roll)

    draw_projection(truth_corners, proj_matrix, ax, 'orange')  # 真实3D框
    draw_projection(est_corners, proj_matrix, ax, 'red')  # 预测3D框

    draw_2dbbox(bbox, ax, 'green')


def main():

    # 标签文件路径
    LABEL_DIR = '/Users/jellyfive/Desktop/实验/Dataset/BuildingData/training/label_2'
    IMAGE_DIR = '/Users/jellyfive/Desktop/实验/Dataset/BuildingData/training/image_2'
    CALIB_DIR = '/Users/jellyfive/Desktop/实验/Dataset/BuildingData/training/calib'

    # 读取标签文件
    label_reader = Reader(IMAGE_DIR, LABEL_DIR, CALIB_DIR)
    show_indices = label_reader.indices

    for index in show_indices:
        data_label = label_reader.data[index]

        proj_matrix = data_label['camera_to_image']
        image = Image.open(data_label['image_path'])

        for tracklet in data_label['tracklets']:
            bbox, dim, loc, r_x, r_y, r_z = [tracklet['bbox'], tracklet['dimensions'],
                                             tracklet['location'], tracklet['rotation_x'], tracklet['rotation_y'], tracklet['rotation_z']]

            # 画图
            draw(image, bbox, proj_matrix, dim, loc,
                 loc, r_x, r_y, r_z, r_x, r_y, r_z)

        # plt.show()
        plt.savefig(
            '/Users/jellyfive/Desktop/实验/3D-pose-estimation--translation/output_6/{}_proj'.format(index))
        # plt.close()


if __name__ == "__main__":
    main()
