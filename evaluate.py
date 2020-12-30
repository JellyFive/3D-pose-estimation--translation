import numpy as np
import math
from pyquaternion import Quaternion
from Math import get_corners


def iou(gt_box, est_box):
    xA = max(gt_box[0], est_box[0])
    yA = max(gt_box[1], est_box[1])
    xB = min(gt_box[2], est_box[2])
    yB = min(gt_box[3], est_box[3])

    if xB <= xA or yB <= yA:
        return 0.

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    boxBArea = (est_box[2] - est_box[0]) * (est_box[3] - est_box[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def trans_error(gt_trans, est_trans):
    """
    :param gt_trans: 真实的三维坐标，np.array([x,y,z])
    :param est_trans: 预测的三维坐标
    :return trans_err_normL: 平方和误差
    :return trans_err_single: x,y,z方向分别的误差
    """
    # L2范式，平方和
    trans_err_norm = np.linalg.norm(gt_trans - est_trans)
    # 绝对值
    trans_err_single = np.abs(gt_trans - est_trans)

    return trans_err_norm, trans_err_single


def rot_error(gt_rot, est_rot):
    """
    :param gt_rot: 真实的旋转角度，np.array([patch,yaw,roll])
    :param est_rot: 预测的旋转角度
    :return rot_error = 2 * arccos(|gt_pose, est_pose|) * 180 / 3.14, 反余弦距离
    """
    # 将欧拉角转换为四元数
    def eulerAnglesToQu(theta):

        q = np.array([math.cos(theta[0]/2)*math.cos(theta[1]/2)*math.cos(theta[2]/2)+math.sin(theta[0]/2)*math.sin(theta[1]/2)*math.sin(theta[2]/2),
                      math.sin(theta[0]/2)*math.cos(theta[1]/2)*math.cos(theta[2]/2) -
                      math.cos(theta[0]/2)*math.sin(theta[1]/2) *
                      math.sin(theta[2]/2),
                      math.cos(theta[0]/2)*math.sin(theta[1]/2)*math.cos(theta[2]/2) +
                      math.sin(theta[0]/2)*math.cos(theta[1]/2) *
                      math.sin(theta[2]/2),
                      math.cos(theta[0]/2)*math.cos(theta[1]/2)*math.sin(theta[2]/2) -
                      math.sin(theta[0]/2)*math.sin(theta[1]/2) *
                      math.cos(theta[2]/2)
                      ])
        return q

    # gt_quat = eulerAnglesToQu(gt_rot)
    # est_quat = eulerAnglesToQu(est_rot)

    # ans = np.dot(gt_quat, est_quat.T)

    # return np.rad2deg(2*math.acos(np.abs(ans)))

    # 与上述等价
    gt_quat = Quaternion(eulerAnglesToQu(gt_rot))
    est_quat = Quaternion(eulerAnglesToQu(est_rot))

    return np.abs((gt_quat * est_quat.inverse).degrees)


def add_err(dim, gt_trans, est_trans, gt_rot, est_rot):
    """
    :param dim:目标的尺寸
    :param gt_trans, gt_rot: 真实的位姿
    :param est_rot, est_rot: 预测的位姿
    :return add_error = 8个二维顶点的欧氏距离的平均值
    """
    gt_corners_3D = get_corners(dim, gt_trans, gt_rot[0], gt_rot[1], gt_rot[2])
    est_corners_3D = get_corners(
        dim, est_trans, est_rot[0], est_rot[1], est_rot[2])

    add_error = np.mean(np.linalg.norm(gt_corners_3D - est_corners_3D, axis=1))

    return add_error


if __name__ == "__main__":
    trans_errors_norm = []
    trans_errors_single = []
    rot_errors = []
    adds = []

    gt_bbox = np.array([120, 200, 400, 700])
    est_bbox = np.array([120, 200, 400, 650])

    dim = np.array([2, 2, 2])

    gt_trans = np.array([1, 2, 3])
    est_trans = np.array([1, 2.2, 3.5])

    gt_rot = np.array([0.5237, -0.5237, 0])
    est_rot = np.array([0.5237, -0.5537, 0])

    if iou(gt_bbox, est_bbox) >= 0.5:

        trans_error = trans_error(gt_trans, est_trans)
        trans_errors_norm.append(trans_error[0])
        trans_errors_single.append(trans_error[1])
        rot_errors.append(rot_error(gt_rot, est_rot))
        adds.append(add_err(dim, gt_trans, est_trans, gt_rot, est_rot))

    mean_trans_error_norm = np.mean(trans_errors_norm)
    mean_trans_error_single = np.mean(trans_errors_single, axis=0)
    mean_rot_error = np.mean(rot_errors)
    mean_add = np.mean(adds)

    print("\tMean Trans Error Norm: {:.3f}".format(mean_trans_error_norm))
    print("\tMean Rotation Error: {:.3f}".format(mean_rot_error))
    print("\tMean Trans Errors: X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(mean_trans_error_single[0],
                                                                        mean_trans_error_single[1],
                                                                        mean_trans_error_single[2]))
    print("\tMean ADD: {:.3f}".format(mean_add))
