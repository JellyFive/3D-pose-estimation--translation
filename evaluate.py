import numpy as np
from pyquaternion import Quaternion


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


def trans_error(gt_pose, est_pose):
    # L2范式，平方和
    trans_err_norm = np.linalg.norm(gt_pose - est_pose)
    # 绝对值
    trans_err_single = np.abs(gt_pose - est_pose)

    return trans_err_norm, trans_err_single


def rot_error(gt_pose, est_pose):
    # 将欧拉角转换为四元数
    def matrix2quaternion(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    gt_quat = Quaternion(matrix2quaternion(gt_pose[:3, :3]))
    est_quat = Quaternion(matrix2quaternion(est_pose[:3, :3]))

    return np.abs((gt_quat * est_quat.inverse).degrees)


if __name__ == "__main__":
    trans_errors_norm = []
    trans_errors_single = []
    rot_errors = []

    gt_trans = np.array([1, 2, 3])
    est_trans = np.array([1, 2.2, 3.5])

    trans_error = trans_error(gt_trans, est_trans)
    trans_errors_norm.append(trans_error[0])
    trans_errors_single.append(trans_error[1])

    mean_trans_error_norm = np.mean(trans_errors_norm)
    mean_trans_error_single = np.mean(trans_errors_single, axis=0)

    print("\tMean Trans Error Norm: {:.3f}".format(mean_trans_error_norm))
    print("\tMean Trans Errors: X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(mean_trans_error_single[0],
                                                                        mean_trans_error_single[1],
                                                                        mean_trans_error_single[2]))
