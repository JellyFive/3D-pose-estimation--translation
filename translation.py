import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from PIL import Image
from evaluate import trans_error
from drawBox import draw
from Reader import Reader
from Math import get_R
mpl.use('QT5Agg')


def get_location_1(box_2d, dimension, rotation_x, rotation_y, rotation_z, proj_matrix):
    """
    方法1 2Dbbox中心与3Dbbox中心重合
    只存在一个中心点间的对应关系。难以约束。
    若是将Z的值替换成真实值，效果还行。Z方向的值与XY相比差距太大。
    """
    R = get_R(rotation_x, rotation_y)

    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    h, w, l = dimension[0], dimension[1], dimension[2]

    constraints = [0, -h/2, 0]
    corners = [(xmin+xmax)/2, (ymin+ymax)/2]

    # create pre M (the term with I and the R*X)
    M = np.zeros([4, 4])

    for i in range(0, 4):
        M[i][i] = 1

    # create A, b
    A = np.zeros([2, 3], dtype=np.float)
    b = np.zeros([2, 1])

    RX = np.dot(R, constraints)
    M[:3, 3] = RX.reshape(3)

    M = np.dot(proj_matrix, M)

    A[0, :] = M[0, :3] - corners[0] * M[2, :3]  # [540 0 960] - 1116[0 0 1]
    b[0] = corners[0] * M[2, 3] - M[0, 3]

    A[1, :] = M[1, :3] - corners[1] * M[2, :3]  # [540 0 960] - 1116[0 0 1]
    b[1] = corners[1] * M[2, 3] - M[1, 3]

    loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # loc = [loc[0][0], loc[1][0] + dimension[0] / 2, loc[2][0]]
    loc = [loc[0][0], loc[1][0], loc[2][0]]
    return loc


def get_location_2(box_2d, dimension, rotation_x, rotation_y, rotation_z, proj_matrix):
    """
    2D框包含3D框作为约束条件
    """
    R = get_R(rotation_x, rotation_y, rotation_z)
    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system ; 原数据集中第一个是height（车高度），第二个是width（车两侧的距离），第三个是length(车头到车尾)
    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width

    left_mult = 1
    right_mult = -1

    switch_mult = -1  # -1

    for i in (-2, 0):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-2, 0):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i*dx, -dy*2, j*dz])
    for i in (-1, 1):
        for j in (-1, 1):
            bottom_constraints.append([i*dx, 0, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(
        set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]  # 4约束，对应上下左右 ，shape=4,3
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]  # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([4, 3], dtype=np.float)
        b = np.zeros([4, 1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]

            M = M_array[row]  # 一个对角是1的4*4方阵 .shape = 4*4

            # create M for corner Xx
            RX = np.dot(R, X)  # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4
            M[:3, 3] = RX.reshape(3)

            # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1
            M = np.dot(proj_matrix, M)

            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1  # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc


def get_location_3(box_2d, dimension, rotation_x, rotation_y, rotation_z, proj_matrix):
    """
    2D框包含3D框作为约束条件，增加left和right的约束
    """
    R = get_R(rotation_x, rotation_y, rotation_z)
    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    box_corners = [xmin, ymin, xmax, ymax]
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system ; 原数据集中第一个是height（车高度），第二个是width（车两侧的距离），第三个是length(车头到车尾)
    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width

    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-2, 0):
                left_constraints.append([i * dx, k * dy, j * dz])
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-2, 0):
                right_constraints.append([i * dx, k * dy, j * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i*dx, -dy*2, j*dz])
    for i in (-1, 1):
        for j in (-1, 1):
            bottom_constraints.append([i*dx, 0, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(
        set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]  # 4约束，对应上下左右 ，shape=4,3
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]  # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([4, 3], dtype=np.float)
        b = np.zeros([4, 1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]  # 一个对角是1的4*4方阵 .shape = 4*4
            # create M for corner Xx
            RX = np.dot(R, X)  # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4
            M[:3, 3] = RX.reshape(3)
            # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1
            M = np.dot(proj_matrix, M)
            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1  # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc


def get_location_4(box_2d, dimension, rotation_x, rotation_y, rotation_z, proj_matrix):
    """
    2D框包含3D框作为约束条件。
    固定住坐标系，约束为64-30=34
    再考虑长宽置换的情况，最终约束为34*2=68
    考虑000这种特殊情况
    """
    R = get_R(rotation_x, rotation_y, rotation_z)
    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    left_constraints_2 = []
    right_constraints_2 = []
    top_constraints_2 = []
    bottom_constraints_2 = []

    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width

    left_mult = -1
    right_mult = 1

    for i in (-2, 0):
        for j in (-1, 1):
            left_constraints.append([left_mult * dx, i*dy, j * dz])
    for i in (-2, 0):
        for j in (-1, 1):
            right_constraints.append([right_mult * dx, i*dy, j * dz])
    # 考虑长宽的置换
    for i in (-2, 0):
        for j in (-1, 1):
            left_constraints_2.append([left_mult * dz, i*dy, j * dx])
    for i in (-2, 0):
        for j in (-1, 1):
            right_constraints_2.append([right_mult * dz, i*dy, j * dx])

    for i in (-1, 1):
        top_constraints.append([i*dx, -dy*2, dz])
    for i in (-1, 1):
        bottom_constraints.append([i*dx, 0, -dz])
    # 考虑长宽的置换
    for i in (-1, 1):
        top_constraints_2.append([i*dz, -dy*2, dx])
    for i in (-1, 1):
        bottom_constraints_2.append([i*dz, 0, -dx])

    # now, 128 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    for left in left_constraints_2:
        for top in top_constraints_2:
            for right in right_constraints_2:
                for bottom in bottom_constraints_2:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(
        set(tuple(i) for i in x)), constraints)
    # print(len(list(constraints)))

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # print('constraint:',constraint)
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]  # 4约束，对应上下左右 ，shape=4,3
        # print('X_array:',X_array)
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]  # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([4, 3], dtype=np.float)
        b = np.zeros([4, 1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            # x_array is four constrains for up bottom left and right;
            # x is one point in world World coordinate system, .shape = 3
            M = M_array[row]  # 一个对角是1的4*4方阵 .shape = 4*4

            # create M for corner Xx
            RX = np.dot(R, X)  # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4
            M[:3, 3] = RX.reshape(3)

            # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1
            M = np.dot(proj_matrix, M)

            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1  # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc


def get_location_5(box_2d, dimension, rotation_x, rotation_y, rotation_z, proj_matrix):
    """
    2D框包含3D框作为约束条件。
    固定住坐标系，约束为64-30=34
    再考虑长宽置换的情况，最终约束为34*2=68
    不考虑000这种特殊情况
    """
    R = get_R(rotation_x, rotation_y, rotation_z)
    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]
    # print('box_corners:',box_corners)
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    left_constraints_2 = []
    right_constraints_2 = []
    top_constraints_2 = []
    bottom_constraints_2 = []

    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width

    left_mult = -1
    right_mult = 1

    for j in (-1, 1):
        left_constraints.append([left_mult * dx, -2*dy, j * dz])
    for j in (-2, 0):
        right_constraints.append([right_mult * dx, j*dy, -dz])
    left_constraints.append([left_mult * dx, 0, dz])
    # 考虑长宽的置换
    for j in (-1, 1):
        left_constraints_2.append([left_mult * dz, -2*dy, j * dx])
    for j in (-2, 0):
        right_constraints.append([right_mult * dz, j*dy, -dx])
    left_constraints_2.append([left_mult * dz, 0, dx])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1, 1):
        top_constraints.append([i*dx, -dy*2, dz])
    for i in (-1, 1):
        bottom_constraints.append([i*dx, 0, -dz])
    # 考虑长宽的置换
    for i in (-1, 1):
        top_constraints_2.append([i*dz, -dy*2, dx])
    for i in (-1, 1):
        bottom_constraints_2.append([i*dz, 0, -dx])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    for left in left_constraints_2:
        for top in top_constraints_2:
            for right in right_constraints_2:
                for bottom in bottom_constraints_2:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(
        set(tuple(i) for i in x)), constraints)
    # print(len(list(constraints)))

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # print('constraint:',constraint)
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]  # 4约束，对应上下左右 ，shape=4,3
        # print('X_array:',X_array)
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]  # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([4, 3], dtype=np.float)
        b = np.zeros([4, 1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]  # 一个对角是1的4*4方阵 .shape = 4*4

            # create M for corner Xx
            RX = np.dot(R, X)  # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4
            M[:3, 3] = RX.reshape(3)

            # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1
            M = np.dot(proj_matrix, M)

            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1  # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc


def get_location_6(box_2d, dimension, rotation_x, rotation_y, rotation_z, proj_matrix):
    """
    2D框包含3D框作为约束条件。
    固定住坐标系，约束为64-30=34
    再考虑长宽置换的情况，最终约束为34*2=68
    不考虑000这种特殊情况
    2D框中心也是3D框的中心
    """
    R = get_R(rotation_x, rotation_y, rotation_z)
    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]
    # print('box_corners:',box_corners)
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    left_constraints_2 = []
    right_constraints_2 = []
    top_constraints_2 = []
    bottom_constraints_2 = []

    # using a different coord system ; 原数据集中第一个是height（车高度），第二个是width（车两侧的距离），第三个是length(车头到车尾)
    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width

    left_mult = -1
    right_mult = 1

    for j in (-1, 1):
        left_constraints.append([left_mult * dx, -2*dy, j * dz])
    for j in (-2, 0):
        right_constraints.append([right_mult * dx, j*dy, -dz])
    left_constraints.append([left_mult * dx, 0, dz])
    # 考虑长宽的置换
    for j in (-1, 1):
        left_constraints_2.append([left_mult * dz, -2*dy, j * dx])
    for j in (-2, 0):
        right_constraints.append([right_mult * dz, j*dy, -dx])
    left_constraints_2.append([left_mult * dz, 0, dx])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1, 1):
        top_constraints.append([i*dx, -dy*2, dz])
    for i in (-1, 1):
        bottom_constraints.append([i*dx, 0, -dz])
    # 考虑长宽的置换
    for i in (-1, 1):
        top_constraints_2.append([i*dz, -dy*2, dx])
    for i in (-1, 1):
        bottom_constraints_2.append([i*dz, 0, -dx])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    for left in left_constraints_2:
        for top in top_constraints_2:
            for right in right_constraints_2:
                for bottom in bottom_constraints_2:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(
        set(tuple(i) for i in x)), constraints)
    # print(len(list(constraints)))

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # 约束2 中心点
    constraints_center = [0, dy, 0]
    corners = [(xmin+xmax)/2, (ymin+ymax)/2]

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    # 约束1
    count = 0
    for constraint in constraints:
        # print('constraint:',constraint)
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]  # 4约束，对应上下左右 ，shape=4,3
        # print('X_array:',X_array)
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]  # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([6, 3], dtype=np.float)
        b = np.zeros([6, 1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            # x_array is four constrains for up bottom left and right;
            # x is one point in world World coordinate system, .shape = 3
            M = M_array[row]  # 一个对角是1的4*4方阵 .shape = 4*4

            # create M for corner Xx
            RX = np.dot(R, X)  # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4
            M[:3, 3] = RX.reshape(3)

            # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1
            M = np.dot(proj_matrix, M)

            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        # 2D中心是3D的投影中心
        M_cer = np.zeros([4, 4])
        # 1's down diagonal
        for i in range(0, 4):
            M_cer[i][i] = 1

        RX_cer = np.dot(R, constraints_center)
        M_cer[:3, 3] = RX_cer.reshape(3)

        M_cer = np.dot(proj_matrix, M_cer)
        A[4, :] = M_cer[0, :3] - corners[0] * \
            M_cer[2, :3]  # [540 0 960] - 1116[0 0 1]
        b[4] = corners[0] * M_cer[2, 3] - M_cer[0, 3]

        A[5, :] = M_cer[1, :3] - corners[1] * \
            M_cer[2, :3]  # [540 0 960] - 1116[0 0 1]
        b[5] = corners[1] * M_cer[2, 3] - M_cer[1, 3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1  # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc


def main():

    # 标签文件路径
    LABEL_DIR = '/Users/jellyfive/Desktop/实验/Dataset/BuildingData/training/label_2'
    IMAGE_DIR = '/Users/jellyfive/Desktop/实验/Dataset/BuildingData/training/image_2'
    CALIB_DIR = '/Users/jellyfive/Desktop/实验/Dataset/BuildingData/training/calib'

    # 读取标签文件
    label_reader = Reader(IMAGE_DIR, LABEL_DIR, CALIB_DIR)
    show_indices = label_reader.indices
    sum_data = len(show_indices)

    trans_errors_norm = []
    trans_errors_single = []

    count = 0
    sum_data = 0

    for index in show_indices:
        data_label = label_reader.data[index]

        proj_matrix = data_label['camera_to_image']
        image = Image.open(data_label['image_path'])

        for tracklet in data_label['tracklets']:
            bbox, dim, loc, r_x, r_y, r_z = [tracklet['bbox'], tracklet['dimensions'],
                                             tracklet['location'], tracklet['rotation_x'], tracklet['rotation_y'], tracklet['rotation_z']]

            location = get_location_4(bbox, dim, r_x, r_y, r_z, proj_matrix)

            print('Truth pose: %s' % loc)
            print('Estimated pose_2: %s' % location)
            print('-------------')

            error = trans_error(loc, location)
            trans_errors_norm.append(error[0])
            trans_errors_single.append(error[1])

            if error[1][0] <= 3 and error[1][1] <= 3 and error[1][2] <= 10:
                count += 1
            sum_data += 1

            # 画图
            # location = np.array(location)
            # draw(image, bbox, proj_matrix, dim, loc, location, r_x, r_y, r_z)

        # plt.show()
        # plt.savefig(
        #     '/Users/jellyfive/Desktop/实验/Translation/output_5/{}_proj'.format(index))
        # plt.close()

    mean_trans_error_norm = np.mean(trans_errors_norm)
    mean_trans_error_single = np.mean(trans_errors_single, axis=0)

    print("\tMean Trans Error Norm: {:.3f}".format(mean_trans_error_norm))
    print("\tMean Trans Errors: X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(mean_trans_error_single[0],
                                                                        mean_trans_error_single[1],
                                                                        mean_trans_error_single[2]))
    print("\tMean Average Precision: {:.3f}".format(count / sum_data))


if __name__ == "__main__":
    main()
