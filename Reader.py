import os
import numpy as np
import matplotlib as mpl
mpl.use('QT5Agg')


class Reader(object):

    def __init__(self, image, label, calib):

        # 判断路径是否存在
        assert os.path.exists(image)
        assert os.path.exists(label)
        assert os.path.exists(calib)

        # 存放每一个图片的路径、标签、内参
        self.data = {}
        # 存放标签的索引，是全局变量
        self.indices = []

        # 通过标签文件确定文件的索引，找到对应的图片以及相机内参
        for label_file in os.listdir(label):
            if not label_file[0] == '0':
                continue
            # 字典方式存放标签文件
            data = {}
            data['tracklets'] = []
            # 取标签的序号作为索引
            index = label_file.split('.')[0]
            self.indices.append(index)
            # 每张图片数据的路径
            data['image_path'] = os.path.join(image, index + '.png')
            # 每张图片的相机内参
            calib_path = os.path.join(calib, index + '.txt')
            with open(calib_path) as calib_file:
                lines = calib_file.readlines()
                # 读取P2相机的参数3X4的矩阵
                data['camera_to_image'] = np.reshape(lines[2].strip().split(' ')[
                                                     1:], (3, 4)).astype(np.float32)
                calib_file.close()

            label_path = os.path.join(label, index + '.txt')
            with open(label_path) as label_file:
                lines = label_file.readlines()
                for line in lines:
                    elements = line.split(' ')
                    bbox = np.array(elements[4: 8], dtype=np.float32)
                    dimensions = np.array(elements[8: 11], dtype=np.float32)
                    location = np.array(elements[11: 14], dtype=np.float32)
                    rotation_x = np.array(elements[14], dtype=np.float32)
                    rotation_y = np.array(elements[15], dtype=np.float32)
                    rotation_z = np.array(elements[16], dtype=np.float32)
                    data['tracklets'].append({
                        'bbox': bbox,
                        'dimensions': dimensions,
                        'location': location,
                        'rotation_x': rotation_x,
                        'rotation_y': rotation_y,
                        'rotation_z': rotation_z
                    })
            # 每一个索引对应一个图片的路径、相机内参、标签。字典里面存放的字典
            self.data[index] = data
        return
