# -*- coding:utf8 -*-
# import pcl
from struct import pack, unpack
import numpy as np
import h5py
import os
import random
import traceback
import matplotlib.image as mpimg


ITEMS = ['lemon', 'apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap',
        'cell_phone', 'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag', 'food_box',
        'food_can', 'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles',
        'keyboard', 'kleenex', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach',
        'pear', 'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can',
        'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle']

item10 = ['lemon', 'apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap']
item4 = ['lemon', 'apple', 'onion', 'pear']  # similar to 60*70
item1 = ['banana']

read_root = '/media/gaojiefeng/新加卷2/lab/rgbd-dataset/'
read_root2 = '/media/gaojiefeng/7AB60EF3B60EB0251/rgbd-dataset/'
read_root3 = '/home/data/dynamic_filter_dataset/'
save_root = "/media/gaojiefeng/新加卷2/lab/dynamic_filter_dataset/"
save_root2 = "/media/gaojiefeng/7AB60EF3B60EB0251/dynamic_filter_dataset/"
save_root3 = '/home/data/dynamic_filter_dataset/'


'''
这个函数把pcd文件中用一个浮点数表示的rgb值转换为rgb三个整数值
'''
def frgb_rgb(frgb):
    rgb = pack('f', float(frgb))
    rgb = unpack('i', rgb)[0]
    # print(rgb)
    nr = (rgb >> 16) & 0x0000ff
    ng = (rgb >> 8) & 0x0000ff
    nb = (rgb) & 0x0000ff
    return nr, ng, nb


'''
转.xyz文件，也可以用来把pcd转到数组但是不保存
'''
def pcd_xyz(pcd_path, save_path):
    xyz = np.zeros((1, 6)).astype(np.float32)
    with open(pcd_path) as f:
        for i in range(10):
            line = f.readline()
            line = f.readline()
        while line:
            x, y, z, rgb, _, _ = line.split(' ')
            r, g, b = frgb_rgb(rgb)
            xyz = np.concatenate((xyz, np.array([[float(x), float(y), float(z), r, g, b]])), axis=0)
            line = f.readline()
        xyz = xyz[1:, :]
        if save_path == None:
            return xyz
        else:
            np.savetxt(save_path, xyz)


'''
从pcd转h5，根据输入参数决定处理的数量
item:物体类别名， 列表
object:物体序号， 列表
video:角度（1,2,4）， 列表
frame:每视频流的采样帧数（均匀采样）， 数字
point_num:每个文件点云采样数
'''
def pcd_h5(item, object, video, frame, point_num):
    total = len(item)*len(object)*len(video)*frame
    # 类数-物体数-角度数-帧数
    with h5py.File(save_root + '{}-{}-{}-{}-{}.h5'.format(len(item), len(object), len(video), frame, point_num), 'w') as f:
        xyz = f.create_dataset('xyz', (total, point_num, 3))  # 点云采样到1024
        rgb = f.create_dataset('rgb', (total, point_num, 3))  # 点云采样到1024
        label = f.create_dataset('label', (total, 1))
        count = 0
        for it in item:
            print(it)
            for ob in object:
                for vi in video:
                    try:
                        # 帧采样
                        # 逐文件访问，读取xyz和rgb，存到h5
                        command = 'ls -lR {3}{0}/{0}_{1}|grep {0}_{1}_{2}_|wc -l'.format(it, ob, vi, read_root)
                        r = os.popen(command)  # 执行命令行
                        frame_num = r.readlines()[0]  # 读取命令行的输出到一个list，是帧数
                        frame_indices = random.sample(range(1, int(frame_num)+1), frame)
                        for fr in frame_indices:
                            xyzrgb = pcd_xyz(read_root+'{0}/{0}_{1}/{0}_{1}_{2}_{3}.pcd'
                                             .format(it, ob, vi, fr), None)
                            slice = random.sample(range(xyzrgb.shape[0]), point_num)
                            xyz[count, ...] = xyzrgb[slice, 0:3]
                            rgb[count, ...] = xyzrgb[slice, 3:6]
                            label[count, :] = ITEMS.index(it)
                            count += 1
                    except:
                        traceback.print_exc()
                        print(command)
                        print(read_root+'{0}/{0}_{1}/{0}_{1}_{2}_{3}.pcd'.format(it, ob, vi, fr))
                        exit()
    return


'''
把深度图转换成点云，返回M*C的矩阵，M是点数，C是xyz坐标数3。
警告：
由于这个函数经常出bug，所以把debug记录记下来：
    当使用(480, 640)展开时，转点云结果在meshlab中是水平翻转，这是因为转点云的公式中，点云的xy坐标系与图像是一致的，
    而meshlab则采用的是常见的解析几何xyz坐标系（已证实），所以x轴镜像对称，但其实每个点和img和label的对应并没有问题，
    且只要在同一坐标系下，点距离的计算也不会有问题；
    当使用(640, 480)展开时，转点云结果也是(640, 480)展开，只不过在meshlab中不再和原图镜像，因为相当于图像的xy轴对换了。
'''
def depth_pc(depth):
    # depth: width, height
    # KI: 3,3
    # Z: width*height,3
    # pixel_matrix: width*height,3
    # xyz: width*height,3
    # print(depth.shape)
    length = depth.shape[1]
    width = depth.shape[0]
    # scale = width / 480
    scale = 1
    pixel_matrix = np.zeros((width*length, 3))
    for i in range(width):
        pixel_matrix[i*length:(i+1)*length, 0] = range(0, length)
        pixel_matrix[i*length:(i+1)*length, 1] = i
        pixel_matrix[i*length:(i+1)*length, 2] = 1
    cx = float(length/2)
    cy = float(width/2)
    f = 570 * scale

    KI = np.linalg.inv(np.array([[f, 0, 0], [0, f, 0], [cx, cy, 1]]))
    depth = depth.reshape(width*length, 1)
    Z = np.tile(depth, 3)
    # print(Z.shape)
    # print(self.pixel_matrix.shape)
    xyz = np.matmul(Z*pixel_matrix, KI)
    return xyz


'''
深度图转点云，然后和rgb一起存成图片格式的h5
'''
def rgbd_h5(item, object, video, frame, size):
    # 直接一个dataset包括所有维度
    total = len(item)*len(object)*len(video)*frame
    # 类数-物体数-角度数-帧数
    with h5py.File(save_root2 + 'img-{}-{}-{}-{}.h5'.format(len(item), len(object), len(video), frame), 'w') as f:
        xyz = f.create_dataset('xyz', (total, size[0], size[1], 3))
        rgb = f.create_dataset('rgb', (total, size[0], size[1], 3))
        label = f.create_dataset('label', (total, size[0], size[1], 1))
        count = 0
        for it in item:
            print(it)
            for ob in object:
                for vi in video:
                    try:
                        # 帧采样
                        # 逐文件访问，读取xyz和rgb，存到h5
                        command = 'ls -lR {3}{0}/{0}_{1}|grep {0}_{1}_{2}_|wc -l'.format(it, ob, vi, read_root2)
                        r = os.popen(command)  # 执行命令行
                        frame_num = r.readlines()[0]  # 读取命令行的输出到一个list，是帧数
                        frame_indices = random.sample(range(1, int(frame_num)//4+1), frame)
                        for fr in frame_indices:
                            rgb_ = mpimg.imread(read_root2 + '{0}/{0}_{1}/{0}_{1}_{2}_{3}_crop.png'.
                                                format(it, ob, vi, fr))
                            dep = mpimg.imread(read_root2 + '{0}/{0}_{1}/{0}_{1}_{2}_{3}_depthcrop.png'.
                                                format(it, ob, vi, fr))
                            label_ = mpimg.imread(read_root2 + '{0}/{0}_{1}/{0}_{1}_{2}_{3}_maskcrop.png'.
                                                format(it, ob, vi, fr))
                            tmp = depth_pc(dep[0:size[0], 0:size[1]])
                            xyz[count, ...] = np.reshape(tmp, (size[0], size[1], 3))
                            rgb[count, ...] = rgb_[0:size[0], 0:size[1], ...]
                            label[count, :, :, 0] = item.index(it)*label_[0:size[0], 0:size[1]]
                            count += 1
                    except:
                        traceback.print_exc()
                        print(command)
                        print(read_root2 + '{0}/{0}_{1}/{0}_{1}_{2}_{3}'.format(it, ob, vi, fr))
                        exit()
    return


"""
按比例裁切图片，如果为真，保存所有图片，否则保存中心那张
"""
def crop(read_path, save_path, scale, save_all_flag=True):
    with h5py.File(read_root3 + read_path, 'r') as f1:
        with h5py.File(save_root3 + save_path, 'w') as f2:
            print('rgb')
            tmp = f1['rgb'][:]
            scale = int(1/scale)
            rgb_ = f2.create_dataset('rgb', (tmp.shape[0]*(scale**2), tmp.shape[1]/scale, tmp.shape[2]/scale, 3))
            dwidth = int(tmp.shape[1]/scale)
            dlength = int(tmp.shape[2]/scale)
            for k in range(tmp.shape[0]):
                for i in range(scale):
                    for j in range(scale):
                        rgb_[k*(scale**2)+(i*5+j), ...] = tmp[k, i*dwidth:(i+1)*dwidth, i*dlength:(i+1)*dlength, :]
            print('xyz')
            tmp = f1['xyz'][:]
            xyz_ = f2.create_dataset('xyz', (tmp.shape[0]*(scale**2), tmp.shape[1]/scale, tmp.shape[2]/scale, 3))
            for k in range(tmp.shape[0]):
                for i in range(scale):
                    for j in range(scale):
                        xyz_[k*(scale**2)+(i*5+j), ...] = tmp[k, i*dwidth:(i+1)*dwidth, i*dlength:(i+1)*dlength, :]
            print('label')
            tmp = f1['label'][:]
            label_ = f2.create_dataset('label', (tmp.shape[0]*(scale**2), tmp.shape[1]/scale, tmp.shape[2]/scale))
            for k in range(tmp.shape[0]):
                for i in range(scale):
                    for j in range(scale):
                        label_[k*(scale**2)+(i*5+j), ...] = tmp[k, i*dwidth:(i+1)*dwidth, i*dlength:(i+1)*dlength]


"""
按比例从图片集中随机抽出一部分作为新数据集
"""
def pick(read_path, save_path, scale):
    with h5py.File(read_root3 + read_path, 'r') as f1:
        with h5py.File(save_root3 + save_path, 'w') as f2:
            tmp = f1['rgb'][:]
            indices = random.sample(range(tmp.shape[0]), int(tmp.shape[0]*scale))
            f2.create_dataset('rgb', data=tmp[indices, ...])
            tmp = f1['xyz'][:]
            f2.create_dataset('xyz', data=tmp[indices, ...])
            tmp = f1['label'][:]
            f2.create_dataset('label', data=tmp[indices, ...])


if __name__ == '__main__':
    # pcd_xyz('banana_1_1_1.pcd', 'banana_1_1_1.xyz')

    # 10个类，每类1个物体，每物体3视角，每视角50帧
    #
    #
    # make_h5(item10, [1], [1, 2, 4], 50, 900)  # item, object, video, frame, point_num

    # rgbd_h5(item4, [1], [1, 2, 4], 50, (55, 55))  # item, object, video, frame

    # crop('train_pc_in_2d.h5', 'train_pc_in_2d_crop_all.h5', 0.2, save_all_flag=True)
    # crop('test_pc_in_2d.h5', 'test_pc_in_2d_crop_all.h5', 0.2, save_all_flag=True)

    pick('train_pc_in_2d.h5', 'train_pc_in_2d_picked.h5', 0.2)
    pick('test_pc_in_2d.h5', 'test_pc_in_2d_picked.h5', 0.2)
