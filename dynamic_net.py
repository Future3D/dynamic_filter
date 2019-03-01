# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.utils.data
import torch.utils.checkpoint as checkpoint
import inspect
import scipy.io as scio
import h5py
import numpy as np
import sys
import cv2
import random
import os
import math
import time
import sys
import scipy


class demo_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path=None, indices=None):
        self.data_path = data_path
        indices = sorted(indices)
        self.indices = np.array(indices)
        with h5py.File(self.data_path, 'r') as f:
            self.xyz = f['xyz'][:][self.indices]
            self.rgb = f['rgb'][:][self.indices]
            self.label = f['label'][:][self.indices]
        self.num_classes = 4
        print('data set init.')

    def __getitem__(self, index):  # 返回的是tensor
        xyz = self.xyz[index, ...]
        xyz = torch.from_numpy(xyz)
        rgb = self.rgb[index, ...]
        rgb = torch.from_numpy(rgb)
        label = self.label[index, ...]
        label = torch.from_numpy(label).long()
        return rgb, xyz, label

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            data = f['rgb'][:][self.indices]
            return int(data.shape[0])


class nyud_dataset(torch.utils.data.Dataset):
    # assume that the dataset is 1449,425,560,4 and 1449,425,560,40 and
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.num_classes = 40
        with h5py.File(self.data_path, 'r') as f:
            self.xyz = f['depth'][:]
            self.rgb = f['rgb'][:]
            self.label = f['label'][:]
        print('data set init.')

    def __getitem__(self, index):  # 返回的是tensor
        xyz_ = torch.from_numpy(self.xyz[index, ...])
        rgb_ = torch.from_numpy(self.rgb[index, ...])
        label_ = torch.from_numpy(self.label[index, ...]).long()
        return rgb_.permute(2, 0, 1), xyz_, label_

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            data = f['label'][:, 0, 0]
            return int(data.shape[0])


class nyud_dataset2(torch.utils.data.Dataset):
    # assume that the dataset is 1449,425,560,4 and 1449,425,560,40 and
    def __init__(self, data_path=None, augmentation=False):
        self.data_path = data_path
        self.num_classes = 40
        with h5py.File(self.data_path, 'r') as f:
            self.depth = f['depth'][:]
            self.rgb = f['rgb'][:]
            self.label = f['label'][:]
        self.augmentation = augmentation
        print('data set init.')

    def __getitem__(self, index):  # 返回的是tensor
        if self.augmentation:
            rgb, depth, label = self._scale_and_crop(self.rgb[index, ...], self.depth[index, ...], self.label[index, ...])
            rgb = self._color_jitter(rgb)
        else:
            depth = self.depth[index, ...]
            rgb = self.rgb[index, ...]
            label = self.label[index, ...]
        depth = torch.from_numpy(depth)
        rgb = torch.from_numpy(rgb)
        label = torch.from_numpy(label).long()
        return rgb, depth, label

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            data = f['label'][:, 0, 0]
            return int(data.shape[0])

    def _scale_and_crop(self, img, depth, label):
        l = img.shape[1]
        w = img.shape[0]

        sw = int((random.random() + 0.5)*l)  # 0.5~1.5 * w
        sl = int((random.random() + 0.5)*w)  # 0.5~1.5 * l
        img = cv2.resize(img, (sw, sl), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (sw, sl), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (sw, sl), interpolation=cv2.INTER_NEAREST)

        cropscale = random.uniform(0.6, 0.9)
        cropsizeh = int(l * cropscale)
        cropsizew = int(w * cropscale)
        x1 = random.randint(0, np.maximum(0, w - cropsizew))
        y1 = random.randint(0, np.maximum(0, l - cropsizeh))
        x2 = x1 + cropsizew - 1
        y2 = y1 + cropsizeh - 1

        return img[x1:x2, y1:y2, :], depth[x1:x2, y1:y2], label[x1:x2, y1:y2]

    def _color_jitter(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] += np.random.rand() * 70 - 35
        hsv[:, :, 1] += np.random.rand() * 0.3 - 0.15
        hsv[:, :, 2] += np.random.rand() * 50 - 25
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 360.)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255.)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img


#############
# 输入是rgb和xyz，以每一小块为单位，xyz通过mlp算出输出，rgb先和卷积变换矩阵元素乘，再卷积
#############
class depth_transform(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1):
        super(depth_transform, self).__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.mlp = nn.Sequential(
            nn.Conv1d(27, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(64, 9, 1, stride=1, padding=0, dilation=1, groups=1, bias=True),
        )
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

    def forward(self, xyz):
        batch = xyz.shape[0]
        unfold_xyz = self.unfold(xyz).view(batch, 3, self.kernel_size**2, -1)
        unfold_xyz = unfold_xyz - unfold_xyz[:, :, [4], :]
        # vector = unfold_xyz[:, :, [0, 1, 2, 3, 5, 6, 7, 8], :] - unfold_xyz[:, :, [4], :]
        # vector_ = vector[:, :, [1, 2, 3, 4, 5, 6, 7, 0], :]
        # cross = torch.cross(vector, vector_, dim=1).view(batch, 3*(self.kernel_size**2-1), -1)  # batch, 24, L
        return self.mlp(
                    # torch.cat((unfold_xyz.view(batch, 3*self.kernel_size**2, -1), cross), dim=1)
                    unfold_xyz.view(batch, 3*self.kernel_size**2, -1)
                )


class tramsform_conv(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1):
        super(tramsform_conv, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channel, in_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channel))
        nn.init.xavier_normal_(self.weight.data)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, rgb, transform_mat):
        # batch, C_, L -> batch, C, 9, L -> batch, L, C*9 -> # batch, L, out_c -> batch, out_c, width, length
        batch = rgb.shape[0]
        width = rgb.shape[2] - (self.kernel_size-1) + 2*self.padding - 2*(self.dilation - 1)
        length = rgb.shape[3] - (self.kernel_size-1) + 2*self.padding - 2*(self.dilation - 1)
        return (torch.matmul(
                    torch.mul(
                        transform_mat.unsqueeze(1),
                        self.unfold(rgb).view(batch, rgb.shape[1], self.kernel_size**2, -1)
                    ).view(batch, rgb.shape[1]*(self.kernel_size**2), -1).permute(0, 2, 1),
                    self.weight.permute(1, 2, 3, 0).contiguous().view(-1, self.out_channel)
                ).view(batch, width, length, self.out_channel) + self.bias.view(1, 1, 1, self.out_channel)).permute(0, 3, 1, 2)


class xyz_pool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, type=None):
        super(xyz_pool, self).__init__()
        self.kernel_size = kernel_size
        self.type = type
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        if type == 'max':
            self.unpool = nn.MaxUnpool2d(kernel_size, stride=stride, padding=padding)
        elif type != 'ave':
            print('wrong pool type!!')
            exit()

    def forward(self, xyz, feat, indices, size):  # feat: batch, C, width, length
        batch = xyz.shape[0]
        C = feat.shape[1]
        block_size = self.kernel_size**2
        width = feat.shape[2]
        length = feat.shape[3]
        # if self.type == 'max':
            # unpool_feat = self.unpool(feat, indices, output_size=size)
            # indices = self.unfold((unpool_feat != 0).float()).view(batch, C, block_size, -1)  # batch, C*blocksize, L -> batch, C, blocksize, L
            # # batch, C, blocksize, L -> (batch, blocksize, L)/len(C)
            # weight = torch.sum(indices, 1) / C  # batch, C, blocksize, L -> batch, blocksize, L
            # # batch, blocksize, L * batch, blocksize, L, C'
            # xyz_ = self.unfold(xyz).view(batch, 3, block_size, -1).permute(0, 2, 3, 1)  # batch, C*blocksize, L -> batch, blocksize, L, C'
            # xyz_ = torch.mul(xyz_, weight.unsqueeze(3))  # batch, blocksize, L, C'
            # # batch, blocksize, L, C' -> batch, L, C'
            # xyz_pooled = torch.sum(xyz_, 1).view(batch, width, length, -1).permute(0, 3, 1, 2)
        # else:
        #     xyz_ = self.unfold(xyz).view(batch, 3, block_size, -1).permute(0, 2, 3, 1)  # batch, C*blocksize, L -> batch, blocksize, L, C'
        #     xyz_pooled = torch.sum(xyz_, 1).view(batch, width, length, -1).permute(0, 3, 1, 2) / block_size
        # return xyz_pooled
        if self.type == 'max':
            weight = torch.sum(self.unfold((self.unpool(feat, indices, output_size=size) != 0).float()).view(batch, C, block_size, -1), 1) / C
            xyz_ = self.unfold(xyz).view(batch, 3, block_size, -1).permute(0, 2, 3, 1)
            xyz_pooled = torch.sum(torch.mul(xyz_, weight.unsqueeze(3)), 1).view(batch, width, length, -1).permute(0, 3, 1, 2)
        return xyz_pooled


class DynamicNet(Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.trans1_1 = depth_transform(3, 64, 3, padding=1)
        self.conv1_1 = tramsform_conv(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        # self.xyz_pool1 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.avepool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.trans2_1 = depth_transform(64, 128, 3, padding=1)
        self.conv2_1 = tramsform_conv(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        # self.xyz_pool2 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.avepool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.trans3_1 = depth_transform(128, 256, 3, padding=1)
        self.conv3_1 = tramsform_conv(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        # self.xyz_pool3 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.avepool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.trans4_1 = depth_transform(256, 512, 3, padding=1)
        self.conv4_1 = tramsform_conv(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        # self.xyz_pool4 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.avepool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.trans5_1 = depth_transform(512, 512, 3, padding=2, dilation=2)
        self.conv5_1 = tramsform_conv(512, 512, 3, padding=2, dilation=2)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # self.fc6_ = nn.Conv2d(512, 1024, 1)
        # self.drop6 = nn.Dropout2d(p=0.5)
        # self.fc7_ = nn.Conv2d(1024, 1024, 1)
        # self.drop7 = nn.Dropout2d(p=0.5)
        # self.fc8_ = nn.Conv2d(1024, 40, 1)
        self.fc6 = nn.Conv2d(1024, 1024, 1)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Conv2d(1024, 40, 1)
        print('network DynamicNet init.')

    def forward(self, rgb=None, xyz=None):
        # xyz = xyz.permute(0, 3, 1, 2)
        # feat = rgb.permute(0, 3, 1, 2)
        feat = rgb

        transform_mat = checkpoint.checkpoint(self.trans1_1, xyz)
        feat = F.relu(checkpoint.checkpoint(self.conv1_1, feat, transform_mat), inplace=True)
        feat = F.relu(self.conv1_2(feat), inplace=True)
        # feat = F.relu(self.bn1_1(checkpoint.checkpoint(self.conv1_1, feat, transform_mat)), inplace=True)
        # feat = F.relu(self.bn1_2(self.conv1_2(feat), inplace=True))
        feat, indices = self.pool1(feat)
        xyz = self.avepool1(xyz)

        transform_mat = checkpoint.checkpoint(self.trans2_1, xyz)
        feat = F.relu(checkpoint.checkpoint(self.conv2_1, feat, transform_mat), inplace=True)
        feat = F.relu(self.conv2_2(feat), inplace=True)
        # feat = F.relu(self.bn2_1(checkpoint.checkpoint(self.conv2_1, feat, transform_mat)), inplace=True)
        # feat = F.relu(self.bn2_2(self.conv2_2(feat), inplace=True))
        feat, indices = self.pool2(feat)
        xyz = self.avepool2(xyz)

        transform_mat = checkpoint.checkpoint(self.trans3_1, xyz)
        feat = F.relu(checkpoint.checkpoint(self.conv3_1, feat, transform_mat), inplace=True)
        feat = F.relu(self.conv3_2(feat), inplace=True)
        feat = F.relu(self.conv3_3(feat), inplace=True)
        # feat = F.relu(self.bn3_1(checkpoint.checkpoint(self.conv3_1, feat, transform_mat)), inplace=True)
        # feat = F.relu(self.bn3_2(self.conv3_2(feat), inplace=True))
        # feat = F.relu(self.bn3_3(self.conv3_3(feat), inplace=True))
        feat, indices = self.pool3(feat)
        xyz = self.avepool3(xyz)

        transform_mat = checkpoint.checkpoint(self.trans4_1, xyz)
        feat = F.relu(checkpoint.checkpoint(self.conv4_1, feat, transform_mat), inplace=True)
        feat = F.relu(self.conv4_2(feat), inplace=True)
        feat = F.relu(self.conv4_3(feat), inplace=True)
        # feat = F.relu(self.bn4_1(checkpoint.checkpoint(self.conv4_1, feat, transform_mat)), inplace=True)
        # feat = F.relu(self.bn4_2(self.conv4_2(feat), inplace=True))
        # feat = F.relu(self.bn4_3(self.conv4_3(feat), inplace=True))
        feat, indices = self.pool4(feat)
        xyz = self.avepool4(xyz)

        transform_mat = checkpoint.checkpoint(self.trans5_1, xyz)
        feat = F.relu(checkpoint.checkpoint(self.conv5_1, feat, transform_mat), inplace=True)
        feat = F.relu(self.conv5_2(feat), inplace=True)
        feat = F.relu(self.conv5_3(feat), inplace=True)
        # feat = F.relu(self.bn5_1(checkpoint.checkpoint(self.conv5_1, feat, transform_mat)), inplace=True)
        # feat = F.relu(self.bn5_2(self.conv5_2(feat), inplace=True))
        # feat = F.relu(self.bn5_3(self.conv5_3(feat), inplace=True))
        feat = self.pool5a(self.pool5(feat))

        global_feat = torch.mean(feat.view(feat.shape[0], feat.shape[1], -1), 2).view(feat.shape[0], feat.shape[1], 1, 1)
        feat = torch.cat((feat, global_feat.repeat(1, 1, feat.shape[2], feat.shape[3])), 1)

        feat = self.drop6(F.relu(self.fc6(feat)))
        feat = F.relu(self.fc7(feat))
        feat = self.fc8(self.drop7(feat))

        return feat


class NormalNet(Module):
    def __init__(self):
        super(NormalNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.mclassifier = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 40, 1),
        )
        # self.interp = nn.UpsamplingBilinear2d(size=(480, 640))
        print('network NormalNet init.')

    def forward(self, rgb=None, xyz=None):
        feat = rgb.permute(0, 3, 1, 2)

        feat = self.features(feat)
        feat = self.mclassifier(feat)

        return feat
