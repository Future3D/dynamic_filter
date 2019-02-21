# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.utils.data
from torch.utils.cpp_extension import load
trans_conv = load(name="trans_conv", sources=["trans_conv.cpp"])
import inspect
import scipy.io as scio
import h5py
import numpy as np
import sys
# import cv2
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
        print('data set init.')

    def __getitem__(self, index):  # 返回的是tensor
        with h5py.File(self.data_path, 'r') as f:
            xyz = f['xyz'][index, ...]
            xyz = torch.from_numpy(xyz)
            rgb = f['rgb'][index, ...]
            rgb = torch.from_numpy(rgb)
            label = f['label'][index, ...]
            label = torch.from_numpy(label).long()
        return rgb, xyz, label

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            data = f['label'][:]
            return int(data.shape[0])


#############
# 输入是rgb和xyz，以每一小块为单位，xyz通过mlp算出输出，rgb先和卷积变换矩阵元素乘，再卷积
#############
class depth_transform(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1):
        super(depth_transform, self).__init__()
        # self.l1 = torch.nn.Linear(kernel_size*kernel_size*3, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.relu1 = nn.ReLU()
        # self.l2 = torch.nn.Linear(64, 9)
        # self.bn2 = nn.BatchNorm1d(9)
        # self.relu2 = nn.ReLU()
        # self.padding = padding
        # self.stride = stride
        # self.dilation = dilation
        # self.in_channel = in_channel
        # self.out_channel = out_channel
        # self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.l1 = nn.Linear(kernel_size*kernel_size*3, 64)  # 输入是batch,L,C_
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(64, 9)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, xyz):
        # xyz_ = self.unfold(xyz).permute(0, 2, 1)  # batch, L, C_，其中c_是先把kernel展开，再把多个channel的kerneal接起来
        # tmp = self.l1(xyz_)
        # tmp = self.bn1(tmp.permute(0, 2, 1))
        # tmp = self.relu1(tmp)
        # tmp = self.l2(tmp.permute(0, 2, 1))
        # tmp = self.bn2(tmp.permute(0, 2, 1))
        # transforms_mat = self.relu2(tmp)  # batch, L, 9
        return self.l2(self.relu(self.bn1(self.l1(self.unfold(xyz).permute(0, 2, 1)).permute(0, 2, 1))).permute(0, 2, 1)).permute(0, 2, 1)  # batch, C_, L


class trans_conv_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_channel, out_channel, kernel_size, stride, padding, dilation, input,
                depth_transform, weight):
        output, = trans_conv.forward(in_channel, out_channel, kernel_size, stride, padding, dilation, input,
                                     depth_transform, weight)  # return list, to unpack, add comma
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.in_channel = in_channel
        ctx.out_channel = out_channel
        ctx.kernel_size = kernel_size
        ctx.save_for_backward(input, depth_transform, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, depth_transform, weight = ctx.saved_tensors
        ###########################
        # grad_input: batch, in_channel, width, length
        # grad_de_tr: in_channel, kernel_size, kernel_size, out_channel
        # grad_weight: batch, L, kernel_size**2
        # grad_output: batch, in_channel, width', length'
        ##########################
        grad_input, grad_de_tr, grad_weight = trans_conv.backward(
                                                ctx.in_channel,
                                                ctx.out_channel,
                                                ctx.kernel_size,
                                                ctx.stride,
                                                ctx.padding,
                                                ctx.dilation,
                                                input,
                                                depth_transform,
                                                weight,
                                                grad_output)
        return None, None, None, None, None, None, grad_input, grad_de_tr, grad_weight


class tramsform_conv(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1):
        super(tramsform_conv, self).__init__()
        self.W = nn.Parameter(torch.empty(in_channel, kernel_size, kernel_size, out_channel))
        nn.init.xavier_normal_(self.W.data)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, feat, transform_mat):
        return trans_conv_Function.apply(
                self.in_channel,
                self.out_channel,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                feat,
                transform_mat.permute(0, 2, 1),
                self.W)


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
        self.trans1_2 = depth_transform(64, 64, 3, padding=1)
        self.conv1_2 = tramsform_conv(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.xyz_pool1 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.trans2_1 = depth_transform(64, 128, 3, padding=1)
        self.conv2_1 = tramsform_conv(64, 128, 3, padding=1)
        self.trans2_2 = depth_transform(128, 128, 3, padding=1)
        self.conv2_2 = tramsform_conv(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.xyz_pool2 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.trans3_1 = depth_transform(128, 256, 3, padding=1)
        self.conv3_1 = tramsform_conv(128, 256, 3, padding=1)
        self.trans3_2 = depth_transform(256, 256, 3, padding=1)
        self.conv3_2 = tramsform_conv(256, 256, 3, padding=1)
        self.trans3_3 = depth_transform(256, 256, 3, padding=1)
        self.conv3_3 = tramsform_conv(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.xyz_pool3 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.trans4_1 = depth_transform(256, 512, 3, padding=1)
        self.conv4_1 = tramsform_conv(256, 512, 3, padding=1)
        self.trans4_2 = depth_transform(512, 512, 3, padding=1)
        self.conv4_2 = tramsform_conv(512, 512, 3, padding=1)
        self.trans4_3 = depth_transform(512, 512, 3, padding=1)
        self.conv4_3 = tramsform_conv(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.xyz_pool4 = xyz_pool(kernel_size=3, stride=2, padding=1, type='max')
        self.trans5_1 = depth_transform(512, 512, 3, padding=2, dilation=2)
        self.conv5_1 = tramsform_conv(512, 512, 3, padding=2, dilation=2)
        self.trans5_2 = depth_transform(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = tramsform_conv(512, 512, 3, padding=2, dilation=2)
        self.trans5_3 = depth_transform(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = tramsform_conv(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear(512, 1024)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Linear(1024, 1024)
        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Linear(1024, 40)
        self.interp = nn.UpsamplingBilinear2d(size=(480, 640))
        print('network DynamicNet init.')

    def forward(self, rgb=None, xyz=None):
        xyz = xyz.permute(0, 3, 1, 2)
        feat = rgb.permute(0, 3, 1, 2)

        print("first group")
        print(torch.cuda.memory_allocated())
        transform_mat = self.trans1_1(xyz)
        # print('- ', transform_mat.nelement()*transform_mat.element_size())
        print(torch.cuda.memory_allocated())
        feat = F.relu(self.conv1_1(feat, transform_mat))
        # print('- ', feat.nelement()*feat.element_size())
        print(torch.cuda.memory_allocated())
        feat = F.relu(self.conv1_2(feat, transform_mat))
        # print('- ', feat.nelement()*feat.element_size())
        size = feat.size()
        print(torch.cuda.memory_allocated())
        feat, indices = self.pool1(feat)
        print(torch.cuda.memory_allocated())
        xyz = self.xyz_pool1(xyz, feat, indices, size)
        print(torch.cuda.memory_allocated())

        print("second group")
        transform_mat = self.trans2_1(xyz)
        feat = F.relu(self.conv2_1(feat, transform_mat))
        feat = F.relu(self.conv2_2(feat, transform_mat))
        size = feat.size()
        feat, indices = self.pool2(feat)
        xyz = self.xyz_pool2(xyz, feat, indices, size)

        print("third group")
        transform_mat = self.trans3_1(xyz)
        feat = F.relu(self.conv3_1(feat, transform_mat))
        feat = F.relu(self.conv3_2(feat, transform_mat))
        size = feat.size()
        feat, indices = self.pool3(feat)
        xyz = self.xyz_pool3(xyz, feat, indices, size)

        print("fourth group")
        transform_mat = self.trans4_1(xyz)
        feat = F.relu(self.conv4_1(feat, transform_mat))
        feat = F.relu(self.conv4_2(feat, transform_mat))
        size = feat.size()
        feat, indices = self.pool4(feat)
        xyz = self.xyz_pool4(xyz, feat, indices, size)

        print("fifth group")
        transform_mat = self.trans5_1(xyz)
        feat = F.relu(self.conv5_1(feat, transform_mat))
        feat = F.relu(self.conv5_2(feat, transform_mat))
        feat = self.pool5a(self.pool5(feat))

        print("sixth group")
        feat = self.drop6(F.relu(self.fc6(feat.permute(0, 2, 3, 1))))
        feat = F.relu(self.fc7(feat))
        feat = self.fc8(self.drop7(feat))

        prediction = self.interp(feat.permute(0, 3, 1, 2))
        return prediction


class deepNet(Module):
    def __init__(self):
        super(deepNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear(512, 1024)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Linear(1024, 1024)
        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Linear(1024, 40)
        self.interp = nn.UpsamplingBilinear2d(size=(480, 640))
        print('network DynamicNet init.')

    def forward(self, rgb=None, xyz=None):
        xyz = xyz.permute(0, 3, 1, 2)
        feat = rgb.permute(0, 3, 1, 2)

        feat = F.relu(self.conv1_1(feat))
        feat = F.relu(self.conv1_2(feat))
        feat, indices = self.pool1(feat)

        feat = F.relu(self.conv2_1(feat))
        feat = F.relu(self.conv2_2(feat))
        feat, indices = self.pool2(feat)

        feat = F.relu(self.conv3_1(feat))
        feat = F.relu(self.conv3_2(feat))
        feat, indices = self.pool3(feat)

        feat = F.relu(self.conv4_1(feat))
        feat = F.relu(self.conv4_2(feat))
        feat, indices = self.pool4(feat)

        feat = F.relu(self.conv5_1(feat))
        feat = F.relu(self.conv5_2(feat))
        feat = self.pool5a(self.pool5(feat))

        feat = self.drop6(F.relu(self.fc6(feat.permute(0, 2, 3, 1))))
        feat = F.relu(self.fc7(feat))
        feat = self.fc8(self.drop7(feat))

        prediction = self.interp(feat.permute(0, 3, 1, 2))
        return prediction
