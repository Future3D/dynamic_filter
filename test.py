# -*- coding:utf-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
from dynamic_net import DynamicNet, demo_dataset, nyud_dataset, NormalNet
import time
from datetime import datetime
import visdom


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GCN_RGBD2')
    parser.add_argument('--model', type=str, default=32)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--opt', type=str, default='Adam', help='the optimizer')
    parser.add_argument('--restore_path', type=str, default=None, help='path to the saved model')

    args = parser.parse_args()
    print(args)

    args.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d_picked.h5"
    test_data_path = "/home/data/dynamic_filter_dataset/test_pc_in_2d_picked.h5"
    # train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d_crop_all.h5"
    # test_data_path = "/home/data/dynamic_filter_dataset/test_pc_in_2d_crop_all.h5"
    # train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d.h5"
    # test_data_path = "/home/data/dynamic_filter_dataset/test_pc_in_2d.h5"
    # train_data_path = "../dynamic_filter_dataset/train_pc_in_2d.h5"
    # test_data_path = "../dynamic_filter_dataset/test_pc_in_2d.h5"
    # train_data_path = "/media/gaojiefeng/7AB60EF3B60EB0252/data/train_pc_in_2d.h5"
    # test_data_path = "/media/gaojiefeng/7AB60EF3B60EB0252/data/test_pc_in_2d.h5"

    train_set = nyud_dataset(data_path=train_data_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=int(args.workers))

    test_set = nyud_dataset(data_path=test_data_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=int(args.workers))

    print(len(train_set), len(test_set))
    num_classes = train_set.num_classes
    print('classes', num_classes)

    if args.model == 'DynamicNet':
        model = DynamicNet().cuda()
    elif args.model == 'NormalNet':
        model = NormalNet().cuda()
    elif args.model == 'vgg16':
        model = models.vgg16().cuda()
    else:
        print('unknown model')
    # if args.model == 'DynamicNet':
    #     model = DynamicNet()

    sign = args.restore_path.split('/')[-1]

    vis = visdom.Visdom(env='{}_{}_{}'.format(args.model, 'test', sign))
    loss_f = nn.CrossEntropyLoss()
    interpolate_f = nn.functional.interpolate
    num_batch = len(train_set)/args.batch_size
    total_correct = 0
    total_accuracy = 0
    start = time.time()
    files = os.listdir(args.restore_path).sort()
    for file in files:
        model.load_state_dict(torch.load(file))
        for cnt, (rgb, xyz, target) in enumerate(test_loader, 0):  # 不知道要不要预处理数据到一定数值
            rgb, xyz = rgb.cuda(), xyz.cuda()
            target = target.squeeze().cuda()
            # rgb, xyz = rgb, xyz
            # target = target.squeeze()
            model = model.eval()
            pred = model(xyz, rgb)
            pred = interpolate_f(pred.permute(0, 3, 1, 2), (rgb.shape[1], rgb.shape[2]), mode='bilinear').permute(0, 2, 3, 1)
            pred = pred.view(pred.shape[0], rgb.shape[1], rgb.shape[2],
                             num_classes).view(-1, num_classes)
            target = target.view(-1, 1).squeeze() - 1
            loss = loss_f(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum()
            total_correct += correct
            accuracy = correct.item()/float(rgb.shape[0] * rgb.shape[1] * rgb.shape[2])
            print('[%d: %d/%d] loss: %f accuracy: %f' % (int(file), cnt, num_batch, loss.item(), accuracy))
        print('test_interval', time.time() - start)
        total_accuracy = float(total_correct)/float(test_set.__len__() * rgb.shape[1] * rgb.shape[2])
        print('[{}] accuracy: {}'.format(int(file), total_accuracy))
        vis.line(X=torch.FloatTensor([int(file)]), Y=torch.FloatTensor([total_accuracy]), win='test accuracy',
                 opts=dict(title='test accuracy'), update='append' if int(file) > 0 else None)
