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
# import sys
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# import numpy as np


def adjust_lr(optimizer, times, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.6 ** times)


def adjust_momentum(optimizer, times, momentum):
    for param_group in optimizer.param_groups:
        param_group['momentum'] = momentum + 0.049 * times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GCN_RGBD2')
    parser.add_argument('--model', type=str, default=32)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epoches', type=int, default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument('--sample_size', type=int, default=8, help='num of sample from 425*560')
    parser.add_argument('--shrink_factor', type=int, default=8, help='')
    # parser.add_argument('--layers', nargs='+')
    parser.add_argument('--opt', type=str, default='Adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--restore_path', type=str, default=None, help='path to the saved model')

    args = parser.parse_args()
    print(args)

    args.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    sign = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.system('mkdir log/{}'.format(args.model+'_'+sign))
    os.system('rm -rf log/{}/events/*'.format(args.model+'_'+sign))
    os.system('mkdir model/{}'.format(args.model+'_'+sign))

    f = open("log/{}/print_log".format(args.model+'_'+sign), 'w')
    # sys.stdout = f

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

    with open('log/{}/configuration'.format(args.model+'_'+sign), 'w') as f:
        f.write('model: {}\n'.format(args.model))
        f.write('data_size: {}\n'.format(len(train_set)+len(test_set)))
        f.write('batch_size: {}\n'.format(args.batch_size))
        f.write('optimizer: {}\n'.format(args.opt))
        f.write('learn rate: {}\n'.format(args.lr))
        f.write('restore_path: {}\n'.format(args.restore_path))
        if args.restore_path is None:
            start_num = 0
        else:
            start_num = args.restore_path.split('/')[-1]
        f.write('start_num: {}\n'.format(start_num))

    blue = lambda x: '\033[94m' + x + '\033[0m'  # 就是个打印变色功能

    if args.model == 'DynamicNet':
        model = DynamicNet().cuda()
        model_dict = model.state_dict()
        model_dict.update(torch.load('params.pkl'))
        model.load_state_dict(model_dict)
        momentum = 0.5
        all_parameters = model.parameters()
        finetune_parameters = []
        for pname, p in model.named_parameters():
            if pname.find('conv') >= 0:
                finetune_parameters.append(p)

        finetune_parameters_id = list(map(id, finetune_parameters))
        other_parameters = list(filter(lambda p: id(p) not in finetune_parameters_id, all_parameters))
        optimizer = optim.SGD(
            [
                {'params': finetune_parameters},
                {'params': other_parameters, 'lr': args.lr*10}
            ], lr=args.lr, momentum=momentum)
        loss_f = nn.CrossEntropyLoss()
    elif args.model == 'NormalNet':
        model = NormalNet().cuda()
        vgg16 = models.vgg16(pretrained=True)
        pretrained_dict = vgg16.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        momentum = 0.5
        optimizer = optim.SGD(
            [
                {'params': model.features.parameters()},
                {'params': model.mclassifier.parameters(), 'lr': args.lr*10}
            ], lr=args.lr, momentum=momentum)
    elif args.model == 'vgg16':
        model = models.vgg16().cuda()
    else:
        print('unknown model')

    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum)

    # if args.opt == 'SGD':
    #     momentum = 0.5
    #     optimizer = optim.SGD(
    #         [
    #             {'params': model.features.parameters()},
    #             {'params': model.mclassifier.parameters(), 'lr': args.lr*10}
    #         ], lr=args.lr, momentum=momentum)
    # elif args.opt == 'Adam':
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)

    vis = visdom.Visdom(env='{}_{}'.format(args.model, sign))
    loss_f = nn.CrossEntropyLoss()
    interpolate_f = nn.functional.interpolate
    num_batch = len(train_set)/args.batch_size

    for epoch in range(1, args.epoches+1):
        if epoch % (args.epoches*0.1) == 0:
            adjust_lr(optimizer, epoch // (args.epoches*0.1), args.lr)
            if args.opt == 'SGD':
                adjust_momentum(optimizer, epoch // (args.epoches*0.1), momentum)

        total_correct = 0
        total_accuracy = 0
        total_loss = 0
        start = time.time()
        for cnt, (rgb, xyz, target) in enumerate(train_loader, 0):
            rgb, xyz = rgb.cuda(), xyz.cuda()
            target = target.squeeze().cuda()
            # rgb, xyz = rgb, xyz
            # target = target.squeeze()
            optimizer.zero_grad()
            model = model.train()
            # start = time.time()
            pred = model(xyz, rgb)
            pred = interpolate_f(pred.permute(0, 3, 1, 2), (rgb.shape[1], rgb.shape[2]), mode='bilinear').permute(0, 2, 3, 1)
            # print('forward', time.time()-start)
            pred = pred.view(pred.shape[0], rgb.shape[1], rgb.shape[2],
                             num_classes).view(-1, num_classes)
            target = target.view(-1, 1).squeeze() - 1
            loss = loss_f(pred, target)
            total_loss += loss.item()
            loss.backward()
            # print('whole', time.time() - start)
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum()
            total_correct += correct
            accuracy = correct.item()/float(rgb.shape[0] * rgb.shape[1] * rgb.shape[2])
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, cnt, num_batch, loss.item(), accuracy))
        print('train_interval', time.time() - start)
        total_accuracy = float(total_correct)/float(train_set.__len__() * rgb.shape[1] * rgb.shape[2])
        average_loss = total_loss / len(train_loader)
        print('[{}] accuracy: {} average_loss: {}'.format(epoch, total_accuracy, average_loss))
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([total_accuracy]), win='train accuracy',
                 opts=dict(title='train accuracy'), update='append' if epoch > 0 else None)
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([average_loss]), win='train loss',
                 opts=dict(title='train loss'), update='append' if epoch > 0 else None)

        # if epoch % 10 == 0 and epoch != 0:
        # if epoch % 1 == 0:
        #     total_correct = 0
        #     total_accuracy = 0
        #     start = time.time()
        #     for cnt, (rgb, xyz, target) in enumerate(test_loader, 0):  # 不知道要不要预处理数据到一定数值
        #         rgb, xyz = rgb.cuda(), xyz.cuda()
        #         target = target.squeeze().cuda()
        #         # rgb, xyz = rgb, xyz
        #         # target = target.squeeze()
        #         optimizer.zero_grad()
        #         model = model.eval()
        #         pred = model(xyz, rgb)
        #         pred = interpolate_f(pred.permute(0, 3, 1, 2), (rgb.shape[1], rgb.shape[2]), mode='bilinear').permute(0, 2, 3, 1)
        #         pred = pred.view(pred.shape[0], rgb.shape[1], rgb.shape[2],
        #                          num_classes).view(-1, num_classes)
        #         target = target.view(-1, 1).squeeze() - 1
        #         loss = loss_f(pred, target)
        #         pred_choice = pred.data.max(1)[1]
        #         correct = pred_choice.eq(target.data).sum()
        #         total_correct += correct
        #         accuracy = correct.item()/float(rgb.shape[0] * rgb.shape[1] * rgb.shape[2])
        #         print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, cnt, num_batch, blue('test'),
        #               loss.item(), accuracy))
        #     print('test_interval', time.time() - start)
        #     total_accuracy = float(total_correct)/float(test_set.__len__() * rgb.shape[1] * rgb.shape[2])
        #     print('[{}] accuracy: {}'.format(epoch, total_accuracy))
        #     vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([total_accuracy]), win='test accuracy',
        #              opts=dict(title='test accuracy'), update='append' if epoch > 0 else None)

        if epoch % (args.epoches*0.1) == 0 and epoch != 0:
            torch.save(model.state_dict(), 'model/{}/{}'.format(args.model+'_'+sign, epoch))
    vis.save(['{}_{}'.format(args.model, sign)])
