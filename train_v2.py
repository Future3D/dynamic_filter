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
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as tfs
from dynamic_net import DynamicNet, demo_dataset, nyud_dataset, NormalNet, nyud_dataset2
import time
from datetime import datetime
import visdom


def depth_pc(depth):
    # depth: width, height
    # KI: 3,3
    # Z: width*height,3
    # pixel_matrix: width*height,3
    # xyz: width*height,3
    # print(depth.shape)
    width = depth.shape[1]
    length = depth.shape[2]
    # if width > length:
    #     print('width: {}, length: {}, width > length, may be wrong input'.format(width, length))
    # scale = width / 480
    scale = 1
    pixel_matrix = torch.zeros(width*length, 3)
    for i in range(width):
        pixel_matrix[i*length:(i+1)*length, 0] = torch.Tensor(range(0, length))
        pixel_matrix[i*length:(i+1)*length, 1] = i
        pixel_matrix[i*length:(i+1)*length, 2] = 1
    cx = float(length/2)
    cy = float(width/2)
    f = 570 * scale
    KI = torch.Tensor([[f, 0, 0], [0, f, 0], [cx, cy, 1]]).inverse()
    depth = depth.reshape(depth.shape[0], width*length, 1)
    Z = depth.repeat(1, 1, 3).float()
    xyz = torch.matmul(torch.mul(Z, pixel_matrix), KI)
    return xyz.view(depth.shape[0], width, length, 3).permute(0, 3, 1, 2)


def scale_and_crop(img, depth, label):
    w = img.shape[2]
    l = img.shape[3]

    sw = int((random.random()/2.5 + 0.4)*l)  # 0.8~1.2 * w
    sl = int((random.random()/2.5 + 0.4)*w)  # 0.8~1.2 * l
    img = F.interpolate(img, (sw, sl), mode='bilinear')
    depth = F.interpolate(depth.unsqueeze(1), (sw, sl), mode='bilinear').squeeze()
    label = F.interpolate(label.unsqueeze(1).float(), (sw, sl), mode='nearest').long().squeeze()

    cropscale = random.uniform(0.8, 0.9)
    cropsizeh = int(sl * cropscale)
    cropsizew = int(sw * cropscale)
    x1 = random.randint(0, max(0, sw - cropsizew))
    y1 = random.randint(0, max(0, sl - cropsizeh))
    x2 = x1 + cropsizew - 1
    y2 = y1 + cropsizeh - 1

    return img[:, :, x1:x2, y1:y2], depth[:, x1:x2, y1:y2], label[:, x1:x2, y1:y2]


def adjust_lr(optimizer, times, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2


def adjust_momentum(optimizer, times, momentum):
    for param_group in optimizer.param_groups:
        param_group['momentum'] += 0.5*(1-param_group['momentum'])


def train(model, train_loader, epoch, loss_f, interpolate_f):
    rate = 0.25
    if epoch % (args.epoches*rate) == 0:
        adjust_lr(optimizer, epoch // (args.epoches*rate), args.lr)
        if args.opt == 'SGD':
            adjust_momentum(optimizer, epoch // (args.epoches*rate), momentum)

    pixel_correct = 0
    pixel_all = 0
    pixel_accuracy = 0
    class_correct = torch.zeros(40).float()
    class_truth = torch.zeros(40).float()
    class_accuracy = torch.zeros(40).float()
    class_pred = torch.zeros(40).float()
    mean_accuracy = 0
    mean_IOU = 0
    total_loss = 0
    start = time.time()
    train_num_batch = len(train_loader)
    for cnt, (rgb, xyz, target) in enumerate(train_loader, 0):
        rgb, xyz, target = scale_and_crop(rgb, xyz, target)
        xyz = depth_pc(xyz)  # 这里传入的xyz是depth，只是为了方便不改了
        rgb, xyz = rgb.cuda(), xyz.cuda()
        target = target.squeeze().cuda()
        optimizer.zero_grad()
        model = model.train()
        pred = model(xyz=xyz, rgb=rgb)
        pred = interpolate_f(pred, (rgb.shape[2], rgb.shape[3]), mode='bilinear').permute(0, 2, 3, 1)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1).squeeze() - 1
        loss = loss_f(pred, target)
        loss.backward()
        optimizer.step()

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).sum()
        accuracy = correct.item()/float(rgb.shape[0] * rgb.shape[2] * rgb.shape[3])
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, cnt, train_num_batch, loss.item(), accuracy))

        for i in range(40):
            class_correct[i] += torch.mul(pred_choice.eq(target), (target == i)).sum()
            class_truth[i] += (target == i).sum()
            class_pred[i] += (pred_choice == i).sum()
        pixel_correct += correct
        pixel_all += rgb.shape[0] * rgb.shape[2] * rgb.shape[3]
        total_loss += loss.item()

    print('train_interval', time.time() - start)

    average_loss = total_loss / len(train_loader)
    pixel_accuracy = float(pixel_correct)/float(pixel_all)
    class_accuracy = torch.mul(class_correct, 1.0/class_truth)
    mean_accuracy = class_accuracy.sum()/40.0
    mean_IOU = torch.mul(class_correct, 1.0/(class_pred + class_truth - class_correct)).sum()/40.0

    print('[{}] accuracy: {} average_loss: {}'.format(epoch, pixel_accuracy, average_loss))
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([average_loss]), win='train loss',
             opts=dict(title='train loss'), update='append' if epoch > 0 else None)

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([pixel_accuracy]), win='train accuracy',
             opts=dict(title='train accuracy'), update='append' if epoch > 0 else None)

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([mean_accuracy]), win='mean accuracy',
            opts=dict(title='mean accuracy'), update='append' if epoch > 0 else None)

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([mean_IOU]), win='mean IOU',
            opts=dict(title='mean IOU'), update='append' if epoch > 0 else None)

    for i in range(40):
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([class_accuracy[i]]), win='{} accuracy'.format(i),
                 opts=dict(title='{} accuracy'.format(i)), update='append' if epoch > 0 else None)


def test(model, test_loader, epoch, loss_f, interpolate_f):
    blue = lambda x: '\033[94m' + x + '\033[0m'  # 就是个打印变色功能

    pixel_correct = 0
    pixel_all = 0
    pixel_accuracy = 0
    total_loss = 0
    start = time.time()
    test_num_batch = len(test_loader)
    for cnt, (rgb, xyz, target) in enumerate(test_loader, 0):
        rgb, xyz, target = scale_and_crop(rgb, xyz, target)
        xyz = depth_pc(xyz)  # 这里传入的xyz是depth，只是为了方便不改了
        rgb, xyz = rgb.cuda(), xyz.cuda()
        target = target.squeeze().cuda()
        optimizer.zero_grad()
        model = model.eval()
        pred = model(xyz=xyz, rgb=rgb)
        pred = interpolate_f(pred, (rgb.shape[2], rgb.shape[3]), mode='bilinear').permute(0, 2, 3, 1)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1).squeeze() - 1
        loss = loss_f(pred, target)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).sum()
        accuracy = correct.item()/float(rgb.shape[0] * rgb.shape[2] * rgb.shape[3])
        print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, cnt, test_num_batch, blue('test'),
              loss.item(), accuracy))

        total_loss += loss.item()
        pixel_correct += correct
        pixel_all += rgb.shape[0] * rgb.shape[2] * rgb.shape[3]

    print('test_interval', time.time() - start)

    pixel_accuracy = float(pixel_correct)/float(pixel_all)
    average_loss = total_loss / len(test_loader)
    print('[{}] accuracy: {} average_loss: {}'.format(epoch, pixel_accuracy, average_loss))

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([pixel_accuracy]), win='test accuracy',
             opts=dict(title='test accuracy'), update='append' if epoch > 0 else None)

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([average_loss]), win='test loss',
             opts=dict(title='test loss'), update='append' if epoch > 0 else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DynamicNet')
    parser.add_argument('--model', type=str)
    parser.add_argument('--train_batch', type=int, default=3, help='input batch size for training')
    parser.add_argument('--test_batch', type=int, default=1, help='input batch size for test ')
    parser.add_argument('--epoches', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--opt', type=str, default='Adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--restore_path', type=str, default=None, help='path to the saved model')

    global args
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

    # train_data_path = "/media/gaojiefeng/7AB60EF3B60EB0252/data/train_pc_in_2d.h5"
    # test_data_path = "/media/gaojiefeng/7AB60EF3B60EB0252/data/test_pc_in_2d.h5"
    # train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d_picked.h5"
    # test_data_path = "/home/data/dynamic_filter_dataset/test_pc_in_2d_picked.h5"
    # train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d_crop_all.h5"
    # test_data_path = "/home/data/dynamic_filter_dataset/test_pc_in_2d_crop_all.h5"
    # train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d.h5"
    # test_data_path = "/home/data/dynamic_filter_dataset/test_pc_in_2d.h5"
    # train_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d_p1.h5"
    # test_data_path = "/home/data/dynamic_filter_dataset/train_pc_in_2d_p2.h5"
    # train_data_path = "../dynamic_filter_dataset/train_pc_in_2d.h5"
    # test_data_path = "../dynamic_filter_dataset/test_pc_in_2d.h5"
    train_data_path = "/home/data/dynamic_filter_dataset/train.h5"
    test_data_path = "/home/data/dynamic_filter_dataset/test.h5"
    # train_data_path = "../dynamic_filter_dataset/train.h5"
    # test_data_path = "../dynamic_filter_dataset/test.h5"

    data_size = 0
    if args.train_batch > 0:
        train_set = nyud_dataset(data_path=train_data_path)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch,
                                                   shuffle=True, num_workers=int(args.workers))

        num_classes = train_set.num_classes
        data_size += len(train_set)

    if args.test_batch > 0:
        test_set = nyud_dataset(data_path=test_data_path)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
                                                  shuffle=True, num_workers=int(args.workers))
        num_classes = test_set.num_classes
        data_size += len(test_set)

    with open('log/{}/configuration'.format(args.model+'_'+sign), 'w') as f:
        f.write('model: {}\n'.format(args.model))
        f.write('data_size: {}\n'.format(data_size))
        f.write('train_batch: {}\n'.format(args.train_batch))
        f.write('test_batch: {}\n'.format(args.test_batch))
        f.write('optimizer: {}\n'.format(args.opt))
        f.write('learn rate: {}\n'.format(args.lr))
        f.write('weight decay: {}\n'.format(args.weight_decay))
        f.write('restore_path: {}\n'.format(args.restore_path))
        if args.restore_path is None:
            start_num = 0
        else:
            start_num = args.restore_path.split('/')[-1]
        f.write('start_num: {}\n'.format(start_num))

    momentum = 0.5
    if args.model == 'DynamicNet':
        model = DynamicNet().cuda()
        model_dict = model.state_dict()
        model_dict.update(torch.load('vgg16_modify_params.pkl'))
        model.load_state_dict(model_dict)
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
                {'params': other_parameters, 'lr': args.lr*2}
            ], lr=args.lr, momentum=momentum, weight_decay=args.weight_decay)
    elif args.model == 'NormalNet':
        model = NormalNet().cuda()
        vgg16 = models.vgg16(pretrained=True)
        pretrained_dict = vgg16.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        optimizer = optim.SGD(
            [
                {'params': model.features.parameters()},
                {'params': model.mclassifier.parameters(), 'lr': args.lr*2}
            ], lr=args.lr, momentum=momentum)
    else:
        print('unknown model')

    if args.restore_path is not None:
        # 由于网络结构改动频繁，有时候上次训练的权重不适合这次，所以用选择性加载。
        pretrained_dict = torch.load(args.restore_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum)

    global vis
    vis = visdom.Visdom(env='{}_{}'.format(args.model, sign))

    loss_f = nn.CrossEntropyLoss()
    interpolate_f = nn.functional.interpolate
    for epoch in range(1, args.epoches+1):
        if args.train_batch > 0:
            train(model, train_loader, epoch, loss_f, interpolate_f)

            if epoch % (args.epoches*0.1) == 0 and epoch != 0:
                torch.save(model.state_dict(), 'model/{}/{}'.format(args.model+'_'+sign, epoch))

        if args.test_batch > 0:
            test(model, test_loader, epoch, loss_f, interpolate_f)

    vis.save(['{}_{}'.format(args.model, sign)])
