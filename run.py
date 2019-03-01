import os
# 参数类别和顺序：
'''
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epoches', type=int, default=100,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--shrink_factor', type=int, default=8, help='')
parser.add_argument('--opt', type=str, default='Adam', help='the optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--restore_path', type=str, default=None, help='path to the saved model')
'''
os.system('python train_v2.py --model DynamicNet --train_batch 5 --test_batch 5 --epoches 500 --workers 4 --opt SGD --lr 0.001 --weight_decay 0.001 --restore_path model/DynamicNet_2019-02-28-16-38-31/200')
# os.system('python train_v2.py --model DynamicNet --train_batch 5 --test_batch 5 --epoches 200 --workers 4 --opt SGD --lr 0.01 --weight_decay 0.001')
# os.system('python train_v2.py --model NormalNet --train_batch 4 --test_batch 4 --epoches 200 --workers 4 --opt SGD --lr 0.001')
