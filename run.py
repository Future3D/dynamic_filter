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

# os.system('python train.py --model NormalNet --batch_size 50 --epoches 500 --workers 4 --opt SGD --lr 0.01 \
#             --restore_path model/2019-02-03-08-30-37/500')

# os.system('python train.py --model DynamicNet --batch_size 1 --epoches 200 --workers 4 --opt SGD --lr 0.001')  # 因为显存不够，只能先训再测然后再训，所以把epoch调小一点
os.system('python train.py --model NormalNet --batch_size 1 --epoches 1600 --workers 4 --opt SGD --lr 0.001')
