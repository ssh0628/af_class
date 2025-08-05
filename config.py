import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--multiLabel', type=bool, default=False, help='enable multiple label')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--recordLength', type=int, default=2500, help='the length of input record')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. ')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# os.environ['CUDA_VISIBLE_DEVICES'] = '8 '

opt = parser.parse_args(['--dataroot',r"C:\Users\cream\OneDrive\Desktop\AF_Emu\Data\MAT",'--cuda'])

# print(opt)