import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler
import random
from copy import deepcopy
import ray
import argparse
from tensorboardX import SummaryWriter
from dirichlet_data import data_from_dirichlet
from sam import SAM
from models.resnet import ResNet18, ResNet50, ResNet10
from models.resnet_bn import ResNet18BN, ResNet50BN, ResNet10BN, ResNet34BN
from model import swin_tiny_patch4_window7_224 as swin_tiny
from model import swin_small_patch4_window7_224 as swin_small
from model import swin_large_patch4_window7_224_in22k as swin_large
from model import swin_base_patch4_window7_224_in22k as swin_base
from vit_model import vit_base_patch16_224_in21k as vit_B
from vit_model import vit_large_patch16_224_in21k as vit_L
from peft import LoraConfig, get_peft_model, TaskType

torch.backends.cudnn.benchmark = True  # 提高 CNN 计算速度

from models.DeiTTiny import deit_tiny


os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"

# 加入存档，log
#python  DP_adamw.py --alg DP-FedAvg-adamw --lr 0.0003 --data_name CIFAR100 --alpha_value 0.6 --alpha  0.9  --epoch 101  --extname CIFAR100 --lr_decay 2 --gamma 0.3 --CNN resnet18 --E 5 --batch_size 50  --gpu 0 --p 1 --num_gpus_per 0.1 --selection 0.2 --dp_sigma 0.56 --rho 0.01 --C 0.1 --num_workers 50 --pre 1 --preprint 5 --lora 1  --ls_sigma 0 --r 16 --privacy 1 --freeze 0 --K 20 --optimizer SGD

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=1001, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=50, type=int, help='# batch_size')
parser.add_argument('--E', default=5, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedAvg', type=str, help='FedAvg')  # FedMoment cddplus cdd SCAF atte

parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.998', type=float, help='lr_decay')
parser.add_argument('--data_name', default='CIFAR100', type=str, help='imagenet,CIFAR100')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')

parser.add_argument('--lr_ps', default='1', type=float, help='only for FedAdam ')

parser.add_argument('--alpha_value', default='0.1', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.1', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=0.01, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='lenet5', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.85, type=float, help=' for mom_step')
parser.add_argument('--p', default=10, type=float, help=' for mom_step')
# parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--freeze-layers', type=bool, default=False)

parser.add_argument('--datapath', type=str, default="./data")
parser.add_argument('--num_gpus_per', default=1, type=float, help=' for mom_step')
parser.add_argument('--normalization', default='BN', type=str, help=' for mom_step')
parser.add_argument('--pre', default=1, type=int, help=' for mom_step')
parser.add_argument('--print', default=0, type=int, help=' for mom_step')

parser.add_argument('--momentum', type=float, default=0.5, metavar='N', help='momentum')
parser.add_argument("--laplacian", type=bool, default=True, help="Laplacian Smoothing")
parser.add_argument("--ls_sigma", type=float, default=1.0)

parser.add_argument('--dp_sigma', default=0.2, type=float, help='noise multiplier for DP')
parser.add_argument('--privacy', default=1, type=int, help='whether to use differential privacy')
parser.add_argument('--C', type=float, default=0.2)

# FedSAM
parser.add_argument("--rho", type=float, default=0.05, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--adaptive", type=bool, default=True, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--preprint", type=int, default=10, help="")
parser.add_argument("--sparse_rate", type=float, default=0.6, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=bool, default=True, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
parser.add_argument("--maxnorm", type=float, default=10, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--clip", type=bool, default=True, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=50, type=int, help='#workers')
parser.add_argument('--optimizer', default='SGD', type=str, help='adam')

args = parser.parse_args()
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16

num_gpus = len(gpu_idx.split(','))
# num_gpus_per = 1
data_name = args.data_name
CNN = args.CNN
print(CNN)

if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base']:
    lora_config = LoraConfig(
        r=args.r,  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
        lora_alpha=args.r,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
        lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
        bias="none",  # 不训练偏置项[^18^]
        task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
        target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
    )

if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base']:
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # '''
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # CIFAR10的标准化参数：mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616]

    # CIFAR100的标准化参数：mean = [0.5071, 0.4865, 0.4409], std = [0.2673, 0.2564, 0.2762]
    # '''
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]预训练
else:
    if data_name == 'CIFAR10' or data_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

import dataset as local_datasets

if data_name == 'SVHN':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)

if data_name == 'FashionMNIST':
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)

if data_name == 'imagenet':
    train_dataset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.datapath, 'tiny-imagenet-200'),
        split='train',
        transform=transform_train
    )

if data_name == 'CIFAR10':

    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=False,
        transform=transform_train)


elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train
    )

elif data_name == 'EMNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        # split='mnist',
        split='balanced',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )
elif data_name == 'MNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        split='mnist',
        # split='balanced',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )


def get_data_loader(pid, data_idx, batch_size, data_name):
    """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
    sample_chosed = data_idx[pid]
    train_sampler = SubsetRandomSampler(sample_chosed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler, num_workers=0, generator=torch.Generator().manual_seed(42))
    return train_loader


def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""

    if data_name == 'imagenet':
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='test',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_train)

    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_train
                                               )
    if data_name == 'SVHN':
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_train)

    if data_name == 'FashionMNIST':
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_train)

    if data_name == 'EMNIST':
        test_dataset = datasets.EMNIST("./data", split='balanced', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    if data_name == 'MNIST':
        test_dataset = datasets.EMNIST("./data", split='mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        shuffle=False,
        num_workers=4,  # 增加线程数以加快数据加载
        pin_memory=True  # 提高 GPU 训练时的数据加载效率
    )

    return test_loader


def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    if data_name == 'imagenet':
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10("./data", train=True, transform=transform_train)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)

    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_train
                                                )

    if data_name == 'SVHN':
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)

    if data_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)

    if data_name == 'EMNIST':
        # test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        train_dataset = datasets.EMNIST("./data", split='balanced', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    if data_name == 'MNIST':
        train_dataset = datasets.EMNIST("./data", split='mnist', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))

    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=False,
        num_workers=4)
    return train_loader


def evaluate2(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    # model = torch.compile(model)
    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            predicted = torch.argmax(outputs, dim=1)
            # _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100. * correct / total, torch.tensor(0), torch.tensor(0)

def evaluate(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            predicted = torch.argmax(outputs, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss += criterion(outputs, target)

        for batch_idx, (data, target) in enumerate(train_loader):
            data_train = data.to(device)
            target_train = target.to(device)
            outputs_train = model(data_train)
            train_loss += criterion(outputs_train, target_train)
    return 100. * correct / total, test_loss / len(test_loader), train_loss / len(train_loader)

import torch.nn as nn
import torch.nn as nn
import torchvision.models as models




class ResNet50pre(nn.Module):
    def __init__(self, num_classes=10, l2_norm=False):
        super(ResNet50pre, self).__init__()
        if args.pre == 1:
            resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet50 = models.resnet50()
        resnet50.fc = nn.Linear(2048, num_classes)
        # nn.Linear(2048, 100)
        self.model = resnet50

    def forward(self, x):
        x = self.model(x)
        return x

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class ResNet18pre(nn.Module):
    def __init__(self, num_classes=10, l2_norm=False):
        super(ResNet18pre, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64

        if args.pre == 1:
            # resnet18=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet18 = replace_bn_with_gn(resnet18, num_groups=32)
        else:
            resnet18 = models.resnet18()
        resnet18.fc = nn.Linear(512, num_classes)
        self.model = resnet18

    def forward(self, x):
        x = self.model(x)
        return x

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class Lenet5(nn.Module):
    """TF Tutorial for CIFAR."""

    def __init__(self, num_classes=10):
        super(Lenet5, self).__init__()
        self.n_cls = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class SimpleCNN_3(nn.Module):
    def __init__(self):
        super(SimpleCNN_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class ConvNet_EMNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 47)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class ConvNet_MNIST(nn.Module):
    """TF Tutorial for MNIST."""

    def __init__(self):
        super(ConvNet_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


import torch.nn as nn
import torchvision.models as models
from torch import nn
import math

if CNN == 'ConvNet_EMNIST':
    def ConvNet():
        if data_name == 'EMNIST':
            return ConvNet_EMNIST()

if CNN == 'ConvNet_MNIST':
    def ConvNet():
        if data_name == 'MNIST':
            return ConvNet_MNIST()

if CNN == 'SimpleCNN':
    def ConvNet():
        if data_name == 'FashionMNIST':
            return SimpleCNN()
        # if data_name == 'EMNIST':
        #    return SimpleCNN()
        if data_name == 'SVHN':
            return SimpleCNN_3()

if CNN == 'swin_tiny':
    def ConvNet():
        return swin_tiny(num_classes=10)


    def ConvNet100():
        return swin_tiny(num_classes=100)


    def ConvNet200():
        return swin_tiny(num_classes=200)

if CNN == 'swin_large':
    def ConvNet():
        return swin_large(num_classes=10)


    def ConvNet100():
        return swin_large(num_classes=100)


    def ConvNet200():
        return swin_large(num_classes=200)
if CNN == 'swin_small':
    def ConvNet():
        return swin_small(num_classes=10)


    def ConvNet100():
        return swin_small(num_classes=100)


    def ConvNet200():
        return swin_small(num_classes=200)

if CNN == 'swin_base':
    def ConvNet():
        return swin_base(num_classes=10)


    def ConvNet100():
        return swin_base(num_classes=100)


    def ConvNet200():
        return swin_base(num_classes=200)

if CNN == 'VIT-B':
    def ConvNet():
        return vit_B(num_classes=10)


    def ConvNet100():
        return vit_B(num_classes=100)


    def ConvNet200():
        return vit_B(num_classes=200)
if CNN == 'VIT-L':
    def ConvNet():
        return vit_L(num_classes=10)


    def ConvNet100():
        return vit_L(num_classes=100)


    def ConvNet200():
        return vit_L(num_classes=200)

if CNN == 'lenet5':
    def ConvNet():
        return Lenet5(num_classes=10)


    def ConvNet100():
        return Lenet5(num_classes=100)

if CNN == 'resnet10':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10):
            return ResNet10BN(num_classes=10)


        def ConvNet100(num_classes=100):
            return ResNet10BN(num_classes=100)


        def ConvNet200(num_classes=200):
            return ResNet10BN(num_classes=200)
    if args.normalization == 'GN':
        def ConvNet(num_classes=10):
            return ResNet10(num_classes=10)


        def ConvNet100(num_classes=100):
            return ResNet10(num_classes=100)


        def ConvNet200(num_classes=200):
            return ResNet10(num_classes=200)

if CNN == 'resnet18':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10, l2_norm=False):
            return ResNet18BN(num_classes=10)


        def ConvNet100(num_classes=100, l2_norm=False):
            return ResNet18BN(num_classes=100)


        def ConvNet200(num_classes=200, l2_norm=False):
            return ResNet18BN(num_classes=200)
    if args.normalization == 'GN':
        def ConvNet(num_classes=10):
            return ResNet18(num_classes=10)


        def ConvNet100(num_classes=100):
            return ResNet18(num_classes=100)


        def ConvNet200(num_classes=200):
            return ResNet18(num_classes=200)
# '''

# '''
if CNN == 'resnet18pre':
    def ConvNet(num_classes=10):
        return ResNet18pre(num_classes=10)


    def ConvNet100(num_classes=100):
        return ResNet18pre(num_classes=100)


    def ConvNet200(num_classes=200):
        return ResNet18pre(num_classes=200)

if CNN == 'resnet50pre':
    def ConvNet(num_classes=10):
        return ResNet50pre(num_classes=10)


    def ConvNet100(num_classes=100):
        return ResNet50pre(num_classes=100)


    def ConvNet200(num_classes=200):
        return ResNet50pre(num_classes=200)


if CNN == 'deit_tiny':
    def ConvNet(num_classes=10):
        return deit_tiny(num_classes=10, img_size=32)
    def ConvNet100(num_classes=100):
        return deit_tiny(num_classes=100, img_size=32)
    def ConvNet200(num_classes=200):
        return deit_tiny(num_classes=200, img_size=64)

import torch
import torch.nn as nn




@ray.remote
# @ray.remote(num_gpus=args.num_gpus_per)
class ParameterServer(object):
    def __init__(self, lr, alg, tau, selection, data_name, num_workers):
        if data_name == 'CIFAR10':
            self.model = ConvNet()
        elif data_name == 'CIFAR100':
            self.model = ConvNet100()
        if data_name == 'imagenet':
            self.model = ConvNet200()
        if data_name == 'FashionMNIST':
            self.model = ConvNet()
        if data_name == 'SVHN':
            self.model = ConvNet()
        if data_name == 'EMNIST':
            self.model = ConvNet()
        if data_name == 'MNIST':
            self.model = ConvNet()
        # self.model = torch.compile(self.model)
        # if args.lora == True:
        #    self.model = get_peft_model(self.model, lora_config)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.momen_v = None
        # self.gamma = 0.9
        self.gamma = args.gamma
        # self.gamma = 0.9
        self.beta = 0.99  # 论文是0.99
        self.alg = alg
        self.num_workers = num_workers

        self.lr_ps = lr
        self.lg = 1.0
        self.ps_c = None
        self.c_all = None
        # 上一代的c
        self.c_all_pre = None
        self.tau = tau
        self.selection = selection
        self.cnt = 0
        self.alpha = args.alpha
        self.h = {}
        self.momen_m = {}
        self.momen_v = {}

    def set_pre_c(self, c):
        self.c_all_pre = c

    def apply_weights_avg(self, num_workers, *weights):

        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        global_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] += v / (num_workers * self.selection)
                else:
                    sum_weights[k] = v / (num_workers * self.selection)
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + sum_weights[k]
        self.model.set_weights(global_weights)
        return self.model.get_weights()



    def load_dict(self):
        self.func_dict = {


            'DP-FedAvg': self.apply_weights_avg,
            'DP-FedAvg-LS': self.apply_weights_avg,
            'DP-SCAFFOLD': self.apply_weights_avg,
            'DP-FedSAM': self.apply_weights_avg,
            'DP-FedAvg-adamw': self.apply_weights_avg,



        }

    def apply_weights_func(self, alg, num_workers, *weights):
        self.load_dict()
        return self.func_dict.get(alg, None)(num_workers, *weights)

    def apply_ci(self, alg, num_workers, *cis):

        args.gamma = 0.2
        sum_c = {}  # delta_c :sum_c
        for ci in cis:
            for k, v in ci.items():
                if k in sum_c.keys():
                    sum_c[k] += v / (num_workers * selection)
                else:
                    sum_c[k] = v / (num_workers * selection)

        if self.ps_c == None:
            self.ps_c = sum_c
            return self.ps_c

        for k, v in self.ps_c.items():

            if alg in {'FedSTORM', 'FedNesterov', 'DP-FedLESAM'}:
                self.ps_c[k] = sum_c[k]
            if alg in {'IGFL_prox'}:
                self.ps_c[k] = v * args.gamma + sum_c[k]
            if alg in {'IGFL_prox'}:
                self.ps_c[k] = v * (1 - args.gamma) + sum_c[k] * args.gamma
            if alg in {'FedCM', 'IGFL_prox', 'FedAGM', 'IGFL', 'MoFedSAM', 'stem', 'DP-FedPGN', 'DP-FedPGN-per',
                       'DP-MoFedSAM', 'DP-FedPGN-LS'}:
                self.ps_c[k] = v + sum_c[k]
            if alg in {'DP-FedPGN-LS'}:
                self.ps_c[k] = v + sum_c[k]
            else:
                self.ps_c[k] = v + sum_c[k] * args.gamma
        return self.ps_c

    def get_weights(self):
        return self.model.get_weights()

    def get_ps_c(self):
        return self.ps_c

    def get_state(self):
        return self.ps_c, self.c_all

    def set_state(self, c_tuple):
        self.ps_c = c_tuple[0]
        self.c_all = c_tuple[1]

    def set_weights(self, weights):
        self.model.set_weights(weights)


def LaplacianSmoothing(data, sigma, device):
    """ d = ifft(fft(g)/(1-sigma*fft(v))) """
    size = torch.numel(data)
    c = np.zeros(shape=(1, size))
    c[0, 0] = -2.
    c[0, 1] = 1.
    c[0, -1] = 1.
    c = torch.Tensor(c).to(device)
    c_fft = torch.view_as_real(torch.fft.fft(c))
    coeff = 1. / (1. - sigma * c_fft[..., 0])
    tmp = data.view(-1, size).to(device)
    ft_tmp = torch.fft.fft(tmp)
    ft_tmp = torch.view_as_real(ft_tmp)
    tmp = torch.zeros_like(ft_tmp)
    tmp[..., 0] = ft_tmp[..., 0] * coeff
    tmp[..., 1] = ft_tmp[..., 1] * coeff
    tmp = torch.view_as_complex(tmp)
    tmp = torch.fft.ifft(tmp)
    tmp = tmp.view(data.size())
    return tmp.real





@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)
        if data_name == 'FashionMNIST':
            self.model = ConvNet().to(device)
        if data_name == 'SVHN':
            self.model = ConvNet().to(device)
        if data_name == 'EMNIST':
            self.model = ConvNet().to(device)
        if data_name == 'MNIST':
            self.model = ConvNet().to(device)
        # torch.set_float32_matmul_precision('high')
        # self.model = torch.compile(self.model)
        # if args.lora == True:
        #    self.model = get_peft_model(self.model, lora_config)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = None
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.dp_clip = 10

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]

    def state_hi_loader(self, index):
        if not hi_dict.get(index):
            return
        self.hi = hi_dict[index]

    def state_Li_loader(self, index):
        if not Li_dict.get(index):
            return
        self.Li = Li_dict[index]

    def get_train_loss(self):
        return self.loss
    def get_param_name(self, param):
        # 获取参数的名称
        for name, p in self.model.named_parameters():
            if p is param:
                return name
        return None



    def update_FedAvg(self, weights, E, index, lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if args.privacy == 1:
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        noise = torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape).to(device)
                        param.grad += noise
                self.optimizer.step()
                self.optimizer.zero_grad()

        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w


    def update_FedAvg_adamw(self, weights, E, index, lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if args.privacy == 1:
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        # 梯度裁剪
                        param.grad *= min(1, args.C / norm_value)
                        # 添加差分隐私噪声
                        noise = torch.normal(0, args.dp_sigma * args.C / args.batch_size, size=param.grad.shape).to(
                            device)
                        param.grad += noise
                self.optimizer.step()
                self.optimizer.zero_grad()

        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_FedAdamW(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                   param.requires_grad = True
        if momen_m=={}:
            momen_m = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        for k, v in self.model.named_parameters():
            momen_m[k] = momen_m[k].to(device)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                           weight_decay=0.01)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step'] = step.to(device)
                self.optimizer.state[p]['exp_avg'] = torch.zeros_like(p.data).to(device)
                self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)
        step = 0  # 新增步数计数
        self.loss = 0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                if args.privacy == 1:
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        param.grad += torch.normal(0, args.dp_sigma * args.C / args.batch_size, size=param.grad.shape).to(
                            device)
                self.optimizer.step()
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma*lr/(args.K)))
        momen_v = {}  # 用字典存每个参数的动量项
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                state = self.optimizer.state.get(p, None)
                momen_v[param_name] = state['exp_avg_sq'].clone().detach().to('cpu')
        delta_w = {}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w, momen_v

    def update_FedAvg_LS(self, weights, E, index, lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.laplacian = args.laplacian
        self.ls_sigma = args.ls_sigma
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if args.privacy == 1:
                    # 添加差分隐私噪声
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        noise = torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape)
                        noise = noise.to(device)
                        param.grad += noise
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            p.data.add_(other=LaplacianSmoothing(p.grad.data, self.ls_sigma, device),
                                        alpha=-group['lr'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        delta_w = deepcopy(self.model.state_dict())
        for k, v in self.model.state_dict().items():
            delta_w[k] = delta_w[k].to('cpu')
            delta_w[k] = delta_w[k] - weights[k]
        return delta_w

    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        if self.ci == None:
            self.ci = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        if ps_c == None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for n, p in model.named_parameters():
            ps_c[n] = ps_c[n].to(device)
            self.ci[n] = self.ci[n].to(device)
            weights[n] = weights[n].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                lg_loss = 0
                loss_c = self.criterion(output, target)
                for n, p in model.named_parameters():
                    lossh = (p * (self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                if args.privacy == 1:
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        noise = torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape)
                        noise = noise.to(device)
                        param.grad += noise
                    self.optimizer.step()
        send_ci = {}
        ci = deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k] = weights[k].to('cpu')
            ci[k] = ci[k].to('cpu')
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator) * lr) + ci[k] - ps_c[k]
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        c_dict[index] = deepcopy(self.ci)
        return delta_w, send_ci




    def update_FedSAM(self, weights, E, index, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        base_optimizer = torch.optim.SGD
        # self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0.5, rho=args.rho,adaptive=False)
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0, rho=args.rho, adaptive=False)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if args.privacy == 1:
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        noise = torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape)
                        noise = noise.to(device)
                        param.grad += noise
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                if args.privacy == 1:
                    for name, param in self.model.named_parameters():
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        noise = torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape)
                        noise = noise.to(device)
                        param.grad += noise
                self.optimizer.second_step(zero_grad=True)

        delta_w = deepcopy(self.model.state_dict())

        for k in delta_w.keys():
            delta_w[k] = delta_w[k].to('cpu')
            delta_w[k] = delta_w[k] - weights[k]
        return delta_w




    def load_dict(self):
        self.func_dict = {
            'DP-FedAvg': self.update_FedAvg,
            'DP-FedAvg-adamw': self.update_FedAvg_adamw,
            'DP-FedAvg-LS': self.update_FedAvg_LS,
            'DP-SCAFFOLD': self.update_scaf,  # scaf
            'DP-FedSAM': self.update_FedSAM,
            'DP-FedAdamW': self.update_FedAdamW,

        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None, step=None,ci=None):
        self.load_dict()
        if alg in {'DP-SCAFFOLD'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'DP-FedAdamW'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, v, lr, step)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)




def set_random_seed(seed=42):
    """
    设置随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def apply_weights_FedLADA(num_workers, weights,model,momen_m):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    scale = 1.0 / (num_workers * selection)
    momen_v = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）
    for k, v in m[0].items():
        momen_v[k] =v / (num_workers * selection)
    # 之后叠加剩余客户端的梯度
    for ci in m[1:]:
        for k, v in ci.items():
            momen_v[k]+= v / (num_workers * selection)


    ps_w = model.state_dict()  # w : ps_w
    sum_weights = {}  # delta_w : sum_weights
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)

    momen_m=momen_m
    for k, v in sum_weights.items():
        if k not in momen_m.keys():
            momen_m[k]=sum_weights[k]/lr
        else:
            momen_m[k] = sum_weights[k] / lr
            #momen_m[k] =args.alpha*momen_m[k]+(1-args.alpha)*sum_weights[k]/lr

    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),momen_m,momen_v

if __name__ == "__main__":
    # 获取args
    set_random_seed(seed=42)
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    # lr_decay = 1
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay = args.lr_decay
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    momen_m={}
    momen_v = {}
    ps_c = {}
    m={}
    v = {}

    hi_dict = {}
    Li_dict = {}
    import time

    localtime = time.asctime(time.localtime(time.time()))

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'DP-FedAvg',
        'DP-SCAFFOLD',
        'DP-FedSAM',
        'DP-FedAvg-LS',
        'DP-FedAvg-adamw',
        'DP-FedAdamW',


    }
    #  配置logger
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                  .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(comment=alg)

    nums_cls = 100
    if data_name == 'CIFAR10':
        nums_cls = 10
    if data_name == 'CIFAR100':
        nums_cls = 100
    if data_name == 'imagenet':
        nums_cls = 200
    if data_name == 'FashionMNIST':
        nums_cls = 10
    if data_name == 'SVHN':
        nums_cls = 10
    if data_name == 'EMNIST':
        nums_cls = 47
    if data_name == 'MNIST':
        nums_cls = 10

    nums_sample = 500
    if data_name == 'CIFAR10':
        nums_sample = int(50000 / (args.num_workers))
        nums_sample = 500
    if data_name == 'CIFAR100':
        nums_sample = int(50000 / (args.num_workers))
        nums_sample = 500
    if data_name == 'imagenet':
        nums_sample = int(100000 / (args.num_workers))
    if data_name == 'FashionMNIST':
        nums_sample = int(60000 / (args.num_workers))
    if data_name == 'SVHN':
        nums_sample = int(70000 / (args.num_workers))
    if data_name == 'EMNIST':
        nums_sample = 500
    if data_name == 'MNIST':
        nums_sample = 500

    import pickle

    if args.data_name == 'imagenet':
        # 存储变量的文件的名字
        if args.alpha_value == 0.6:
            filename = 'data_idx.data'
        if args.alpha_value == 0.1:
            filename = 'data_idx100000_0.1.data'
        f = open(filename, 'rb')
        # 将文件中的变量加载到当前工作区
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)
        logger.info('std:{}'.format(std))
    #
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

    ps = ParameterServer.remote(lr_ps, alg, tau, selection, data_name, num_workers)
    if data_name == 'imagenet':
        model = ConvNet200().to(device)
    if data_name == 'CIFAR10':
        model = ConvNet().to(device)
    elif data_name == 'CIFAR100':
        model = ConvNet100().to(device)
    if data_name == 'FashionMNIST':
        model = ConvNet().to(device)
    if data_name == 'SVHN':
        model = ConvNet().to(device)
    if data_name == 'EMNIST':
        model = ConvNet().to(device)
    if data_name == 'MNIST':
        model = ConvNet().to(device)

    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection / args.p))]
    logger.info(
        'extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},rho:{},C:{},sigma:{},name:{}'
        .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.rho, args.C, args.dp_sigma,
                args.alg))
    # logger.info('data_idx{}'.format(data_idx))
    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.CNN == 'VIT-B':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('vit_base_patch16_224_in21k.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'VIT-L':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('jx_vit_large_patch16_224_in21k-606da67d.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'swin_tiny':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_tiny_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'swin_small':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_small_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'swin_base':

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_base_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
        # '''

    if args.CNN == 'swin_large':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_large_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    # if args.lora == True:
    #    model = get_peft_model(model, lora_config)
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)

    ps.set_weights.remote(model.get_weights())
    current_weights = ps.get_weights.remote()
    ps_c = ps.get_ps_c.remote()

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    zero = model.get_weights()
    # print(delta_g_sum)
    for k, v in model.get_weights().items():
        zero[k] = zero[k] - zero[k]
    ps_c = deepcopy(zero)

    del zero
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()
        index = np.arange(num_workers)  # 100
        lr = lr * lr_decay
        np.random.shuffle(index)
        if args.lr_decay==2:
            eta_max=args.lr
            #eta_min=args.lr*0.01
            eta_min =0
            t=epochidx
            T=args.epoch
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))

        index = index[:int(num_workers * selection)]  # 10id
        if alg in {'DP-SCAFFOLD'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                # weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                #                                   worker, idx in
                #                                   zip(workers, index_sel)]
                weights_and_ci.extend(
                    [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c)
                     for worker, idx in zip(workers, index_sel)]
                )

            weights_and_ci = ray.get(weights_and_ci)

            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)

            weights = [w for w, ci in weights_and_ci]
            ci = [ci for w, ci in weights_and_ci]
            ps_c = ps.apply_ci.remote(alg, num_workers, *ci)
            current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            ps_c, current_weights = ray.get([ps_c, current_weights])
            # ps_c, current_weights = ray.get([future_ps_c, future_weights])
            # current_weights = deepcopy(current_weights)
            ps_c = deepcopy(ps_c)
            model.set_weights(current_weights)
            del weights_and_ci
            del weights
            del ci
        if alg in {'DP-FedAdamW'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=m, v=v,step=step) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            current_weights, m, v = apply_weights_FedLADA(num_workers, weights_and_ci, model,m)

            model.load_state_dict(current_weights)
            step.add_( args.K)

        elif alg in {'DP-FedAvg', 'DP-FedSAM', 'DP-FedAvg-LS','DP-FedAvg-adamw'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                # worker_sel = workers[i:i + int(n / 2)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]

            time3 = time.time()
            #print(epochidx, '    ', time3 - start_time1)
            current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            current_weights = ray.get(current_weights)
            model.set_weights(current_weights)


        end_time1 = time.time()
        #print(epochidx, '    ', end_time1 - time3)
        print(epochidx, '    ', end_time1 - start_time1)

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            model.set_weights(current_weights)
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                ps_state = ps.get_state.remote()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},C:{},sigma:{},lr:{:.5f},CNN:{},GPU:{},gamma:{},rho:{},alpha_value:{},ls_sigma:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, args.C, args.dp_sigma, lr, args.CNN, args.gpu, args.gamma, args.rho,
                    args.alpha_value, args.ls_sigma))

            print(
                "Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},C:{},sigma:{},lr:{:.5f},CNN:{},GPU:{},data:{},gamma:{},rho:{},alpha_value:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, args.C, args.dp_sigma, lr, args.CNN, args.gpu, args.data_name, args.gamma,
                    args.rho, args.alpha_value))

            # logger.info('attention:{}'.format(ray.get(ps.get_attention.remote())))
            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)

    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)

    result_loss = np.array(result_list_loss)
    test_list_loss = np.array(test_list_loss)
    # x2 = np.array(X_list)
    # div = np.array(div)

    save_name = './plot/alg_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}-C{}-sigma{}-ls_sigma{}'.format(
        alg, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        args.data_name, args.gamma, args.rho, args.CNN, endtime, args.C, args.dp_sigma, args.ls_sigma)

    save_name2 = './model/model_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}-C{}-sigma{}'.format(
        alg, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        args.data_name, args.gamma, args.rho, args.CNN, endtime, args.C, args.dp_sigma)
    torch.save(model.state_dict(), save_name2)
    save_name = save_name + '.npy'
    # save_name2 = save_name2 + '.pth'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()