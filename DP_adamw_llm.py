import math
import os
import statistics
import torch
import torch.nn as nn
import os
os.environ['MPLBACKEND'] = 'Agg'  # 在导入matplotlib之前设置
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import SubsetRandomSampler, random_split
import random
import ray
import argparse
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    RobertaForSequenceClassification, RobertaTokenizer
from dirichlet_data import data_from_dirichlet
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
import torch, gc
from datasets import load_dataset
from tqdm.auto import tqdm  # 或者 from tqdm import tqdm

from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, random_split

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=16, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedMoment', type=str, help='alg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0,1', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.99', type=float, help='lr_decay')
parser.add_argument('--data_name', default='imagenet', type=str, help='lr_decay')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='0.15', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.6', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.06', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=1, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='VIT-L', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.9, type=float, help=' for mom_step')
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--p', default=1, type=int, help=' for mom_step')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--datapath', type=str,
                    default="./data")
parser.add_argument('--num_gpus_per', default=0.5, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD,AdamW')
parser.add_argument("--preprint", type=int, default=5, help="")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")

parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")

parser.add_argument("--clip", type=bool, default=True, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--ls_sigma", type=float, default=1.0)
parser.add_argument('--K', default=20, type=int, help='#workers')
parser.add_argument("--maxnorm", type=float, default=10, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--freeze', default=0, type=int, help='# batch_size')
parser.add_argument('--dp_sigma', default=0.2, type=float, help='noise multiplier for DP')
parser.add_argument('--privacy', default=1, type=int, help='whether to use differential privacy')
parser.add_argument('--C', type=float, default=0.2)

args = parser.parse_args()
print(args.lora)
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16
# num_gpus_per = 0.5
num_gpus = len(gpu_idx.split(','))

data_name = args.data_name
CNN = args.CNN

if args.CNN == 'bert':
    model_path = '../glfl/BERT'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    lora_config = LoraConfig(
        r=args.r,  # LoRA attention dimension
        lora_alpha=args.r,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        target_modules=["query", "value", 'key', 'dense']  # Target modules to apply LoRA
    )

if args.CNN == 'roberta_base':
    model_path = './roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    lora_config = LoraConfig(
        r=args.r,  # LoRA attention dimension
        lora_alpha=args.r * 2,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        target_modules=["query", "value", 'key', 'dense']  # Target modules to apply LoRA
    )

if args.data_name == 'QQP':
    dataset_path = './data/QQP'
    # 加载数据集
    dataset = load_dataset(dataset_path)


    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)


    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name == 'MNLI':
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    dataset_path = './data/MNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)


    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)


    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

if args.data_name == 'STS-B':
    dataset_path = './data/sts-b'
    # 加载数据集
    dataset = load_dataset(dataset_path)

    dataset = dataset.rename_column("score", "label")


    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length",
                         max_length=128)


    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name == 'WNLI':
    dataset_path = './data/WNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)


    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)


    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name == 'RTE':
    dataset_path = './data/RTE'
    dataset = load_dataset(dataset_path)


    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)


    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name == 'MRPC':
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length", max_length=128)


    dataset_path = './data/MRPC'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name == 'qnli':
    def preprocess_function(examples):
        # 拼接问题和句子
        inputs = tokenizer(
            examples["text1"],
            examples["text2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        labels = [1 if label == "entailment" else 0 for label in examples["label"]]
        inputs["labels"] = labels
        return inputs


    dataset_path = './data/qnli'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name == 'sst2':
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", return_tensors="pt",
                         max_length=64)


    dataset_path = './data/sst2'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name == 'cola':
    dataset_path = './data/cola'


    def preprocess_function(examples):
        return tokenizer(examples["Sentence"], padding="max_length", truncation=True, max_length=64)


    dataset = load_dataset(dataset_path)
    dataset = dataset.rename_column("Acceptability", "label")
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # tokenized_dataset = tokenized_dataset.rename_column("Acceptability", "label")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

# '''

# '''
seed = 42
if args.alpha_value == 1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        generator = torch.Generator().manual_seed(42)
        train_dataset = tokenized_dataset["train"]
        total_size = len(train_dataset)
        # print(total_size)
        subset_size = total_size // args.num_workers
        remainder = total_size % args.num_workers  # 计算剩余的样本数

        # 创建分割大小列表
        split_sizes = [subset_size] * (args.num_workers - 1) + [subset_size + remainder]
        subsets = random_split(train_dataset, split_sizes, generator=generator)
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_dataset = tokenized_dataset["train"]
        train_loader = DataLoader(subsets[pid], batch_size=args.batch_size, shuffle=True)
        return train_loader

if args.alpha_value != 1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler, num_workers=0, generator=torch.Generator().manual_seed(seed))
        return train_loader


def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    # train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4)
    return test_loader


def get_data_loader_train(data_name):
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4)
    return test_loader


def evaluate2(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # test_loss+= criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    return accuracy, torch.tensor(0), torch.tensor(0)


def evaluate(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss += criterion(logits, target)
        for batch in tqdm(train_loader, disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            train_loss += criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    return accuracy, test_loss / len(test_loader), train_loss / len(train_loader)


def laplacian_smoothing(update, lambda_smooth=args.ls_sigma):
    """
    对一维模型参数差分进行拉普拉斯平滑
    """
    smoothed = update.clone()
    smoothed[1:-1] = update[1:-1] - lambda_smooth * (2 * update[1:-1] - update[:-2] - update[2:])
    return smoothed


import torch.nn.functional as F


def laplacian_smoothing_2d(update, lambda_smooth=args.ls_sigma):
    """
    针对2D参数如Conv层进行平滑
    """
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=update.dtype, device=update.device).unsqueeze(0).unsqueeze(0)

    laplace = F.conv2d(update.unsqueeze(0), kernel, padding=1)
    smoothed = update - lambda_smooth * laplace.squeeze(0)
    return smoothed


import torch
import torch.nn.functional as F


def laplacian_smoothing_4d(update, lambda_smooth=args.ls_sigma):
    """
    针对 4D Conv2D 参数进行拉普拉斯平滑
    update: 形状 [out_channels, in_channels, kernel_size, kernel_size]
    lambda_smooth: 平滑系数
    """
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=update.dtype, device=update.device).unsqueeze(0).unsqueeze(0)

    # 适配 Conv2D 的多通道输入
    kernel = kernel.expand(update.size(1), 1, 3, 3)  # [in_channels, 1, 3, 3]

    # 计算拉普拉斯变换
    laplace = F.conv2d(update, kernel, padding=1, groups=update.size(1))  # 按 in_channels 计算
    smoothed = update - lambda_smooth * laplace

    return smoothed




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


import torch
import numpy as np


def LaplacianSmoothing2D(data, sigma, device):
    """
    Applies 2D Laplacian smoothing in the frequency domain.
    data: Tensor of shape (H, W)
    sigma: smoothing strength
    """
    H, W = data.shape
    c = torch.zeros((H, W), dtype=torch.float32, device=device)

    # 2D discrete Laplacian kernel (periodic BC)
    c[0, 0] = -4.
    c[0, 1] = 1.
    c[1, 0] = 1.
    c[-1, 0] = 1.
    c[0, -1] = 1.

    # FFT of the Laplacian kernel
    c_fft = torch.fft.fft2(c)
    coeff = 1. / (1. - sigma * c_fft)

    # FFT of the input data
    data = data.to(device)
    data_fft = torch.fft.fft2(data)

    # Apply filter
    smoothed_fft = data_fft * coeff
    smoothed = torch.fft.ifft2(smoothed_fft)

    return smoothed.real


import torch
from torch.optim.optimizer import Optimizer


@ray.remote(num_cpus=1, num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if args.CNN == 'bert':
            model_path = '../glfl/BERT'
            self.model = BertForSequenceClassification.from_pretrained(model_path)

        if args.CNN == 'roberta_base':
            model_path = './roberta_base'
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            if args.data_name == 'MNLI':
                self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

        if args.lora == 1 and args.alg != "FLORA":
            self.model = get_peft_model(self.model, lora_config)
            # print(args.lora)
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
        self.momen_v = {}
        self.momen_m = {}
        self.R = 0
        self.t = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in self.model.named_parameters()}

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


    def update_FedAvg_per(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                         weight_decay=1e-3)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                step += 1
                if args.privacy == 1:
                    layer_clip_norms = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        layer_clip_norms[name] = norm_value
                    values = list(layer_clip_norms.values())
                    median_value = statistics.median(values)
                    args.C = min(median_value, 0.4)
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        param.grad += torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape).to(device)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w
    def update_FedAvg_adamw(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                step += 1
                if args.privacy == 1:
                    layer_clip_norms = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        layer_clip_norms[name] = norm_value
                    values = list(layer_clip_norms.values())
                    median_value = statistics.median(values)
                    args.C = min(median_value, 0.4)
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        param.grad += torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape).to(device)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        # 6. 模型迁回 CPU，清显存（如果后面这一段时间不用它算梯度的话）
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w

    def update_FedAvg_LS_per(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze == 0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.ls_sigma = args.ls_sigma

        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                step += 1
                if args.privacy == 1:
                    layer_clip_norms = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        layer_clip_norms[name] = norm_value
                    values = list(layer_clip_norms.values())
                    median_value = statistics.median(values)
                    args.C = min(median_value, 0.4)
                    # 添加差分隐私噪声
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        param.grad += torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape).to(device)
                    #for name, param in self.model.named_parameters():
                    #    if param.grad is not None:
                    #        if 'lora' in name:
                    #            param.data.add_(other=LaplacianSmoothing(param.grad.data, args.ls_sigma, device),
                    #                            alpha=-lr)
                    #        else:
                    #            param.data.add_(other=LaplacianSmoothing(param.grad.data, args.ls_sigma, device),
                    #                            alpha=-lr)
                self.optimizer.step()
                self.optimizer.zero_grad()
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w


    def update_FedAdamW(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
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
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                step += 1
                if args.privacy == 1:
                    layer_clip_norms = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        layer_clip_norms[name] = norm_value
                    values = list(layer_clip_norms.values())
                    median_value = statistics.median(values)
                    args.C = min(median_value, 0.4)
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        norm_value = torch.norm(param.grad, 2)
                        param.grad *= min(1, args.C / norm_value)
                        param.grad += torch.normal(0, args.dp_sigma * args.C / args.batch_size,
                                             size=param.grad.shape).to(device)

                    #for group in self.optimizer.param_groups:
                    #    for p in group['params']:
                    #        p.data.add_(other=LaplacianSmoothing(p.grad.data, args.ls_sigma, device),
                    #                    alpha=-group['lr'])
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
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return delta_w, momen_v



    def load_dict(self):
        self.func_dict = {
            'DP-FedAvg': self.update_FedAvg_per,
            'DP-FedAvg-adamw': self.update_FedAvg_adamw,
            'DP-FedAvg-LS': self.update_FedAvg_LS_per,
            'DP-FedAdamW': self.update_FedAdamW


        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None, step=None, shared_state=None):
        self.load_dict()
        if alg in {'FedCM', 'MoFedSAM',
                   'FedACG'}:
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


@torch.no_grad()
def apply_weights_avg(num_workers, weights, model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora == 1:
                sum_weights[k].add_(v, alpha=scale)  # inplace 加法
            else:
                sum_weights[k].add_(v, alpha=scale)
    # 将 server 模型加上 delta_w
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])  # inplace 加法
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}


def apply_weights_avg_LS(num_workers, weights, model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}  # w : ps_w
    sum_weights = {}  # delta_w : sum_weights
    global_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            sum_weights[name] = laplacian_smoothing(sum_weights[name])
        elif len(param.shape) == 2:
            sum_weights[name] = laplacian_smoothing_2d(sum_weights[name])
        elif len(param.shape) == 4:
            sum_weights[name] = laplacian_smoothing_4d(sum_weights[name])
        else:
            sum_weights[name] = sum_weights[name]
    for k, v in sum_weights.items():  # w = w + delta_w
        global_weights[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(global_weights)
    return {k: v.cpu() for k, v in model.state_dict().items()}


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
    set_random_seed(seed=seed)
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
    mi_dict = {}
    vi_dict = {}
    ti_dict = {}
    import time

    localtime = time.asctime(time.localtime(time.time()))

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'DP-FedAvg-adamw',
        'DP-FedAvg',
        'DP-FedAvg-LS',
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

    nums_cls = 2
    if data_name == 'sst2':
        nums_cls = 2

    nums_sample = 500
    if data_name == 'sst2':
        nums_sample = int(67349 / (args.num_workers))
    if data_name == 'cola':
        nums_sample = int(8500 / (args.num_workers))
    if data_name == 'qnli':
        nums_sample = int(105000 / (args.num_workers))
    if data_name == 'MRPC':
        nums_sample = int(3700 / (args.num_workers))
    if data_name == 'RTE':
        nums_sample = int(2500 / (args.num_workers))

    # if args.data_name!='imagenet':
    # data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)

    import pickle
    if args.alpha_value == 0.6:
        filename = 'data_idx.data'
    if args.alpha_value == 0.1:
        filename = 'data_idx100000_0.1.data'
    filename = 'data_idx100000_0.1.data'
    if args.alpha_value==1:
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)

    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)



    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)

    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path)

        if args.data_name == 'MNLI':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model = model.to(device)
    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection / args.p))]
    logger.info("extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},gamma:{},"
                "freeze:{},ls_sigma:{} "
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.gamma, args.freeze,
                        args.ls_sigma))
    # logger.info('data_idx{}'.format(data_idx))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)

    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        if args.data_name == 'MNLI':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

    if args.lora == 1 and args.alg != 'FLORA':
        model = get_peft_model(model, lora_config)
        # print(args.lora)

    current_weights = model.state_dict()

    ps_c = None

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0

    div = []
    sim = []
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()
        index = np.arange(num_workers)  # 100
        lr = lr * lr_decay
        if args.lr_decay==2:
            eta_max=args.lr
            #eta_min=args.lr*0.01
            eta_min =0
            t=epochidx
            T=args.epoch
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))


        if alg in {'DP-FedAvg-adamw','DP-FedAvg-LS','DP-FedAvg'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            weights = ray.get(weights)
            current_weights = apply_weights_avg(num_workers, weights, model)
            model.load_state_dict(current_weights)
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

        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            model.to('cpu')
            torch.cuda.empty_cache()
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}:  accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.8f},CNN:{},GPU:{},r:{},dp:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.r, args.dp_sigma))

            print(
                "Iter {}:  accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.8f},CNN:{},GPU:{},data:{},r:{},dp:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.data_name, args.r, args.dp_sigma))

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
    save_name = './plot/alg_{}-data_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-optimizer{}-time{}'.format(
        alg, args.data_name, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN, args.optimizer, endtime)
    save_name = save_name + '.npy'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()