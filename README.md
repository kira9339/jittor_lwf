# jittor_lwf
## 环境配置
  首先下载wsl，然后会自动下载ubuntu(需要打开电脑的hyper_v服务)，然后下载conda,配置环境jittor1.3.9，g++11.4.0，python3.8.0（g++和python版本不能太高，否则无法兼容），然后在vscode中连接wsl系统，进入到自己创建的相应虚拟环境，pytorch下载最新版即可。(本代码并未配置cuda，原因是作者的电脑是集显)
## 数据处理脚本
```python
#jittor
import os
import shutil
from jittor.dataset import Dataset
from jittor import jt
from PIL import Image  


data_root = '/mnt/d/tiny-imagenet/tiny-imagenet-200'
old_task_dir = '/mnt/d/tiny-imagenet/old_task'
new_task_dir = '/mnt/d/tiny-imagenet/new_task'

def split_tinyimagenet():
    print("正在创建目录")
    os.makedirs(old_task_dir, exist_ok=True)
    os.makedirs(new_task_dir, exist_ok=True)

    print("正在扫描类别")
    try:
        classes = sorted(os.listdir(os.path.join(data_root, 'train')))
        print(f"找到{len(classes)}个类别")
    except Exception as e:
        print(f"扫描类别失败: {e}")
        return

    print("正在划分数据")
    for i, cls in enumerate(classes):
        dst_dir = old_task_dir if i < 100 else new_task_dir
        src = os.path.join(data_root, 'train', cls)
        dst = os.path.join(dst_dir, 'train', cls)
        
        try:
            shutil.copytree(src, dst)
            print(f"已复制: {cls} -> {dst}")
        except Exception as e:
            print(f"复制失败 {cls}: {e}")

    print("划分完成")

if __name__ == '__main__':
    split_tinyimagenet()

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, debug=False):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.debug = debug

        if self.train:
            self.data_dir = os.path.join(self.root, 'train') if 'train' not in self.root else self.root
        else:
            self.data_dir = os.path.join(self.root, 'val', 'images')
            self.annotation_file = os.path.join(self.root, 'val', 'val_annotations.txt')

        print(f"[DEBUG] 开始扫描数据集目录: {self.data_dir}")  
        self.samples = []
        if self.train:
            print(f"[DEBUG] 开始扫描训练集类别...")  
            self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"[DEBUG] 找到 {len(self.classes)} 个类别")  
            for cls in self.classes:
                cls_dir = os.path.join(self.data_dir, cls, 'images')
                if not os.path.exists(cls_dir):
                    print(f"类 {cls} 的 images 目录不存在: {cls_dir}")
                    continue
                print(f"[DEBUG] 开始扫描类别 {cls} ")  
                for img_name in os.listdir(cls_dir):
                    full_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(full_path):
                        self.samples.append((full_path, self.class_to_idx[cls]))
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")  
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            self.img_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in lines}
            self.classes = sorted(list(set(self.img_to_class.values())))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.samples = [
                (os.path.join(self.data_dir, img_name), self.class_to_idx[self.img_to_class[img_name]])
                for img_name in self.img_to_class.keys()
                if os.path.isfile(os.path.join(self.data_dir, img_name))
            ]

        self.set_attrs(total_len=len(self.samples), batch_size=128, shuffle=True)


    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # return img, label
        return img, jt.int32(label)
#---------------------------------------------------
#pytorch
import os
import shutil
from torch.utils.data import Dataset
from PIL import Image
import torch

data_root = '/mnt/d/tiny-imagenet/tiny-imagenet-200'
old_task_dir = '/mnt/d/tiny-imagenet/old_task'
new_task_dir = '/mnt/d/tiny-imagenet/new_task'

def split_tinyimagenet():
    print("正在创建目录")
    os.makedirs(old_task_dir, exist_ok=True)
    os.makedirs(new_task_dir, exist_ok=True)

    print("正在扫描类别")
    try:
        classes = sorted(os.listdir(os.path.join(data_root, 'train')))
        print(f"找到{len(classes)}个类别")
    except Exception as e:
        print(f"扫描类别失败: {e}")
        return

    print("正在划分数据")
    for i, cls in enumerate(classes):
        dst_dir = old_task_dir if i < 100 else new_task_dir
        src = os.path.join(data_root, 'train', cls)
        dst = os.path.join(dst_dir, 'train', cls)
        
        try:
            shutil.copytree(src, dst)
            print(f"已复制: {cls} -> {dst}")
        except Exception as e:
            print(f"复制失败 {cls}: {e}")

    print("划分完成")

if __name__ == '__main__':
    split_tinyimagenet()


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, debug=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.debug = debug

        if self.train:
            self.data_dir = os.path.join(self.root, 'train') if 'train' not in self.root else self.root
        else:
            self.data_dir = os.path.join(self.root, 'val', 'images')
            self.annotation_file = os.path.join(self.root, 'val', 'val_annotations.txt')

        print(f"[DEBUG] 开始扫描数据集目录: {self.data_dir}")
        self.samples = []
        if self.train:
            print(f"[DEBUG] 开始扫描训练集类别...")
            self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"[DEBUG] 找到 {len(self.classes)} 个类别")
            for cls in self.classes:
                cls_dir = os.path.join(self.data_dir, cls, 'images')
                if not os.path.exists(cls_dir):
                    print(f"类 {cls} 的 images 目录不存在: {cls_dir}")
                    continue
                print(f"[DEBUG] 开始扫描类别 {cls}")
                for img_name in os.listdir(cls_dir):
                    full_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(full_path):
                        self.samples.append((full_path, self.class_to_idx[cls]))
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            self.img_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in lines}
            self.classes = sorted(list(set(self.img_to_class.values())))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.samples = [
                (os.path.join(self.data_dir, img_name), self.class_to_idx[self.img_to_class[img_name]])
                for img_name in self.img_to_class.keys()
                if os.path.isfile(os.path.join(self.data_dir, img_name))
            ]



    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label
```
## 训练脚本
```python
# jittor
import jittor.nn as nn
import jittor as jt
from jittor.transform import Compose, Resize, RandomCrop, ToTensor, RandomHorizontalFlip
from jt_ds import TinyImageNet
import os
import time
import argparse
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    
    def execute(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        return self.classifier(x)


def train_old_model(resume=False):
    def custom_normalize(tensor):
        mean = jt.array([0.485, 0.456, 0.406]).view(3,1,1)
        std = jt.array([0.229, 0.224, 0.225]).view(3,1,1)
        return (tensor - mean) / std

    transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),  
        ToTensor(),
        custom_normalize
    ])

    print("正在加载数据集")
    train_data = TinyImageNet(
        root='/mnt/d/tiny-imagenet/old_task',
        train=True,
        transform=transform,
        debug=True
    )
    print("数据集加载完成，正在创建数据加载器")
    train_loader = train_data.set_attrs(batch_size=128, shuffle=True)  

    print("测试数据加载")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"成功加载 Batch {i}: inputs.shape={inputs.shape}, targets.shape={targets.shape}")
        if i >= 2:
            break

    model = AlexNet(num_classes=100)

    def xavier_normal_(tensor, gain=1.0):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
        jt.init.gauss_(tensor, mean=0.0, std=std)

    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()
        if dimensions == 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            fan_in = tensor.size(1) * tensor[0][0].numel()
            fan_out = tensor.size(0) * tensor[0][0].numel()
        return fan_in, fan_out

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal_(m.weight, gain=1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                jt.init.constant_(m.bias, 0.0)

    model.apply(init_weights)

    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_list = []

    start_epoch = 0
    start_batch = 0


    start_time = time.time()
    total_batches = len(train_loader)

    for epoch in range(start_epoch, 10):

        if epoch == 0:
            optimizer.lr = 0.01
        elif epoch == 1:
            optimizer.lr = 0.005
        elif epoch >=4:
            optimizer.lr = 0.0001
        elif epoch >= 2:
            optimizer.lr = 0.0025

        print(f"\nEpoch {epoch+1} 开始，当前学习率: {optimizer.lr}")

        epoch_start = time.time()
        train_iter = enumerate(train_loader)
        

        for i, (inputs, targets) in train_iter:
            if epoch == start_epoch and i < start_batch:
                continue

            try:
                outputs = model(inputs)
                loss = nn.cross_entropy_loss(outputs, targets)
                optimizer.step(loss)

                print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")


            except Exception as e:
                print(f"训练第{i+1}个 batch 时出错: {e}")
                raise
        print(f'Epoch {epoch+1} 完成, 耗时: {time.time() - epoch_start:.2f}秒\n')

    total_time = time.time() - start_time
    print(f"总训练耗时: {total_time:.2f} 秒 ≈ {total_time/60:.2f} 分钟")
    plt.figure(figsize=(10,5))
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve of Old Model Training")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    print("loss 曲线已保存为 loss_curve.png")

    jt.save(model.state_dict(), 'pretrained_old_model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("开始训练...")
    jt.flags.log_silent = False
    jt.flags.use_cuda = 0

    
    train_old_model(resume=args.resume)
# --------------------------------------------------------------
#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_ds import TinyImageNet
import os
import time
import argparse
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        return self.classifier(x)


def train_old_model(resume=False):
    def custom_normalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return (tensor - mean) / std

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        custom_normalize
    ])

    print("正在加载数据集")
    train_data = TinyImageNet(
        root='/mnt/d/tiny-imagenet/old_task',
        train=True,
        transform=transform,
        debug=True
    )
    print("数据集加载完成，正在创建数据加载器")
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)

    print("测试数据加载")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"成功加载 Batch {i}: inputs.shape={inputs.shape}, targets.shape={targets.shape}")
        if i >= 2:
            break

    model = AlexNet(num_classes=100)
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_list = []

    start_epoch = 0

    start_time = time.time()
    total_batches = len(train_loader)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
      if epoch >= 4:
        for param_group in optimizer.param_groups:
          param_group['lr'] = 0.0001
      elif epoch >= 2:
        for param_group in optimizer.param_groups:
         param_group['lr'] = 0.0025


        print(f"\nEpoch {epoch+1} 开始，当前学习率: {optimizer.lr}")

        epoch_start = time.time()
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())  

                print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {i+1}: {e}")

                raise

        print(f'Epoch {epoch+1} completed, time: {time.time() - epoch_start:.2f} seconds\n')


        try:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")

        except Exception as e:
                print(f"Error in batch {i+1}: {e}")

                raise


        print(f'Epoch {epoch+1} completed, time: {time.time() - epoch_start:.2f} seconds\n')

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ≈ {total_time/60:.2f} minutes")
    
    plt.figure(figsize=(10,5))
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve of Old Model Training")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    print("loss 曲线已保存为 loss_curve.png")

    torch.save(model.state_dict(), 'pretrained_old_model.pth')
    save_path = os.path.abspath("pretrained_old_model.pth")
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("开始训练...")
    train_old_model()
```
## 测试脚本
```python
import jittor.nn as nn
import jittor as jt
from jittor.transform import Compose, Resize, CenterCrop, ToTensor, RandomCrop
from jt_ds import TinyImageNet
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


# 强制使用GPU（如果可用）并关闭冗余日志
jt.flags.use_cuda = 0  
jt.flags.log_silent = True

# 旧模型定义（与pretrain_old_model.py完全一致）
class OldAlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)  # 单输出层
        )
    
    def execute(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        return self.classifier(x)  # 仅返回旧任务输出（形状：[batch, 100]）

# 新模型定义（支持新旧任务双输出）
class AlexNet(nn.Module):
    def __init__(self, num_old_classes=100, num_new_classes=100):
        super().__init__()
        # 共享特征层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 旧任务头
        self.old_classifier = nn.Linear(256*6*6, num_old_classes)
        # 新任务头
        self.new_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_new_classes)
        )
    
    def execute(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        y_old = self.old_classifier(x)  # 形状：[batch, 100]
        y_new = self.new_classifier(x)  # 形状：[batch, 100]
        return (y_old, y_new)  # 明确返回元组

def calculate_accuracy(model, data_loader, task='new'):
    model.eval()
    correct = 0
    total = 0

    with jt.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            # 处理新旧任务双输出的情况
            if isinstance(outputs, tuple):
                outputs_old, outputs_new = outputs
            else:
                outputs_old = outputs_new = outputs

            # 选择当前任务的输出
            outputs_task = outputs_new if task == 'new' else outputs_old
            predicted = outputs_task.argmax(1)
            # 检查predicted是否为元组
            if isinstance(predicted, tuple):
                # 如果是元组，取第二个元素（索引）
                if len(predicted) >= 2:
                    predicted = predicted[1]  # 取索引部分
                else:
                    raise RuntimeError("argmax返回的元组长度不足，无法获取索引")
                            # 确保 predicted 是 jittor.Var 类型
            if not isinstance(predicted, jt.Var):
                raise RuntimeError(f"predicted 不是 jittor.Var 类型，而是 {type(predicted)} 类型")
            if total == 0:
                print(f"[Debug] predicted type: {type(predicted)}")
                print(f"[Debug] predicted shape: {predicted.shape if hasattr(predicted, 'shape') else 'N/A'}")
                print(f"[Debug] targets shape: {targets.shape}")      
            # 计算正确预测数
            predicted = predicted.int32()
            targets = targets.int32()
            jt.sync_all()
            correct += (predicted == targets).sum().item()

            total += targets.size(0)
            

    return 100. * correct / total


def kl_div_loss(p, q):
    """处理新旧模型输出的KL散度损失"""
    if isinstance(q, tuple):  # 兼容旧模型可能返回的元组
        q = q[0]
    p = nn.softmax(p/2, dim=1)
    q = nn.softmax(q/2, dim=1)
    return (p * (p.log() - q.log())).sum(dim=1).mean()

def plot_metrics(train_losses, old_accs, new_accs, test_old_accs, test_new_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(old_accs, label='Old Task Accuracy')
    plt.plot(new_accs, label='New Task Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(test_old_accs, label='Old Task Test Accuracy')
    plt.plot(test_new_accs, label='New Task Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
CHECKPOINT_PATH = "lwf_checkpoint.pkl"

def save_checkpoint(model, optimizer, epoch, history):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'history': history
    }
    jt.save(checkpoint, CHECKPOINT_PATH)
    print(f"[Checkpoint] 已保存至 {CHECKPOINT_PATH}")

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = jt.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f"[Checkpoint] 恢复自 epoch {epoch}")
        return epoch, history
    return 0, {
        'train_losses': [],
        'old_accuracies': [],
        'new_accuracies': [],
        'test_old_accuracies': [],
        'test_new_accuracies': []
    }

def train(): 
    # 数据加载（确保路径正确）
    train_data = TinyImageNet(
        root='/mnt/d/tiny-imagenet/new_task',
        train=True,
        transform=Compose([
            Resize(256),
            RandomCrop(224),
            ToTensor()
        ])
    )
    train_loader = train_data.set_attrs(batch_size=32, shuffle=True, num_workers=4)
    
    test_data = TinyImageNet(
        root='/mnt/d/tiny-imagenet/new_task',
        train=False,
        transform=Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor()
        ])
    )
    test_loader = test_data.set_attrs(batch_size=32, shuffle=False, num_workers=4)
    
    # 加载旧模型（严格使用OldAlexNet）
    old_model = OldAlexNet(num_classes=100)
    old_model.load_state_dict(jt.load('pretrained_old_model.pkl'))
    old_model.eval()
    
    # 初始化新模型
    model = AlexNet(num_old_classes=100, num_new_classes=100)
    optimizer = nn.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0005)
        # 加载或初始化历史记录
    start_epoch, history = load_checkpoint(model, optimizer)
    train_losses = history['train_losses']
    old_accuracies = history['old_accuracies']
    new_accuracies = history['new_accuracies']
    test_old_accuracies = history['test_old_accuracies']
    test_new_accuracies = history['test_new_accuracies']
    
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs):

        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10', ncols=100)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 前向传播
            outputs_old, outputs_new = model(inputs)
            
            # 检查输出形状
            if batch_idx == 0 and epoch == 0:
                print(f"\n[Debug][Epoch {epoch+1}] outputs_old shape: {outputs_old.shape}")
                print(f"[Debug][Epoch {epoch+1}] outputs_new shape: {outputs_new.shape}")
                print(f"[Debug][Epoch {epoch+1}] targets shape: {targets.shape}\n")
            
            # 计算损失
            loss_new = nn.cross_entropy_loss(outputs_new, targets)
            with jt.no_grad():
                old_outputs = old_model(inputs)  # 旧模型返回单输出
            loss_old = kl_div_loss(outputs_old, old_outputs) * 4
            loss = loss_new + 1.0 * loss_old
            
            # 反向传播
            optimizer.step(loss)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        
        # 计算准确率
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        old_acc = calculate_accuracy(model, train_loader, task='old')
        new_acc = calculate_accuracy(model, train_loader, task='new')
        old_accuracies.append(old_acc)
        new_accuracies.append(new_acc)
        
        test_old_acc = calculate_accuracy(model, test_loader, task='old')
        test_new_acc = calculate_accuracy(model, test_loader, task='new')
        test_old_accuracies.append(test_old_acc)
        test_new_accuracies.append(test_new_acc)
                # 保存checkpoint
        save_checkpoint(model, optimizer, epoch + 1, {
            'train_losses': train_losses,
            'old_accuracies': old_accuracies,
            'new_accuracies': new_accuracies,
            'test_old_accuracies': test_old_accuracies,
            'test_new_accuracies': test_new_accuracies
        })

        
        print(f"Epoch {epoch+1}: "
              f"Loss: {avg_loss:.4f} | "
              f"Old Acc: {old_acc:.2f}% | "
              f"New Acc: {new_acc:.2f}% | "
              f"Test Old: {test_old_acc:.2f}% | "
              f"Test New: {test_new_acc:.2f}%")
    
    
    jt.save(model.state_dict(), 'lwf_model.pkl')
    plot_metrics(train_losses, old_accuracies, new_accuracies, 
                test_old_accuracies, test_new_accuracies)

if __name__ == '__main__':
    print("===== 开始训练 =====")
    train()
#--------------------------------------------------
#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_ds import TinyImageNet  
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OldAlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class AlexNet(nn.Module):
    def __init__(self, num_old_classes=100, num_new_classes=100):
        super().__init__()
        self.features = OldAlexNet(num_old_classes).features
        self.old_classifier = nn.Linear(256 * 6 * 6, num_old_classes)
        self.new_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_new_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.old_classifier(x), self.new_classifier(x)

def calculate_accuracy(model, data_loader, task='new'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs_old, outputs_new = outputs
            else:
                outputs_old = outputs_new = outputs

            outputs_task = outputs_new if task == 'new' else outputs_old
            predicted = outputs_task.argmax(1)

            if isinstance(predicted, tuple):

                if len(predicted) >= 2:
                    predicted = predicted[1]  
                else:
                    raise RuntimeError("argmax返回的元组长度不足，无法获取索引")
                            
            if not isinstance(predicted, torch.Tensor):
                raise RuntimeError(f"predicted 不是 jittor.Var 类型，而是 {type(predicted)} 类型")
            if total == 0:
                print(f"[Debug] predicted type: {type(predicted)}")
                print(f"[Debug] predicted shape: {predicted.shape if hasattr(predicted, 'shape') else 'N/A'}")
                print(f"[Debug] targets shape: {targets.shape}")      

            predicted = predicted.to(torch.int32)
            targets = targets.to(torch.int32)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            

    return 100. * correct / total

def kl_div_loss(p, q):
    if isinstance(q, tuple):  
        q = q[0]
    p = torch.softmax(p / 2, dim=1)
    q = torch.softmax(q / 2, dim=1)
    return torch.mean(torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=1))

def plot_metrics(train_losses, old_accs, new_accs, test_old_accs, test_new_accs):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss'); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(old_accs, label='Old Task Accuracy')
    plt.plot(new_accs, label='New Task Accuracy')
    plt.title('Training Accuracy'); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(test_old_accs, label='Test Old Acc')
    plt.plot(test_new_accs, label='Test New Acc')
    plt.title('Test Accuracy'); plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

CHECKPOINT_PATH = "checkpoints/lwf_checkpoint_py.pt"

def save_checkpoint(model, optimizer, epoch, history):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'history': history
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"[Checkpoint] 已保存至 {CHECKPOINT_PATH}")

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f"[Checkpoint] 恢复自 epoch {epoch}")
        return epoch,history
    
    return 0, {
            'train_losses': [],
            'old_accuracies': [],
            'new_accuracies': [],
            'test_old_accuracies': [],
            'test_new_accuracies': []
        }

# ==== 训练主函数 ====
def train():
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_data = TinyImageNet(root='/mnt/d/tiny-imagenet/new_task', train=True, transform=transform_train)
    test_data = TinyImageNet(root='/mnt/d/tiny-imagenet/new_task', train=False, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    old_model = OldAlexNet(num_classes=100).to(device)
    old_model.load_state_dict(torch.load('pretrained_old_model.pth'))
    old_model.eval()

    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0005)

    start_epoch, history = load_checkpoint(model, optimizer)
    train_losses = history['train_losses']
    old_accs = history['old_accuracies']
    new_accs = history['new_accuracies']
    test_old_accs = history['test_old_accuracies']
    test_new_accs = history['test_new_accuracies']

    num_epochs = 5

    for epoch in range(start_epoch, num_epochs):

            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                out_old, out_new = model(inputs)
                with torch.no_grad():
                    old_outputs = old_model(inputs)

                loss_new = nn.CrossEntropyLoss()(out_new, targets)
                loss_old = kl_div_loss(out_old, old_outputs) * 4
                loss = loss_new + loss_old

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            old_acc = calculate_accuracy(model, train_loader, task='old')
            new_acc = calculate_accuracy(model, train_loader, task='new')
            old_accs.append(old_acc)
            new_accs.append(new_acc)

            test_old_acc = calculate_accuracy(model, test_loader, task='old')
            test_new_acc = calculate_accuracy(model, test_loader, task='new')
            test_old_accs.append(test_old_acc)
            test_new_accs.append(test_new_acc)

            # 保存 checkpoint
            save_checkpoint(model, optimizer, epoch + 1, {
                'train_losses': train_losses,
                'old_accuracies': old_accs,
                'new_accuracies': new_accs,
                'test_old_accuracies': test_old_accs,
                'test_new_accuracies': test_new_accs
            })

            print(f"Epoch {epoch+1}: "
              f"Loss: {avg_loss:.4f} | "
              f"Old Acc: {old_acc:.2f}% | "
              f"New Acc: {new_acc:.2f}% | "
              f"Test Old: {test_old_acc:.2f}% | "
              f"Test New: {test_new_acc:.2f}%")
    
    plot_metrics(train_losses, old_accs, new_accs, test_old_accs, test_new_accs)
    torch.save(model.state_dict(), "lwf_model_py.pt")


if __name__ == '__main__':
    print("===== 开始训练（PyTorch） =====")
    train()
```
## 训练结果（jittor与pytorch对齐）
### jittor
![image](https://github.com/kira9339/jittor_lwf/blob/main/training_metrics.png)
### pytorch
![image](https://github.com/kira9339/jittor_lwf/blob/main/training_metrics_pytorch.png)
## 注意事项

直接在windows系统中下载jittor会出现无法运行的报错，jittor在windows系统下兼容性不是很好。

安装中会出现ImportError: libstdc++.so.6: version `GLIBCXX_3.4.30'报错，如果电脑里面有对应的版本，是因为 Conda 环境使用了自带的旧版 libstdc++.so.6，需要删除旧的库，手动安装新库。

数据准备脚本是jt_ds.py和pytorch_ds.py，预训练脚本是jt_pretrain_old_model.py和pytorch_pretrain_old_model.py，测试脚本是jt_Alexnet_train.py和alext_pytorch.py，按顺序进行运行。

数据集是Tiny-imagenet（论文中是imagenet，但是受限于电脑性能以及下载速度选用的小数据集），将这个数据集随机平均分成两份，作为新旧任务。
## 最后
论文复现有诸多瑕疵，一切以原文为主。原文作者的代码使用matlab工具箱进行编写的
在数据集的剪裁方法选用过于直接，原文的数据选择方法会更好


