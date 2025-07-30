# pretrain_old_model.py
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