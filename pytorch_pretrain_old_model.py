# pretrain_old_model.py
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