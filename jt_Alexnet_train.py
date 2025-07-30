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
    
    # 保存结果
    jt.save(model.state_dict(), 'lwf_model.pkl')
    plot_metrics(train_losses, old_accuracies, new_accuracies, 
                test_old_accuracies, test_new_accuracies)

if __name__ == '__main__':
    print("===== 开始训练 =====")
    train()