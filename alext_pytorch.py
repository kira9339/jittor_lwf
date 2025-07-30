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
