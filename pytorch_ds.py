
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