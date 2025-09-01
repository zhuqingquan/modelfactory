import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.autograd import Variable


import torch.nn as nn
import torch.nn.functional as F
import torch

from model_utils import *
from MobileNetV3 import MobileNetV3, MobileNetV3Large

# 改进的数据增强
def get_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


# 修改数据加载部分
def get_imagenet_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def load_weight(net, pth_file, map_location = None):
    # 加载预训练权重（如果提供了checkpoint文件）
    start_epoch = 0
    loaded = False
    if pth_file and os.path.exists(pth_file):
        print(f"Loading pretrained weights from {pth_file}")
        checkpoint = torch.load(pth_file, map_location=map_location)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整checkpoint格式（包含优化器状态等）
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            # 只有模型权重的格式
            net.load_state_dict(checkpoint)
            print("Loaded model weights only")
        loaded = True
    return loaded,net,start_epoch

def test_net(net, test_loader):
    net.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            output = net.forward(img)
            pre = torch.argmax(output, 1)
            test_correct += (pre == label).sum().item()
            test_total += label.size(0)
    
    test_accuracy = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_correct}/{test_total})")

def train_with_CIFAR10(net, num_classes=10):
    batch_size = 148  # CIFAR10数据集
    epoch_max = 100
    #num_classes = 10 #CIFAR10数据集包含10中类型的图片
    transform = get_transforms()
    #==================训练==================================
    train_data = CIFAR10('cifar', download=True, train=True, transform=transform)
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    cross = nn.CrossEntropyLoss().to(device)
    # --------------- for CIFAR10 ----------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(epoch_max):
        for img, label in data:
            img = Variable(img).to(device)
            label = Variable(label).to(device)
            output = net.forward(img)
            loss = cross(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre = torch.argmax(output, 1)
            #print(f'pre={pre} len={len(pre)}')
            num = (pre == label).sum().item()
            acc = num / img.shape[0]
        #scheduler.step()
        print(f"epoch: {epoch + 1} loss: {loss.item()} Accuracy: {acc}")
    #========================测试=============================
    test_data = CIFAR10('cifar', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # Test the model
    test_net(net, test_loader=test_loader)

def train_with_ImageNet(net, num_classes=1000):
    batch_size = 256    # ImageNet数据集
    epoch_max = 100
    #num_classes = 1000 #ImageNet数据集包含1000中类型的图片
    #==================加载ImageNet数据集==============================
    # 使用ImageNet特定的数据增强
    train_transform = get_imagenet_transforms(train=True)
    test_transform = get_imagenet_transforms(train=False)
    train_data = ImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    test_data = ImageFolder(
        root=os.path.join(args.data_dir, 'val'),
        transform=test_transform
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # ---------------for ImageNet------------------
    # 使用更适合ImageNet的优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    cross = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_max):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            output = net.forward(img)
            loss = cross(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        scheduler.step()
    # Test the model
    test_net(net, test_loader=test_loader)

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MobileNetV3 and optionally export to ONNX')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--pth-file', type=str, required=False, help='Path to pretrained checkpoint file')
    parser.add_argument('--export-onnx-path', type=str, help='Path to save ONNX model')
    args = parser.parse_args()

    device = torch.device('cuda')

    #==================创建net==============================
    #net = MobileNetV3Large(num_classes=10).to(device)
    net = MobileNetV3(num_classes=1000).to(device)
    loaded,net,startEpoch = load_weight(net, args.pth_file, device)
    if loaded:
        print(f"load model weight success. file={args.pth_file}")
    print(net)
    #train_with_CIFAR10(net)
    train_with_ImageNet(net)
    
    #======================保存pth==========================
    # Create output directory and save model
    os.makedirs('output/models', exist_ok=True)
    torch.save(net.state_dict(), 'output/models/mobilenetv3.pth')
    print("Model saved to output/models/mobilenetv3.pth")

    #=====================导出onnx==========================
    # Export to ONNX if requested
    #input_picture_shape = (3, 224, 224) # ImageNet数据集标准训练输入尺寸
    input_picture_shape = (3, 32, 32) # CIFAR10数据集的图片是32x32的rgb
    if args.export_onnx_path:
        os.makedirs(os.path.dirname(args.export_onnx_path), exist_ok=True)
        dummy_input = torch.randn(1, *input_picture_shape).to(device)  # CIFAR10 input size
        torch.onnx.export(
            net,
            dummy_input,
            f'{args.export_onnx_path}/mobilenetv3.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to {args.export_onnx_path}/mobilenetv3.onnx")