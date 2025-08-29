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

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, in_channel // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(in_channel, reduced_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.se_expend = nn.Conv2d(reduced_channels, out_channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.se_reduce(out)
        out = self.relu(out)
        out = self.se_expend(out)
        out = x * self.sigmoid(out)
        #out = self.sigmoid(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, kernel_size//2)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, kernel_size//2)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = out * self.se(out)
        out += self.shortcut(x)
        out = nn.functional.relu(out, inplace=True)
        return out

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels,se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size,stride=1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            hswish(inplace=True),
        )
 
    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x
    
class SEInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride,activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        # mid_channels = (in_channels * expansion_factor)
 
        self.conv = Conv1x1BNActivation(in_channels, mid_channels,activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size,stride,activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)
 
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels,activate)
 
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)
 
    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out
    

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000,type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type
 
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            hswish(inplace=True),
        )
 
        if type=='large':
            self.large_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2,activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1,activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1,activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2,activate='hswish', use_se=True,se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1,activate='hswish', use_se=True,se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1,activate='hswish', use_se=True,se_kernel_size=7),
            )
 
            self.large_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1),
                nn.BatchNorm2d(960),
                hswish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
                hswish(inplace=True),
            )
        else:
            self.small_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2,activate='relu', use_se=True, se_kernel_size=56),
                SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
            )
            self.small_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1),
                nn.BatchNorm2d(576),
                hswish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1),
                hswish(inplace=True),
            )
 
        self.classifier = nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1, stride=1)
 
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        x = self.first_conv(x)
        if self.type == 'large':
            x = self.large_bottleneck(x)
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)
        out = self.classifier(x)
        out = out.view(out.size(0), -1)
        return out

class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__()#

        self.conv1 = ConvBlock(3, 16, 3, 2, 1)     # 1/2
        self.bottlenecks = nn.Sequential(
            ResidualBlock(16, 16, 3, 1, False),
            ResidualBlock(16, 24, 3, 2, False),     # 1/4
            ResidualBlock(24, 24, 3, 1, False),
            ResidualBlock(24, 40, 5, 2, True),      # 1/8
            ResidualBlock(40, 40, 5, 1, True),
            ResidualBlock(40, 40, 5, 1, True),
            ResidualBlock(40, 80, 3, 2, False),     # 1/16
            ResidualBlock(80, 80, 3, 1, False),
            ResidualBlock(80, 80, 3, 1, False),
            ResidualBlock(80, 112, 5, 1, True),
            ResidualBlock(112, 112, 5, 1, True),
            ResidualBlock(112, 160, 5, 2, True),    # 1/32
            ResidualBlock(160, 160, 5, 1, True),
            ResidualBlock(160, 160, 5, 1, True)
        )
        self.conv2 = ConvBlock(160, 960, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(960, 1280),
            nn.BatchNorm1d(1280),
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bottlenecks(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Small, self).__init__()

        self.conv1 = ConvBlock(3, 16, 3, 2, 1)     # 1/2
        self.bottlenecks = nn.Sequential(
            ResidualBlock(16, 16, 3, 2, False),     # 1/4
            ResidualBlock(16, 72, 3, 2, False),     # 1/8
            ResidualBlock(72, 72, 3, 1, False),
            ResidualBlock(72, 72, 3, 1, True),
            ResidualBlock(72, 96, 3, 2, True),      # 1/16
            ResidualBlock(96, 96, 3, 1, True),
            ResidualBlock(96, 96, 3, 1, True),
            ResidualBlock(96, 240, 5, 2, True),     # 1/32
            ResidualBlock(240, 240, 5, 1, True),
            ResidualBlock(240, 240, 5, 1, True),
            ResidualBlock(240, 480, 5, 1, True),
            ResidualBlock(480, 480, 5, 1, True),
            ResidualBlock(480, 480, 5, 1, True),
        )
        self.conv2 = ConvBlock(480, 576, 1, 1, 0, groups=2)
        self.conv3 = nn.Conv2d(576, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1024)
        self.act = hswish()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bottlenecks(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

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

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MobileNetV3 and optionally export to ONNX')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--pth-file', type=str, required=False, help='Path to pretrained checkpoint file')
    parser.add_argument('--export-onnx-path', type=str, help='Path to save ONNX model')
    args = parser.parse_args()

    device = torch.device('cuda')
    batch_size = 148  # CIFAR10数据集
    #batch_size = 256    # ImageNet数据集
    epoch_max = 100
    num_classes = 10 #CIFAR10数据集包含10中类型的图片
    #num_classes = 1000 #ImageNet数据集包含1000中类型的图片
    input_picture_shape = (3, 32, 32) # CIFAR10数据集的图片是32x32的rgb
    #input_picture_shape = (3, 224, 224) # ImageNet数据集标准训练输入尺寸
    transform = get_transforms()

    #==================加载ImageNet数据集==============================
    # 使用ImageNet特定的数据增强
    #train_transform = get_imagenet_transforms(train=True)
    #test_transform = get_imagenet_transforms(train=False)
    #train_data = ImageFolder(
    #    root=os.path.join(args.data_dir, 'train'),
    #    transform=train_transform
    #)
    #test_data = ImageFolder(
    #    root=os.path.join(args.data_dir, 'val'),
    #    transform=test_transform
    #)
    
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    #==================创建net==============================
    net = MobileNetV3Large(num_classes=num_classes).to(device)
    loaded,net,startEpoch = load_weight(net, args.pth_file, device)
    if loaded:
        print(f"load model weight success. file={args.pth_file}")
    print(net)
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
    
    # ---------------for ImageNet------------------
    # 使用更适合ImageNet的优化器设置
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #for epoch in range(epoch_max):
    #    net.train()
    #    running_loss = 0.0
    #    correct = 0
    #    total = 0
    #    
    #    for img, label in train_loader:
    #        img = img.to(device)
    #        label = label.to(device)
    #        
    #        output = net.forward(img)
    #        loss = cross(output, label)
    #        loss.backward()
    #        optimizer.step()
    #        optimizer.zero_grad()
    #        
    #        running_loss += loss.item()
    #        _, predicted = output.max(1)
    #        total += label.size(0)
    #        correct += predicted.eq(label).sum().item()
    #    
    #    train_acc = 100. * correct / total
    #    avg_loss = running_loss / len(train_loader)
    #    
    #    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
    #    scheduler.step()
    #========================测试=============================
    test_data = CIFAR10('cifar', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        # Test the model
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
    #======================保存pth==========================
    # Create output directory and save model
    os.makedirs('output/models', exist_ok=True)
    torch.save(net.state_dict(), 'output/models/mobilenetv3.pth')
    print("Model saved to output/models/mobilenetv3.pth")

    #=====================导出onnx==========================
    # Export to ONNX if requested
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