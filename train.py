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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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

writer = SummaryWriter('runs/mobilenetv3_imagenet_dataparallel_batchsize_20_train_amp_epoch_384')

def test_net(net, test_loader, epoch = 0):
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
    writer.add_scalar('Acc/val', test_accuracy, epoch)

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

from torch.amp import autocast, GradScaler
def train_with_ImageNet(net, num_classes=1000, batch_size=200, epoch_max=20):
    #num_classes = 1000 #ImageNet数据集包含1000中类型的图片
    #==================加载ImageNet数据集==============================
    # 使用ImageNet特定的数据增强
    print(f"start load data for train. path={args.data_dir}")
    train_transform = get_imagenet_transforms(train=True)
    test_transform = get_imagenet_transforms(train=False)
    train_data = ImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    print(f"load data for train done. path={args.data_dir}")
    print(f"start load data for val. path={args.data_dir}")
    train_transform = get_imagenet_transforms(train=True)
    test_data = ImageFolder(
        root=os.path.join(args.data_dir, 'val'),
        transform=test_transform
    )
    print(f"load data for val done. path={args.data_dir}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12)
    # ---------------for ImageNet------------------
    # 使用更适合ImageNet的优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = GradScaler() #使用梯度缩放器，用于混合精度訓練
    cross = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_max):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epoch_max}')
        for batch_idx, (img, label) in enumerate(pbar):
        #for batch_idx,(img, label) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = net.forward(img)
                loss = cross(output, label)
            #loss.backward()
            scaler.scale(loss).backward() # 先对loss进行scale，再backward()
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            #print(f'Epoch: {epoch} batch_idx: {batch_idx} running_loss: {running_loss} running_acc: {100. * correct / total}')
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        scheduler.step()
        os.makedirs('temp', exist_ok=True)
        torch.save(net.state_dict(), f'temp/mobilenetv3_epoch{epoch}.pth')
        print(f"Model saved to temp/mobilenetv3_epoch{epoch}.pth")

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    # Test the model
    test_net(net, test_loader=test_loader)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
if __name__ == '__main__':
    print(torch.cuda.is_available())
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MobileNetV3 and optionally export to ONNX')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--pth-file', type=str, required=False, help='Path to pretrained checkpoint file')
    parser.add_argument('--export-onnx-path', type=str, help='Path to save ONNX model')
    args = parser.parse_args()

    ##==================创建net==============================
    ##net = MobileNetV3Large(num_classes=10)
    net = MobileNetV3(num_classes=1000)

    device_ids = list(range(torch.cuda.device_count()))
    print(f'可用GPU：{device_ids}')
    #for i in device_ids:
    #    prop = torch.cuda.get_device_properties(i)
    #    print(f'GPU {i} : {prop.name}, 显存： {prop.total_memory / 1024**3:.2f}')
    #net = nn.DataParallel(net, device_ids=device_ids)
    #device = torch.device(f'cuda:{device_ids[0]}')
    #net.to(device)

    device = torch.device(f'cuda')
    net = net.to(device)
    loaded,net,startEpoch = load_weight(net, args.pth_file, device)
    if loaded:
        print(f"load model weight success. file={args.pth_file}")
    print(net)
    #train_with_CIFAR10(net)
    train_with_ImageNet(net, batch_size=200)
    
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
