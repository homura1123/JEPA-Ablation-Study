import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

#造数据,将图片变成 4 帧的“移动视频”
class MovingMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset
        
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx] # img 形状: [1, 28, 28]
        # 制造一个 4 帧的视频，每帧数字向右平移
        # 形状变成 [1, 4, 28, 28] (Channel, Time, H, W)
        frames = []
        for t in range(4):
            shifted_img = torch.roll(img, shifts=t * 3, dims=2) 
            frames.append(shifted_img)
        video = torch.stack(frames, dim=1) 
        return video, label


#Encoder变成3D版本
class VideoEncoder(nn.Module):
    """把 [Batch, 1, 4, 28, 28] 的视频压缩成 128 维向量"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # 3D 卷积，同时看时间(4)和空间(28x28)
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.BatchNorm3d(32),nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.BatchNorm3d(64),nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.BatchNorm3d(128),nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, x):
        return self.net(x)


def vicreg_loss(z, std_coeff=1.0, cov_coeff=0.04):
    batch_size, dim = z.shape
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std))
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)
    cov_loss = (cov.fill_diagonal_(0) ** 2).sum() / dim
    return std_coeff * std_loss + cov_coeff * cov_loss


context_encoder = VideoEncoder().to(device)
target_encoder = copy.deepcopy(context_encoder).to(device)
predictor = Predictor().to(device)

for p in target_encoder.parameters():
    p.requires_grad = False

optimizer = optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-4)
ema_decay = 0.995

transform = transforms.Compose([transforms.ToTensor()])
base_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
video_dataset = MovingMNIST(base_dataset)
dataloader = DataLoader(video_dataset, batch_size=64, shuffle=True)

#训练循环
epochs = 10
for epoch in range(epochs):
    total_loss, total_pred_loss, total_reg_loss = 0, 0, 0
    
    for video, _ in dataloader:
        video = video.to(device) # 形状: [Batch, 1, 4, 28, 28]
        

        view2 = video.clone() #
        view1 = video.clone() #
        view1[:, :, 2:, :, :] = 0.0 # 让模型看不到未来的帧
        
        z1 = context_encoder(view1)
        
        with torch.no_grad():
            z2 = target_encoder(view2)
            
        pred_z2 = predictor(z1)
        
        pred_loss = F.mse_loss(pred_z2, z2)
        reg_loss = vicreg_loss(z1) + vicreg_loss(pred_z2)
        loss = pred_loss + 0.01 * reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
                param_k.data = param_k.data * ema_decay + param_q.data * (1. - ema_decay)
                
        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_reg_loss += reg_loss.item()
        
    print(f"Epoch {epoch+1} | Total: {total_loss/len(dataloader):.4f} | Pred: {total_pred_loss/len(dataloader):.4f} | Reg: {total_reg_loss/len(dataloader):.4f}")

#线性探针验证
print("\n---线性探针测试---")
context_encoder.eval()
for p in context_encoder.parameters():
    p.requires_grad = False

classifier = nn.Linear(128, 10).to(device)
optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

test_base = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_video_dataset = MovingMNIST(test_base)
test_loader = DataLoader(test_video_dataset, batch_size=256, shuffle=True)

correct = 0
total = 0
classifier.train()
for test_video, y in test_loader:
    test_video, y = test_video.to(device), y.to(device)


    view1_test = test_video.clone()
    view1_test[:, :, 2:, :, :] = 0.0

    
    with torch.no_grad():
        z = context_encoder(test_video) 
    
    logits = classifier(z)
    loss = criterion(logits, y)
    optimizer_cls.zero_grad()
    loss.backward()
    optimizer_cls.step()
    
    total += y.size(0)
    correct += (logits.argmax(1) == y).sum().item()

acc = 100 * correct / total
print(f"线性探针测试分类准确率: {acc:.2f}%")