import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
import csv
import os
import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


LR = 1e-4         # 学习率 (1e-4,5e-4,1e-3)
ARCH_NAME = "deep" # 架构 (tiny,base,deep),在下面加注释来修改
REG_W = 0.3        # 正则权重 (0.0,0.1,0.3)


#架构定义
class Encoder(nn.Module):
    def __init__(self, arch_type,latent_dim=128):
        super().__init__()
        if arch_type == "tiny":
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(64, latent_dim) # tiny 输出 64 维
            )
        elif arch_type == "base":
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(128, latent_dim) # base 输出 128 维
            )
        else: # deep
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(256, latent_dim)
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


#VICReg 正则化（防止表征坍塌）
def vicreg_loss(z, std_coeff=1.0, cov_coeff=0.04):
    """
    Variance: 每个维度的标准差必须大于阈值（防止把所有东西映射成同一个向量）
    Covariance: 不同维度之间必须不相关（强迫维度独立）
    """
    batch_size, dim = z.shape
    #方差惩罚
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std)) # 小于1就惩罚
    
    #协方差惩罚
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)
    #把对角线（也就是方差本身）去掉，只惩罚非对角线元素
    cov_loss = (cov.fill_diagonal_(0) ** 2).sum() / dim
    
    return std_coeff * std_loss + cov_coeff * cov_loss



transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


def run_experiment(LR, ARCH_NAME, REG_W, epochs=10): 
    torch.manual_seed(42)
    context_encoder = Encoder(ARCH_NAME).to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    predictor = Predictor().to(device) 
    for p in target_encoder.parameters(): p.requires_grad = False

    optimizer = optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()), lr=LR)
    ema_decay = 0.998

    for epoch in range(epochs):
        for x, _ in dataloader:
            x = x.to(device)
            view2 = x.clone()
            view1 = x + 0.1*torch.randn_like(x)
            view1[:, :, 10:18, 10:18] = 0.0 
            
            z1 = context_encoder(view1)
            with torch.no_grad(): z2 = target_encoder(view2)
            pred_z2 = predictor(z1)
            
            loss = F.mse_loss(pred_z2, z2) + REG_W * (vicreg_loss(z1) + vicreg_loss(pred_z2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
                    param_k.data = param_k.data * ema_decay + param_q.data * (1. - ema_decay)

            #线性探针验证 
    print("\n--- 线性探针测试 ---")
    context_encoder.eval()
    for p in context_encoder.parameters():
        p.requires_grad = False

    #一层线性层
    classifier = nn.Linear(128, 10).to(device)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

#分类
    correct = 0
    total = 0
    classifier.train()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
    
        with torch.no_grad():
            z = context_encoder(x)
    
        #训练分类头
        logits = classifier(z)
        loss = criterion(logits, y)
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()
    
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    acc = 100 * correct / total
    print(f"线性探针测试分类准确率: {acc:.2f}% ")
    return acc



if __name__ == "__main__":
    SAVE_DIR = r"C:\Python学习\JEPA"
    os.makedirs(SAVE_DIR, exist_ok=True)
    CSV_FILE = os.path.join(SAVE_DIR, "JEPA_Ablation_Results.csv") 
    if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["LR", "Architecture", "Reg_Weight", "Final_Accuracy(%)"])

    lrs = [1e-4, 5e-4, 1e-3]
    archs = ["tiny", "base", "deep"]
    reg_ws = [0.0, 0.1, 0.3]
    experiments = list(itertools.product(lrs, archs, reg_ws))

    print(f"开始自动跑27组实验")
    
    for i, (LR, ARCH_NAME, REG_W) in enumerate(experiments):
        print(f"[{i+1}/27] 正在跑: LR={LR}, ARCH={ARCH_NAME}, REG_W={REG_W} ...", end="")
        acc = run_experiment(LR, ARCH_NAME, REG_W, epochs=10)
        print(f" 准确率: {acc:.2f}%")
        
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([LR, ARCH_NAME, REG_W, round(acc, 2)])

    print(f"\n27组全部完成,数据保存至: {CSV_FILE}")
