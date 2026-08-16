import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import STGCN  
import utils
from utils import build_adjacency  
import matplotlib.pyplot as plt

# === 超參數設定 ===
NUM_CLASSES = 3  # grab, assemble, release
WINDOW_SIZE = utils.WINDOW_SIZE
JOINT_NUM =utils.JOINTS
CHANNEL = utils.CHANNELS
BATCH_SIZE = 4
EPOCHS = 100
VAL_RATIO = 0.2
LEARNING_RATE = 0.001
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SAVE_DIR = os.path.join(os.path.dirname(DATASET_DIR), 'results')  # 上一層的 results
os.makedirs(SAVE_DIR, exist_ok=True)


label_map = {
    'grab': 0,
    'assemble': 1,
    'release': 2
}

# === 自訂資料集 ===
class SkeletonDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = [] #儲存所有檔案的 檔案路徑與對應標籤（label）
        for filename in os.listdir(data_dir):
            if filename.endswith('.npy'):
                for label_str in label_map:
                    if filename.startswith(label_str):
                        label = label_map[label_str]
                        self.samples.append((os.path.join(data_dir, filename), label))
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)  # 讀取 .npy 檔案，內容是形狀為 [3, T, V] 的骨架資料
        return torch.tensor(data), torch.tensor(label)


# === 訓練主程式 ===
def train():
    dataset = SkeletonDataset(DATASET_DIR)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    A_np = build_adjacency()
    A = torch.tensor(A_np, dtype=torch.float32)

    model = STGCN(in_channels=CHANNEL, num_class=NUM_CLASSES, A=A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_state = None
    train_acc_list, val_acc_list = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        # training
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()

        train_acc = correct / len(train_set)
        train_acc_list.append(train_acc)

        # validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = output.argmax(dim=1)
                val_correct += (pred == y).sum().item()

        val_acc = val_correct / len(val_set)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | Loss: {total_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # === 儲存最佳模型 ===
    torch.save(best_model_state, os.path.join(SAVE_DIR, 'stgcn_best.pth'))
    print(f"\n最佳模型已儲存（Validation Acc: {best_val_acc:.2%}）為 stgcn_best.pth")

    # === 繪製 acc 曲線圖 ===
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, train_acc_list, label='Train Acc')
    plt.plot(epochs_range, val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_plot.png"))

    
if __name__ == '__main__':
    train()
