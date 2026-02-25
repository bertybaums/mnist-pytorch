import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

# ── Config ──────────────────────────────────────────────────────────────────
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 1e-3
VAL_FRAC    = 0.1       # fraction of training set used for validation
DATA_DIR    = "./data"
CKPT_PATH   = "./best_model.pt"

# ── Model ────────────────────────────────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                    # 14×14
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                    # 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Data ─────────────────────────────────────────────────────────────────────
def get_loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test_ds    = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

    n_val   = int(len(full_train) * VAL_FRAC)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = get_loaders()

    model     = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  {vl_loss:>8.4f}  {vl_acc:>7.2%}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_PATH)

    # ── Test ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    _, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"\nTest accuracy: {test_acc:.2%}  (best val acc: {best_val_acc:.2%})")
    print(f"Model saved to {CKPT_PATH}")


if __name__ == "__main__":
    main()
