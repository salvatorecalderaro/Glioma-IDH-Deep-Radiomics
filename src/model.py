import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)

class CNNMatrixClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = ResidualBlock(128, 256)

        self.dropout = nn.Dropout(0.4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 8, 107)
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.dropout(x)
        return self.head(x)


# =========================
# TRAIN FUNCTION
# =========================
def train_model(device, model, trainloader, epochs=20, lr=1e-3):

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(xb)

            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(trainloader):.4f}")
    path = "../models/cnn.pth"
    torch.save(model.state_dict(), path)
    return model


# =========================
# PREDICT FUNCTION
# =========================
def predict(device, model, testloader):

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_proba = []

    with torch.no_grad():
        for xb, yb in testloader:
            xb = xb.to(device)

            proba = torch.sigmoid(model(xb))
            preds = (proba > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(yb)
            all_proba.append(proba.cpu())

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    all_proba = torch.cat(all_proba).numpy().flatten()

    return all_preds, all_labels, all_proba

