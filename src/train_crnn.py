import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    def __init__(self, rows, chars, width=160, height=48):
        self.rows, self.char_to_id = rows, {c: i + 1 for i, c in enumerate(chars)}
        self.transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row["image"]).convert("L")
        image = self.transform(image)
        target = torch.tensor([self.char_to_id[c] for c in row["text"] if c in self.char_to_id], dtype=torch.long)
        return image, target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), pad_sequence(targets, batch_first=True, padding_value=0)


class TeluguCRNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        self.rnn = nn.LSTM(256 * 12, 256, 2, bidirectional=True, batch_first=True, dropout=.2)
        self.fc = nn.Linear(512, classes + 1)

    def forward(self, x):
        x = self.cnn(x).permute(0, 3, 1, 2).flatten(2)
        x, _ = self.rnn(x)
        return self.fc(x).log_softmax(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to a CSV manifest with image and text columns")
    ap.add_argument("--output", default="models/telugu_crnn.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    with open(args.manifest, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["text"]]
    random.Random(42).shuffle(rows)
    if args.limit: rows = rows[:args.limit]
    chars = sorted(set("".join(r["text"] for r in rows)))
    split = max(1, int(len(rows) * .1))
    train, valid = rows[:-split], rows[-split:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(OCRDataset(train, chars), args.batch_size, shuffle=True, collate_fn=collate)
    model = TeluguCRNN(len(chars)).to(device)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
            target_lengths = torch.tensor([int((t != 0).sum()) for t in targets], dtype=torch.long)
            flat_targets = torch.cat([t[:n] for t, n in zip(targets, target_lengths)])
            loss = loss_fn(logits.transpose(0, 1), flat_targets, input_lengths, target_lengths)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(train_loader)):.4f}")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "chars": chars, "width": 160, "height": 48}, out)
    print(f"Saved custom Telugu model: {out}; validation samples: {len(valid)}; device: {device}")


if __name__ == "__main__": main()
