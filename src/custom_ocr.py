"""Inference helpers for the locally trained Telugu CRNN."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class TeluguCRNN(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        self.rnn = nn.LSTM(256 * 12, 256, 2, bidirectional=True,
                           batch_first=True, dropout=.2)
        self.fc = nn.Linear(512, classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).permute(0, 3, 1, 2).flatten(2)
        x, _ = self.rnn(x)
        return self.fc(x).log_softmax(2)


class CustomTeluguOCR:
    def __init__(self, checkpoint: Path) -> None:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.chars = list(data["chars"])
        self.model = TeluguCRNN(len(self.chars))
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((int(data.get("height", 48)), int(data.get("width", 160)))),
            transforms.ToTensor(), transforms.Normalize((.5,), (.5,)),
        ])

    def parse(self, image_path: str) -> dict[str, str]:
        image = Image.open(image_path).convert("L")
        with torch.inference_mode():
            logits = self.model(self.transform(image).unsqueeze(0))[0]
        ids = logits.argmax(1).tolist()
        text: list[str] = []
        previous = 0
        for index in ids:
            if index != 0 and index != previous:
                text.append(self.chars[index - 1])
            previous = index
        return {"markdown": "".join(text), "blocks": []}
