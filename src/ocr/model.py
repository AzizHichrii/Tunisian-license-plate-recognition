"""CRNN = CNN (visual features) + BiLSTM (sequence) + Linear (characters)."""
import torch
from torch import nn

from src.ocr.dataset import ALPHABET


def conv_block(c_in: int, c_out: int, pool=None) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class CRNN(nn.Module):
    def __init__(self, num_classes: int = len(ALPHABET) + 1, hidden: int = 128, seq_len: int = 32):
        super().__init__()
        w_pool = (2, 2) if seq_len == 32 else (2, 1)   # (2,1) keeps width -> 64 steps
        self.cnn = nn.Sequential(
            conv_block(1, 64, pool=(2, 2)),      # 32x128 -> 16x64
            conv_block(64, 128, pool=w_pool),    # -> 8x32  or  8x64
            conv_block(128, 256),
            conv_block(256, 256, pool=(2, 1)),   # -> 4xW
            conv_block(256, 256, pool=(4, 1)),   # -> 1xW   (W = seq_len)
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True,
                           dropout=0.2, batch_first=False)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                    # (B, 256, 1, 32)
        f = f.squeeze(2).permute(2, 0, 1)  # (T=32, B, 256)  sequence first for CTC
        f, _ = self.rnn(f)                 # (T, B, 2*hidden)
        return self.fc(f).log_softmax(2)   # (T, B, num_classes)


if __name__ == "__main__":
    model = CRNN()
    out = model(torch.randn(4, 1, 32, 128))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output={tuple(out.shape)}  (T, batch, classes) | params={n_params / 1e6:.2f}M")