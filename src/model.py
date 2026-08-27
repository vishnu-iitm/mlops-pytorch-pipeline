import torch.nn as nn


class TinyCNN(nn.Module):
    # small 3-block cnn for cifar-10
    def __init__(self, num_classes):
        super().__init__()
        # conv blocks: 3 -> 16 -> 32 -> 64 channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


def get_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == 'tinycnn':
        return TinyCNN(num_classes)
    raise ValueError(f"architecture {architecture} not supported yet")