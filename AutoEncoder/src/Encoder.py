from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(in_features=12 * 4 * 4, out_features=12 * 4 * 4),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.layers(data)


if __name__ == "__main__":
    import torch

    encoder = Encoder()

    data = torch.randn((32, 1, 28, 28))

    print(encoder(data).shape)
