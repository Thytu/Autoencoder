from torch import nn


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=5),

            nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=5),
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=5),

            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=5),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=5),

            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, data):
        data = data.reshape(-1, 12, 4, 4)

        return self.layers(data)


if __name__ == "__main__":
    import torch

    decoder = Decoder()

    data = torch.randn((32, 12 * 4 * 4))

    print(decoder(data).shape)
