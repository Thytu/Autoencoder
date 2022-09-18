from torch import nn


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.__latent_to_data = nn.Sequential(
           nn.Linear(latent_dim, hidden_dim),
           nn.ReLU(inplace=True),

           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(inplace=True),

           nn.Linear(hidden_dim, input_dim),
           nn.Sigmoid(),
        )

    def forward(self, z):
        return self.__latent_to_data(z)


if __name__ == "__main__":
    import torch

    decoder = Decoder(input_dim=28*28, hidden_dim=200, latent_dim=20)

    z = torch.randn((1, 20))

    print(f"{decoder(z).shape=}")
