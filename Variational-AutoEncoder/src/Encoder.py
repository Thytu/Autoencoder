from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.__input_to_hidden = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.__hidden_to_sigma = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
        )

        self.__hidden_to_mu = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, data):
        h = self.__input_to_hidden(data)

        return self.__hidden_to_sigma(h), self.__hidden_to_mu(h)


if __name__ == "__main__":
    import torch

    encoder = Encoder(input_dim=28*28, hidden_dim=200, latent_dim=20)

    x = torch.randn((1, 28 * 28))
    sigma, mu = encoder(x)

    print(f"{sigma.shape=} {mu.shape=}")
