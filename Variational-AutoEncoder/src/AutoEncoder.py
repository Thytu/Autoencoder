from torch import nn, rand_like
from .Encoder import Encoder
from .Decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.__encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.__decoder = Decoder(input_dim, hidden_dim, latent_dim)

    @staticmethod
    def create_latent_space(sigma, mu):
        epsilon = rand_like(sigma)

        z = mu + sigma * epsilon

        return z

    def forward(self, data):
        sigma, mu = self.__encoder(data)

        z = self.create_latent_space(sigma=sigma, mu=mu)

        return self.__decoder(z), sigma, mu

    def encode(self, data):
        return self.__encoder(data)

    def decode(self, z):
        return self.__decoder(z)
