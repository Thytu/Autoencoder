from torch import nn
from Encoder import Encoder
from Decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            Encoder(),
            Decoder()
        )

    def forward(self, data):
        return self.model(data)
