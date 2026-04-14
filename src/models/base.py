from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import Tensor
import torch.nn as nn

@dataclass
class Latent:
    bottleneck: Tensor
    skips: list[Tensor]

class BaseSeparator(nn.Module, ABC):
    @abstractmethod
    def forward(self, mixture: Tensor) -> dict[str, Tensor]:
        """
        mixture shape should be [B, C, T]
        """
        raise NotImplementedError
    
class BaseEncDecSeparator(BaseSeparator, ABC):
    @abstractmethod
    def encode(self, mixture: Tensor) -> Latent:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, latent: Latent) -> dict[str, Tensor]:
        raise NotImplementedError