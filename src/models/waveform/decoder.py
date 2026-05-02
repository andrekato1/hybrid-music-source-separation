"""
Waveform decoder for music source separation.

Model inspired by
@article{defossez2019music,
  title={Music Source Separation in the Waveform Domain},
  author={D{\'e}fossez, Alexandre and Usunier, Nicolas and Bottou, L{\'e}on and Bach, Francis},
  journal={arXiv preprint arXiv:1911.13254},
  year={2019}
}
"""
from torch import nn

## center_trim utility function directly leveraged from demucs (https://github.com/facebookresearch/demucs/blob/v2/demucs/utils.py)
def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

class DemucsDecoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, context=3):
        super().__init__()
        C = base_channels

        # going the opposite direction to encoder
        self.layer4 = nn.Sequential(
            nn.Conv1d(8*C, 16*C, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(8*C, 4*C, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(4*C, 8*C, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(4*C, 2*C, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(2*C, 4*C, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(2*C, C, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv1d(C, 2*C, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(C, in_channels, kernel_size=8, stride=4),
        )

    def forward(self, x, skips):
        # skips coming from encoder
        skip1, skip2, skip3, skip4 = skips
        x = x + center_trim(skip4, x)
        x = self.layer4(x)

        x = x + center_trim(skip3, x)
        x = self.layer3(x)

        x = x + center_trim(skip2, x)
        x = self.layer2(x)

        x= x +center_trim(skip1, x)
        x = self.layer1(x)

        return x

