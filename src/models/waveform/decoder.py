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
    def __init__(self, in_channels=2, context=3):
        super().__init__()

        # going the opposite direction to encoder
        self.layer4 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(512, 256, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=context, stride=1),
            nn.GLU(dim=1),
            nn.ConvTranspose1d(64, in_channels, kernel_size=8, stride=4),
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

