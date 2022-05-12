import torch
from torch import nn

def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss
class GaussianSmearing(nn.Module):
    r"""Smear layer using a set of Gaussian functions.
    使用一组扩散函数的高斯层

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        第一高斯函数中心
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        最后一个高斯函数的中心
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        高斯函数的总数
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        如果为真，则高斯函数以原点为中心 偏移量被用来作为它们的宽度(例如，用于角函数)。
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.
        如果为真，表示高斯函数的宽度和偏移量在训练过程中进行调整。

    """

    def __init__(
        self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        #切成n段
        offset = torch.linspace(start, stop, n_gaussians)
        #间距数组
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )
