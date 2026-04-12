import logging
import math

import torch
import torch.nn as nn


class EntLossClassification(nn.Module):
    r"""
        A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization (Extreme Classification).
        The objective function is defined as

        .. math::

            F(\mathbf{w}) = \frac{1}{n} \sum_{i= 1}^{n} \log \left[\sum_{k= 1}^{K} \exp
            \left( h(\mathbf{x}_{i})^{\top}(\mathbf{w}_{k}- \mathbf{w}_{y_{i}}) \right)\right],

        where :math:`\mathbf{x}` is a data sample, :math:`h(\cdot)` is a pretrained encoder,
        :math:`\mathbf{w}_{k}` denotes the (linear) classifier for class :math:`k`, and
        :math:`y_{i}` denotes the label of data sample :math:`i`.

        Args:
            data_size (int): number of samples in the training dataset (default: ``100000``)
            alpha (float): the step size for SCENT (in log scale, i.e., the real step size is exp(alpha))
                (default: ``10.0``)
            gamma (float): the moving average factor for SOX, in range the range of (0.0, 1.0) (default: ``0.9``)
            is_scent (bool): whether to use SCENT or SOX (default: ``True``)
            alpha_multiplier (float): A parameter that controls how fast nu is updated in SCENT (default: ``1.0``)

        Reference:
            .. [1] Wei, X., Zhou, L., Wang, B., Lin, C.J. and Yang, T., 2026.
                   A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization.
                   arXiv preprint arXiv:2602.02877.
    """
    def __init__(self,
                 data_size: int,
                 alpha: float = 10.0,
                 gamma: float = 0.9,
                 is_scent: bool = True,
                 alpha_multiplier: float = 1.0,
                 ) -> None:
        super().__init__()
        self.data_size = data_size
        self.alpha = alpha
        self.gamma_orig = gamma
        self.gamma = gamma
        self.is_scent = is_scent
        self.nu = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.alpha_multiplier = alpha_multiplier

    def adjust_gamma(self, epoch: int, max_epoch: int) -> None:
        if not self.is_scent:
            self.gamma = 0.5 * (1.0 - self.gamma_orig) * (1 + torch.cos(torch.tensor(epoch / max_epoch * math.pi))) + self.gamma_orig
            logging.info(f"Adjusted gamma to {self.gamma:.6f} at epoch {epoch}")

    def forward(self,
                logits: torch.Tensor,
                indices: torch.Tensor,
                ) -> dict:
        nu = self.nu[indices].to(logits.device)

        # update nu
        # check which nu are not initialized
        uninit_idx = torch.nonzero(nu == 0.0, as_tuple=True)[0]
        exp_logits_mean = torch.sum(torch.exp(logits), dim=-1, keepdim=True).detach() / (logits.shape[1] - 1)
        if self.is_scent:
            nu = nu + torch.log(1 + math.exp(self.alpha) * exp_logits_mean * torch.exp(nu * (self.alpha_multiplier - 1.0))) \
                 - torch.log(1 + math.exp(self.alpha) * torch.exp(nu * self.alpha_multiplier))
        else:
            b = math.log(1 - self.gamma) + nu
            w = math.log(self.gamma) + torch.log(exp_logits_mean)
            nu = torch.max(b, w) - torch.log(torch.sigmoid(torch.abs(b - w)))
        if uninit_idx.shape[0] > 0:
            nu[uninit_idx] = torch.log(exp_logits_mean[uninit_idx])
        self.nu[indices] = nu.cpu()

        # compute loss
        loss = torch.mean(torch.sum(torch.exp(logits - nu), dim=-1, keepdim=True) / (logits.shape[1] - 1))
        loss_dict = {"loss": loss}
        return loss_dict
