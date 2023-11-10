import torch 

__all__ = ['squared_loss', 
           'squared_hinge_loss', 
           'hinge_loss', 
           'logistic_loss', 
           'barrier_hinge_loss', 
           'get_surrogate_loss']


def squared_loss(margin, t):
    r"""
    Squared Loss. The loss can be described as:

    .. math::

        L_\text{squared}(t, m) = (m - t)^2

    where ``m`` is the margin hyper-parameter.
    """
    return (margin - t)** 2

def squared_hinge_loss(margin, t):
    r"""
    Squared Hinge Loss. The loss can be described as:

    .. math::
        L_\text{squared_hinge}(t, m) = \max(m - t, 0)^2

    where ``m`` is the margin hyper-parameter.
    """
    return torch.max(margin - t, torch.zeros_like(t)) ** 2

def hinge_loss(margin, t):
    r"""
    Hinge Loss. The loss can be described as:

    .. math::

        L_\text{hinge}(t, m) = \max(m - t, 0)

    where ``m`` is the margin hyper-parameter.
    """
    return torch.max(margin - t, torch.zeros_like(t))

def logistic_loss(scale, t):
    r"""
    Logistic Loss. The loss can be described as: 

    .. math::
        L_\text{logistic}(t, s) = \log(1 + e^{-st})

    where ``s`` is the scaling hyper-parameter.
    """
    return torch.log(1+torch.exp(-scale*t))

def barrier_hinge_loss(hparam, t):
    r"""
    Barrier Hinge Loss. The loss can be described as: 

    .. math::
        L_\text{barrier_hinge}(t, s, m) = \max(−s(m + t) + m, \max(s(t − m), m − t))
    
    where ``m`` is the margin hyper-parameter and ``s`` is the the scaling hyper-parameter.

    Reference: 
        .. [1] Charoenphakdee, Nontawat, Jongyeong Lee, and Masashi Sugiyama. "On symmetric losses for learning from corrupted labels." International Conference on Machine Learning. PMLR, 2019.
    """
    m,s = hparam
    loss = torch.maximum(-s * (m + t) + m, torch.maximum(m - t, s* (t - rm)))
    return loss

def get_surrogate_loss(loss_name='squared_hinge'):
    r"""
        A wrapper to call a specific surrogate loss function.
    
        Args:
            loss_name (str): type of surrogate loss function to fetch, including 'squared_hinge', 'squared', 'logistic', 'barrier_hinge' (default: ``'squared_hinge'``).
    """
    assert f'{loss_name}_loss' in __all__, f'{loss_name} is not implemented'
    if loss_name == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif loss_name == 'squared':
       surr_loss = squared_loss
    elif loss_name == 'logistic':
       surr_loss = logistic_loss
    elif loss_name == 'barrier_hinge':
       surr_loss = barrier_hinge_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss



