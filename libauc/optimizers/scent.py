import torch
from torch.optim import SGD


class SCENT(SGD):
    r"""
    Optimizer for Compositional Entropic Risk
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if weight_decay != 0 and d_p.abs().sum() > 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        has_grad = (d_p.abs().sum(dim=1) > 0) if d_p.dim() > 1 else (d_p.abs() > 0)
                        buf[has_grad] = buf[has_grad] * momentum + d_p[has_grad] * (1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
