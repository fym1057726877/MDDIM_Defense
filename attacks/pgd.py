import torch
import numpy as np
from torch import nn
from attacks.base import AbstractAttack


# pgd attack(white attack), k-fsgm, k: iter times
class PGDAttack(AbstractAttack):
    def __init__(
            self,
            model,
            loss_fn=None,
            eps=0.3,
            iter_eps=0.01,
            clip_min=0.,
            clip_max=1.,
            targeted=False,
            rand_init=True,
            num_iter=10,
    ):
        super(PGDAttack, self).__init__(model=model, loss_fn=loss_fn, clip_min=clip_min, clip_max=clip_max)
        self.eps = eps
        self.device = next(model.parameters()).device
        self.iter_eps = iter_eps
        self.targeted = targeted
        self.rand_init = rand_init
        self.num_iter = num_iter
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        :param x: image tensor.
        :param y: label tensor.
                  - if y=None and self.targeted=False, compute y as predicted labels.
                  - if y is not None and self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = x.to(self.device), y.to(self.device)
        if self.rand_init:
            x_adv = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).to(self.device)
            x_adv.clamp(min=self.clip_min, max=self.clip_max)
        else:
            x_adv = x.clone().to(self.device)

        for i in range(self.num_iter):
            x_adv = self.one_step_attack(x, x_adv, y)

        return x_adv

    def one_step_attack(self, x_ori, x_adv, label):
        x_adv = x_adv.requires_grad_(True)
        pred = self.model(x_adv)
        loss = self.loss_fn(pred, label)
        if self.targeted:
            loss = -loss
        self.model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.sign()
        # get the pertubation of an iter_eps
        x_adv = x_adv + self.iter_eps * grad_sign

        x_adv = x_adv.clamp(min=x_ori - self.eps, max=x_ori + self.eps)
        x_adv = x_adv.clamp(min=self.clip_min, max=self.clip_max)

        return x_adv.detach()
