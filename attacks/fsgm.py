from torch import nn
from attacks.base import AbstractAttack


# FSGM attack(white attack)
class GradientSignAttack(AbstractAttack):
    def __init__(
            self,
            model,
            loss_fn=None,
            eps=0.3,
            clip_min=0.,
            clip_max=1.,
            targeted=False
    ):
        super(GradientSignAttack, self).__init__(model=model, loss_fn=loss_fn, clip_min=clip_min, clip_max=clip_max)
        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        :param x: image tensor.
        :param y: label tensor.
                  - if y=None and self.targeted=False, compute y as predicted labels.
                  - if y is not None and self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x_adv = x.requires_grad_()
        # -----------------------------
        # pred = self.model(x_adv, y)["logit_x"]
        pred = self.model(x_adv)
        loss = self.loss_fn(pred, y)
        if self.targeted:
            loss = -loss
        # -----------------------------
        loss.backward()
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + self.eps * grad_sign
        x_adv = x_adv.clamp(min=self.clip_min, max=self.clip_max)
        return x_adv.detach()
