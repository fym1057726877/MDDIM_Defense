from models.ddim.unet import UNetModel
from memoryunit import MemoryUnit
import torch as th


class MemoryUnet(UNetModel):
    def __init__(self, MEM_DIM=200, addressing="sparse", **kwargs):
        super().__init__(**kwargs)

        self.features = self.model_channels * 2 * 7 * 7
        self.middle_block.add_module("memory", MemoryUnit(MEM_DIM=MEM_DIM, features=self.features, addressing=addressing))

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)


if __name__ == '__main__':
    model = MemoryUnet()
    x = th.randn((16, 1, 28, 28))
    t = th.randint(0, 10, (16, ))
    out = model(x, t)
    print(out.shape)


