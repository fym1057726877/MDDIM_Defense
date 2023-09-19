import torch
from torch import optim
from models.gaussiandiffusion import create_ddim_and_unet
from utils import draw_ori_and_recon_images32
from dataset.fashion_mnist import get_fashion_mnist_dataloader
from tqdm import tqdm
from time import time

train_loader, test_loader = get_fashion_mnist_dataloader(batch_size=64, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
noises = torch.randn((64, 1, 28, 28), device=device)


def train(epochs):
    start = time()

    timesteps = 1000

    ddim, model = create_ddim_and_unet(device=device)
    model.load_state_dict(torch.load("./pretrained/ddim_fmnist.pth"))
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        count = len(train_loader)
        epoch_loss = 0
        model.train()
        for step, (images, _) in tqdm(enumerate(train_loader), desc=f"train step {epoch + 1}/{epochs}", total=count):
            # for step, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            batch_size = images.shape[0]

            t1 = torch.randint(0, timesteps // 2, (batch_size // 2,), device=device).long()
            t2 = torch.randint(timesteps // 2, timesteps, (batch_size // 2,), device=device).long()
            t = torch.cat([t1, t2], dim=-1)

            # out, mem_weight = memdiff(images, t)
            #
            # loss = mse(out, images) + entropy_loss(mem_weight)

            loss = ddim.training_losses(model, images, t)["loss"]

            epoch_loss += loss
            # if step % 600 == 0:
            #     print(f"Epoch:{epoch + 1}/{epochs}  Batch:{step}/{count}  Loss:{loss:.8f}")

            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), "./pretrained/ddim_fmnist.pth")

        model.eval()
        epoch_loss /= count
        print(f"Epoch:{epoch + 1}/{epochs}  Loss:{epoch_loss:.8f}")

        # show_ddim_results(ddim, model, imgs)

    end = time()
    seconds = int(end - start)
    minutes = seconds // 60
    remain_second = seconds % 60
    print(f"time consumed: {minutes}min{remain_second}s")


def test():
    ddim, model = create_ddim_and_unet(device=device)
    model.load_state_dict(torch.load("./pretrained/ddim_fmnist.pth"))
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    # show_ddim_results(ddim, model, imgs)
    final_sample = ddim.ddim_sample_loop(model, shape=imgs.shape, noise=imgs, progress=True)
    draw_ori_and_recon_images32(imgs, final_sample)


if __name__ == "__main__":
    # train(5)
    test()
