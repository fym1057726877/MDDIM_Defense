import torch
from torch import optim
from models.ddim.ddim import create_ddim_and_unet
from utils import draw_ori_and_recon_images32, draw_fashion_mnist_images16
from dataset.fashion_mnist import get_fashion_mnist_dataloader
from tqdm import tqdm
from time import time

train_loader, test_loader = get_fashion_mnist_dataloader(batch_size=64, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(epochs):
    start = time()

    ddim, model = create_ddim_and_unet(device=device)
    model.load_state_dict(torch.load("./pretrained/ddim_fmnist_eps_new.pth"))
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        count = len(train_loader)
        epoch_loss = 0
        model.train()
        for step, (images, _) in tqdm(enumerate(train_loader), desc=f"train step {epoch + 1}/{epochs}", total=count):
            optimizer.zero_grad()
            images = images.to(device)
            batch_size = images.shape[0]
            t = ddim.get_rand_t(batch_size=batch_size, device=device)
            loss = ddim.training_losses(model, images, t)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), "./pretrained/ddim_fmnist_eps_new.pth")

        model.eval()
        epoch_loss /= count
        print(f"Epoch:{epoch + 1}/{epochs}  Loss:{epoch_loss:.8f}")

    end = time()
    seconds = int(end - start)
    minutes = seconds // 60
    remain_second = seconds % 60
    print(f"time consumed: {minutes}min{remain_second}s")


def ddim_test():
    ddim, model = create_ddim_and_unet(device=device)
    model.load_state_dict(torch.load("./pretrained/ddim_fmnist_eps_new.pth"))
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    final_sample = ddim.ddim_sample_loop(model, shape=imgs.shape, noise=imgs, progress=True)[0]
    draw_ori_and_recon_images32(imgs, final_sample)


def ddpm_generate():
    noise = torch.randn((64, 1, 28, 28), device=device)
    ddim, model = create_ddim_and_unet(device=device)
    model.load_state_dict(torch.load("./pretrained/ddim_fmnist_eps_new.pth"))
    generate_imgs = ddim.p_sample_loop(model=model, shape=noise.shape, progress=True)[0]
    draw_fashion_mnist_images16(generate_imgs)


def ddim_generate():
    noise = torch.randn((64, 1, 28, 28), device=device)
    ddim, model = create_ddim_and_unet(device=device)
    model.load_state_dict(torch.load("./pretrained/ddim_fmnist_eps_new.pth"))
    generate_imgs = ddim.ddim_sample_loop(model=model, shape=noise.shape, progress=True)[0]
    draw_fashion_mnist_images16(generate_imgs)


if __name__ == "__main__":
    # train(5)
    ddim_test()
    # ddim_generate()



