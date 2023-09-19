import torch
import os
from matplotlib import pyplot as plt


def get_project_path(project_name='MDDIM_Defense'):
    """
    :param project_name: 项目名称，如pythonProject
    :return: ******/project_name
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(project_name)] + project_name


def draw_fashion_mnist_images16(images, labels=None):
    label_dict = {
        0: "T-shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    images = images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(4, 4)
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            idx = i * 4 + j
            ax.imshow(images[idx], cmap="gray")
            if labels is not None:
                title = label_dict[int(labels[idx])]
                ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_fashion_mnist_images32(images, labels, predict):
    label_dict = {
        0: "T-shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    images = images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(4, 8)
    for i in range(4):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            k = i if i < 2 else i - 2
            idx = k * 4 + j
            ax.imshow(images[idx], cmap="gray")
            if i < 2:
                title = label_dict[int(labels[idx])]
            else:
                title = label_dict[int(predict[idx])]
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_ori_adv_recon_images48(images, adv_imgs, recon_images):
    images = images.cpu().squeeze(1).detach().numpy()
    adv_imgs = adv_imgs.cpu().squeeze(1).detach().numpy()
    recon_images = recon_images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(6, 8)
    for i in range(6):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            if i < 2:
                idx = i * 4 + j
                ax.imshow((images[idx] + 1) * 255 / 2, cmap="gray")
            elif 2 <= i < 4:
                idx = (i - 2) * 4 + j
                ax.imshow((adv_imgs[idx] + 1) * 255 / 2, cmap="gray")
            else:
                idx = (i - 4) * 4 + j
                ax.imshow((recon_images[idx] + 1) * 255 / 2, cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_ori_and_recon_images32(images, recon_images):
    images = images.cpu().squeeze(1).detach().numpy()
    recon_images = recon_images.cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(4, 8)
    for i in range(4):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            if i < 2:
                idx = i * 4 + j
                ax.imshow((images[idx] + 1) * 255 / 2, cmap="gray")
            else:
                idx = (i - 2) * 4 + j
                ax.imshow((recon_images[idx] + 1) * 255 / 2, cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_diffusion_process(diff_model, unet_model, imgs, num_col=12):
    B, C, H, W = imgs.shape
    assert H == W, "only support H==W images"
    generated_images = diff_model.sample(unet_model, image_size=H, channels=C, batch_size=B,
                                         img=imgs).cpu().detach().numpy()
    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(B, num_col)

    for n_row in range(B):
        for n_col in range(num_col):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (1000 // num_col) * n_col if n_col < num_col - 1 else -1
            img = generated_images[t_idx][n_row].reshape(H, W)
            f_ax.imshow((img + 1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_diffusion_results(diff_model, unet_model, imgs, batch_size=64):
    B, C, H, W = imgs.shape
    assert H == W, "only support H==W images"
    assert imgs.shape[0] == batch_size == 64, "only support batch_size=64"
    generated_images = diff_model.sample(unet_model, image_size=H, channels=C, batch_size=B, img=imgs)
    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(8, 8)
    imgs = generated_images[-1].reshape(8, 8, 28, 28).cpu().detach().numpy()
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_ddim_results(diff_model, unet_model, imgs):
    B, C, H, W = imgs.shape
    assert H == W, "only support H==W images"
    assert imgs.shape[0] == 64, "only support batch_size=64"
    final_sample = diff_model.ddim_sample_loop(unet_model, shape=imgs.shape, noise=imgs, progress=True)
    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(8, 8)
    imgs = final_sample.squeeze(1).reshape(8, 8, 28, 28).cpu().detach().numpy()
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    plt.tight_layout()
    plt.show()


# [(h, w), (h, w), ……, (h, w)] -> [(batch_size, h, w), (batch_size, h, w), ……, (batch_size, h, w)]
def batch_list(src_list: list, batch_size: int) -> list:
    length = len(src_list)
    assert length >= batch_size
    assert isinstance(src_list[0], torch.Tensor)

    batch_num = (length + batch_size - 1) // batch_size

    target_list = []

    for bx in range(batch_num):
        start = bx * batch_size
        end = min((bx + 1) * batch_size, length)
        batch_tensor = src_list[start:end]
        batch = torch.stack(batch_tensor)
        target_list.append(batch)

    return target_list


if __name__ == '__main__':
    print(get_project_path())
