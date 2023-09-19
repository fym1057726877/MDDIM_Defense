import torch
from dataset.fashion_mnist import get_fashion_mnist_dataloader, transform_adv
from utils import batch_list, get_project_path, draw_ori_and_recon_images32, draw_ori_adv_recon_images48
from classifier.resnet import resnet50
from tqdm import tqdm
from models.ddim_munet import create_ddim_and_unet

root_dir = get_project_path()
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, test_loader = get_fashion_mnist_dataloader(batch_size=batch_size, shuffle=True)


def adv_recon_test():
    ddim, unetmodel = create_ddim_and_unet(device=device)
    unetmodel.load_state_dict(torch.load("./pretrained/mddim_fmnist.pth"))
    classifier = resnet50(num_classes=10, pretrained=True)
    classifier = classifier.to(device)

    test_adv_imgs = torch.load(root_dir + "/dataset/attacked/fmnist/test/fgsm_eps_0.03_resnet50/images.pth")
    test_adv_labs = torch.load(root_dir + "/dataset/attacked/fmnist/test/fgsm_eps_0.03_resnet50/labels.pth")
    test_adv_imgs, test_adv_labs = batch_list(test_adv_imgs, batch_size), batch_list(test_adv_labs, batch_size)

    unetmodel.eval()
    classifier.eval()

    adv_correct = 0
    total = 0
    clean_correct = 0
    adv_recon_correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(zip(test_adv_imgs, test_adv_labs), desc='adv test step', total=len(test_adv_imgs)):
            imgs = transform_adv(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            out = classifier(imgs)
            predict = torch.max(out.data, dim=1)[1]
            adv_correct += (predict == labels).sum()

        for batch_idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='clean test step', total=len(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            out = classifier(imgs)
            predict = torch.max(out.data, dim=1)[1]
            total += labels.shape[0]
            clean_correct += (predict == labels).sum()

        for imgs, labels in tqdm(zip(test_adv_imgs, test_adv_labs), desc='adv recon test step',
                                 total=len(test_adv_imgs)):
            imgs = transform_adv(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            recon_imgs = ddim.ddim_sample_loop(unetmodel, shape=imgs.shape, noise=imgs, progress=False)
            recon_imgs = transform_adv(recon_imgs)
            out = classifier(recon_imgs)
            predict = torch.max(out.data, dim=1)[1]
            adv_recon_correct += (predict == labels).sum()

        adv_accuracy = adv_correct / total
        clean_accuracy = clean_correct / total
        adv_recon_accuracy = adv_recon_correct / total
        print(f"------------------------------------\n"
              f"clean_accuracy:{clean_accuracy}\n"
              f"adv_accuracy:{adv_accuracy}\n"
              f"adv_recon_accuracy:{adv_recon_accuracy}\n"
              f"------------------------------------")


def show_recon_img():
    ddim, unetmodel = create_ddim_and_unet(device=device)
    unetmodel.load_state_dict(torch.load("./pretrained/mddim_fmnist.pth"))

    classifier = resnet50(num_classes=10, pretrained=True)
    classifier = classifier.to(device)

    test_adv_imgs = torch.load(root_dir + "/dataset/attacked/fmnist/test/fgsm_eps_0.03_resnet50/images.pth")
    test_adv_labs = torch.load(root_dir + "/dataset/attacked/fmnist/test/fgsm_eps_0.03_resnet50/labels.pth")
    test_adv_imgs, test_adv_labs = batch_list(test_adv_imgs, batch_size), batch_list(test_adv_labs, batch_size)

    it = iter(test_loader)
    ori_imgs, ori_labs = next(it)
    unetmodel.eval()
    classifier.eval()
    imgs_adv, labs_adv = test_adv_imgs[0].to(device), test_adv_labs[0].to(device)
    imgs_adv = transform_adv(imgs_adv)
    ori_imgs, ori_labs = ori_imgs.to(device), ori_labs.to(device)

    out_ori = classifier(ori_imgs)
    predict_ori = torch.max(out_ori.data, dim=1)[1]
    correct_ori = (predict_ori == ori_labs).sum()

    out_adv = classifier(imgs_adv)
    predict_adv = torch.max(out_adv.data, dim=1)[1]
    correct_adv = (predict_adv == labs_adv).sum()

    recon_imgs = ddim.ddim_sample_loop(unetmodel, shape=imgs_adv.shape, noise=imgs_adv, progress=False)

    out_recon = classifier(recon_imgs)
    predict_recon = torch.max(out_recon.data, dim=1)[1]
    correct_recon = (predict_recon == labs_adv).sum()

    print(f"acc_adv_imgs:{correct_ori}/{batch_size}\n"
          f"acc_adv_imgs:{correct_adv}/{batch_size}\n"
          f"acc_adv_recon_imgs:{correct_recon}/{batch_size}")

    draw_ori_adv_recon_images48(ori_imgs, imgs_adv, recon_imgs)


if __name__ == '__main__':
    # adv_recon_test()
    show_recon_img()
