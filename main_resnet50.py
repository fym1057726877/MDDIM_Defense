import torch
import time
from torch import nn, optim
from dataset.fashion_mnist import get_fashion_mnist_dataloader, transform_adv
from utils import batch_list, get_project_path, draw_fashion_mnist_images32
from classifier.resnet import resnet50
from tqdm import tqdm

root_dir = get_project_path()
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, test_loader = get_fashion_mnist_dataloader(batch_size=batch_size, shuffle=True)


def train(epochs):
    start = time.time()

    model = resnet50(num_classes=10, pretrained=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    accs = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = len(train_loader)
        for batch_idx, (imgs, labels) in tqdm(enumerate(train_loader), desc=f"train step {epoch+1}/{epochs}", total=batch_count):
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        # evaluate
        model.eval()
        epoch_loss /= batch_count
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='test step', total=len(test_loader)):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                predict = torch.max(out.data, dim=1)[1]
                total += labels.shape[0]
                correct += (predict == labels).sum()
        accuracy = correct / total
        accs.append(accuracy)
        print(f"epoch:{epoch+1}/{epochs}, loss:{epoch_loss}, test_accuracy:{accuracy}")
        torch.save(model.state_dict(), "./pretrained/resnet50_fashionmnist_classifier.pth")
    end = time.time()
    total_time = end - start
    seconds = total_time % 60
    minutes = total_time // 60
    print(f"train epochs:{epochs}\ntime consumed:{minutes}min{seconds}s")

    best_epoch = max(enumerate(accs), key=lambda x: x[1])[0] + 1
    best_accuracy = max(enumerate(accs), key=lambda x: x[1])[1]
    print(f"the best accuracy is {best_accuracy} at epoch {best_epoch}")


def clean_test():
    model = resnet50(num_classes=10, pretrained=True)
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    for batch_idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='test step', total=len(test_loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        predict = torch.max(out.data, dim=1)[1]
        total += labels.shape[0]
        correct += (predict == labels).sum()
    accuracy = correct / total
    print(f"test_accuracy:{accuracy}")


def adv_test():
    model = resnet50(pretrained=True)
    model = model.to(device)
    test_adv_imgs = torch.load(root_dir + "/dataset/attacked/fmnist/test/fgsm_eps_0.03_resnet50/images.pth")
    test_adv_labs = torch.load(root_dir + "/dataset/attacked/fmnist/test/fgsm_eps_0.03_resnet50/labels.pth")
    test_adv_imgs, test_adv_labs = batch_list(test_adv_imgs, batch_size), batch_list(test_adv_labs, batch_size)
    model.eval()
    adv_correct = 0
    total = 0
    clean_correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(zip(test_adv_imgs, test_adv_labs), desc='adv test step', total=len(test_adv_imgs)):
            imgs = transform_adv(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            predict = torch.max(out.data, dim=1)[1]
            adv_correct += (predict == labels).sum()

        for batch_idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='clean test step', total=len(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            predict = torch.max(out.data, dim=1)[1]
            total += labels.shape[0]
            clean_correct += (predict == labels).sum()

        adv_accuracy = adv_correct / total
        clean_accuracy = clean_correct / total
        # accs.append(accuracy)
        print(f"------------------------------------\n"
              f"clean_accuracy:{clean_accuracy}\n"
              f"adv_accuracy:{adv_accuracy}\n"
              f"------------------------------------")


if __name__ == '__main__':
    # train(5)
    adv_test()

