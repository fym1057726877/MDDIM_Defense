from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils import get_project_path

__all__ = ["transform_train", "transform_test", "transform_adv", "get_fashion_mnist_dataloader"]

transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

transform_adv = transforms.Compose([
    transforms.Normalize((0.2860,), (0.3530,)),
])


def get_fashion_mnist_dataloader(batch_size=64, shuffle=True):
    """
    1.T-shirt / top（T恤）
    2.Trouser（裤子）
    3.Pullover（套衫）
    4.Dress（连衣裙）
    5.Coat（外套）
    6.Sandal（凉鞋）
    7.Shirt（衬衫）
    8.Sneaker（运动鞋）
    9.Bag（包）
    10.Ankle boot（短靴）
    """
    path = get_project_path() + "/dataset/fashionmnist"
    traindata = datasets.FashionMNIST(root=path, train=True, transform=transform_train, download=True)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle)
    testdata = datasets.FashionMNIST(root=path, train=False, transform=transform_test, download=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


if __name__ == '__main__':
    train, test = get_fashion_mnist_dataloader(batch_size=64, shuffle=True)