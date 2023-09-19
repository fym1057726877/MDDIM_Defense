from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils import get_project_path

__all__ = ["transform_train", "transform_test", "get_mnist_dataloader"]

transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_mnist_dataloader(batch_size=32, shuffle=True):
    path = get_project_path() + "/dataset/mnist"
    traindata = datasets.MNIST(root=path, train=True, transform=transform_train, download=True)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle)
    testdata = datasets.MNIST(root=path, train=False, transform=transform_test, download=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader


if __name__ == '__main__':
    train, test = get_mnist_dataloader(batch_size=64, shuffle=True)