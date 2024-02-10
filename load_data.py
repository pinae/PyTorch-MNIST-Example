from torchvision import datasets
from torchvision.transforms import ToTensor


def training_data():
    return datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )


def test_data():
    return datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
