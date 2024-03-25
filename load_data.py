from torchvision.datasets import MNIST
from MNIST_mods.confusing_MNIST import Confusing_MNIST
from torchvision.transforms import ToTensor


def training_data():
    return MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )


def test_data():
    return MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )


def confusing_training_data():
    return Confusing_MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        match_ratio=0.5
    )


def confusing_test_data():
    return Confusing_MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        match_ratio=0.5
    )
