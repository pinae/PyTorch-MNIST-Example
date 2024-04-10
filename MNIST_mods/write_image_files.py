from torchvision.datasets import MNIST
from os import path, mkdir
import json

if __name__ == '__main__':
    train_data = MNIST(
        root=path.join("..", "data"),
        train=True,
        download=True)
    test_data = MNIST(
        root=path.join("..", "data"),
        train=False,
        download=True)
    if not path.exists(path.join("..", "data")):
        mkdir(path.join("..", "data"))
    if not path.exists(path.join("..", "data", "MNIST_image_files")):
        mkdir(path.join("..", "data", "MNIST_image_files"))
    if not path.exists(path.join("..", "data", "MNIST_image_files", "train")):
        mkdir(path.join("..", "data", "MNIST_image_files", "train"))
    if not path.exists(path.join("..", "data", "MNIST_image_files", "test")):
        mkdir(path.join("..", "data", "MNIST_image_files", "test"))
    train_labels = {}
    for img_no, (img, label) in enumerate(train_data):
        img.save(path.join("..", "data", "MNIST_image_files", "train", f"{img_no:05d}.png"), "PNG")
        train_labels[img_no] = label
    with open(path.join("..", "data", "MNIST_image_files", "train", "labels.json"), "w") as f:
        json.dump(train_labels, f)
    test_labels = {}
    for img_no, (img, label) in enumerate(test_data):
        img.save(path.join("..", "data", "MNIST_image_files", "test", f"{img_no:05d}.png"), "PNG")
        test_labels[img_no] = label
    with open(path.join("..", "data", "MNIST_image_files", "test", "labels.json"), "w") as f:
        json.dump(test_labels, f)
