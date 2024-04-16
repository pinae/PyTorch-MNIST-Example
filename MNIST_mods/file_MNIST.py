from typing import Optional, Callable, Tuple, Any
from torch.utils.data import Dataset
from os import path
from PIL import Image
from json import load


class MNIST(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform
        self.folder = path.join(root, 'MNIST_image_files', 'train' if train else 'test')
        if not path.exists(path.join(self.folder, "labels.json")):
            raise FileExistsError(f"The file {path.join(self.folder, 'labels.json')} needs to exist.")
        with open(path.join(self.folder, "labels.json"), 'r') as f:
            json_data = load(f)
            self.labels = dict((int(key), value) for key, value in json_data.items())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if not path.exists(path.join(self.folder, f"{index:05d}.png")):
            raise FileExistsError("The file " + path.join(self.folder, f"{index:05d}.png") + " is missing.")
        with Image.open(path.join(self.folder, f"{index:05d}.png")) as img:
            target = self.labels[index]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self) -> int:
        return len(self.labels)
